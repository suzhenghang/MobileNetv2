/*
* Integrate imgResize and blobAlign python layers together.
*
* Author: Wei Zhen @ IIE, CAS
* Last modified: 2017-06-11
*/

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/resize_layer.hpp"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace caffe {

template <typename Dtype>
__global__ void nearestForwardGPU(const int nthreads,
	const int input_channels, const int input_size, const int output_size, const float resize_factor,
	const Dtype* bottom_data, Dtype* top_data) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		// resume channel idx and pixel idx
		int rw = index % output_size;
		int rh = (index / output_size) % output_size;
		int c = (index / output_size / output_size) % input_channels;
		int n = index / output_size / output_size / input_channels;
		// indexing and sampling
		int h = int(rh / resize_factor);
		int w = int(rw / resize_factor);
		h = min(h, input_size);
		w = min(w, input_size);
		const Dtype* bottom_data_channel = bottom_data + (n * input_channels + c) * input_size * input_size;
		top_data[index] = bottom_data_channel[h * input_size + w];
	}
}

template <typename Dtype>
__global__  void bilinearForwardGPU(const int nthreads,
	const int input_channels, const int input_size, const int output_size, const float resize_factor,
	const Dtype *bottom_data, Dtype *top_data) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		// resume channel idx and pixel idx
		int rw = index % output_size;
		int rh = (index / output_size) % output_size;
		int c = (index / output_size / output_size) % input_channels;
		int num = index / output_size / output_size / input_channels;
		// indexing and sampling
		float h = rh / resize_factor;
		float w = rw / resize_factor;
		//h = min(h, float(input_size));
		//w = min(w, float(input_size));
		const Dtype* bottom_data_channel = bottom_data + (num * input_channels + c) * input_size * input_size;
		top_data[index] = 0;		// DO NOT forget to reset before accumulation
		for (int n = MAX(floor(h-1) + 1, 0); n < MIN(h + 1, input_size); n++) {
		    for (int m = MAX(floor(w-1) + 1, 0); m < MIN(w + 1, input_size); m++) {
			top_data[index] += bottom_data_channel[n * input_size + m] * (1 - abs(w - m)) * (1 - abs(h - n));
		    }
		}
	}
}

template <typename Dtype>
void ResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  // perform interpolation based on different interpolation types
  switch (this->layer_param_.resize_param().intepolation_type()) {
	case SegResizeParameter_InterpolationType_NEAREST:
		// parallel at pixel level
		nearestForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
			(count, bottom[0]->channels(), this->input_size_, this->output_size_,
			 this->resize_factor_, bottom_data, top_data);
		break;
	case SegResizeParameter_InterpolationType_BILINEAR:
		// parallel at pixel level
		bilinearForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
			(count, bottom[0]->channels(), this->input_size_, this->output_size_,
			 this->resize_factor_, bottom_data, top_data);
		break;
	default:
		LOG(FATAL) << "Unknown interpolation type.";
  }
}

template <typename Dtype>
__global__ void nearestBackwardGPU(const int nthreads,
	const int input_channels, const int input_size, const int output_size, const float resize_factor,
	Dtype* bottom_diff, const Dtype* top_diff) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		// resume channel idx and pixel idx
		int w = index % input_size;
		int h = (index / input_size) % input_size;
		int c = (index / input_size / input_size) % input_channels;
		int n = index / input_size / input_size / input_channels;
		// indexing and sampling
		int rh = int(h * resize_factor);
		int rw = int(w * resize_factor);
		rh = min(rh, output_size - 1);
		rw = min(rw, output_size - 1);
		top_diff += (n*input_channels + c) * output_size * output_size;
		bottom_diff[index] = top_diff[rh * output_size + rw];
	}
}

template <typename Dtype>
__global__ void bilinearBackwardGPU(const int nthreads,
	const int input_channels, const int input_size, const int output_size, const float resize_factor,
	Dtype* bottom_diff, const Dtype* top_diff) {

	/*CUDA_KERNEL_LOOP(index, nthreads) {
		// resume channel idx
		int c = index % input_channels;
		int n = index / input_channels;

		// indexing and sampling
		for (int rh = 0; rh < output_size; ++rh){
		    for (int rw = 0; rw < output_size; ++rw){
			float h = rh / resize_factor;
			float w = rw / resize_factor;
			int top_idx = ((n*input_channels + c) * output_size + rh ) * output_size + rw;
			Dtype* bottom_diff_channel = bottom_diff + (n*input_channels + c) * input_size * input_size;

			int h0 = max(int(h), 0);
			int w0 = max(int(w), 0);
			int h1 = 0;
			float weight_h_h1 = 0;

			float weight_w_w0 = max(1 - abs(w - w0), 0.);
			float weight_h_h0 = max(1 - abs(h - h0), 0.);
			if (h < input_size-1) {				// h1 does not exceed input_size
				h1 = min(int(h)+1, input_size - 1);
				weight_h_h1 = max(1 - std::abs(h - h1), float (0));
				if (w0 == 0) {				// the first column
					bottom_diff_channel[h1 * input_size + w0] += top_diff[top_idx] * weight_w_w0 * weight_h_h1 * 2;
				} else {
					bottom_diff_channel[h1 * input_size + w0] += top_diff[top_idx] * weight_w_w0 * weight_h_h1;
				}
			}
			if (w < input_size-1) {				// w1 does not exceed input_size
				int w1 = min(int(w)+1, input_size - 1);
				float weight_w_w1 = max(1 - std::abs(w - w1), float (0));
				if (h0 == 0) {				// the first row
					bottom_diff_channel[h0 * input_size + w1] += top_diff[top_idx] * weight_w_w1 * weight_h_h0 * 2;
				} else {
					bottom_diff_channel[h0 * input_size + w1] += top_diff[top_idx] * weight_w_w1 * weight_h_h0;
				}
				if (h < input_size-1) {			// for the very left-bottom one
					bottom_diff_channel[h1 * input_size + w1] += top_diff[top_idx] * weight_w_w1 * weight_h_h1;
				}
			}
		    }
		}
		// the normalization is outside this function
	}*/
	CUDA_KERNEL_LOOP(index, nthreads) {
		// resume channel idx and pixel idx
		int w = index % input_size;
		int h = (index / input_size) % input_size;
		int c = (index / input_size / input_size) % input_channels;
		int n = index / input_size / input_size / input_channels;
		// indexing and sampling
		float hstart = (h - 1) * resize_factor + 1;
		float wstart = (w - 1) * resize_factor + 1;
		float hend = (h + 1) * resize_factor;
		float wend = (w + 1) * resize_factor;
		hstart = max(hstart, 0.);
		wstart = max(wstart, 0.);
		hend = min(hend, float(output_size));
		wend = min(wend, float(output_size));
		const Dtype* top_diff_channel = top_diff + (n*input_channels + c) * output_size * output_size;
		bottom_diff[index] = 0;
		for (int rh = hstart; rh < hend; ++rh) {
		    for (int rw = wstart; rw < wend; ++rw) {
			bottom_diff[index] += top_diff_channel[rh * output_size + rw]
                                          * (1 - abs((rw / resize_factor) - w)) * (1 - abs((rh / resize_factor) - h));
		    }
		}
		bottom_diff[index] /= resize_factor*resize_factor;
	}
}

template <typename Dtype>
void ResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {    return;  }

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int bottom_count = bottom[0]->count();
  //caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  //const int top_count = top[0]->count();
  int nthreads = top[0]->num() * top[0]->channels();

  // data gradient
  switch (this->layer_param_.resize_param().intepolation_type()) {
	case SegResizeParameter_InterpolationType_NEAREST:
		// parallel at pixel level
		nearestBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>
			(bottom_count, bottom[0]->channels(), this->input_size_, this->output_size_,
			 this->resize_factor_, bottom_diff, top_diff);
		break;
	case SegResizeParameter_InterpolationType_BILINEAR:
		// parallel at channel level
		bilinearBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>
			(bottom_count, bottom[0]->channels(), this->input_size_, this->output_size_,
			 this->resize_factor_, bottom_diff, top_diff);
		// normalize
		//caffe_gpu_scal(bottom_count, this->resize_factor_>1 ? 1/this->resize_factor_/this->resize_factor_ : 1, bottom_diff);
		break;
	default:
		LOG(FATAL) << "Unknown interpolation type.";
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);

}// namespace caffe
