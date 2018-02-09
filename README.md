# MobileNetv2

The caffe inference prototxt for mobilenetv2.  
I will share the trained model soon.

Amazing happens when I distill the small mobilenetv2; The top1/top5 of the small mobilenetv2 is state of the art.

Training details for ImageNet2012 :
                                   type: "SGD"
                                   lr_policy: "poly"
                                   base_lr: 0.045
                                   power: 1
                                   momentum: 0.9
                                   weight_decay: 0.00004

