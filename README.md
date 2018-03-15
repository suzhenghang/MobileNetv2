# MobileNetv2

Run the script benchmark_mobilenetv2.sh and you can get the result: top1/top5: 0.7123/0.9018,
you can use this model to do a lot of things such as training a smaller mobilenetv2 (By moving params or knowledge distillation)

Training details for ImageNet2012 :
                                   type: "SGD"
                                   lr_policy: "poly"
                                   base_lr: 0.045
                                   power: 1
                                   momentum: 0.9
                                   weight_decay: 0.00004

