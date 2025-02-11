python main.py -c paddlex/configs/modules/small_object_detection/PP-YOLOE_plus_SOD-S.yaml \
    -o Global.mode=train -o Global.dataset_dir=../visdrone \
    -o Train.num_classes=10 -o Train.epochs_iters=80 -o Train.batch_size=8 \
    -o Train.learning_rate=0.005 -o Global.device=gpu:0,1,2,3,4,5,6,7
    -o Global.output='./output/small_object_detection/PP-YOLOE_plus_SOD-S0_dy'