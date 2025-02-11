python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml \
    -o Global.mode=train -o Global.dataset_dir=../DOTA-v1.0 \
    -o Train.num_classes=10 -o Train.epochs_iters=36 -o Train.batch_size=1 \
    -o Train.warmup_steps=1000 -o Train.learning_rate=0.008 \
    -o Global.device=gpu:0,1,2,3,4,5,6,7 \
    -o Global.output='./output/rotated_object_detection/PP-YOLOE-R-L_CINN' \
    -o Train.dy2st=True