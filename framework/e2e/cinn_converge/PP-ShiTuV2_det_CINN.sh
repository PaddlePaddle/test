python main.py -c paddlex/configs/modules/mainbody_detection/PP-ShiTuV2_det.yaml \
    -o Global.mode=train -o Global.dataset_dir=../mainbody \
    -o Train.num_classes=1 -o Train.epochs_iters=100 -o Train.batch_size=32 \
    -o Train.learning_rate=0.32 -o Global.device=gpu:0,1,2,3,4,5,6,7 -o Global.output='./output/mainbody_detection/PP-ShiTuV2_det_CINN' -o Train.dy2st=True