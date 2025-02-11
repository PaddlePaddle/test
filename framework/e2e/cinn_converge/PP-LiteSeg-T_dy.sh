python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=train -o Global.dataset_dir=../cityscapes \
    -o Train.num_classes=19 -o Train.epochs_iters=160000 -o Train.batch_size=2 \
    -o Train.warmup_steps=1000 -o Train.learning_rate=0.005 \
    -o Global.device=gpu:0,1,2,3,4,5,6,7 -o Global.output='./output/semantic_segmentation/PP-LiteSeg-T_dy'