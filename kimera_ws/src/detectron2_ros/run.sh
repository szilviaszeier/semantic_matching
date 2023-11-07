CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
    --config-file configs/mask_rcnn_R_101_FPN_3x_transfiner_deform.yaml \
    --input 'samples/habitat/*.png' \
    --output 'output/habitat/' \
    --opts MODEL.WEIGHTS ./pretrained_models/model_final.pth
