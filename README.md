python train.py --model resnet50 --max-samples 10000 --epochs 10 --phase1-epochs 3 --workers 0
python evaluate.py --model resnet50 --checkpoint models/resnet50_best.pt --workers 0
python train.py --model efficientnet_b0 --max-samples 10000 --epochs 10 --phase1-epochs 3 --workers 0
python evaluate.py --model efficientnet_b0 --checkpoint models/efficientnet_b0_best.pt --workers 0
python gradcam.py --model resnet50 --checkpoint models/resnet50_best.pt --workers 0
