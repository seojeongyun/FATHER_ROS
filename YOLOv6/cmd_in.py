import argparse

import sys
import os

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Training', add_help=add_help)

    parser.add_argument('--data-path', default='./data/WIDER_FACE.yaml', type=str, help='path of dataset')
    parser.add_argument('--conf-file', default='./configs/repopt/yolov6_lite_s.py', type=str, help='experiments description file')
    # parser.add_argument('--conf-file', default='./configs/face/yolov6l_finetune.py', type=str,
    #                     help='experiments description file')
    parser.add_argument('--img-size', default=360, type=int, help='train, val image size (pixels)')
    parser.add_argument('--batch-size', default=4, type=int, help='total batch size for all GPUs')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--device', default='1', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--eval-interval', default=1, type=int, help='evaluate at every interval epochs')
    parser.add_argument('--eval-final-only', action='store_true', help='only evaluate at the final epoch')
    parser.add_argument('--heavy-eval-range', default=50, type=int,
                        help='evaluating every epoch for last such epochs (can be jointly used with --eval-interval)')
    parser.add_argument('--check-images', action='store_true', help='check images when initializing datasets')
    parser.add_argument('--check-labels', action='store_true', help='check label files when initializing datasets')
    # parser.add_argument('--check-images', type=bool, default=False, help='check images when initializing datasets')
    # parser.add_argument('--check-labels', type=bool, default=False, help='check label files when initializing datasets')
    parser.add_argument('--output-dir', default='./runs/train', type=str, help='path to save outputs')
    parser.add_argument('--name', default='6s_opt', type=str, help='experiment name, saved to output_dir/name')
    parser.add_argument('--gpu_count', type=int, default=0)
    parser.add_argument('--gpu-id', type=str, default='1', help='None or gpu number')
    # parser.add_argument('--resume', nargs='?', const=True, default='./runs/train/ours_320_CW/weights/coco_150.pt', help='pretrained weight path')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='pretrained weight path')
    parser.add_argument('--write_trainbatch_tb', action='store_true',
                        help='write train_batch image to tensorboard once an epoch, may slightly slower train speed if open')
    parser.add_argument('--stop_aug_last_n_epoch', default=15, type=int,
                        help='stop strong aug at last n epoch, neg value not stop, default 15')
    parser.add_argument('--save_ckpt_on_last_n_epoch', default=-1, type=int,
                        help='save last n epoch even not best or last, neg value not save')
    # --- for debugging of hyperparameter search and training target network
    parser.add_argument('--distill', action='store_true', help='distill or not')
    parser.add_argument('--distill_feat', action='store_true', help='distill featmap or not')
    parser.add_argument('--quant', action='store_true', help='quant or not, for QAT')
    parser.add_argument('--calib', action='store_true', help='run ptq')
    parser.add_argument('--teacher_model_path', type=str, default=None, help='teacher model path')
    # --- for debugging of PTQ
    # parser.add_argument('--distill', action='store_true', help='distill or not')
    # parser.add_argument('--distill_feat', action='store_true', help='distill featmap or not')
    # parser.add_argument('--quant', type=bool, default=True, help='quant or not, for QAT')
    # parser.add_argument('--calib', type=bool, default=True, help='run ptq')
    # parser.add_argument('--teacher_model_path', type=str, default=None, help='teacher model path')
    # --- for debugging of QAT
    # parser.add_argument('--distill',  type=bool, default=True, help='distill or not')
    # parser.add_argument('--distill_feat', type=bool, default=True, help='distill featmap or not')
    # parser.add_argument('--quant', type=bool, default=True, help='quant or not, for QAT')
    # parser.add_argument('--calib', type=bool, default=False, help='run ptq')
    # parser.add_argument('--teacher_model_path', type=str, default='./weights/opt/yolov6s_opt.pt', help='teacher model path')
    # #
    parser.add_argument('--temperature', type=int, default=20, help='distill temperature')
    # parser.add_argument('--fuse_ab', action='store_true', help='fuse ab branch in training process or not')
    parser.add_argument('--fuse_ab', type=bool, default=False, help='fuse ab branch in training process or not')
    parser.add_argument('--height', type=int, default=None, help='image height of model input')
    parser.add_argument('--width', type=int, default=None, help='image width of model input')
    return parser