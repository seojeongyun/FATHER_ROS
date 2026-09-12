from tqdm import tqdm
import torch
import torch.nn as nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import tensor_quant
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from tools.partial_quantization.utils import set_module, module_quant_disable


# ======== QAT ========
def qat_init_model_manu(model, cfg, args):
    # Information of quantization
    # https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html#quantization-aware-training
    # tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL == QuantDescriptor(num_bits=8, calib_method='max')
    conv2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
    conv2d_input_default_desc = QuantDescriptor(num_bits=cfg.ptq.num_bits, calib_method=cfg.ptq.calib_method)

    convtrans2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL
    convtrans2d_input_default_desc = QuantDescriptor(num_bits=cfg.ptq.num_bits, calib_method=cfg.ptq.calib_method)

    for k, m in model.named_modules():
        if 'proj_conv' in k:
            print("Skip Layer {}".format(k))
            continue
        if args.calib is True and cfg.ptq.sensitive_layers_skip is True:
            if k in cfg.ptq.sensitive_layers_list:
                print("Skip Layer {}".format(k))
                continue

        if isinstance(m, nn.Conv2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_conv = quant_nn.QuantConv2d(in_channels, out_channels,
                                              kernel_size, stride, padding,
                                              quant_desc_input=conv2d_input_default_desc,
                                              quant_desc_weight=conv2d_weight_default_desc)
            quant_conv.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_conv.bias.data.copy_(m.bias.detach())
            else:
                quant_conv.bias = None
            set_module(model, k, quant_conv)

        elif isinstance(m, nn.ConvTranspose2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_convtrans = quant_nn.QuantConvTranspose2d(in_channels,
                                                            out_channels,
                                                            kernel_size,
                                                            stride,
                                                            padding,
                                                            quant_desc_input=convtrans2d_input_default_desc,
                                                            quant_desc_weight=convtrans2d_weight_default_desc)
            quant_convtrans.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_convtrans.bias.data.copy_(m.bias.detach())
            else:
                quant_convtrans.bias = None
            set_module(model, k, quant_convtrans)
        elif isinstance(m, nn.MaxPool2d):
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            dilation = m.dilation
            ceil_mode = m.ceil_mode
            quant_maxpool2d = quant_nn.QuantMaxPool2d(kernel_size,
                                                      stride,
                                                      padding,
                                                      dilation,
                                                      ceil_mode,
                                                      quant_desc_input=conv2d_input_default_desc)
            set_module(model, k, quant_maxpool2d)
        else:
            # module can not be quantized, continue
            continue


def skip_sensitive_layers(model, sensitive_layers):
    print('Skip sensitive layers...')
    for name, module in model.named_modules():
        if name in sensitive_layers:
            print(F"Disable {name}")
            module_quant_disable(model, name)


# ======== PTQ ========
def ptq_calibrate(model, train_loader, cfg):
    model.eval()
    model.cuda()
    with torch.no_grad():
        collect_stats(model, train_loader, cfg.ptq.calib_batches)
        # amax: abs().max()
        compute_amax(model, method=cfg.ptq.histogram_amax_method, percentile=cfg.ptq.histogram_amax_percentile)


def collect_stats(model, data_loader, num_batches):
    """ Feed data to the netework and collect statistic """

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()  # Use full precision data to calibrate
                module.enable_calib()
            else:
                module.disable()
    for i, (image, _, _, _) in tqdm(enumerate(data_loader), total=num_batches):
        image = image.float() / 255.0
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load Calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print(F"{name:40}: {module}")
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    # MinMaxCalib
                    module.load_calib_amax()
                else:
                    # HistogramCalib
                    module.load_calib_amax(**kwargs)
    model.cuda()


if __name__ == '__main__':
    from core_train import get_args_parser, check_and_init

    args = get_args_parser().parse_args()
    args.conf_file = '/home/hrlee/PycharmProjects/YOLOv6/configs/report/yolov6s_opt_qat.py'
    cfg, device, args = check_and_init(args)
    device = torch.device('cpu')

    from yolov6.utils.events import load_yaml

    data_dict = load_yaml('/home/hrlee/PycharmProjects/YOLOv6/data/coco.yaml')

    from yolov6.models.yolo import build_model

    distill_ns = True if args.distill and cfg.model.type in ['YOLOv6n', 'YOLOv6s'] else False
    model = build_model(cfg, data_dict['nc'], device, fuse_ab=args.fuse_ab, distill_ns=distill_ns)

    # Debugging Quantization related to QAT
    qat_init_model_manu(model, cfg, args)
    if args.calib is False:
        if cfg.qat.sensitive_layers_skip:
            skip_sensitive_layers(model, cfg.qat.sensitive_layers_list)
            # QAT flow load calibrated model
            assert cfg.qat.calib_pt is not None, 'Please provide calibrated model'
            model.load_state_dict(torch.load(cfg.qat.calib_pt)['model'].float().state_dict())
        model.to(device)

    # Debugging Quantization related to PTQ
    import glob
    from yolov6.data.delme.datasets import img2label_paths

    img_paths = glob.glob('/storage/hrlee/coco/images/train2014/' + '*.jpg')
    label_paths = img2label_paths(img_paths)
    # Data class and Loader
    from yolov6.utils.events import load_yaml

    path = data_dict['val']
    nc = int(data_dict['nc'])
    class_names = data_dict['names']
    assert len(class_names) == nc, f'the length of class_names should match the number of classes defined'
    grid_size = max(int(max(cfg.model.head.strides)), 32)

    img_dir = path
    img_size = 640
    batch_size = 16
    stride = 32
    hyp = dict(cfg.data_aug)
    augment = True
    workers = 0
    shuffle = True
    check_images = True
    check_labels = True
    task = 'train'

    from yolov6.data.data_load import create_dataloader

    loader = create_dataloader(
        path=img_dir,
        img_size=img_size,
        batch_size=batch_size,
        stride=grid_size,
        hyp=hyp,
        augment=augment,
        workers=8,
        shuffle=shuffle,
        check_images=check_images,
        check_labels=check_labels,
        data_dict=data_dict,
        task='train'
    )
    ptq_calibrate(model, loader, cfg)
