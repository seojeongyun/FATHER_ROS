def dataset_info(dataset_name='demo'):
    if dataset_name == 'demo':
        #
        train_path = "./demo_ims/"
        cache_file = "./cache/demo_train_img.pickle"
        num_classes = 10

    elif dataset_name == 'VGGFace2':
        #
        # train_path = "/storage/sjpark/VGGFace2/train"
        train_path = "/storage/sjpark/VGGFace2/vgg_train_6s_hs_CBAM_9601"
        #
        # cache_file = "/storage/hrlee/vggface2/cache/vgg_train_6s_hs_CBAM_9601_2593_10.pickle"
        # cache_file = "/storage/hrlee/vggface2/cache/vgg_train_6s_hs_CBAM_9601_2593_100.pickle"
        # cache_file = "/storage/hrlee/vggface2/cache/vgg_train_6s_hs_CBAM_9601_2593_300.pickle"
        cache_file = "/storage/hrlee/vggface2/cache/vgg_train_6s_hs_CBAM_9601_2593_500.pickle"
        # cache_file = "/storage/hrlee/vggface2/cache/train_2593_100.pickle"
        #
        # num_classes = 777
        # num_classes = 2074
        num_classes = 2593

    elif dataset_name == 'total_data':
        train_path = "/storage/sjpark/total_data/train"
        cache_file = "/storage/sjpark/total_data/cache/total_train_img.pickle"
        num_classes = 79259

    else:
        NotImplementedError('Not Implemented...{}'.format(dataset_name))
    #
    gallery_path = "/storage/hrlee/groupface/demo_eval/gallery/"
    probe_path = "/storage/hrlee/groupface/demo_eval/probe/"

    return train_path, cache_file, gallery_path, probe_path, num_classes


class Config(object):
    epoch = 100
    #
    dataset_name = 'VGGFace2'  # [VGGFace2 | total_data]
    train_path, cache_file, gallery_path, probe_path, num_classes = dataset_info(dataset_name)
    img_size = 224
    #
    batch_size = 32
    #
    resnet = 50  # [18, 50, 101]
    feature_dim = 1024
    groups = 5
    #
    num_class = cache_file.split('.')[:-1][0].split('_')[-2]
    img_file_per_folder = cache_file.split('.')[:-1][0].split('_')[-1]
    # log_dir_name = f'Resnet{resnet}_groups{groups}_feadim_{feature_dim}_img_{img_size}_vggface2_{num_class}_{img_file_per_folder}'
    log_dir_name = f'Caps_groups{groups}_feadim_{feature_dim}_img_{img_size}_vggface2_{num_class}_{img_file_per_folder}'
    #
    loss = 'focal_loss'
    fc_metric = 'arc'  # [arc]
    easy_margin = False
    #
    optimizer = 'adam'
    scheduler = 'cyclelr'  # [cosine | constant | cyclelr | steplr]
    lr = 1e-4  # initial learning rate
    lr_base = 1e-5
    lr_max = 0.5e-7
    lr_gamma = 0.9
    lrf = 1e-4
    T_up = 10
    T_down = 10
    weight_decay = 5e-4
    #
    save_checkpoints_every_epoch = True
    #
    checkpoints_save_path = "./checkpoints/"
    checkpoints_file = None
    # checkpoints_file = "vggface2_yolo_checkpoints/caps_primdim48_preddim_64_512_64/best_rescaps50_group5_featdim1024_top1_0_891_224_2593_500.pth"
    #
    use_gpu = True  # use GPU or nots
    gpu_id = '0'
    num_workers = 10  # how many workers for loading data
    print_freq = 100  # print info every N batch
    eval_interval = 25000 * 4
    num_thres = 120
    num_up_down = 50
