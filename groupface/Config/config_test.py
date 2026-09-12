def dataset_info(dataset_name='demo'):
    if dataset_name == 'demo':
        #
        train_path = "./demo_ims/"
        cache_file = "./cache/demo_train_img.pickle"
        num_classes = 10
    elif dataset_name =='VGGFace2':
        #
        # train_path = "/storage/sjpark/VGGFace2/train"
        # cache_file = "/storage/sjpark/VGGFace2/cache/vggface2_train_img.pickle"
        train_path = "/storage/sjpark/VGGFace2/vgg_train_6s_hs_CBAM_9601"
        cache_file = "/storage/sjpark/VGGFace2/cache/vggface2_train_img_6s_hs_CBAM_9601.pickle"
        num_classes = 2593
    else:
        NotImplementedError('Not Implemented...{}'.format(dataset_name))
    #
    # test_path = "/storage/sjpark/VGGFace2/test"
    test_path = "/storage/sjpark/VGGFace2/vgg_test_6s_hs_CBAM_9601"

    return train_path, cache_file, test_path, num_classes


class Config(object):
    epoch = 100
    #
    dataset_name = 'VGGFace2'  # [VGGFace2 | total_data]
    train_path, cache_file, test_path,  num_classes = dataset_info(dataset_name)
    save_path = './checkpoints/'
    img_size = 224
    #
    batch_size = 32
    #
    resnet = 50  # [18, 50, 101]
    feature_dim = 1024
    groups = 5
    #
    loss = 'focal_loss'
    fc_metric = 'arc'  # [arc | sphere | add]
    easy_margin = False
    #
    save_checkpoints_every_epoch = False
    #
    checkpoints_save_path = "/home/sjpark/PycharmProjects/groupface-main/checkpoints"
    checkpoints_file = "total_data_vgg+digi_256_test_opt=adam,lr=1e-4_8_ num_workers = 8 .pth"
    #
    use_gpu = True  # use GPU or not
    gpu_id = "0"
    num_workers = 5  # how many workers for loading data
    print_freq = 100  # print info every N batch
    eval_freq = 100
    num_thres = 120

# 86 -> 0.33, 85 -> 0.34, 84 -> ,83 ->
