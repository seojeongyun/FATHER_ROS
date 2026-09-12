import pickle
from tqdm import tqdm
import os
import sys


def make_cache_file(max_data_per_folder, file_path, cache_pth, num_class):
    file_paths = []
    file_IDs = []
    file_labels = []
    IDs = []
    IDsLabels = {}

    print(f"Check IDs and Labels of IDs from {file_path}.")
    max_num_class = num_class if isinstance(num_class, int) else int(len(os.listdir(file_path)) * num_class)
    label_idx = 0
    for dir in tqdm(os.listdir(file_path), total=len(os.listdir(file_path)), leave=True):
        if os.path.isdir(os.path.join(file_path, dir)) is False:
            raise print("DIR Error")
        #
        cnt_data = 0
        for file in os.listdir(os.path.join(file_path, dir)):
            if os.path.splitext(file)[1] in [".jpg", ".bmp", ".png"]:
                file_paths.append(os.path.join(file_path, dir, file))
                file_IDs.append(dir)
                file_labels.append(label_idx)
                #
                cnt_data += 1
                if cnt_data >= max_data_per_folder:
                    break
        label_idx += 1
        IDsLabels[dir] = label_idx
        IDs.append(dir)
        if label_idx >= max_num_class:
            break
    print('=' * 5)
    print(f'Num. of Cached Folders: {label_idx}')
    print(f'Max Num. of Cached Image Files per a Cached Folder: {cnt_data}')
    print('=' * 5)
    print("data set loaded from scratch len: {}".format(len(file_paths)))
    sys.stdout.flush()
    #
    if not os.path.exists(cache_pth):
        os.makedirs(cache_pth)
    #
    cache_pth = os.path.join(cache_pth, file_path.split('/')[-1] + f'_{max_num_class}_{max_data_per_folder}.pickle')
    with open(cache_pth, "wb") as f:
        pickle.dump(
            [
                file_paths,
                file_IDs,
                file_labels,
                IDs,
                IDsLabels
            ], f)


if __name__ == '__main__':
    print('-*' * 5 + ' Start making a cache file having sample data ' + '-*' * 5)
    make_cache_file(max_data_per_folder=800,
                    # file_path='/storage/sjpark/VGGFace2/train',
                    file_path='/storage/sjpark/VGGFace2/vgg_train_6s_hs_CBAM_9601',
                    cache_pth='/storage/hrlee/vggface2/cache',
                    num_class=1.0)
    print('-*' * 5 + ' End ' + '-*' * 5)
