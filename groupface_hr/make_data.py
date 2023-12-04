import os
import shutil


def VGGFace():  ### training dataset
    dir_path = "/storage/sjpark/VGGFace2/train"
    file_list = os.listdir(dir_path)
    print(len(file_list))
    num = 0
    for i in range(len(file_list)):
        num += len(os.listdir("/storage/sjpark/VGGFace2/train" + "/" + file_list[i]))
    print(num)


def digiface():
    num_path = 1
    num_path2 = 0
    while num_path != 9:
        dir_path = "/storage/sjpark/digi_data/digi_data" + str(num_path)
        num = 0
        x = os.listdir(dir_path)
        for j in range(len(x)):
            num += len(os.listdir("/storage/sjpark/digi_data/digi_data" + str(num_path) + "/" + x[j]))
            num_path2 += 1
        num_path += 1
    print(num)


def total_data():
    dir_path = "/storage/sjpark/total_data/train"
    num = 2782
    file_list = os.listdir(dir_path)
    file_list.sort()
    for i in range(len(file_list)):
        if file_list[i][0] != 'n':
            if num >= 10000:
                source_path = os.path.join("/storage/sjpark/total_data/train/", file_list[i])
                new_name = "n0" + str(num)
                dst_path = os.path.join("/storage/sjpark/total_data/train/", new_name)
                os.rename("/storage/sjpark/total_data/train/" + file_list[i], dst_path)
                num += 1
            else:
                source_path = os.path.join("/storage/sjpark/total_data/train/", file_list[i])
                new_name = "n00" + str(num)
                dst_path = os.path.join("/storage/sjpark/total_data/train/", new_name)
                os.rename("/storage/sjpark/total_data/train/" + file_list[i], dst_path)
                num += 1


def copy_and_rename_directories(source_directory, destination_directory):
    # 원본 디렉토리 목록 가져오기
    directories = os.listdir(destination_directory)

    # 원본 디렉토리 이름을 변경하기 위한 반복문
    for directory in directories:
        source_path = os.path.join(destination_directory, directory)
        new_name = directory + "_new"  # 여기에 원하는 새로운 이름을 정의해주세요.
        destination_path = os.path.join(destination_directory, new_name)
        os.rename(source_path, destination_path)


if __name__ == '__main__':
    # digiface()
    # os.makedirs("/storage/sjpark/total_data")
    # target_path = "/storage/sjpark/VGGFace2/train"
    # for i in range(1,9):
    #     target_path2 = "/storage/sjpark/digi_data/digi_data" + str(i) +"/"
    #     pik = os.listdir(target_path2)
    #     for j in range(len(pik)):
    #         target_path2 = "/storage/sjpark/digi_data/digi_data" + str(i) +"/" + str(pik[j])
    # destination_path = "/storage/sjpark/total_data/train"
    # shutil.copytree(target_path, destination_path)
    VGGFace()
    # digiface()
    # total_data()
