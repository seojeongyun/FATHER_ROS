import os


def VGGFace():  ### training dataset
    dir_path = "/storage/sjpark/VGGFace2/test"
    file_list = os.listdir(dir_path)
    print(len(file_list))
    num = 0
    for i in range(len(file_list)):
        num += len(os.listdir("/storage/sjpark/VGGFace2/test" + "/" + file_list[i]))
    print(num)


def LFW():
    dir_path = "/storage/sjpark/LFW/lfw-funneled/lfw_funneled"
    file_list = os.listdir(dir_path)
    print(len(file_list))
    img_num = 0
    txt_num = 0
    for i in range(len(file_list)):
        if len(file_list[i].split(".")) == 1:
            img_num += len(os.listdir("/storage/sjpark/LFW/lfw-funneled/lfw_funneled" + "/" + file_list[i]))
        else:
            txt_num += 1
    print(img_num)
    print(txt_num)


def CelebFace():
    dir_path = "/storage/sjpark/CelebFace/img_align_celeba/img_align_celeba"
    file_list = os.listdir(dir_path)
    print(len(file_list))


def CPLFW():
    dir_path1 = "/storage/sjpark/CPLFW/aligned images/aligned images"
    dir_path2 = "/storage/sjpark/CPLFW/images/images"
    dir_path3 = "/storage/sjpark/CPLFW/CP_landmarks/CP_landmarks"

    file_list1 = os.listdir(dir_path1)
    file_list2 = os.listdir(dir_path2)
    file_list3 = os.listdir(dir_path3)

    print(len(file_list1))
    print(len(file_list2))
    print(len(file_list3))


def YouTubeFace():
    dir_path1 = "/storage/sjpark/YouTubeFaces/aligned_images_DB"
    dir_path2 = "/storage/sjpark/YouTubeFaces/frame_images_DB"
    dir_path3 = "/storage/sjpark/YouTubeFaces/meta_data"
    dir_path4 = "/storage/sjpark/YouTubeFaces/headpose_DB"

    file_path1 = os.listdir(dir_path1)
    file_path2 = os.listdir(dir_path2)
    file_path3 = os.listdir(dir_path3)
    file_path4 = os.listdir(dir_path4)
    num1 = 0
    print(len(file_path1))
    # aligned_folder
    for i in range(len(file_path1)):
        file = os.listdir("/storage/sjpark/YouTubeFaces/aligned_images_DB" + "/" + file_path1[i])
        for j in range(len(file)):
            num1 += len(
                os.listdir("/storage/sjpark/YouTubeFaces/aligned_images_DB" + "/" + file_path1[i] + "/" + file[j]))
    print(num1)

    # frame_images

    img_num = 0
    txt_num = 0
    for i in range(len(file_path2)):
        if len(file_path2[i].split(".")) == 1:
            file2 = os.listdir("/storage/sjpark/YouTubeFaces/frame_images_DB" + "/" + file_path2[i])
            for j in range(len(file2)):
                img_num += len(
                    os.listdir("/storage/sjpark/YouTubeFaces/frame_images_DB" + "/" + file_path2[i] + "/" + file2[j]))
        else:
            txt_num += 1
    print(img_num)
    print(txt_num)
    # print(len(file_path4))


def YouTubeFace1():
    dir_path1 = "/storage/sjpark/YouTubeFaces (1)/descriptors_DB"
    dir_path2 = "/storage/sjpark/YouTubeFaces (1)/frame_images_DB"
    dir_path3 = "/storage/sjpark/YouTubeFaces (1)/headpose_DB"

    file_path1 = os.listdir(dir_path1)
    file_path2 = os.listdir(dir_path2)
    file_path3 = os.listdir(dir_path3)

    print(len(file_path2))
    img_num = 0
    txt_num = 0
    for i in range(len(file_path2)):
        if len(file_path2[i].split(".")) == 1:
            file2 = os.listdir("/storage/sjpark/YouTubeFaces (1)/frame_images_DB" + "/" + file_path2[i])
            for j in range(len(file2)):
                img_num += len(
                    os.listdir(
                        "/storage/sjpark/YouTubeFaces (1)/frame_images_DB" + "/" + file_path2[i] + "/" + file2[j]))
        else:
            txt_num += 1

    print(img_num)
    print(txt_num)
    print(len(file_path3))


def WiderFace():
    dir_path1 = "/storage/sjpark/WiderFace/WIDER_train/WIDER_train/images"
    dir_path2 = "/storage/sjpark/WiderFace/WIDER_test/WIDER_test/images"
    dir_path3 = "/storage/sjpark/WiderFace/WIDER_val/WIDER_val/images"

    file_path1 = os.listdir(dir_path1)
    file_path2 = os.listdir(dir_path2)
    file_path3 = os.listdir(dir_path3)

    # print(len(file_path1))
    # num = 0
    # for i in range(len(file_path1)):
    #     num += len("/storage/sjpark/WiderFace/WIDER_train/WIDER_train/images" + "/" + file_path1[i])
    # print(num)

    # print(len(file_path2))
    # num = 0
    # for i in range(len(file_path2)):
    #     num += len("/storage/sjpark/WiderFace/WIDER_test/WIDER_test/images" + "/" + file_path2[i])
    # print(num)

    print(len(file_path3))
    num = 0
    for i in range(len(file_path3)):
        num += len("/storage/sjpark/WiderFace/WIDER_val/WIDER_val/images" + "/" + file_path3[i])
    print(num)


if __name__ == '__main__':
    # VGGFace()
    # LFW()
    # CelebFace()
    # CPLFW()
    YouTubeFace1()
# WiderFace()
