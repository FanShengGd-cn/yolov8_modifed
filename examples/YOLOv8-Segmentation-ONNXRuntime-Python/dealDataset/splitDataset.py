import os
import shutil
import random
from tqdm import tqdm


def split_img(img_path, label_path, split_list):
    Data = r'D:\selectImg\2024_2_17_dataset\split'
    train_img_dir = Data + '/images/train'
    val_img_dir = Data + '/images/val'
    # test_img_dir = Data + '/images/test'

    train_label_dir = Data + '/labels/train'
    val_label_dir = Data + '/labels/val'
    try:  # 创建数据集文件夹

        os.mkdir(Data)


        # test_label_dir = Data + '/labels/test'

        # 创建文件夹
        os.makedirs(train_img_dir)
        os.makedirs(train_label_dir)
        os.makedirs(val_img_dir)
        os.makedirs(val_label_dir)
        # os.makedirs(test_img_dir)
        # os.makedirs(test_label_dir)

    except:
        print('文件目录已存在')

    train, val = split_list
    # all_img = os.listdir(img_path)
    # all_img_path = [os.path.join(img_path, img) for img in all_img]
    all_label = os.listdir(label_path)
    all_label_path = [os.path.join(label_path, label) for label in all_label]
    # train_img = random.sample(all_img_path, int(train * len(all_img_path)))
    train_label = random.sample(all_label_path, int(train * len(all_label)))
    # train_img_copy = [os.path.join(train_img_dir, img.split('\\')[-1]) for img in train_img]
    train_img = [toImgPath(label, img_path) for label in train_label]
    # train_label_copy = [os.path.join(train_label_dir, label.split('\\')[-1]) for label in train_label]
    for i in tqdm(range(len(train_img)), desc='train ', ncols=11, unit='img'):
        _copy(train_img[i], train_img_dir)
        _copy(train_label[i], train_label_dir)
        all_label_path.remove(train_label[i])
    val_label = all_label_path
    val_img = [toImgPath(label, img_path) for label in val_label]
    for i in tqdm(range(len(val_img)), desc='val ', ncols=11, unit='img'):
        _copy(val_img[i], val_img_dir)
        _copy(val_label[i], val_label_dir)


def _copy(from_path, to_path):
    shutil.copy(from_path, to_path)


def toLabelPath(img_path, label_path):
    img = img_path.split('\\')[-1]
    label = img.lower().split('.jpg')[0] + '.txt'
    return os.path.join(label_path, label)


def toImgPath(label_path, img_path):
    label = label_path.split('\\')[-1]
    img = label.lower().split('.txt')[0] + '.jpg'
    return os.path.join(img_path, img)


def main():
    img_path = r'D:\selectImg\2024_2_17_dataset\images'
    label_path = r'D:\selectImg\2024_2_17_dataset\labels'
    split_list = [0.8, 0.2]  # 数据集划分比例[train:val]
    split_img(img_path, label_path, split_list)


if __name__ == '__main__':
    main()
