import os
import json
import re
import shutil
import torch
from PIL import Image
from torchvision import transforms
import logging
from model import resnet34


class_indict = {0:'0_1-0_2', 1:'ge_0.3', 2:'ge_0.4', 3:'ge_0.5', 4:'ge_0.6',5:'ge_0.7', 6:'ge_0.8', 7:'ge_0.9' ,8:'no_snow'}

data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def read_imgs(imgs_root='data_set/harmo/ge_0_5'):
    """load image"""
    # 指向需要遍历预测的图像文件夹
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]
    return img_path_list


def predict_batch_imgs(model, img_path_list, log_file='ge_0.5_.log', batch_size=16):
    f = open(log_file, 'w')
    model.eval()
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                # print(img_path)
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (prob, class_name) in enumerate(zip(probs, classes)):
                f.write("image: {}  class: {}  prob: {:.3}\n"
                        .format(img_path_list[ids * batch_size + idx],
                                class_indict[int(class_name.numpy())],
                                prob.numpy()))
    f.close()




if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # create model
    # weights_path = "checkpoint/ResNet34-99-v2.pth"
    # model = resnet34(num_classes=9).to(device)
    # # load model weights
    # assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    # model.load_state_dict(torch.load(weights_path, map_location=device))
    # img_path_list = read_imgs(imgs_root='data_set/harmo/ge_0.3')#在哪里读图
    # print("Reading image:", read_imgs)
    # predict_batch_imgs(model, img_path_list, log_file='results/filter_ge_0.5_.log')#log放在哪里

    # 打开 log 文件并读取内容
    with open("C:\\Users\\a5630\\Project-10-26\\results\\filter_ge_0.5_.log", "r") as f:

        lines = f.readlines()

    # 遍历每一行
    for line in lines:
        print("Line:", line)
        # 将行按照空格分割成多个字符串
        tokens = line.split()
        # 使用正则表达式匹配 class 和 prob 的值
        class_ = re.search("class: ([\w\.]+)", line).group(1)
        prob = float(re.search("prob: ([\d\.]+)", line).group(1))
        # 如果 class 等于 "ge_0.5" 且 prob 大于 0.95，则复制图片 class_ == '0_8-0_9' and prob <= 0.98
        if class_ == 'ge_0.4' and prob >= 0.98:
            image_path = re.search("image: (\S+)", line).group(1)
            print("Copying image:", image_path)
            shutil.move(image_path, "C:\\Users\\a5630\\Project-10-26\\data_set\\harmo\\ge_0.4" )






