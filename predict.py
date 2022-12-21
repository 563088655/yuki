import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34

data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class_indict = {0: "daisy",
              1: "dandelion",
              2: "roses",
              3: "sunflowers",
              4: "tulips"}


def eval(device, img):
    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "./ResNet34-best.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_dataloder(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        predict_cla = int(predict_cla)
    print("class: {:10}   prob: {:.3}".format(class_indict[predict_cla],
                                              predict[predict_cla].numpy()))

    for i in range(len(predict)):
        print("class: {:10} ---  prob: {:.3}".format(class_indict[i], predict[i].numpy()))


def read_img(img_path="../tulip.jpg"):
    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    return img


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = read_img(img_path='data_set/flower_data/val/dandelion/16987075_9a690a2183.jpg')
    eval(device, img=img)

