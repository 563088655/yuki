import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse
from model import resnet34

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

data_transform = {
    "train": transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(192, 192),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--version', type=int, default=2)
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str, default="data_set/harmo")

    parser.add_argument('--weights', type=str, default='checkpoint/ResNet34-99-v2.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    # 默认不冻结，也就是默认训练模型
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()
    return args


def train_model(net, args, train_loader):
    r""" 训练模型"""
    net.to(args.device)  # 传到对应设备
    # define loss function
    loss_function = nn.CrossEntropyLoss()
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.00005)
    train_steps = len(train_loader)
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)  # 进度条
        for step, data in enumerate(train_bar):
            # 前向传播
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(args.device))  # logits 预测的结果
            loss = loss_function(logits, labels.to(args.device))  # 计算损失
            # 后向传播
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}" \
                .format(epoch + 1, args.epochs, loss)
        print('[epoch %d] train_loss: %.3f' % (epoch + 1, running_loss / train_steps))
        save_path = 'checkpoint/ResNet34-{}-v{}.pth'.format(epoch, args.version)
        torch.save(net.state_dict(), save_path)
    print('Finished Training')


def get_loader(args):
    print("using {} device.".format(args.device))
    image_path = args.data_path  # data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    class_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=nw)
    print("using {} images for training".format(train_num))
    return train_loader


def load_model(args):
    """模型加载"""
    net = resnet34()
    net.fc = nn.Linear(net.fc.in_features, args.num_classes - 1)  # 更改全连接层
    # assert os.path.exists(args.weights), "file {} does not exist.".format(args.weights)
    state_dict = torch.load(args.weights, map_location='cpu')
    net.load_state_dict(state_dict, strict=False)
    net.fc = nn.Linear(net.fc.in_features, args.num_classes )
    return net


if __name__ == '__main__':
    args = parse_args()
    train_loader = get_loader(args)
    net = load_model(args)
    train_model(net=net, args=args, train_loader=train_loader)


