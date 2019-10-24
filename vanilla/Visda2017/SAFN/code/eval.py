import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

from utils import VisDAImage, print_args
from model import ResBase101, ResBase50, ResClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="")
parser.add_argument("--target", default="validation")
parser.add_argument("--batch_size", default=64)
parser.add_argument("--shuffle", default=False)
parser.add_argument("--num_workers", default=4)
parser.add_argument("--snapshot", default="")
parser.add_argument("--epoch", default=10, type=int)
parser.add_argument("--result", default="")
parser.add_argument("--class_num", default=12)
parser.add_argument("--model", default='-1', type=str)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
args = parser.parse_args()

print_args(args)

result = open(os.path.join(args.result, "VisDA_IAFN_" + args.target + '_' + args.post + '.' + args.repeat + "_score.txt"), "a")

t_root = os.path.join(args.data_root, args.target)
t_label = os.path.join(args.data_root, args.target + "_list"  +".txt")
data_transform = transforms.Compose([
    transforms.Scale((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
t_set = VisDAImage(t_root, t_label, data_transform)
# assert len(t_set) == 55388
t_loader = torch.utils.data.DataLoader(t_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)

if args.model == 'resnet101':
    netG = ResBase101().cuda()
elif args.model == 'resnet50':
    netG = ResBase50().cuda()
else:
    raise ValueError('Unexpected value of args.model')

netF = ResClassifier(class_num=args.class_num).cuda()
netG.eval()
netF.eval()


for epoch in range(args.epoch/2, args.epoch+1):
    if epoch % 10 != 0:
        continue
    
    netG.load_state_dict(torch.load(os.path.join(args.snapshot, "VisDA_IAFN_"+ args.model + "_netG_" + args.post + '.' + args.repeat + '_' + str(epoch) + ".pth")))
    netF.load_state_dict(torch.load(os.path.join(args.snapshot, "VisDA_IAFN_"+ args.model + "_netF_" + args.post + '.' + args.repeat + '_' + str(epoch) + ".pth")))
    correct = 0
    tick = 0
    subclasses_correct = np.zeros(args.class_num)
    subclasses_tick = np.zeros(args.class_num)
    
    for (imgs, labels) in t_loader:
        tick += 1
        imgs = Variable(imgs.cuda())
        pred = netF(netG(imgs))
        pred = F.softmax(pred)
        pred = pred.data.cpu().numpy()
        pred = pred.argmax(axis=1)
        labels = labels.numpy()
        for i in range(pred.size):
            subclasses_tick[labels[i]] += 1
            if pred[i] == labels[i]:
                correct += 1
                subclasses_correct[pred[i]] += 1

    correct = correct * 1.0 / len(t_set)
    subclasses_result = np.divide(subclasses_correct, subclasses_tick)
    # overall accuracy
    print "Epoch {0}: {1}".format(epoch, correct)
    result.write("Epoch " + str(epoch) + ": " + str(correct) + "\n")
    # pre class accuracy
    for i in range(args.class_num):
        print "\tClass {0} : {1}".format(i, subclasses_result[i])
        result.write("\tClass " + str(i) + ": " + str(subclasses_result[i]) + "\n")

    result.write("\tAvg : " + str(subclasses_result.mean()) + "\n")
    print "\tAvg : {0}\n".format(subclasses_result.mean())

result.close()
