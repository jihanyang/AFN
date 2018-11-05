import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

from utils import VisDAImage, print_args
from model import ResBase50, ResClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="")
parser.add_argument("--t", default="validation")
parser.add_argument("--batch_size", default=64)
parser.add_argument("--shuffle", default=False)
parser.add_argument("--num_workers", default=4)
parser.add_argument("--snapshot", default="")
parser.add_argument("--epoch", default=39, type=int)
parser.add_argument("--result", default="")
parser.add_argument("--class_num", default=12)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
args = parser.parse_args()
print_args(args)

result = open(os.path.join(args.result, "resnet50_" + args.t + '_' + args.post + '.' + args.repeat + "_score.txt"), "a")

t_root = os.path.join(args.data_root, args.t)
t_label = os.path.join(args.data_root, args.t + "6_list.txt")

data_transform = transforms.Compose([
    transforms.Scale((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

t_set = VisDAImage(t_root, t_label, data_transform)
assert len(t_set) == 28978
t_loader = torch.utils.data.DataLoader(t_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)

netG = ResBase50().cuda()
netF = ResClassifier(class_num=args.class_num).cuda()
netG.eval()
netF.eval()

for epoch in range(args.epoch/2 , args.epoch + 1):
    if epoch % 10 != 0:
        continue
    
    netG.load_state_dict(torch.load(os.path.join(args.snapshot, "VisDA_resnet50_netG_par_" + args.post + '.' + str(args.repeat) + '_'  + str(epoch) + ".pth")))
    netF.load_state_dict(torch.load(os.path.join(args.snapshot, "VisDA_resnet50_netF_par_" + args.post + '.' + str(args.repeat) + '_'  + str(epoch) + ".pth")))

    correct = 0
    tick = 0
    
    for (imgs, labels) in t_loader:
        tick += 1
        imgs = Variable(imgs.cuda())
        pred = netF(netG(imgs))
        pred = F.softmax(pred)
        pred = pred.data.cpu().numpy()
        pred = pred.argmax(axis=1)
        labels = labels.numpy()
        correct += np.equal(labels, pred).sum()
        
    correct = correct * 1.0 / len(t_set)
    # overall accuracy
    print "Epoch {0}: {1}".format(epoch, correct)
    result.write("Epoch " + str(epoch) + ": " + str(correct) + "\n")
result.close()
