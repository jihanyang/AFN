import os
import argparse
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from utils import VisDAImage, weights_init, print_args
from model import ResBase50, ResClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="")
parser.add_argument("--source", default="train")
parser.add_argument("--target", default="validation")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=8)
parser.add_argument("--epoch", default=120, type=int)
parser.add_argument("--snapshot", default="")
parser.add_argument("--lr", default=0.001)
parser.add_argument("--class_num", default=12)
parser.add_argument("--extract", default=True)
parser.add_argument("--weight_L2norm", default=0.01)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
args = parser.parse_args()
print_args(args)

source_root = os.path.join(args.data_root, args.source)
source_label = os.path.join(args.data_root, args.source + "_list.txt")
target_root = os.path.join(args.data_root, args.target)
target_label = os.path.join(args.data_root, args.target + "6_list.txt")

train_transform = transforms.Compose([
    transforms.Scale((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

source_set = VisDAImage(source_root, source_label, train_transform)
target_set = VisDAImage(target_root, target_label, train_transform)

assert len(source_set) == 152397
assert len(target_set) == 28978

source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)
target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)

netG = ResBase50()
netF = ResClassifier(class_num=args.class_num, extract=args.extract)
netF.apply(weights_init)
netG.cuda()
netF.cuda()


def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

def get_L2norm_loss_self_driven(x):
    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 0.3
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return args.weight_L2norm * l


opt_g = optim.SGD(netG.parameters(), lr=args.lr, weight_decay=0.0005)
opt_f = optim.SGD(netF.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

for epoch in range(1, args.epoch +1):
    source_loader_iter = iter(source_loader)
    target_loader_iter = iter(target_loader)
    print(">>trianing epoch : " + str(epoch))
    for i, (t_imgs, _) in tqdm.tqdm(enumerate(target_loader_iter)):
        try:
            s_imgs, s_labels = source_loader_iter.next()
        except:
            source_loader_iter = iter(source_loader)
            s_imgs, s_labels = source_loader_iter.next()

        if s_imgs.size(0) != args.batch_size or t_imgs.size(0) != args.batch_size:
            continue        

        s_imgs = Variable(s_imgs.cuda())
        s_labels = Variable(s_labels.cuda())     
        t_imgs = Variable(t_imgs.cuda())
        
        opt_g.zero_grad()
        opt_f.zero_grad()

        s_bottleneck = netG(s_imgs)
        t_bottleneck = netG(t_imgs)
        
        s_fc2_emb, s_logit = netF(s_bottleneck)
        t_fc2_emb, _ = netF(t_bottleneck)

        s_cls_loss = get_cls_loss(s_logit, s_labels)
        s_fc2_L2norm_loss = get_L2norm_loss_self_driven(s_fc2_emb)
        t_fc2_L2norm_loss = get_L2norm_loss_self_driven(t_fc2_emb)

        loss = s_cls_loss + s_fc2_L2norm_loss + t_fc2_L2norm_loss
        loss.backward()

        opt_g.step()
        opt_f.step()
    if epoch % 10 == 0:   
        torch.save(netG.state_dict(), os.path.join(args.snapshot, "VisDA_IAFN_" + args.post + '.' + str(args.repeat) + '_' + str(epoch)  + ".pth"))
        torch.save(netF.state_dict(), os.path.join(args.snapshot, "VisDA_IAFN_" + args.post + '.' + str(args.repeat) + '_' + str(epoch)  + ".pth"))
