import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/qyd/code/SSL4MIS/data/ACDC', help='Name of Experiment')

parser.add_argument('--exp', type=str,
                    default='ACDC/try1_1_entmask', help='experiment_name')

parser.add_argument('--model', type=str,
                    default='unet', help='model_name')

parser.add_argument('--max_iterations', type=int,
                    default=5, help='maximum epoch number to train')

parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2024, help='random seed') # init seed = 1337
parser.add_argument('--gpu', type=str, default='3', help='gpu id')


parser.add_argument('--num_classes', type=int,  default=4,   # >= 4
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,    
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# 取entropyMap像素的初始阈值
parser.add_argument('--Ent_th', type=float,
                    default=0.75, help='entropy_threshold')

# mask
parser.add_argument('--mask_patch_size', type=int, default=16, help='patch_size:4,8,16,24,32') # mask的小方块有多大
parser.add_argument('--mask_ratio', type=float, default=0.50, help='ratio') # mask的比例
parser.add_argument('--lambda_mask', type=float, default=1.5, help='lambda_mask') # 对应论文Eq.6的γ，一个可调的超参数，用于放大mask盖住的部分的贡献

args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,   
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

# 计算熵
def entropy_map(p, C = 4):
    # p N*C*W*H，n是样本数量，c是类别数量，w*h是图片大小
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / torch.tensor(np.log(C)).cuda()
    return y1

def ent_mask_patch_countnum(data, emap, thres, patch_size, image_size=256):
    ratio = 0.5
    new_data = data.clone()
    # 此处打了entmask的取值是1，不打的是0
    entmask = (emap >= thres).float()
    entmask = torch.unsqueeze(entmask, 1)

    # 用于乘data以得到打了mask的new_data
    entmask_multi = torch.ones_like(entmask)

    # 每个patch里像素个数
    patch_pixel_num = patch_size * patch_size   
    # 每行patch个数
    patch_num = image_size // patch_size
    # s是patch总个数
    s = [x for x in range(patch_num*patch_num)]

    for i in range(len(s)):
        row = s[i] // patch_num
        col = s[i] % patch_num

        mask_sum = entmask[:,:,row*patch_size:(row+1)*patch_size,col*patch_size:(col+1)*patch_size].sum()
        
        if mask_sum >= patch_pixel_num * ratio :
            # 此处打了mask的取值是0，不打mask的是1
            entmask_multi[:,:,row*patch_size:(row+1)*patch_size,col*patch_size:(col+1)*patch_size] = 0
    
    
    new_data = new_data * entmask_multi

    # 此处打了mask的取值是1，不打mask的是0
    entmask = entmask_multi * (-1) + 1

    return new_data, entmask




def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()  # student model
    ema_model = create_model(ema=True) # teacher model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, 
                            split="train", 
                            num=None, 
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))

    db_val = BaseDataSets(base_dir=args.root_path, split="test")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    ce_loss_mask = CrossEntropyLoss(reduction='none')

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            ema_inputs = unlabeled_volume_batch

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            
            consistency_weight = 1

            # Entropy Selection
            EMap = entropy_map(ema_output_soft, C=num_classes)
            # 取高熵值作为mask，此处阈值可调
            # threshold = args.Ent_th - 0.15*ramps.sigmoid_rampup(iter_num, max_iterations)  # 0.75->0.6
            # threshold = args.Ent_th + (0.95-args.Ent_th)*ramps.sigmoid_rampup(iter_num, max_iterations)  # 0.75->0.95
            # threshold = args.Ent_th + (0.9-args.Ent_th)*ramps.sigmoid_rampup(iter_num, max_iterations)  # 0.75->0.9
            threshold = 0.7

            mask_unlabeled_volume_batch, ent_mask = ent_mask_patch_countnum(unlabeled_volume_batch, EMap, threshold, args.mask_patch_size, args.patch_size[0])

            logits_masked_unlabeled_volume = model(mask_unlabeled_volume_batch)  # [12, 4, 256, 256]
            pseudo_label = torch.argmax(torch.softmax(ema_output, dim=1), dim=1).squeeze(0)           
            loss_ce_2 = ce_loss_mask(logits_masked_unlabeled_volume, pseudo_label[:].long())

            ent_mask_1 = ent_mask.squeeze(1)   # ([12, 256, 256])
            ent_mask_2 = 1 - ent_mask_1         
            masksum = ent_mask_1.sum()
            nomasksum = ent_mask_2.sum()

            loss_ce_2_mask_1 = loss_ce_2 * ent_mask_1.float()
            loss_ce_2_mask_1 = loss_ce_2_mask_1.sum() / (ent_mask_1.sum()+10e-5)
            loss_ce_2_mask_2 = loss_ce_2 * ent_mask_2.float()
            loss_ce_2_mask_2 = loss_ce_2_mask_2.sum() / (ent_mask_2.sum()+10e-5)

            loss_ce_2 = args.lambda_mask * loss_ce_2_mask_1 + loss_ce_2_mask_2
            
            loss_dice_2 = dice_loss(logits_masked_unlabeled_volume, pseudo_label.unsqueeze(1), softmax=True)

            
            if iter_num < 1000:
                consistency_loss = 0.0
            else: 
                consistency_loss = 0.5 * (loss_ce_2 + loss_dice_2)
                
            loss = supervised_loss + consistency_weight * consistency_loss
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('info/threshold', threshold, iter_num)
            writer.add_scalar('info/masksum', ent_mask_1.sum(), iter_num)            
            writer.add_scalar('info/nomasksum', ent_mask_2.sum(), iter_num)

            
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, threshold: %f, masksum: %d, nomasksum: %d' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), threshold, masksum, nomasksum ))
            
            # masked image 可视化
            masked_img = mask_unlabeled_volume_batch
            masked_img[masked_img == 0] = 255
            masked_img_vis = masked_img[1, :, :, :]
            writer.add_image('maskVis/MaskImage', masked_img_vis, iter_num)

            # mask对应的image原图可视化
            vis_volume_batch = volume_batch[args.labeled_bs:]           
            vis_image = vis_volume_batch[1, 0:1, :, :]
            writer.add_image('maskVis/MaskOriginImage', vis_image, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)                
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    # -------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/home/qyd/EMCL_results/model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
