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
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.dataset_new import (BaseDataSet, RandomGenerator)
from networks.net_factory import net_factory
from utils import losses
from utils.mask import random_mask, ent_mask_patch_countnum
from utils.val_2D import test_single_volume, test_single_slice

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/qyd/datas/ACDC', help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='/home/qyd/datas/Lung_semi/lists/lists_ACDC_0907', help='list dir')
parser.add_argument('--exp', type=str,
                    default='ACDC/new_entmask_patchCountNum_test', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_epochs', type=int,
                    default=1000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--image_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2024, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

parser.add_argument('--gpu', type=int, default=2, help='gpu id')

# label and unlabel
parser.add_argument('--labeled_batch_size', type=int,
                    default=8, help='batch_size per gpu')    
parser.add_argument('--unlabeled_batch_size', type=int,
                    default=32, help='batch_size per gpu')           

# costs
parser.add_argument('--mask_ratio', type=float, default=0.5,
                    help='mask_ratio')
parser.add_argument('--batch_num', type=int, default=2,
                    help='use how many images to shuffle')
parser.add_argument('--ce_loss_weight', default=[0.01, 1])
parser.add_argument('--loss_type', type=int, default=0,
                    help='ablation')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--lambda_mask', type=float,
                    default=1.5, help='consistency')
parser.add_argument('--lambda_patch', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--start_consistency_epoch', type=float,
                    default=0.1, help='consistency')
args = parser.parse_args()


# 权重在前20%周期内线性增加到max，其后保持max
def get_weight(max_weight, epoch):
    if epoch < args.max_epochs * 0.2:
        return max_weight * epoch / (args.max_epochs * 0.2)
    else:
        return max_weight


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

# 计算混合的最终loss
def mix_loss(prediction, label, loss2, weight):
    ce_loss = CrossEntropyLoss(weight=weight)
    loss_ce = ce_loss(prediction, label[:].long())
    loss_dice = loss2(prediction, label.unsqueeze(1), softmax=True)
    loss = 0.5 * loss_ce + 0.5 * loss_dice
    return loss


# 这样实现的shuffle速度很慢，但暂时不知道怎么优化
def shuffle(data, batch_num, patch_num, image_size, s=None, dim=4):
    data_splits = torch.split(data, data.shape[0]//batch_num, dim=0)
    # print(len(data_splits))
    # print(data_splits[0].shape)
    if s == None:
        s = [x for x in range(patch_num*patch_num*batch_num)]
        random.shuffle(s)
    data_shuffled = []
    for i in range(batch_num):
        data_shuffled.append(torch.zeros(data_splits[0].shape).cuda())
    patch_size = int(image_size / patch_num)
    for i in range(len(s)):
        splits_batch = s[i] // (patch_num*patch_num)
        splits_row = s[i] // patch_num - splits_batch * patch_num
        splits_column = s[i] % patch_num
        shuffled_batch = i // (patch_num*patch_num)
        shuffled_row = i // patch_num - shuffled_batch * patch_num
        shuffled_column = i % patch_num
        # print(splits_batch, splits_row, splits_column, shuffled_batch, shuffled_row, shuffled_column)
        # print(data_shuffled[shuffled_batch][:,:,shuffled_row*patch_size:(shuffled_row+1)*patch_size,shuffled_column*patch_size:(shuffled_column+1)*patch_size])
        # print(data_splits[splits_batch][:,:,splits_row*patch_size:(splits_row+1)*patch_size,splits_column*patch_size:(splits_column+1)*patch_size])
        if dim==4:
            data_shuffled[shuffled_batch][:,:,shuffled_row*patch_size:(shuffled_row+1)*patch_size,shuffled_column*patch_size:(shuffled_column+1)*patch_size] = data_splits[splits_batch][:,:, splits_row*patch_size:(splits_row+1)*patch_size, splits_column*patch_size:(splits_column+1)*patch_size]
        elif dim==3:
            data_shuffled[shuffled_batch][:,shuffled_row*patch_size:(shuffled_row+1)*patch_size,shuffled_column*patch_size:(shuffled_column+1)*patch_size] = data_splits[splits_batch][:,splits_row*patch_size:(splits_row+1)*patch_size, splits_column*patch_size:(splits_column+1)*patch_size]
    data_shuffle = torch.cat(data_shuffled, dim=0)
    return data_shuffle, s

# 计算熵
def entropy_map(p, C = 4):
    # p N*C*W*H，n是样本数量，c是类别数量，w*h是图片大小
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / torch.tensor(np.log(C)).cuda()
    return y1


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    lambda_patch = args.lambda_patch

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes, patch_num=4)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train_labeled = BaseDataSet(base_dir=args.root_path, list_dir=args.list_dir, split="train_labeled",num_classes = args.num_classes,
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=args.image_size)]), data_type=args.data_type)
    db_train_unlabeled = BaseDataSet(base_dir=args.root_path, list_dir=args.list_dir, split="train_unlabeled",num_classes = args.num_classes,
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=args.image_size)]), data_type=args.data_type)
    db_test = BaseDataSet(base_dir=args.root_path, split="test", list_dir=args.list_dir, num_classes = args.num_classes, data_type=args.data_type)
    print("The length of labeled train set is: {}".format(len(db_train_labeled)))
    print("The length of unlabeled train set is: {}".format(len(db_train_unlabeled)))
    print("The length of test set is: {}".format(len(db_test)))

    trainloader_labeled = DataLoader(db_train_labeled, batch_size=args.labeled_batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    trainloader_unlabeled = DataLoader(db_train_unlabeled, batch_size=args.unlabeled_batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    logging.info("label: {}; unlabel: {}; test: {};".format(len(trainloader_labeled), len(trainloader_unlabeled), len(testloader)))

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_iterations = args.max_epochs * len(trainloader_labeled)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader_labeled), max_iterations))
    
    best_performance = 0.0
    repeat_time = args.unlabeled_batch_size // args.labeled_batch_size
    iterator = tqdm(range(max_epochs), ncols=70)
    for epoch_num in iterator:
        dataloader_labeled = iter(trainloader_labeled)
        for i, unlabeled_batch in enumerate(trainloader_unlabeled):
            try:
                labeled_batch = next(dataloader_labeled)
            except StopIteration:
                dataloader_labeled = iter(trainloader_labeled)
                labeled_batch = next(dataloader_labeled)      

            image_batch, label_batch = labeled_batch['image'], labeled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            unlabeled_image_batch = unlabeled_batch['image']
            unlabeled_image_batch = unlabeled_image_batch.cuda()

            p_labeled_1 = model(image_batch)

            loss_ce = ce_loss(p_labeled_1[:], label_batch[:].long())
            loss_dice = dice_loss(p_labeled_1[:], label_batch.unsqueeze(1), softmax=True)
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            total_loss = supervised_loss.clone()
        
            if epoch_num >= max_epochs * args.start_consistency_epoch:

                # mask_unlabeled_image_batch, mask = random_mask(unlabeled_image_batch, 16, 256, args.mask_ratio)
                
                logits_unlabeled_image = ema_model(unlabeled_image_batch)
                pred_unlabeled_image = torch.softmax(logits_unlabeled_image, dim=1)
                
                EMap = entropy_map(pred_unlabeled_image, C=num_classes)
                threshold = 0.7

                mask_unlabeled_image_batch, mask = ent_mask_patch_countnum(unlabeled_image_batch, EMap, threshold, 16, 256)
                logits_masked_unlabeled_image = model(mask_unlabeled_image_batch)
                
                
                pseudo_label = torch.argmax(torch.softmax(logits_unlabeled_image, dim=1), dim=1).squeeze(0)
                # print('pseudo_label ', pseudo_label.shape)
                # loss_ce_2 = ce_loss_nomask(logits_masked_unlabeled_image, pseudo_label[:].long())
                # print('logits_unlabeled_image: ', logits_unlabeled_image.shape)
                loss_ce_2 = ce_loss(logits_masked_unlabeled_image, pseudo_label[:].long())
                # mask = torch.ones_like(pseudo_label)
                # mask[pseudo_label == 0] = 0.1
                
                # print('mask_1: ', mask_1.shape)
                # stop == 1
                
                
                mask_1 = mask.squeeze(1)
                mask_2 = 1 - mask_1
                
                masksum = mask_1.sum()
                nomasksum = mask_2.sum()

                
                '''
                mask_1[((mask_1 == 1).float() * (pseudo_label == 0).float()).bool()] = ce_loss_weight[0]
                loss_ce_2_mask_1 = loss_ce_2 * mask_1.float()
                # loss_ce_2_mask_1 = loss_ce_2_mask_1.sum() / mask_1.sum()
                loss_ce_2_mask_1 = loss_ce_2_mask_1.sum() / (mask_1.sum()+10e-5)
                
                mask_2[((mask_2 == 1).float() * (pseudo_label == 0).float()).bool()] = ce_loss_weight[0]
                loss_ce_2_mask_2 = loss_ce_2 * mask_2.float() 
                # loss_ce_2_mask_2 = loss_ce_2_mask_2.sum() / mask_2.sum()
                loss_ce_2_mask_2 = loss_ce_2_mask_2.sum() /(mask_2.sum()+10e-5)

                loss_ce_2 = args.lambda_mask * loss_ce_2_mask_1 + loss_ce_2_mask_2
                '''
                
                # print('loss_ce_2_mask_1: ', loss_ce_2_mask_1.item())
                # print('loss_ce_2_mask_2: ', loss_ce_2_mask_2.item())
                # print('loss_ce_2_mask: ', loss_ce_2_mask.item())
                # stop == 1
                
                loss_dice_2 = dice_loss(logits_masked_unlabeled_image, pseudo_label.unsqueeze(1), softmax=True)
                loss_consistency = 0.4 * loss_ce_2 + 0.6 * loss_dice_2
                # print("loss_consistency: ", loss_consistency)
                total_loss += loss_consistency * lambda_patch

                

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            if lr_ < base_lr * 0.1:
                lr_ = base_lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', total_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            
            if epoch_num >= max_epochs * args.start_consistency_epoch:
 
                writer.add_scalar('info/masksum', masksum, iter_num)            
                writer.add_scalar('info/nomasksum', nomasksum, iter_num)

                logging.info(
                    'iteration %d : loss : %f ; loss_supervised : %f, masksum: %d, nomasksum: %d ;' % 
                    (iter_num, total_loss.item(), supervised_loss.item(), masksum, nomasksum))

                # masked image 可视化
                masked_img = mask_unlabeled_image_batch
                masked_img[masked_img == 0] = 255
                
                masked_img_vis = masked_img[1, :, :, :]
                writer.add_image('train/MaskImage', masked_img_vis, iter_num)

                # mask对应的image原图可视化
                vis_volume_batch = unlabeled_image_batch           
                vis_image = vis_volume_batch[1, 0:1, :, :]
                writer.add_image('train/MaskOriginImage', vis_image, iter_num)
            else:
                logging.info(
                    'iteration %d : loss : %f ; loss_supervised : %f;' % 
                    (iter_num, total_loss.item(), supervised_loss.item()))


            if iter_num % 500 == 0:
                image = image_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(p_labeled_1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 1000 == 0:
                ema_model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(testloader):
                    if args.dataset == "MLT":
                        metric_i = test_single_slice(
                            sampled_batch["image"], sampled_batch["label"], ema_model, classes=num_classes)
                    elif args.dataset == "ACDC":
                        metric_i = test_single_volume(
                            sampled_batch["image"], sampled_batch["label"], ema_model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_test)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_asd'.format(class_i+1), metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_nsd'.format(class_i+1), metric_list[class_i, 3], iter_num)
                performance = np.mean(metric_list, axis=0)
                writer.add_scalar('info/val_mean_dice', performance[0], iter_num)
                writer.add_scalar('info/val_mean_hd95', performance[1], iter_num)
                writer.add_scalar('info/val_mean_asd', performance[2], iter_num)
                writer.add_scalar('info/val_mean_nsd', performance[3], iter_num)
                
                if performance[0] > best_performance:
                    best_performance = performance[0]
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best)
                    logging.info('The best val epoch is: %d' % (iter_num))
                logging.info('epoch %d : mean_dice : %f mean_hd95 : %f asd : %f nsd : %f' % (epoch_num, performance[0], performance[1], performance[2], performance[3]))
                ema_model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
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
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    dataset_name = args.dataset

    dataset_config = {
        # 'MLT': {
        #     'root_path': '/data/hjj/dataset/MLT/',
        #     # 'list_dir': '/data/hjj/dataset/Lung_semi/lists/lists_MLT/10%',
        #     'list_dir': '/data/hjj/dataset/Lung_semi/lists/lists_MLT/5%',
        #     'num_classes': 2,
        #     'ce_loss_weight': [0.01, 1],
        #     'unlabeled_batch_size': 18,
        #     'labeled_batch_size': 2,
        #     'data_type': 'npz',
        # },
        'ACDC': {
            'root_path': '/home/qyd/datas/ACDC/data/slices/',
            # 'list_dir': '/home/qyd/datas/Lung_semi/lists/lists_ACDC_0907/5%/',
            'list_dir': '/home/qyd/datas/Lung_semi/lists/lists_ACDC_0907/10%/',
            # 'list_dir': '/home/qyd/datas/Lung_semi/lists/lists_ACDC_0907/20%/',
            'num_classes': 4,
            'ce_loss_weight': [0.1, 1, 1, 1],
            'unlabeled_batch_size': 18,       
            'labeled_batch_size': 2,
            'data_type': 'h5',
        },
    }

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.ce_loss_weight = dataset_config[dataset_name]['ce_loss_weight']
    # if dataset_name == 'LIDC':
    args.unlabeled_batch_size = dataset_config[dataset_name]['unlabeled_batch_size']
    args.labeled_batch_size = dataset_config[dataset_name]['labeled_batch_size']
    args.data_type = dataset_config[dataset_name]['data_type']

    # snapshot_path = "/home/qyd/EMCL_results/model/{}/{}".format(args.mask_ratio, args.exp)
    snapshot_path = "/home/qyd/EMCL_results/model/{}".format(args.exp)
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
