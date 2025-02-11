import random
import numpy as np
import torch
torch.set_printoptions(threshold=np.inf)


from .resize import resize

output_debug_file = "/home/qyd/MCL_try/debug/debug.log"


def random_mask(data, patch_size=16, image_size=256, mask_ratio=0.75):
    patch_num = image_size // patch_size
    s = [x for x in range(patch_num*patch_num)]
    random.shuffle(s)
    s = s[0:int(len(s)*mask_ratio)]     # 不要的，将这些patch置为0
    mask = torch.zeros_like(data)
    new_data = data.clone()
    for i in range(len(s)):
        row = s[i] // patch_num
        col = s[i] % patch_num
        new_data[:,:,row*patch_size:(row+1)*patch_size,col*patch_size:(col+1)*patch_size] = 0
        mask[:,:,row*patch_size:(row+1)*patch_size,col*patch_size:(col+1)*patch_size] = 1
    return new_data, mask



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


def cal_patch_mean(emap, patchnum, patchsize):
    batch_size, H, W = emap.shape
    meanmap = torch.zeros(batch_size, patchnum, patchnum)
    meanmap.cuda()

    # with open(output_debug_file, 'a') as f:
    #     print('emap[0] = ', emap[0], file = f)

    for b in range(batch_size):
        tmp = emap[b]
        for i in range(0, patchnum):
            for j in range(0, patchnum):
                # 计算每个块左上角的索引
                start_row = i * patchsize
                start_col = j * patchsize
                # 获取当前块
                block = tmp[start_row : start_row + patchsize, start_col : start_col + patchsize]
                
                # 计算当前块的平均值并存储到对应的位置
                meanmap[b, i, j] = block.mean()


    # with open(output_debug_file, 'a') as f:
    #     print('emap[0] = ', emap[0], file = f)
    #     print('meanmap[0]: ', meanmap[0], file = f)
    #     print('meanmapshape = ', meanmap.shape, file = f)
    
    return meanmap


# entmap分块取均值后按升序排列，前百分之γ的分位线作为选取的阈值
def ent_mask_patch_mean(data, emap, maskratio, patch_size=16, image_size=256, hint = 0):
    ndevice = torch.device('cuda')
    _, _, H, W = data.shape
    new_data = data.clone()
    # 每个patch里像素个数
    patch_pixel_num = patch_size * patch_size   
    # 每行patch个数
    patch_num = image_size // patch_size
    # 储存每一块的熵值均值
    mean_map = cal_patch_mean(emap, patch_num, patch_size)

    gamma1 = 100 * (1 - maskratio)
    thresh1 = np.percentile(mean_map.cpu().numpy().flatten(), gamma1)
    
    # 用于保留最高ent值的部分，即前95%的低熵值不被选中，后5%的高熵值被选为保留部分
    gamma2 = 95
    thresh2 = np.percentile(mean_map.cpu().numpy().flatten(), gamma2)
        
    mean_map = mean_map.to(ndevice)
    # mean_map.cuda()
    # 小于thr1的值为1，此时前gamma1%的低平均熵值被选中为1
    ent_mask_mul1 = mean_map.le(thresh1).float()

    if hint == 1:      
        # 大于thr2的值为1，此时后（100-gamma2）%的高平均熵值被选中为1
        ent_mask_mul2 = mean_map.ge(thresh2).float()

        ent_mask_mul = ent_mask_mul1 + ent_mask_mul2
    else:
        ent_mask_mul = ent_mask_mul1
    

    # with open(output_debug_file, 'a') as f:
    #     print('ent_mask_mul0: ', ent_mask_mul.shape, file = f)

    ent_mask_mul = torch.unsqueeze(ent_mask_mul, 1)    
    # with open(output_debug_file, 'a') as f:
    #     print('ent_mask_mul1: ', ent_mask_mul.shape, file = f)

    # 打了mask的为0，不打的为1
    ent_mask_mul = resize(ent_mask_mul, size=(H, W))    
    # with open(output_debug_file, 'a') as f:
    #     print('ent_mask_mul2: ', ent_mask_mul.shape, file = f)

    # 此时打了mask的为1，不打的为0
    ent_mask = (-1) * ent_mask_mul + 1

    # new_data.cuda()
    ent_mask_mul = ent_mask_mul.to(ndevice)

    # print('nd: ', new_data.device)
    # print('mm: ', mean_map.device)
    # print('emm: ', ent_mask_mul.device)

    new_data = new_data * ent_mask_mul

    return new_data, ent_mask