


def random_mask(data, patch_size, image_size=256, mask_ratio=0.75, mask_approach='zero'):       
    
    patch_num = image_size // patch_size
    s = [x for x in range(patch_num*patch_num)]
    random.shuffle(s)
    s = s[0:int(len(s)*mask_ratio)]     # 不要的部分，将这些patch打上mask
    mask = torch.zeros_like(data)
    new_data = data.clone()
    for i in range(len(s)):
        row = s[i] // patch_num
        col = s[i] % patch_num
        if mask_approach == 'zero':     # 将mask的部分置为0(black)
            new_data[:,:,row*patch_size:(row+1)*patch_size,col*patch_size:(col+1)*patch_size] = 0
        elif mask_approach == '1-mean':     # 将mask的部分置为（1-均值）
            mean = torch.mean(new_data[:,:,row*patch_size:(row+1)*patch_size,col*patch_size:(col+1)*patch_size])
            new_data[:,:,row*patch_size:(row+1)*patch_size,col*patch_size:(col+1)*patch_size] = 1-mean
            # print("1-mean: ", 1-mean)
        elif mask_approach == 'one':    # 将mask的部分置为1(white)
            new_data[:,:,row*patch_size:(row+1)*patch_size,col*patch_size:(col+1)*patch_size] = 1
        mask[:,:,row*patch_size:(row+1)*patch_size,col*patch_size:(col+1)*patch_size] = 1

    # print('new_data_shape:', new_data.shape)  # [12, 1, 256, 256]
    # print('mask_shape:', mask.shape)  # [12, 1, 256, 256]

    return new_data, mask
