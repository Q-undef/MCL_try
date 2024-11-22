# 笔记记录：
 
## MCL在val和test上的结果对比
- mcl_val_7_labeled，max_mean_dice：90.41
![49cc74f9f1fdd708ebf61d8103fc07e](https://github.com/user-attachments/assets/08067a8c-cea6-49cb-88c6-e26e199648d4)
- mcl_test_7_labeled，max_mean_dice：90.33
![0c0a61c2e8737f5eecbff3bf424742f](https://github.com/user-attachments/assets/f5053b9e-14e9-4d5d-b169-7a27ca0f60d5)



## 使用原图传入teacher模型得到概率输出，生成entMap之后，将Map作为掩码经过student，按照MCL的方法计算loss（try1_1.py）
### 出现问题1：数据消失
![7c88a00e745183686cdba636ed20d23](https://github.com/user-attachments/assets/91a2b00a-b202-4821-963a-36c1ed20bc6b)
![04ac12f6a34c67b00822968a45583f7](https://github.com/user-attachments/assets/d13bdca4-6275-4760-a206-760739daf231)
- 查找：是计算mask weight时出现问题
- 尝试解决1：注释掉weight计算，会报错: RuntimeError: grad can be implicitly created only for scalar output
![image](https://github.com/user-attachments/assets/ee251392-b6ac-42af-9862-49ee217998b8)
  - 原因：是ce_loss_mask = CrossEntropyLoss(reduction='none')的问题（将reduction设置为'none'计算出的loss直接返回n个样本的loss，即是一个元素个数和样本数相等的向量）reduction标记为none时不会求平均值，而是对于每个样本求一个loss，设置为none时是在weight计算时手动求平均，因此会出现注释掉weight计算过程后报错的现象。
- 尝试解决2：恢复reduction为默认，并注释掉weight计算，可以正常运行
  - 不添加weight计算模块的运行结果：
  - entmask_noMaskWeight_val_7_labeled，max_mean_dice：89.17
  ![7f47ef2a031498ad47f1bdf5953a9ae](https://github.com/user-attachments/assets/8c919b3b-867e-4259-9970-0af822f99e65)
  - entmask_noMaskWeight_test_7_labeled，max_mean_dice：89.22
  ![cf8e96a3c49878d72e1f00ca580dcde](https://github.com/user-attachments/assets/69820a5e-927e-499a-8823-8e740cdbd7a1)

### 可能原因1：数据消失是因为weight计算时有除法，分母可能过于小，导致出现除0现象
- 尝试优化1：查看mask的像素数量，根据高于阈值的数量占比，修改阈值
- ...

