# 笔记记录：

## 使用原图传入teacher模型得到概率输出，生成entMap之后，将Map作为掩码经过student，按照MCL的方法计算loss（try1_1.py）
### 出现问题1：数据消失
![7c88a00e745183686cdba636ed20d23](https://github.com/user-attachments/assets/91a2b00a-b202-4821-963a-36c1ed20bc6b)
![04ac12f6a34c67b00822968a45583f7](https://github.com/user-attachments/assets/d13bdca4-6275-4760-a206-760739daf231)
- 原因：是计算mask weight时出现问题，正在思考如何修改...
- 尝试解决1：注释掉weight计算，会报错: RuntimeError: grad can be implicitly created only for scalar output
![image](https://github.com/user-attachments/assets/ee251392-b6ac-42af-9862-49ee217998b8)
  - 原因：是ce_loss_mask = CrossEntropyLoss(reduction='none')的问题（将reduction设置为'none'计算出的loss直接返回n个样本的loss，即是一个元素个数和样本数相等的向量）
- 尝试解决2：恢复reduction为默认，并注释掉weight计算，可以正常运行
  - 运行结果：
  - entmask_noMaskWeight_val_7_labeled，max_mean_dice：89.17
  ![7f47ef2a031498ad47f1bdf5953a9ae](https://github.com/user-attachments/assets/8c919b3b-867e-4259-9970-0af822f99e65)
  - entmask_noMaskWeight_test_7_labeled，max_mean_dice：89.22
  ![cf8e96a3c49878d72e1f00ca580dcde](https://github.com/user-attachments/assets/69820a5e-927e-499a-8823-8e740cdbd7a1)

    
