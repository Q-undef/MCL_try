# MCL_try
MCL相关尝试

## Try1：Entropy
- 尝试按照熵的计算方式生成模糊目标选择图，以像素为单位生成掩码（而不是以patch为单位）

  问题：

  - ACMT的模糊目标生成是在训练过程中生成的，每轮次的阈值会发生变化，生成的entMap也不同；而MCL的随机mask是在训练前生成的。
  - ACMT中加噪的图像是送入teacher模型，而MCL中mask的图像是被送入student模型训练生成概率预测；

  尝试想法：

  - 将随机掩码作为ACMT中teacher的噪声传入，并修改consistent loss的定义公式（怎样修改？）
  - 取消ACMT中的随机噪声，在生成entMap之后，将Map作为掩码重新经过student，按照MCL的方法计算loss，需要修改计算方法（因为Map是以像素为单位生成的，MCL中参与loss运算的mask是以patch为单位）
  - ......



## Try2：...
.......
