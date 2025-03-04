# MCL_try
MCL-Entropy mask相关尝试

## try1 pixel entmask
- 尝试按照熵的计算方式生成模糊目标选择图，以像素为单位生成掩码

  问题：

  - ACMT的模糊目标生成是在训练过程中生成的，每轮次的阈值会发生变化，生成的entMap也不同；而MCL的随机mask是在训练前生成的。
  - ACMT中加噪的图像是送入teacher模型，而MCL中mask的图像是被送入student模型训练生成概率预测；

  尝试想法：

  - 将随机掩码作为ACMT中teacher的噪声传入，并修改consistent loss的定义公式（try1.py）（pass）
  - 使用原图传入teacher模型得到概率输出，生成entMap之后，将Map作为掩码经过student，按照MCL的方法计算loss

## try2 patch entmask
- 一：先按像素筛选出mask像素，然后通过分块计数将mask化为patch
- 二：计算每一个patch的熵值均值，将patch按熵值均值从高到低排序，按照一定比例选取前百分之r的patch作为mask
- 三：对于教师模型encoder输出的特征图，使其通过一个另外的注意力decoder计算自注意力，引导mask的生成
- ......

# 问题、笔记、结果详见'notes.md'



### 在vscode管理github时遇到的报错（持续更新ing）
- error: RPC failed; curl 16 Error in the HTTP2 framing layer fatal: expected flush after ref listing
  - 强制 git 使用 HTTP 1.1：git config --global http.version HTTP/1.1
  - 然后使用git操作
  - 将其设置回 HTTP2：git config --global http.version HTTP/2
- fatal: unable to access https://github.com/xxxx/: gnutls_handshake() failed: The TLS connection was non-properly terminated.
  - git config --global  --unset https.https://github.com.proxy
  - git config --global  --unset http.https://github.com.proxy
- fatal: unable to access https://github.com/xxx/: Failed to connect to github.com port 443: Connection timed out
  - git config --global --unset http.proxy
  - git config --global --unset https.proxy
- fatal: Not possible to fast-forward, aborting.
  - git pull origin main --rebase // main指的是当前修改的分支，请修改当前所修改的分支名称
- ...
