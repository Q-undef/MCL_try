# MCL_try
MCL相关尝试

## Try1：Entropy
- 尝试按照熵的计算方式生成模糊目标选择图，以像素为单位生成掩码（而不是以patch为单位）

  问题：

  - ACMT的模糊目标生成是在训练过程中生成的，每轮次的阈值会发生变化，生成的entMap也不同；而MCL的随机mask是在训练前生成的。
  - ACMT中加噪的图像是送入teacher模型，而MCL中mask的图像是被送入student模型训练生成概率预测；

  尝试想法：

  - 将随机掩码作为ACMT中teacher的噪声传入，并修改consistent loss的定义公式（try1.py）（pass）
  - 使用原图传入teacher模型得到概率输出，生成entMap之后，将Map作为掩码经过student，按照MCL的方法计算loss（try1_1.py）
    - 出现问题：数据消失
     ![7c88a00e745183686cdba636ed20d23](https://github.com/user-attachments/assets/91a2b00a-b202-4821-963a-36c1ed20bc6b)
     ![04ac12f6a34c67b00822968a45583f7](https://github.com/user-attachments/assets/d13bdca4-6275-4760-a206-760739daf231)
    - 是计算mask weight时出现问题，正在思考如何修改...
     ![image](https://github.com/user-attachments/assets/ee251392-b6ac-42af-9862-49ee217998b8)
    - 注释掉weight计算，会报错: RuntimeError: grad can be implicitly created only for scalar output
    - 是reduction='none'的问题，why?（将reduction设置为'none'计算出的loss直接返回n个样本的loss，即是一个元素个数和样本数相等的向量）
  - ......



## Try2：...
.......




## 备注
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
