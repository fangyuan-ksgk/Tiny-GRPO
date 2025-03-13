<div align="center">

![Tiny_po](https://github.com/user-attachments/assets/82a6488e-9434-4192-a97c-0d4af4823f8d)

</div>

Minimal implementation of GRPO (https://arxiv.org/abs/2402.03300) from scratch. Inspired by implementation by @aburkov (https://github.com/aburkov/theLMbook). I optimize the memory usage in the training process, specifically the softmax operation into chunk-wise operations and uses mixed precision training, together they reduces memory usage by 50% and one could run GRPO on consumer-grade GPUs and obtain nice results on math datasets. No complicated file structure, just a simple implementation that's easy to hack to your need and further understanding of the algorithm. 

set up environment 
```bash
bash set.sh 
```

train GRPO on gsm8k dataset (Qwen-2.5-Instruct-1.5B)
```bash
python train.py
```

test model output 
```bash
python test.py
```


