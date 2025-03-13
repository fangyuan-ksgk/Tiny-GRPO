<div align="center">

![Tiny_po](https://github.com/user-attachments/assets/82a6488e-9434-4192-a97c-0d4af4823f8d)

</div>

Minimal implementation of GRPO (https://arxiv.org/abs/2402.03300) from scratch. Inspired by implementation by @aburkov. I optimize the memory usage in the training process, specifically the softmax operation into chunk-wise operations and uses mixed precision training, together they reduces memory usage by 50% and one could run GRPO with ~20GB GPU memory and obtain nice results on math datasets. No complicated file structure, just a simple implementation that's easy to hack to your need and further understanding of the algorithm. 

🔥 Features
✅ Lightweight & Easy to Understand – Simple structure, no complex file organization.
✅ Memory Optimized – Chunk-wise softmax operations & mixed precision training reduce memory usage by 50%.
✅ Efficient Training – Run GRPO with ~20GB GPU memory and achieve strong results on math datasets.
✅ Hackable – Easily modify and experiment with the core algorithm.

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

🤝 Contributing
Feel free to submit issues, PRs, or suggestions to improve the implementation!

⚡ Acknowledgments
Inspired by @aburkov's work in The LM Book (https://github.com/aburkov/theLMbook).

