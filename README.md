<div align="center">

![Tiny_po](https://github.com/user-attachments/assets/82a6488e-9434-4192-a97c-0d4af4823f8d)

</div>

Minimal implementation of **Group Relative Policy Optimization (GRPO)** (DeepSeek) from scratch. No complicated file structure—just a **simple, hackable implementation** with few scripts for better understanding of the algorithm.  

Inspired by the implementation by [@aburkov](https://github.com/aburkov). This implementation optimizes **memory usage** during training by:  
- Using **chunk-wise softmax operations**  
- Leveraging **mixed precision training**  
Together, these techniques reduce memory usage by **50%**, enabling GRPO to run on singel GPU while achieving strong results on **math datasets**. 

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

