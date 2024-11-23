# Efficient Training of Spiking Neural Networks with Multi-Parallel Implicit Stream Architecture  
## 1 Abstract  
  Spiking neural networks (SNNs) are a novel type of bio-plausible neural network with energy efficiency. However, SNNs are non-differentiable and the training memory costs increase with the number of simulation steps. To address these challenges, this work introduces an implicit training method for SNNs inspired by equilibrium models. Our method relies on the multi-parallel implicit stream architecture (MPIS-SNNs). In the forward process, MPIS-SNNs drive multiple fused parallel implicit streams (ISs) to reach equilibrium state simultaneously. In the backward process, MPIS-SNNs solely rely on a single-time-step simulation of SNNs, avoiding the storage of a large number of activations. Extensive experiments on N-MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 demonstrate that MPIS-SNNs exhibit excellent characteristics such as low latency, low memory cost, low firing rates, and fast convergence speed, and are competitive among latest efficient training methods for SNNs.  
## 2 Run  
For static datasets, train/test using "static_code".  
  
For neuromorphic datasets, train/test using "neuro_code".  
  
## 3 Credits  
Our work uses the code from [IDE-FSNN](https://github.com/pkuxmq/IDE-FSNN) as a foundation, and we have expanded upon this basis.
