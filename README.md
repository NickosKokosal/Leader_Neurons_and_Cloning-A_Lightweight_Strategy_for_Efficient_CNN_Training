Boiled Egg Theory: Efficient CNN Training via Leader Neuron Cloning
Overview

This project introduces a novel training strategy for Convolutional Neural Networks (CNNs) inspired by 
The Bully Algorithm which is a distributed systems process for dynamically electing a coordinator (leader). 
The core idea is to reduce computational cost during training by electing a small number of leader neurons to learn directly from the data 
and then cloning their learned weights to the remaining neurons and in parallel I keep some neurons independent.

The approach aims to preserve model accuracy while significantly lowering the number of training updates—saving time and energy.

Key Concepts

Elected Leaders: A subset of neurons (e.g. filters) are selected to act as "leaders" during training.

Weight Cloning: Once trained, leader weights are copied to non-leader neurons, reducing redundant computations.

Independent Neurons : Some neurons are allowed to train freely alongside the leaders to introduce diversity.

Bottleneck Layer: A structural addition to limit dimensionality and force feature compression, improving efficiency.

Experimental Scenarios

Six different training scenarios were evaluated on the CIFAR-100 dataset:


scenarios = [
    {"desc": "(1)baseline training", "leaders": 0, "independents": 0, "clone": False},
    {"desc": "(2)leaders only", "leaders": 20 , "independents": 0, "clone": True},
    {"desc": "(3)leaders + independents", "leaders": 30, "independents": 10, "clone": True},  # 40
    {"desc": "(4)leaders + independents", "leaders": 25, "independents": 19, "clone": True},  # 44
    {"desc": "(5)leaders + independents", "leaders": 40, "independents": 15, "clone": True},  # 55
    {"desc": "(6)leaders + independents", "leaders": 50, "independents": 10, "clone": True},  # 60
]

What we gain?

Fewer updates → Reduced compute

Faster training

New neural architecture idea applicable to resource-constrained environments

Easy to integrate into PyTorch workflows



Requirements

Python 3.9+

PyTorch

NumPy

Matplotlib

Citation & Acknowledgements

If you use this idea or code, please cite this repository or mention:

Boiled Egg Theory: Efficient CNN Training via Leader Neuron Cloning
Nikolaos Anastasios Kokosalakis, 2025