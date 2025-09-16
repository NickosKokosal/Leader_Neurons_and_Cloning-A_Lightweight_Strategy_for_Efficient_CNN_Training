Boiled Egg Theory: Efficient CNN Training via Leader Neuron Cloning
Overview

Imagine a boiled egg.
The yolk at the center represents the core leaders — they are few but responsible for learning and guiding the rest.
The egg white symbolizes the clones — abundant, surrounding the yolk, and shaped by it.
The shell protects the egg but acts independently — like our independent neurons, which are updated but not cloned.

This "Boiled Egg Theory" proposes a computationally efficient training strategy for CNNs, 
where only a subset of core filters (leaders) are updated and cloned to the rest, 
drastically reducing computational load without compromising learning capacity.

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

Three different training scenarios were evaluated on the CIFAR-100 dataset:

scenarios = [
    # (1) Baseline: without roles -> full conv
    {"desc": "(1)baseline training",      "per_layer_p": None,                "clone": False},
    # (2) Mild per-layer
    {"desc": "(2)mild per-layer",         "per_layer_p": mild_per_layer,      "clone": True},
    # (3) Aggresive per-layer
    {"desc": "(3)aggressive per-layer",   "per_layer_p": aggressive_per_layer,"clone": True},
]





Three different training scenarios were evaluated on the MNIST dataset:

scenarios = [
    # (1) Baseline: without roles -> full conv
    {"desc": "(1)baseline training",      "per_layer_p": None,                "clone": False},
    # (2) Mild per-layer
    {"desc": "(2)mild per-layer",         "per_layer_p": mild_per_layer,      "clone": True},
    # (3) Aggresive per-layer
    {"desc": "(3)aggressive per-layer",   "per_layer_p": aggressive_per_layer,"clone": True},
]




What we gain with the 128 channels version?

Fewer updates 

We can observe that the neurons which have the role of clone don't affect or change so much the behavior of the NN.

New neural architecture idea applicable to resource-constrained environments

Easy to integrate into PyTorch workflows



Requirements

Python 3.9+

PyTorch

NumPy

Matplotlib

Last but not least

I will continue trying new scenarios and CNN versions, and I will update the results.

If you ever use this idea or code, please cite this repository or mention:

Boiled Egg Theory: Efficient CNN Training via Leader Neuron Cloning
Nikolaos Anastasios Kokosalakis, 2025