# CMU10-714

Implementations of Needle, a deep learning library from scratch.

- Course Website: <https://dlsyscourse.org/>

## Course Introduction

The goal of this course is to provide students an understanding and overview of the “full stack” of deep learning systems, ranging from the high-level modeling design of modern deep learning systems, to the basic implementation of automatic differentiation tools, to the underlying device-level implementation of efficient algorithms. Throughout the course, students will design and build from scratch a complete deep learning library, capable of efficient GPU-based operations, automatic differentiation of all implemented functions, and the necessary modules to support parameterized layers, loss functions, data loaders, and optimizers. Using these tools, students will then build several state-of-the-art modeling methods, including convolutional networks for image classification and segmentation, recurrent networks and self-attention models for sequential tasks such as language modeling, and generative models for image generation.

## Demo

Using Needle, a CNN model (ResNet9) can be trained and evaluated locally:

<div style="display: flex; gap: 10px;">
    <img src="hw4/ResNet9.png" alt="ResNet9" style="height: 400px; width: calc(50% - 5px);">
    <img src="hw4/Demo.png" alt="demo" style="height: 400px; width: calc(50% - 5px);">
</div>

Also works for language model (RNN/LSTM/Transformer), but w. way longer time to train.


## Final Project

Plan: Add efficient transformer inference backend / multi-GPU training w. MPI.

(*Co-project for Brown APMA2822B)

## Reference

I got to know this course thanks to [csdiy](https://csdiy.wiki/) and I referenced some implementations in hw4 & hw4_extra from [this repo](https://github.com/PKUFlyingPig/CMU10-714) and this [PR](https://github.com/dlsyscourse/hw4_extra/pull/1).
