# CMU10-714

Implementations of Needle, a deep learning library from scratch.

- Course Website: <https://dlsyscourse.org/>

## Course Introduction

The goal of this course is to provide students an understanding and overview of the “full stack” of deep learning systems, ranging from the high-level modeling design of modern deep learning systems, to the basic implementation of automatic differentiation tools, to the underlying device-level implementation of efficient algorithms. Throughout the course, students will design and build from scratch a complete deep learning library, capable of efficient GPU-based operations, automatic differentiation of all implemented functions, and the necessary modules to support parameterized layers, loss functions, data loaders, and optimizers. Using these tools, students will then build several state-of-the-art modeling methods, including convolutional networks for image classification and segmentation, recurrent networks and self-attention models for sequential tasks such as language modeling, and generative models for image generation.

## Demo

Using Needle, a ResNet9 model can be trained and evaluated locally:

<div style="display: flex; justify-content: space-around;">
    <img src="hw4/ResNet9.png" alt="ResNet9" style="width: 30%;">
    <img src="hw4/demo.png" alt="demo" style="width: 60%;">
</div>

## Final Project

*Plan: Add multi-GPU training backend using MPI/NCCL. 
