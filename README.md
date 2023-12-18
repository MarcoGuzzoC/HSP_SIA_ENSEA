# CNN Implementation on GPU

Our goal is to implement the very famous LeNet-5 CNN on GPU using Cuda. 

![image](https://github.com/MarcoGuzzoC/HSP_SIA_ENSEA/assets/107397770/891ceaac-5d27-4ebf-9fea-e09f9dfe8c05)

To do so we need to implement several functions. At least we need to understand how to deal with matrices : initializing, adding, multiplying...
Then it's important to understand a CNN structure and so code the mandatory functions just as 2D-convolution, pooling (max / average), activation function (tanh here for simplicity), the flattening function and a way to create dense layers. 

**Consider the following links depending your profile xor your interest on our project**
- [Project follow-up](#progression-and-work-done-during-this-project)
- [How to run the code](#how-to-run-the-code)

# Progression and work done during this project
## Part 0 : Objectives

### Obj. 1: Learning CUDA
Because it's a CUDA project, it seems important to learn how we can compile a function on a NVIDIA GPU - which is the specificity of CUDA - and not only on CPU as all the code written until now. The most important point is to learn how to parallelize tasks dealing with blocks and threads.

### Obj. 2: Study algorithm complexity and GPU acceleration
Understand the advantages of using a GPU to execute complex algorithms or very high-dimensional calculations.

### Obj. 3: Discuss the limitations of CPU/GPU
Involves observing the limits of using a GPU, including data size, algorithm complexity and resource availability.

### Obj. 4: Implement CNN on GPU
Involves implementing a CNN, a type of Deep Learning algorithm commonly used for computer vision.

## Part 1 : Dealing with CUDA, Matrix Multiplication

### Sub-part 1.1 : First conception on CPU, adding, multiplying, displaying

### Sub-part 1.2 : Conversion into GPU functions

### Sub-part 1.3 : Complexity and comparaison

## Part 2 : First layers of our CNN, convolution and subsampling

### Sub-part 2.1 : Generating test data

### Sub-part 2.2 : 2D convolution

### Sub-part 2.3 : Subsampling

### Sub-part 2.4 : Activation function 

## Part 3 : Using Python to import the weights

### Sub-part 3.1 : Generating weights

### Sub-part 3.2 : Missing functions ? 

### Sub-part 3.3 : Putting everything together

# How to run the code

The best way to run this code is to have in the same folder the files [](),[]() and []() and write directly in the terminal :
```
nvcc blabla -o main && ./main 
```
