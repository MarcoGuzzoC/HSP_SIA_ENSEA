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

_All the functions are available in [this file](multMatrix.cu)._
_Each time we allocate the memory using malloc for the work on CPU or cudamalloc for the work on GPU._

### Sub-part 1.1 : First conception on CPU, initializing, adding, multiplying, displaying

_**Initialization :**_

First of all, the function `MatrixInit` allows us to generate an initialize a $n \times p$ matrix with random values between $-1$ and $1$. 
We consider then a function `zeros` doing exactly as the same name function in Python, filling with $0$ the given vector/matrix.
Finally, we have a last `Init2` function for initializing $n \times n$ matrices with random values between $0$ and $1$. 

_Code output :_

`MatrixInit :`

![matrix_print](https://github.com/MarcoGuzzoC/HSP_SIA_ENSEA/assets/107397770/329b9690-3c0d-493f-bf34-17f627d3fc69)


_**Displaying :**_

To display any $n \times p$ matrix, we use the function `MatrixPrint` (upgraded in during the 3rd session to deal with $n \times p \times q$ matrices. Nothing more to say here, it just display all the elements with a matrix shape (tabulations & back to line). 

We already saw it working during the past section to display our initialized matrices. 

_**Adding :**_

The function `MatrixAdd` allows us to do the simple term to term matrix sum for two $n \times p$ matrices.

_Example :_

$$\begin{pmatrix} 0.68 & -0.21 \\
0.57 & 0.60 \end{pmatrix} + \begin{pmatrix} 0.82 & -0.60 \\ 
-0.33 & 0.54 \end{pmatrix} = \begin{pmatrix} 1.5 & -0.82 \\
0.24 & 1.13 \end{pmatrix}$$

_Code output :_

![image](https://github.com/MarcoGuzzoC/HSP_SIA_ENSEA/assets/107397770/1bfe880e-fc6a-4058-8559-4a6661feb727)

_Duration :_ 5ms

_**Multiplying :**_

The function `MatrixMult` allows us to do the matrix product between two $n \times n$ matrices. 

_Example :_

$$\begin{pmatrix} 0.68 & -0.21 \\
0.57 & 0.60 \end{pmatrix} + \begin{pmatrix} 0.82 & -0.60 \\ 
-0.33 & 0.54 \end{pmatrix} = \begin{pmatrix} 0.63 & -0.52 \\
0.27 & -0.02 \end{pmatrix}$$

_Code output :_

![image](https://github.com/MarcoGuzzoC/HSP_SIA_ENSEA/assets/107397770/340b160d-52d7-43ce-ae7d-bd9769b6cd28)

_Duration :_ 4ms

### Sub-part 1.2 : Conversion into GPU functions

_**Adding :**_

Nothing very different here from what we did before, the function `cudaMatrixAdd` is exactly the same as the simple `MatrixAdd`. The only different thing is that pointers and dimensions are managed by the gridDim and the bockDim playing with threads and blocks.

_**Multiplying :**_

Nothing very different here from what we did before, the function `cudaMatrixAdd` is exactly the same as the simple `MatrixAdd`. The only different thing is that pointers and dimensions are managed by the gridDim and the bockDim playing with threads and blocks.

### Sub-part 1.3 : Complexity and comparaison

We compare the duration of each fuction to discuss the interest and limitation of the both methods. To do so, we will use the `nvprof` method. Then we change the size of our matrices and do it again to see how it changes. 

_Code output :_

$N=30 \times 30$

![image](https://github.com/MarcoGuzzoC/HSP_SIA_ENSEA/assets/107397770/65709fa9-4bae-482b-b5d3-1473feeb2f3b)

We can see that the most time consumming task is the data transfer between CPU and GPU. 
For a matrix with that size, it's useless to use the GPU and the operation is slower than on a regular CPU.
Nonetheless, when you increase the size, (ex: $10^6 \times 10^6$) then it's better to use the GPU functions.

## Part 2 : First layers of our CNN, convolution and subsampling

### Sub-part 2.1 : Generating test data

We create some matrices mandatory for the rest of our code :
- `raw_data` : $32 \times 32$ matrix representing an input image initialized with some ones and zeros.
- `C1_data` : $6 \times 28 \times 28$ matrix full of $0$ corresponding to the output of the first convolution operation.
- `C1_kernel` : $6 \times 5 \times 5$ matrix initialized with ones and zeros 

### Sub-part 2.2 : 2D convolution

It's mandatory to know how to convolve things when you deal with neural networks. So we create a function to do it on GPU.

### Sub-part 2.3 : Subsampling

To subsample our images, we do a $2\times 2$ average pooling. 

### Sub-part 2.4 : Activation function 

Several activation functions were done : we use the `tanh` for the intermediate layers and then `softmax` for the final classification. 

## Part 3 : Using Python to import the weights

We train our model on Python, using Keras and the traditional architecture given by LeNet5.
The input images are $32 \times 32$ gray scale images.
The model is given by the following table :

|Layer (type) |Output shape| Number of parameters|
|:-----------:|:----------:|:-------------------:|
|conv2d|$28 \times 28 \times 6$|156|
|average_pooling2d|$14 \times 14 \times 6$|0|
|conv2d|$10 \times 10 \times 16$|2416|
|average_pooling2d|$5 \times 5 \times 16$|0|
|flatten|$1 \times 400$|0|
|dense|$1 \times 120$|48120|
|dense|$1 \times 84$|10164|
|dense|$1 \times 10$|850|


### Sub-part 3.1 : Generating weights

We train our model on a [Python notebook](LeNet5.py) to obtain a Tensor of values corresponding to the trained weights that we can put in our matrices to initialize them instead of using $0$, $1$ and $-1$.

### Sub-part 3.2 : Missing functions ? 

As we saw while running the model, we didn't have all the needed function to run our entire model. We still need to code the fuctions `flatten` and `dense` to have the entire pipeline. 

The `flatten` function is quite simple to implement, we just convert our tensor to a list putting every element on the same level. We consider it line by line and layer by layer (just as keras or TensorFlow does).

The `dense` function is then just a convolution and some biais given by the calculated weights.  

### Sub-part 3.3 : Putting everything together

When the different weights were generated by our primary training on Python, we modify directly our matrices in our code to make our model more accurate. 
The complete and corrected version of the code done during session can be found in the final file [main](main.cu) not to have scattered functions etc...

# How to run the code

Simply download the [main](main.cu) file and run the following command in your terminal :

```
nvcc main.cu -o prog && ./prog
```
