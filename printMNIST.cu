#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "multMatrix.cu"

#define WIDTH 32
#define HEIGHT 32

void charBckgrndPrint(char *str, int rgb[3]){
  printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
  printf("%s\033[0m",str);
}

void imgColorPrint(int height, int width, int ***img){
  int row, col;
  char *str="  ";
  for(row=0; row<height; row++){
    for(col=0; col<width; col++){
      charBckgrndPrint(str,img[row][col]);
    }
    printf("\n");
  }
}

void MatrixCopy(float *M, int *Mcopy, int n, int p){
  for(int i=0; i<n; i++){
    for(int j=0;j<p;j++){
      *(M+i*p+j)=(float) *(Mcopy+i*p+j);
    }
  }
}

__global__ void cudaFlatten(float *M, float *v, int n, int p){
  int i=blockIdx.x;
  int j=blockIdx.y;
  int z=blockIdx.z;
  for(int k=0;k<n;k++){
    for(int l=0;l<p;l++){
      *(v+i*n*p+j*p+z)=*(M+i*n*p*j*p+z);
    }
  } 
}

float* load_image(int width, int height){
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;
    FILE *fptr;
    float *arr;

    //Open File
    if((fptr = fopen("train-images.idx3-ubyte","rb")) == NULL){
      printf("Can't open file");
      exit(1);
    }

    //Read File
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);
    arr = (float*) malloc(height*width*sizeof(float));

    for (int i=0; i<height*width; i++){
      arr[i] = 0;
    }

    for (int i=0; i<nbRows*nbCols; i++){
      fread(&val, sizeof(unsigned char), 1, fptr);  
      arr[i] = (float) (((int) val) / 255.0);
    }

    return arr;
}

int main() {

  float *raw_data; 
	float *C1_data = (float *)malloc(6*28*28*sizeof(float));
	float *S1_data = (float *)malloc(6*14*14*sizeof(float));
	float *C1_kernel = (float *)malloc(6*5*5*sizeof(float));
	float *C1conv2 = (float *)malloc(16*10*10*sizeof(float));
	float *C1pool2 = (float *)malloc(16*5*5*sizeof(float));
	float *C1flatten = (float *)malloc(1*400*sizeof(float));
	float *C1dense = (float *)malloc(1*120*sizeof(float));
	float *C1dense1 = (float *)malloc(1*84*sizeof(float));
	float *C1dense2 = (float *)malloc(1*10*sizeof(float));	

	float *raw_data_gpu;
	float *C1_data_gpu;
	float *S1_data_gpu;
	float *C1_kernel_gpu;
	float *C1conv2_gpu;
	float *C1pool2_gpu;
	float *C1flatten_gpu;
	float *C1dense_gpu;
	float *C1dense1_gpu;
	float *C1dense2_gpu;

	dim3 gridsize1(28,28,6);
	dim3 gridsize2(14,14,6);
	dim3 gridsize3(5,5,6);
	dim3 gridsize4(10,10,16);
	dim3 gridsize5(5,5,16);


	//Init2(raw_data,32*32);
  raw_data = load_image(WIDTH, HEIGHT);


	zeros(C1_data,6*28*28);
	zeros(S1_data,6*14*14);
	Init2(C1_kernel,6*5*5);
	Init2(C1conv2,16*10*10);
	Init2(C1pool2,16*5*5);
	zeros(C1flatten,1*400);
	zeros(C1dense,1*120);
	zeros(C1dense1,1*84);
	zeros(C1dense2,1*10);


	cudaMalloc((void **)&raw_data_gpu,32*32*sizeof(float));
	cudaMalloc((void **)&C1_data_gpu,6*28*28*sizeof(float));
	cudaMalloc((void **)&S1_data_gpu,6*14*14*sizeof(float));
	cudaMalloc((void **)&C1_kernel_gpu,6*5*5*sizeof(float));
	cudaMalloc((void **)&C1conv2_gpu,16*10*10*sizeof(float));
	cudaMalloc((void **)&C1pool2_gpu,16*5*5*sizeof(float));
	cudaMalloc((void **)&C1flatten_gpu,1*400*sizeof(float));
	cudaMalloc((void **)&C1dense_gpu,1*120*sizeof(float));
	cudaMalloc((void **)&C1dense1_gpu,1*84*sizeof(float));
	cudaMalloc((void **)&C1dense2_gpu,1*10*sizeof(float));

	cudaMemcpy(raw_data_gpu,raw_data,32*32*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C1_data_gpu,C1_data,6*28*28*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(S1_data_gpu,S1_data,6*14*14*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C1_kernel_gpu,C1_kernel,6*5*5*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C1conv2_gpu,C1conv2,16*10*10*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C1pool2_gpu,C1pool2,16*5*5*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C1flatten_gpu,C1flatten,1*400*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C1dense_gpu,C1dense,1*120*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C1dense1_gpu,C1dense1,1*84*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C1dense2_gpu,C1dense2,1*10*sizeof(float), cudaMemcpyHostToDevice);

  MatrixPrint(raw_data,32,32,1);
  exit(EXIT_SUCCESS);
}
