#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1



//Création d'une matrice sur CPU
void MatrixInit(float *M, int n, int p){
	// srand(time(NULL));
	for (int i =0; i<n; i++){
		for (int j =0; j<p; j++){
			float nbr = rand();
			float rdVal = nbr/RAND_MAX;
			rdVal = 2*rdVal -1;
			*(M+i*p+j)=rdVal;
		}
	}
}

void zeros(float *M, int N){

	for (int i=0;i<N;i++){
		*(M+i)=0;
	}
}

void Init2(float *M, int N){

	for (int i=0;i<N;i++){
		float nbr = rand();
		float rdVal = nbr/RAND_MAX;
		*(M+i)=rdVal;
	}
}



//Affichage d'une matrice sur CPU
void MatrixPrint(float *M, int n, int p){
	for (int i =0; i<n; i++){
		for (int j =0; j<p; j++){
			printf("%.2f \t ",*(M+i*p+j));
		}
		printf("\n");
	}

}


//Addition de deux matrices sur CPU
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
	for (int i =0; i<n;i++){
		for (int j =0; j<p;j++){
			 *(Mout+i*p+j)=*(M1+i*p+j)+*(M2+i*p+j);
		}
	}
}


//Addition de deux matrices sur GPU
__global__ void cudaMatrixAdd(float *d_M1, float *d_M2, float *d_Mout, int n, int p){
int i = blockIdx.x;
int j = threadIdx.x;
float d_M1_ct = *(d_M1+i*p+j);
float d_M2_ct = *(d_M2+i*p+j);
float d_Mout_ct = d_M1_ct + d_M2_ct;
*(d_Mout+i*p+j)=d_Mout_ct;	
}



//Multiplication de deux matrices sur CPU
void MatrixMult(float *M1, float *M2, float *Mout, int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			float tmp = 0;
			for(int k=0;k<n;k++){
				tmp+=*(M1+i*n + k) * *(M2+k*n+j);	
			}
			*(Mout+i*n+j) = tmp;
		}
	}
}

void HadamardProd(float *M1, float *M2, float *Mout, int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			*(Mout+i*n+j) = *(M1+i*n+j)* *(M2+i*n+j);
		}
	}
}

__global__ void cudaMatrixMult(float *d_M1, float *d_M2, float *d_Mout, int n) 
{
int i = blockIdx.x;
int j = threadIdx.x;
*(d_Mout+i*n+j)=0;
for(int k = 0; k<n;k++){
	float d_M1_ct = *(d_M1+i*n+k);
	float d_M2_ct = *(d_M2+k*n+j);
	float d_Mout_ct = d_M1_ct*d_M2_ct;
	*(d_Mout+i*n+j)=d_Mout_ct;
	}
}


__global__ void cudaMatrixMult2(float *M1, float *M2, float *Mout, int n) 
{
int row = blockIdx.y*blockDim.y+threadIdx.y;
int col = blockIdx.x*blockDim.x+threadIdx.x;
if (row<n && col<n){
	float product=0;
	for (int k=0; k<n; k++){
		product+=M1[row*n+k]*M2[k*n+col];
		}
	Mout[row*n+col]=product;
	}
}


__global__ void cudaHadamardProd(float *M1, float *M2, float *Mout, int n) 
{
int row = blockIdx.y*blockDim.y+threadIdx.y;
int col = blockIdx.x*blockDim.x+threadIdx.x;
if (row<n && col<n){
	Mout[row*n+col]=M1[row*n+col]*M2[row*n+col];
	}
}




void Conv2D(float *Mat, float *Kernel, float *Output, int Mat_size, int Kernel_size){
/*

Returns the valid convolution with stride=1, padding=1 between Matrix and Kernel, computed on CPU.
Inputs:

      -Mat: Input square matrix of size Mat_size*Mat_size

      -Kernel: Input square convolution kernel of size Kernel_size*Kernel_size

      -Output: Output array of shape (Mat_size-Kernel_size+1)*(Mat_size-Kernel_size+1)

      -Mat_size: Matrix size

      -Kernel_size: Kernel size
*/
	int Output_size = Mat_size - Kernel_size +1;
	for (int horizontal_block_id=0; horizontal_block_id<Output_size;horizontal_block_id++){
		for (int vertical_block_id=0;vertical_block_id<Output_size;vertical_block_id++){
			float block_output=0;
			for (int i=0;i<Kernel_size;i++){

				for (int j=0;j<Kernel_size;j++){

					block_output=block_output + *(Mat+(horizontal_block_id+i)*Mat_size+vertical_block_id+j) * *(Kernel+i*Kernel_size+j);
				}
			}
			*(Output+horizontal_block_id*Output_size+vertical_block_id)=block_output;
		}
	}
}


void subsamp(float *img, float *result, int input_size, int output_size)
{
	float sum;
	for(int i = 0; i<output_size ; i++){
		for(int j = 0; j<output_size ; j++){
			sum = 0;
			for(int k = 0; k<2 ; k++){
				for(int l = 0; l<2 ; l++){
					sum +=*(img + (i*2+k)*input_size+j*2+l);
}
}
			*(result+i*output_size+j)=sum/4.0;
}
}
}



int main() {
	// int n=10000;
	// int p=10000;
	
	// float *M1 = (float *)malloc(n*p*sizeof(float));
	// float *M2 = (float *)malloc(n*p*sizeof(float));
	// float *M1gpu; 
	// float *M2gpu; 
	// float *Moutadd = (float *)malloc(n*p*sizeof(float));
	// float *Moutadd_gpu ;
	// float *Moutmult = (float *)malloc(n*n*sizeof(float));
	// float *Moutmult_gpu;

	// MatrixInit(M1,n,p);
	// MatrixInit(M2,n,p);
	
	// MatrixInit(Moutadd,n,p);
	// MatrixInit(Moutmult,n,n);

	
	// cudaMalloc((void**) &M1gpu,n*p*sizeof(float));
	// cudaMalloc((void**) &M2gpu,n*p*sizeof(float));
	// cudaMalloc((void**) &Moutadd_gpu,n*p*sizeof(float));
	// cudaMalloc((void**) &Moutmult_gpu,n*n*sizeof(float));

	// cudaMemcpy(M1gpu,M1,n*p*sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(M2gpu,M2,n*p*sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(Moutadd_gpu,Moutadd,n*p*sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(Moutmult_gpu,Moutmult,n*n*sizeof(float), cudaMemcpyHostToDevice);

	// printf("Matrix M1: \n");
	// MatrixPrint(M1,n,p);
	// printf("\n\n");
	// printf("Matrix M2: \n");
	// MatrixPrint(M2,n,p);
	// printf("\n\n");
	// printf("Matrix M1+M2: \n");
	// MatrixAdd(M1,M2,Moutadd,n,p);
	// MatrixPrint(Moutadd,n,p);
	
	// MatrixMult(M1,M2,Moutmult,n);
	// printf("\n\n");
	// printf("Matrix M1*M2: \n");
	// MatrixPrint(Moutmult,n,n);
	// printf("\n\n");
	
	// int N=4;
	// dim3 dimBlock(N,N);
	// dim3 dimGrid(ceil(N/16.0),ceil(N/16.0));
	// printf("\n\n");
	// cudaMatrixAdd<<<n*n,1>>>(M1gpu,M2gpu,Moutadd_gpu,n,n);
	// cudaMatrixMult<<<n*n,1>>>(M1gpu,M2gpu,Moutmult_gpu,n);
	// MatrixPrint(Moutmult_gpu,n,n);
	// printf("\n\n");


	// cudaFree(M1gpu);
	// cudaFree(M2gpu);
	// cudaFree(Moutadd_gpu);
	// cudaFree(Moutmult_gpu);

	// free(M1);
	// free(M2);
	// free(Moutadd);
	// free(Moutmult);

	float *raw_data = (float *)malloc(32*32*sizeof(float));
	float *C1_data = (float *)malloc(6*28*28*sizeof(float));
	float *S1_data = (float *)malloc(6*14*14*sizeof(float));
	float *C1_kernel = (float *)malloc(6*5*5*sizeof(float));
	
	Init2(raw_data,32*32);
	zeros(C1_data,6*28*28);
	zeros(S1_data,6*14*14);
	Init2(C1_kernel,6*5*5);

	float * input = (float *)malloc(12*12*sizeof(float));
	float * output = (float *)malloc(6*6*sizeof(float));

	MatrixInit(input, 12,12);
	MatrixInit(input, 6,6);

	subsamp(input, output, 12,6);
	MatrixPrint(input, 12,12);
	printf("\n\n");
	MatrixPrint(output,6,6);

}
