#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>



//Création d'une matrice sur CPU
void MatrixInit(float *M, int n, int p, int q){
	// srand(time(NULL));
	for(int h=0;h<q;h++){
		for (int i =0; i<n; i++){
			for (int j =0; j<p; j++){
				float nbr = rand();
				float rdVal = nbr/RAND_MAX;
				rdVal = 2*rdVal -1;
				*(M+i*p+j)=rdVal;
			}
		}
	}
}
void MatrixInit2(float *M, int n, int p){
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

void MatrixPrint2(float *M, int n, int p){
	
	for (int i =0; i<n; i++){
		for (int j =0; j<p; j++){
			printf("%.2f \t ",*(M+i*p+j));
		}
		printf("\n");
	}
	printf("Next \n");	
}

void MatrixPrint(float *M, int n, int p, int q){
	for(int h=0;h<q;h++){
	for (int i =0; i<n; i++){
		for (int j =0; j<p; j++){
			printf("%.2f \t ",*(M+i*p+j));
		}
		printf("\n");
	}
	printf("Next \n");
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


__global__ void cudaMatrixMult2(float *M1, float *M2, float *Mout, int n, int p){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	float product=0;
	if (row<n && col<p){		
		for (int k=0; k<p; k++){
			product+=M1[row*p+k]*M2[k*n+col];
		}
		Mout[row*n+col]=product;
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

__global__ void cudaConv2D(float *Mat, float *Kernel, float *Output, int Mat_size, int Kernel_size){

	 int i = blockIdx.x; //32
	 int j = blockIdx.y; //32
	 int z = blockIdx.z; //6

	 int Output_size= Mat_size-Kernel_size+1;
	 float block_output=0;
	 for (int l=0;l<Kernel_size;l++){
	 	for (int k=0;k<Kernel_size;k++){
	 		block_output += *(Mat+(i+l)*Mat_size+j+k) * *(Kernel+l*Kernel_size+k+z*Kernel_size*Kernel_size);
	 	}
	 }
	 *(Output+i*Output_size+j+z*Kernel_size*Kernel_size)=block_output;
	
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

__device__ float activation_tanh(float M){
	return tanh(M);
}

__global__ void cudasubsamp(float *M_in, float *M_out, int channel, int p=28, int l=14){
	int i=blockIdx.x;
	int a=blockIdx.y*2;
	int b=blockIdx.z*2;
	float num=0;
	for (int k=0; i<2;i++){
		for (int m=0; m<2;m++){
			float numberOut=*(M_in+ i*p*p+(a+k)*p+(b+m));
			num+=numberOut;
		}
	}
	*(M_out+i*(l)*(l)+a/2*(l)+b/2)=activation_tanh(num/4);
}



int main() {
	int n=2;
	int p=2;
	
	float *M1 = (float *)malloc(n*p*sizeof(float));
	float *M2 = (float *)malloc(n*p*sizeof(float));
	float *M1gpu; 
	float *M2gpu; 
	// float *Moutadd = (float *)malloc(n*p*sizeof(float));
	// float *Moutadd_gpu ;
	float *Moutmult = (float *)malloc(n*n*sizeof(float));
	float *Moutmult_gpu;

	MatrixInit2(M1,n,p);
	MatrixInit2(M2,n,p);
	
	// MatrixInit2(Moutadd,n,p);
	MatrixInit2(Moutmult,n,n);

	
	cudaMalloc((void**) &M1gpu,n*p*sizeof(float));
	cudaMalloc((void**) &M2gpu,n*p*sizeof(float));
	// cudaMalloc((void**) &Moutadd_gpu,n*p*sizeof(float));
	cudaMalloc((void**) &Moutmult_gpu,n*n*sizeof(float));

	cudaMemcpy(M1gpu,M1,n*p*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(M2gpu,M2,n*p*sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(Moutadd_gpu,Moutadd,n*p*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Moutmult_gpu,Moutmult,n*n*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(M1,M1gpu,n*p*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(M2,M2gpu,n*p*sizeof(float), cudaMemcpyDeviceToHost);
	// cudaMemcpy(Moutadd,Moutadd_gpu,n*p*sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("Matrix M1: \n");
	MatrixPrint2(M1,n,p);
	printf("\n\n");
	printf("Matrix M2: \n");
	MatrixPrint2(M2,n,p);
	printf("\n\n");
	printf("Matrix M1*M2 (GPU): \n");
	// MatrixAdd(M1,M2,Moutadd,n,p);
	// MatrixPrint2(Moutadd,n,p);
	
	// MatrixMult(M1,M2,Moutmult,n);
	//printf("\n\n");
	//printf("Matrix M1*M2: \n");
	//MatrixPrint2(Moutmult,n,n);
	//printf("\n\n");
	
	// printf("\n\n");
	// cudaMatrixAdd<<<n*n,1>>>(M1gpu,M2gpu,Moutadd_gpu,n,n);
	// cudaMemcpy(Moutadd,Moutadd_gpu,n*p*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMatrixMult2<<<n*p,1>>>(M1gpu,M2gpu,Moutmult_gpu,n,p);
	cudaMemcpy(Moutmult,Moutmult_gpu,n*p*sizeof(float), cudaMemcpyDeviceToHost);
	MatrixPrint2(Moutmult,n,n);
	printf("\n\n");


	cudaFree(M1gpu);
	cudaFree(M2gpu);
	// cudaFree(Moutadd_gpu);
	cudaFree(Moutmult_gpu);

	free(M1);
	free(M2);
	// free(Moutadd);
	free(Moutmult);

	
	// float * input = (float *)malloc(12*12*sizeof(float));
	// float * output = (float *)malloc(6*6*sizeof(float));

	// float * test_data = (float *)malloc(32*32*sizeof(float));
	// float * test_kernel = (float *)malloc(2*5*5*sizeof(float));
	// float * test_output = (float *)malloc(2*28*28*sizeof(float));

	// MatrixInit(test_data,32,32,1);
	// MatrixInit(test_kernel,5,5,2);
	// MatrixInit(test_output,28,28,2);

	// MatrixInit(input, 12,12);
	// MatrixInit(input, 6,6);

	// subsamp(input, output, 12,6);
	// MatrixPrint(input, 12,12);
	// printf("\n\n");
	// MatrixPrint(output,6,6);

	// float *test_datagpu;
	// float *test_kernelgpu;
	// float *test_outgpu;


	// cudaMalloc((void**) &test_datagpu,32*32*sizeof(float));
	// cudaMalloc((void**) &test_kernelgpu,2*5*5*sizeof(float));
	// cudaMalloc((void**) &test_outgpu,2*28*28*sizeof(float));

	// cudaMemcpy(test_datagpu,test_data,32*32*sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(test_kernelgpu,test_kernel,2*5*5*sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(test_outgpu,test_output,2*28*28*sizeof(float), cudaMemcpyHostToDevice);

	// Conv2D(test_data, test_kernel, test_output, 32, 5);
	// MatrixPrint(test_output,28,28);
	// printf("\n\n");
	// printf("PreTest");
	// printf("\n\n");
	// cudaConv2D<<<gridsize,1>>>(test_datagpu,test_kernelgpu,test_outgpu,32,5);
	// printf("\n\n");
	// cudaMemcpy(test_output,test_outgpu,2*28*28*sizeof(float), cudaMemcpyDeviceToHost);
	// MatrixPrint(test_output,28,28,2);


	// cudaFree(test_datagpu);
	// cudaFree(test_kernelgpu);
	// cudaFree(test_outgpu);

	// free(test_data);
	// free(test_kernel);
	// free(test_output);

	// cudaConv2D<<<gridsize1,1>>>(raw_data_gpu,C1_kernel_gpu,C1_data_gpu,28,5);
	// cudasubsamp<<<gridsize2,1>>>(C1_data_gpu,S1_data_gpu,6,28,14);

	




}
