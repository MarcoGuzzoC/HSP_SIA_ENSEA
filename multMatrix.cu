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

__global__ void cudaMatrixMult(float *d_M1, float *d_M2, float *d_Mout, int n) 
{
int i = blockIdx.x;
int j = threadIdx.x;
*(d_Mout+i*n+j)=0;
for(int k = 0; k<n;k++)
{
	float d_M1_ct = *(d_M1+i*n+k);
	float d_M2_ct = *(d_M2+k*n+j);
	float d_Mout_ct = d_M1_ct*d_M2_ct;
	*(d_Mout+i*n+j)=d_Mout_ct;
	
}
}




int main() {
	int n=100;
	int p=8;
	
	float *M1 = (float *)malloc(n*p*sizeof(float));
	float *M2 = (float *)malloc(n*p*sizeof(float));
	float *Moutadd = (float *)malloc(n*p*sizeof(float));
	float *Moutadd_gpu = (float *)malloc(n*p*sizeof(float));
	float *Moutmult = (float *)malloc(n*n*sizeof(float));
	float *Moutmult_gpu = (float *)malloc(n*n*sizeof(float));
	
	MatrixInit(M1,n,p);
	MatrixInit(M2,n,p);
	MatrixInit(Moutadd,n,p);
	MatrixInit(Moutmult,n,p);
	MatrixInit(Moutadd_gpu,n,p);
	MatrixInit(Moutmult_gpu,n,p);
	
	printf("Matrix M1: \n");
	MatrixPrint(M1,n,p);
	printf("\n\n");
	printf("Matrix M2: \n");
	MatrixPrint(M2,n,p);
	printf("\n\n");
	printf("Matrix M1+M2: \n");
	MatrixAdd(M1,M2,Moutadd,n,p);
	MatrixPrint(Moutadd,n,p);
	
	MatrixMult(M1,M2,Moutmult,n);
	printf("\n\n");
	printf("Matrix M1*M2: \n");
	MatrixPrint(Moutmult,n,n);
	printf("\n\n");

	cudaMatrixMult<<<n*n,1>>>(M1,M2,Moutmult_gpu,n);
	MatrixPrint(Moutmult_gpu,n,n);
	printf("\n\n");
}
