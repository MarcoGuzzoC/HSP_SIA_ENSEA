#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>




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


int main() {
	int n=8;
	int p=8;
	
	float *M1 = (float *)malloc(n*p*sizeof(float));
	float *M2 = (float *)malloc(n*p*sizeof(float));
	float *Moutadd = (float *)malloc(n*p*sizeof(float));
	float *Moutmult = (float *)malloc(n*p*sizeof(float));
	
	MatrixInit(M1,n,p);
	MatrixInit(M2,n,p);
	MatrixInit(Moutadd,n,p);
	MatrixInit(Moutmult,n,p);
	
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
}
