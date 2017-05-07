#include <stdio.h>

#define ELEMS 20

__global__ void kernel_vecadd(int *a, int *b, int *c){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        c[index] = a[index] + b[index];
}

int main(){
	int *ha,*hb,*hc;
        int *da,*db,*dc;
        int i;
	int size = ELEMS*sizeof(int);

        ha = (int*)malloc(size);
        hb = (int*)malloc(size);
        hc = (int*)malloc(size);

        cudaMalloc((void **)&da,size);
        cudaMalloc((void **)&db,size);
        cudaMalloc((void **)&dc,size);

        for (i=0;i<ELEMS;i++){
                ha[i] = i;
                hb[i] = ELEMS-i-1;
        }

	cudaMemcpy(da,ha,size,cudaMemcpyHostToDevice);
        cudaMemcpy(db,hb,size,cudaMemcpyHostToDevice);

        kernel_vecadd<<<2,10>>>(da,db,dc);

        cudaMemcpy(hc,dc,size,cudaMemcpyDeviceToHost);

        for (i=0;i<ELEMS;i++)
                printf("%2d + %2d = %2d\n",ha[i],hb[i],hc[i]);

        return 0;
}

