#include <stdio.h>

#define SIZE 1024*1024*4

__global__ void reduce0(unsigned int *g_idata, unsigned int *g_odata, long size){

  // dynamically allocated shared memory                                                                       
  extern __shared__ unsigned int sdata[];

  // set up thread ids: within a block, and across the grid                                                    
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  //initialize the shared sum array                                                                            
  sdata[tid] = 0;

  sdata[tid] = g_idata[i];

  // synchronize: so that all threads within a block have loaded                                               
  // the elements from GDRAM to the shared mem                                                                 
  __syncthreads();

  
  //for(unsigned int s=1; s < blockDim.x; s *= 2) {
  //   // version 1
    // if (tid % (2*s) == 0) {
    //   sdata[tid] += sdata[tid + s];
    // }
  //   //version 2
    // int index = 2 * s * tid;
  //    if (index < blockDim.x) {
  //      sdata[index] += sdata[index + s];
  //    }
  //   __syncthreads(); // let one phase complete before the next starts                            
    
  // }
  // //version 3
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}



#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}


int main(void){

  cudaEvent_t start_event, stop_event;
  float cuda_elapsed_time;

  // define a vector of certain "SIZE" and set values to "1"                                                   
  // This means that the final sum will be equal to size.                                                      
  // Easy for debugging!                                                                                       
  // Dayum, am I smart or am I smart ?!  
  unsigned int *h_i, *h_o;
  long size = SIZE;
  h_i = (unsigned int *) malloc (sizeof(unsigned int)*SIZE);
  for (unsigned int  i = 0; i < SIZE; i ++) 
    h_i[i] = 1;

  // declare a device array and copy the host array to the device array                                        
  // If the size of an array is smallish, one can use the Thrust lib                                           
  // Thrust is kickass!                                                                                        
  unsigned int *d_i;
  unsigned int threadsPerBlock = 128;
  unsigned int  totalBlocks = (SIZE+(threadsPerBlock-1))/threadsPerBlock;
  
 // allocate space for output array on the host                                                                
  h_o = (unsigned int*) malloc(totalBlocks * sizeof(unsigned int));

  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

/* start the timer for GPU code */
  cudaEventRecord(start_event, 0);

  gpuErrChk(cudaMalloc((void**)&d_i, sizeof(unsigned int)*SIZE));
  gpuErrChk(cudaMemcpy(d_i, h_i, sizeof(unsigned int)*SIZE, cudaMemcpyHostToDevice));


  // define an output array on device which will hold the sum from                                             
  // each block                                                                                                
  unsigned int *d_o;
  gpuErrChk(cudaMalloc((void**)&d_o, sizeof(unsigned int)*totalBlocks));


  // Invoke the kernel: by the power of the greyskull!                                                         
  reduce0<<<totalBlocks, threadsPerBlock, 2*threadsPerBlock*sizeof(unsigned int)>>>(d_i, d_o, size);

  // Copy the output array back and reduce on CPU                                                              
  gpuErrChk(cudaMemcpy(h_o, d_o, totalBlocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  /*end the timer for GPU code */
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&cuda_elapsed_time, start_event, stop_event);
  for (unsigned int j = 1; j < totalBlocks; j++)
    {
      h_o[0] += h_o[j];
    }

  printf("Reduced Sum from GPU = %ld \n", h_o[0]);
  printf("Time taken by the kernel: %f ms \n",cuda_elapsed_time);


  return 0;

}
