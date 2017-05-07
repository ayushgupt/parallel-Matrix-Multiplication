#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

string int2str(int num)
{
  ostringstream ss;
  ss << num;
  return(ss.str());
}

int str2int(string str)
{
  int value;
  istringstream (str)>>value;
  return value;
}
__global__ void spmv_csr_vector_kernel ( 
	int * num_rows_p ,
	int * ptr ,
	int * indices ,
	int * data ,
	int * x ,
	int * y)
{
	__shared__ int vals [32];
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
	int warp_id = thread_id / 32; // global warp index
	int lane = thread_id & (32 - 1); // thread index within the warp
	// one warp per row
	int row = warp_id ;
	int num_rows = *num_rows_p;
	//int num_rows = 5;
	if ( row < num_rows ){
		int row_start = ptr [ row ];
		int row_end = ptr [ row +1];
		// compute running sum per thread
		vals [ threadIdx.x ] = 0;
		for ( int jj = row_start + lane ; jj < row_end ; jj += 32)
			vals [ threadIdx.x ] += data [ jj ] * x [ indices [ jj ]];
		// parallel reduction in shared memory
		if ( lane < 16) vals [ threadIdx.x ] += vals [ threadIdx.x + 16];
		if ( lane < 8) vals [ threadIdx.x ] += vals [ threadIdx.x + 8];
		if ( lane < 4) vals [ threadIdx.x ] += vals [ threadIdx.x + 4];
		if ( lane < 2) vals [ threadIdx.x ] += vals [ threadIdx.x + 2];
		if ( lane < 1) vals [ threadIdx.x ] += vals [ threadIdx.x + 1];
		// first thread writes the result
		if ( lane == 0)
			y[ row ] += vals [ threadIdx.x ];
	}

}

int main(int argc, char	*argv[]) {
	int 	comm_sz,my_rank;
	MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int matrix_dimension;
    ifstream inputfile(argv[1]);

	vector<int> ptr_h;
	vector<int> data_h;
	vector<int> indices_h;

	string a,dimension;
	getline(inputfile,a);
	getline(inputfile,dimension);
    matrix_dimension=str2int(dimension.substr(11,dimension.length()-11));
    int b_h[matrix_dimension];
    int c_h[matrix_dimension];
    getline(inputfile,a);
    long long int row_number,col_number,element;
    string b; 
    int oldrowid=-1;
	int newrowid;
	int rowcount=0;
	while(true){
        inputfile>>b;
        if (b == "B")
        {
        	ptr_h.push_back(rowcount);
            break;
        }
        oldrowid = newrowid;
		istringstream (b) >> newrowid;
		//inputfile >> newrowidc;
		//cout<<newrowidc<<endl;
		if(oldrowid==newrowid){
			rowcount++;
		}
		else{
			ptr_h.push_back(rowcount);
			rowcount++;
		}
        inputfile>>col_number;
        inputfile>>element;
        indices_h.push_back(col_number);
        data_h.push_back(element);
    }
    for(int i=0;i<matrix_dimension;i++)
    {
        inputfile>>b_h[i];    
    }
    inputfile.close();

    int NNZ = data_h.size();


    int *data_d, *ptr_d, *indices_d, *b_d, *c_d, *matrix_dimension_d; // device copies of a, b, c
    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&data_d, NNZ*sizeof(int));
    cudaMalloc((void **)&ptr_d, (matrix_dimension+1)*sizeof(int));
    cudaMalloc((void **)&indices_d, NNZ*sizeof(int));
    cudaMalloc((void **)&b_d, (matrix_dimension)*sizeof(int));
	cudaMalloc((void **)&c_d, (matrix_dimension)*sizeof(int));
    cudaMalloc((void **)&matrix_dimension_d, sizeof(int));

    

    cudaMemcpy(data_d, data_h, NNZ*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_d, ptr_h, (matrix_dimension+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(indices_d, indices_h, NNZ*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, (matrix_dimension)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h, (matrix_dimension)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_dimension_d, &matrix_dimension, sizeof(int), cudaMemcpyHostToDevice);

    //spmv_csr_vector_kernel<<<2,64>>>(matrix_dimension_d,ptr_d,indices_d,data_d,b_d,c_d,rf_d);
    spmv_csr_vector_kernel<<<2,32>>>(matrix_dimension_d,ptr_d,indices_d,data_d,b_d,c_d);
    //rf_fn<<<64,64>>>(rf_d);

    cudaMemcpy(c_h, c_d, (matrix_dimension)*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0;i<matrix_dimension;i++){
		cout<<c_h[i] <<"\n";
	}


	MPI_Finalize();	


}
