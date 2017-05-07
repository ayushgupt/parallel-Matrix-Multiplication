#include "mpi.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <ctime>
#include <stack>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <algorithm>
#include <climits>
#include <bitset>
#include <set>
#include <sys/time.h>
#include <bits/stdc++.h>
using namespace std;

#define print_to_err(x) do { std::cerr << #x << ": " << x << std::endl; } while (0)


#define bitcount(x) __builtin_popcount(x)

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


string input_file="";
string output_file="";
vector<long long int> matrix_A_row;
vector<long long int> matrix_A_col;
vector<long long int> matrix_A_data;

vector<long long int> matrix_B;
vector<long long int> matrix_C;
vector<long long int> ptr_vec;


long long int dimension;
long long int per_processor_rows;

int my_proc_rank;


void readfile()
{
    ifstream infile;
    char input_file_array[input_file.length()+1];
    for(int i=0;i<input_file.length();i++)
    {
        input_file_array[i]=input_file.at(i);
    }
    input_file_array[input_file.length()]='\0';


    infile.open(input_file_array);
    //Name Line
    string name_line;
    getline(infile,name_line);

    //Dimension Line
    string dimension_line;
    getline(infile,dimension_line);
    // print_to_err(dimension_line);
    int dimension_line_length=int(dimension_line.length());
    int start_dimension_line;
    for(int i=10;i<dimension_line_length;i++)
    {
        if(dimension_line.at(i)!=' ')
        {
            start_dimension_line=i;
            break;
        }
    }
    string dimension_string=dimension_line.substr(start_dimension_line,(dimension_line_length- start_dimension_line));
    // print_to_err(dimension_string);
    istringstream(dimension_string)>>dimension;

    ptr_vec.resize(dimension+1);
    //A
    string garbage_line;
    getline(infile,garbage_line);
    

    long long int temp_inp_a;
    string b_detect; 
    long long int curr_row=(-1);
    long long int count_ptr=0;
    while(true)
    {
        infile>>b_detect;
        if(b_detect=="B")
        {
            for(long long int j=(curr_row+1);j<=dimension;j++)
            {
                //check this
                ptr_vec[j]=count_ptr;
            }
            break;
        }
        istringstream (b_detect)>>temp_inp_a;
        matrix_A_row.push_back(temp_inp_a);

        if(curr_row<temp_inp_a)
        {
            for(long long int j=(curr_row+1);j<=temp_inp_a;j++)
            {
                ptr_vec[j]=count_ptr;
            }
            curr_row=temp_inp_a;
        }

        infile>>temp_inp_a;
        matrix_A_col.push_back(temp_inp_a);
        infile>>temp_inp_a;
        matrix_A_data.push_back(temp_inp_a);
        count_ptr+=1;
    }
    matrix_B.resize(dimension);
    for(int i=0;i<dimension;i++)
    {
        infile>>matrix_B[i];    
    }
    matrix_C.clear();
    matrix_C.resize(dimension,0);
    infile.close();
}

void outfile()
{
    ofstream outfile;
    char output_file_array[output_file.length()+1];
    for(int i=0;i<output_file.length();i++)
    {
        output_file_array[i]=output_file.at(i);
    }
    output_file_array[output_file.length()]='\0';

    if(my_proc_rank==0)
    {
        outfile.open (output_file_array);    
    }
    else
    {
        outfile.open(output_file_array, std::ios_base::app);
    }
    // outfile<<"my rank:"<<my_rank<<" I will print "<<int(my_numbers.size())<<" numbers"<<endl;


    // for(int i=0;i<matrix_A_data.size();i++)
    // {
    //     outfile<<matrix_A_row[i]<<" "<<matrix_A_col[i]<<" "<<matrix_A_data[i]<<endl;
    // }

    for (int i = 0; i < int(matrix_C.size()); i++)
    {
        outfile<<matrix_C[i]<<endl;
    }
    // outfile<<endl;
    outfile.close();
}


__global__ void spmv_csr_vector_kernel ( long long int num_rows ,long long int * ptr ,long long int * indices ,long long int * data ,long long int * x ,long long int * y)
{

    __shared__ long long int vals [32];
    long long int thread_id = blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
    long long int warp_id = thread_id / 32; // global warp index
    int lane = thread_id & (32 - 1); // thread index within the warp
    // one warp per row
    long long int row = warp_id ;
    //int num_rows = 5;
    if ( row < num_rows ){
        long long int row_start = ptr [ row ];
        long long int row_end = ptr [ row +1];
        // compute running sum per thread
        vals [ threadIdx.x ] = 0;
        for ( long long int jj = row_start + lane ; jj < row_end ; jj += 32)
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

    // if(thread_id==0)
    // {
    //     cout<<"GPU PRINT"<<endl;
    //     for(int i=0; i<11;i++)
    //     {
    //         cout<<indices[i]<<" "<<data[i]<<endl;
    //     }
    //     cout<<endl;
    //     cout<<"num_rows:"<<num_rows<<endl;
    //     cout<<"B is "<<endl;
    //     for(int i=0; i<5;i++)
    //     {
    //         cout<<x[i]<<endl;
    //     }
    //     cout<<"ptr_vec is "<<endl;
    //     for(int i=0; i<6;i++)
    //     {
    //         cout<<ptr[i]<<endl;
    //     }
    // }
}



int main(int argc, char *argv[])
{

    int rc;
    rc = MPI_Init(&argc, &argv);
    if(rc != MPI_SUCCESS) 
    {
            printf("Error starting MPI program. Terminating. \n");
            MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    my_proc_rank=world_rank;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;

    ///////DEBUG//////////
    string debug_file="debug"+int2str(world_rank)+".txt";
    ofstream debugfile;
    char debug_file_array[debug_file.length()+1];
    for(int i=0;i<debug_file.length();i++)
    {
     debug_file_array[i]=debug_file.at(i);
    }
    debug_file_array[debug_file.length()]='\0';
    debugfile.open(debug_file_array);




    string temp_inpfile(argv[1]);
    input_file=temp_inpfile;
    string temp_outfile(argv[2]);
    output_file=temp_outfile;
    debugfile<<1<<endl;
    long long int matrix_A_nonzero_elem;
    if(world_rank==0)
    {
        readfile();
        matrix_A_nonzero_elem=(long long int)(matrix_A_data.size());
        //matrix_C resized, ptr_vec, matrix_B, matrix_A_col, matrix_A_row, matrix_A_data
    }
    debugfile<<2<<endl;
    long long int one_start_index;
    if(world_rank==0)
    {
        per_processor_rows=(dimension/(long long int)world_size);
    }    
    debugfile<<3<<endl;
    MPI_Bcast(&dimension, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&per_processor_rows, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    one_start_index=dimension- (per_processor_rows*(world_size-1));

    debugfile<<4<<endl;

    long long int start_row,end_row;

    if(world_rank==0)
    {
        start_row=0;
        end_row=one_start_index-1;
    }
    else
    {
        start_row=one_start_index+((long long int)world_rank-1)*per_processor_rows;
        end_row=(start_row+(per_processor_rows-1));
    }

    debugfile<<5<<endl;

    if(world_rank!=0)
    {
        matrix_C.clear();
        matrix_C.resize(dimension,0);
        matrix_B.resize(dimension);
    }
    MPI_Bcast(&matrix_B[0], dimension, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    debugfile<<6<<endl;
    if(world_rank==0)
    {
        long long int temp_start_row, temp_end_row;
        for(int j=1;j<world_size;j++)
        {
            temp_start_row=one_start_index+((long long int)j-1)*per_processor_rows;
            temp_end_row=(temp_start_row+(per_processor_rows-1));
            // matrix_A_data[ptr_vec[temp_start_row]] to matrix_A_data[ptr_vec[temp_end_row+1]-1]
            // temp_temp=
            debugfile<<"in1"<<endl;
            MPI_Send(&ptr_vec[temp_start_row],1,MPI_LONG_LONG_INT,j,(j*3),MPI_COMM_WORLD);
            debugfile<<"in2"<<endl;
            MPI_Send(&ptr_vec[temp_end_row+1],1,MPI_LONG_LONG_INT,j,(j*3)+1,MPI_COMM_WORLD);
            debugfile<<"in3"<<endl;
            debugfile<<"temp_start_row"<<temp_start_row<<endl;
            debugfile<<"temp_end_row"<<temp_end_row<<endl;
            debugfile<<"ptr_vec[temp_end_row+1]"<<ptr_vec[temp_end_row+1]<<endl;
            debugfile<<"ptr_vec[temp_start_row]"<<ptr_vec[temp_start_row]<<endl;

            MPI_Send(&matrix_A_data[ptr_vec[temp_start_row]],(ptr_vec[temp_end_row+1]-ptr_vec[temp_start_row]),MPI_LONG_LONG_INT,j,(j*3)+2,MPI_COMM_WORLD);
            debugfile<<"in4"<<endl;
            MPI_Send(&matrix_A_row[ptr_vec[temp_start_row]],(ptr_vec[temp_end_row+1]-ptr_vec[temp_start_row]),MPI_LONG_LONG_INT,j,(j*3)+3,MPI_COMM_WORLD);
            debugfile<<"in5"<<endl;
            MPI_Send(&matrix_A_col[ptr_vec[temp_start_row]],(ptr_vec[temp_end_row+1]-ptr_vec[temp_start_row]),MPI_LONG_LONG_INT,j,(j*3)+4,MPI_COMM_WORLD);
            debugfile<<"in6"<<endl;
            MPI_Bcast(&ptr_vec[0], (dimension+1), MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);   
            debugfile<<"in7"<<endl;
        }
        matrix_A_data.resize(ptr_vec[one_start_index]);
        matrix_A_col.resize(ptr_vec[one_start_index]);
        matrix_A_row.resize(ptr_vec[one_start_index]);

    }
    else
    {
        debugfile<<"in1"<<endl;
        long long int temp_start_ptr,temp_end_ptr;
        MPI_Recv(&temp_start_ptr,1,MPI_LONG_LONG_INT,0,(my_proc_rank*3),MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        debugfile<<"in2"<<endl;
        MPI_Recv(&temp_end_ptr,1,MPI_LONG_LONG_INT,0,(my_proc_rank*3+1),MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        debugfile<<"in3"<<endl;
        matrix_A_data.resize((temp_end_ptr)-(temp_start_ptr));
        matrix_A_row.resize((temp_end_ptr)-(temp_start_ptr));
        matrix_A_col.resize((temp_end_ptr)-(temp_start_ptr));
        MPI_Recv(&matrix_A_data[0],((temp_end_ptr)-(temp_start_ptr)),MPI_LONG_LONG_INT,0,(my_proc_rank*3+2),MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        debugfile<<"in4"<<endl;
        MPI_Recv(&matrix_A_row[0],((temp_end_ptr)-(temp_start_ptr)),MPI_LONG_LONG_INT,0,(my_proc_rank*3+3),MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        debugfile<<"in5"<<endl;
        MPI_Recv(&matrix_A_col[0],((temp_end_ptr)-(temp_start_ptr)),MPI_LONG_LONG_INT,0,(my_proc_rank*3+4),MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        debugfile<<"in6"<<endl;
        ptr_vec.resize(dimension+1);
        MPI_Bcast(&ptr_vec[0], (dimension+1), MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        debugfile<<"in7"<<endl;
    }

    for(int i=0;i<int(matrix_A_row.size());i++)
    {
        matrix_A_row[i]-=start_row;
    }

    debugfile<<7<<endl;
    vector<long long int> local_ptr_vec;
    long long int temp_first=ptr_vec[start_row];
    for(int i=start_row;i<=(end_row+1);i++)
    {
        local_ptr_vec.push_back(ptr_vec[i]-temp_first);
    }

    debugfile<<8<<endl;
    ptr_vec.clear();
    ptr_vec=local_ptr_vec;


    

    matrix_A_nonzero_elem=(long long int)(matrix_A_data.size());
    
    debugfile<<9<<endl;

    long long int *d_ptr_vector,*d_matrixA_col,*d_matrixA_data,*d_matrixB,*d_multiply_answer;

    long long int nonzero_size=matrix_A_nonzero_elem*sizeof(long long int);
    long long int dimension_size=(dimension*sizeof(long long int));

    long long int my_rows_number=(end_row- start_row+1);

    debugfile<<"my_rows_number::"<<my_rows_number<<endl;

    matrix_C.resize(my_rows_number,0);

    long long int row_number_size=(my_rows_number*sizeof(long long int));


    debugfile<<"My rank is "<<my_proc_rank<<endl;
    debugfile<<10<<endl;
    debugfile<<"row_col_data"<<endl;
   for(int i=0; i<matrix_A_nonzero_elem;i++)
   {
       debugfile<<matrix_A_row[i]<<" "<<matrix_A_col[i]<<" "<<matrix_A_data[i]<<endl;
   }
   debugfile<<endl;
   debugfile<<"dimension:"<<dimension<<endl;
   debugfile<<"B is "<<endl;
   for(int i=0; i<dimension;i++)
   {
       debugfile<<matrix_B[i]<<endl;
   }
   debugfile<<"ptr_vec is "<<endl;
   for(int i=0; i<int(ptr_vec.size());i++)
   {
       debugfile<<ptr_vec[i]<<endl;
   }




    cudaMalloc((void **)&d_ptr_vector,((my_rows_number+1)*sizeof(long long int)));
    cudaMalloc((void **)&d_matrixA_col,nonzero_size);
    cudaMalloc((void **)&d_matrixA_data,nonzero_size);
    cudaMalloc((void **)&d_matrixB,dimension_size);
    cudaMalloc((void **)&d_multiply_answer,row_number_size);


    cudaMemcpy(d_ptr_vector,(&ptr_vec[0]),((my_rows_number+1)*sizeof(long long int)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixA_col,(&matrix_A_col[0]),nonzero_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixA_data,(&matrix_A_data[0]),nonzero_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB,(&matrix_B[0]),dimension_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_multiply_answer,(&matrix_C[0]),row_number_size,cudaMemcpyHostToDevice);

    // long long int num_blocks;

    // num_blocks=(dimension/(long long int)(32));

    // if(num_blocks%32!=0)
    // {
    //     num_blocks+=1;
    // }

    spmv_csr_vector_kernel<<<my_rows_number,32>>>(my_rows_number,d_ptr_vector,d_matrixA_col,d_matrixA_data,d_matrixB,d_multiply_answer);
    

    cudaMemcpy((&matrix_C[0]),d_multiply_answer,row_number_size,cudaMemcpyDeviceToHost);
    
    outfile();
    
    if(world_rank==0)
    {
        outfile();
        int one=1;
        MPI_Send(&one,1,MPI_INT,(world_rank+1),1000,MPI_COMM_WORLD);
    }
    else if(world_rank==(world_size-1))
    {
        int upper_printed;
        MPI_Recv(&upper_printed,1,MPI_INT,(world_rank-1),1000,MPI_COMM_WORLD, &status);
        outfile();
    }
    else
    {
        int upper_printed;
        MPI_Recv(&upper_printed,1,MPI_INT,(world_rank-1),1000,MPI_COMM_WORLD,&status);
        outfile();
        int one=1;
        MPI_Send(&one,1,MPI_INT,(world_rank+1),1000,MPI_COMM_WORLD);
    }



    debugfile.close();
    //  Terminate MPI.
    MPI_Finalize();
    return 0;
}

