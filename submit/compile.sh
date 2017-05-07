module load apps/lammps/gpu
nvcc cuda_multiplication.cu -O3 -o main
