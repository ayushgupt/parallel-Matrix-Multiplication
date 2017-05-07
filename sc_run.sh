#!/bin/sh
### Set the job name
#PBS -N temp77
### Set the project name, your department dc by default
#PBS -P cse
### Specify email address to use for notification.
####
#PBS -l select=1:ngpus=1:ncpus=24:mem=2gb
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=01:00:00
#### Get environment variables from submitting shell
#PBS -V
#PBS -l software=ANSYS
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
#job 
module load apps/lammps/gpu
nvcc cuda_multiplication.cu
time ./a.out sample_inp.txt output_inp.txt
