#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include "hip_tools.h"

int main( int argc, char *argv[] ){

  int rank, n_ranks;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);


  if (rank == 0) std::cout << "GPU-Aware MPI. All reduce" << std::endl;
  
  int n_iter = 1;
  if ( argc > 1 ){
    n_iter = atoi( argv[1] );
  }

  int device_id = hip_set_device( rank, n_ranks );
    
  const uint N_total = 256 * 256 * 256 * 8;
  
  double buffer_size_MB = (double)N_total*sizeof(double)/(1024*1024);
  double message_size_MB = (double)N_total*n_ranks*sizeof(double)/(1024*1024);
  double total_transfer_size_MB = (double)N_total*n_ranks*sizeof(double)/(1024*1024);
  if (rank == 0) std::cout  << "Total array size: " << N_total
                            <<  "  Size: " << buffer_size_MB << " MB" 
                            << std::endl;
  
  double *h_send_buffer = (double *)malloc( N_total*sizeof(double));
  double *h_recv_buffer = (double *)malloc( N_total*sizeof(double));

  for (int i=0; i<N_total; i++ ){
    h_send_buffer[i] = (double)(rank+1)*i;
  } 

  if (rank == 0) std::cout << "\nCPU N iterations: " << n_iter << std::endl;

  auto time_start_cpu = std::chrono::high_resolution_clock::now();  

  for (int i=0; i<n_iter; i++ ){

    MPI_Allreduce( h_send_buffer, h_recv_buffer, N_total, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

  }

  auto time_end_cpu = std::chrono::high_resolution_clock::now();  
  double time_total_cpu = std::chrono::duration<double>(time_end_cpu - time_start_cpu).count()  * 1e3 ; // convert to millisecs
  double time_per_iter_cpu = time_total_cpu / n_iter;
  double bandwidth_cpu = ( total_transfer_size_MB / 1024 ) / ( time_per_iter_cpu * 1e-3 ); //GB/s 
  // printf( "CPU Rank %d Time total: %.2f msecs  Time per iteration: %.2f msecs  Bandwidth: %.2f GB/s\n", rank, time_total_cpu, time_per_iter_cpu, bandwidth_cpu );  
  std::cout << "CPU Rank " << rank 
            << " Time total: "<< time_total_cpu << " msecs" 
            << " Time per iteration: " << time_per_iter_cpu << " msecs"  
            // << " Bandwidth: " << bandwidth_cpu << " GB/s" 
            << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);


  double *d_send_buffer, *d_recv_buffer;
  hip_malloc_device( d_send_buffer, N_total*sizeof(double) );
  hip_malloc_device( d_recv_buffer, N_total*sizeof(double) );
  hip_copy_device_to_host( d_send_buffer, h_send_buffer, N_total*sizeof(double) );

  if (rank == 0) std::cout << "\nGPU N iterations: " << n_iter << std::endl;
// 
  auto time_start_gpu = std::chrono::high_resolution_clock::now();  

  for (int i=0; i<n_iter; i++ ){

    MPI_Allreduce( d_send_buffer, d_recv_buffer, N_total, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

  }

  auto time_end_gpu = std::chrono::high_resolution_clock::now();  
  double time_total_gpu = std::chrono::duration<double>(time_end_gpu - time_start_gpu).count()  * 1e3 ; // convert to millisecs
  double time_per_iter_gpu = time_total_gpu / n_iter;
  double bandwidth_gpu = ( total_transfer_size_MB / 1024 ) / ( time_per_iter_gpu * 1e-3 ); //GB/s 
  // printf( "GPU Rank %d Time total: %.2f msecs  Time per iteration: %.2f msecs  Bandwidth: %.2f GB/s\n", rank, time_total_gpu, time_per_iter_gpu, bandwidth_gpu );  
  std::cout << "GPU Rank " << rank 
            << " Time total: "<< time_total_gpu << " msecs" 
            << " Time per iteration: " << time_per_iter_gpu << " msecs"  
            // << " Bandwidth: " << bandwidth_gpu << " GB/s" 
            << std::endl;
  sleep(1);
  MPI_Barrier(MPI_COMM_WORLD);

  bool validation_passed = true;
  for (int i=0; i<N_total; i++ ){
    if ( h_recv_buffer[i] != d_recv_buffer[i] ) validation_passed = false;
  } 

  if (validation_passed) std::cout << "Validation PASSED." << std::endl;
  else std::cout << "Validation FAILED." << std::endl;
  sleep(1);
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) printf( "Finished \n" );
  MPI_Finalize();
  return 0;
}
