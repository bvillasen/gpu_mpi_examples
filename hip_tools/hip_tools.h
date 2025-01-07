#ifndef HIP_TOOLS_H
#define HIP_TOOLS_H

#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"

#define CHECK_GPU_MEMORY 

void hip_get_device_properties( int device_id );

int hip_set_device(int proc_id, int n_procs );


#define CHECK(command) {   \
  hipError_t status = command; \
  if (status!=hipSuccess) {    \
    std::cout << "Error: HIP reports " << hipGetErrorString(status) << std::endl; \
    std::abort(); }}

// void hip_sync_device(){ CHECK( hipDeviceSynchronize() ); }

// void hip_reset_device(){ CHECK( hipDeviceReset() ); }

template <typename T> 
int hip_malloc_device( T *&d_array, size_t size ){

	if ( size <= 0 ){
		std::cout << "WARNING: hipMalloc size of array <=  	0. size: " << size << std::endl; 
		return 0;
	}
	
	#ifdef CHECK_GPU_MEMORY
	size_t global_free, global_total; 
  CHECK( hipMemGetInfo( &global_free, &global_total )	);
	size_t to_MB = 1024*1024;
	if ( global_free < size ){
		std::cout << "ERROR: hip_malloc_device -- Not enough free memory in device: " << std::endl;
		std::cout << "  Total memory:      " << static_cast<double>(global_total) / to_MB << " MB" << std::endl;
		std::cout << "  Available memory:  " << static_cast<double>(global_free) / to_MB << " MB" << std::endl;;
		std::cout << "  Array size:        " << static_cast<double>(size) / to_MB << " MB" << std::endl; 
		return -1;
	}
	#endif

	CHECK( hipMalloc( (void**)&d_array, size ) );
	if ( d_array != nullptr )	return 1;
	else{
		std::cout << "ERROR: hipMalloc failed " << std::endl;
		return -1;
	}
}

template <typename T> 
int hip_malloc_host( T *&h_array, size_t size ){
	CHECK( hipHostMalloc( (void**)&h_array, size ) );
	if ( h_array != nullptr )	return 1;
	else{
		std::cout << "ERROR: hipHostMalloc failed " << std::endl;
		return -1;
	}
}


template <typename T> 
void hip_free_device( T *d_array ){
	CHECK( hipFree( d_array ) );
}

template <typename T> 
void hip_free_host( T *h_array ){
	CHECK( hipHostFree( h_array ) );
}

template <typename T> 
void hip_copy_host_to_device( T *d_array_dst, T *h_array_src, size_t size ){
	CHECK( hipMemcpy(  d_array_dst, h_array_src, size, hipMemcpyHostToDevice ) );
}

template <typename T> 
void hip_copy_device_to_host( T *h_array_dst, T *d_array_src, size_t size ){
	CHECK( hipMemcpy(  h_array_dst, d_array_src, size, hipMemcpyDeviceToHost ) );
}

#endif //HIP_TOOLS_H