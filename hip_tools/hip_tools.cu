// Basic hip functions callable from C/C++ code
#include <stdio.h>
#include <iostream>
#include "hip_tools.h"

void hip_get_device_properties( int device_id ){

	hipDeviceProp_t prop;
	CHECK( hipGetDeviceProperties( &prop, device_id ) );

  std::cout << "Device: " << device_id << " name: " << prop.name  << std::endl;
	std::cout << "Total global memory (bytes): " << prop.totalGlobalMem  << std::endl;
	std::cout << "Shared memory per block (bytes): " << prop.sharedMemPerBlock  << std::endl;
	std::cout << "Registers per block: " << prop.regsPerBlock  << std::endl;	
	std::cout << "Warp size: " << prop.warpSize  << std::endl;	
	std::cout << "Max threads per block: " << prop.maxThreadsPerBlock  << std::endl;	

}

int hip_set_device(int proc_id, int n_procs ){ 
	
  int n_devices; 
	CHECK( hipGetDeviceCount(&n_devices) );

	if (n_devices == 0){
		std::cout << "MPI rank= " << proc_id << " NO DEVICES FOUND!" << std::endl;
		return 0;
	}

	int device = proc_id % n_devices;
	CHECK( hipSetDevice(device) ); 
	std::cout << "MPI rank= " << proc_id << " will use GPU ID " << device << " / " << n_devices << std::endl;
	return device;
} 