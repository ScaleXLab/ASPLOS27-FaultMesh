#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <unistd.h>


//Solid data pinning in device for oversubscription

int main(int argc, char *argv[]) {
    int a,Pinsize;
    
    if(argc > 1)
    	Pinsize = atoi(argv[1]) - 210;
    else Pinsize = 3000;
    
    if(Pinsize < 0)
        return 1;
    
    unsigned char *d_data;
    cudaMalloc(&d_data, sizeof(unsigned char) * Pinsize * 1024 * 1024);
    
    while(1){
        a = 0;
	sleep(1);
    }
}
