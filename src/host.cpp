/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <memory>
#include <string>
typedef std::chrono::high_resolution_clock Clock;//more accurate
typedef std::chrono::system_clock SClock;//less accurate

#include "xcl2.hpp"
#include <vector>
#include "kernel_params.h"

#include "weights/w27.h"
#include "weights/w31.h"
#include "weights/w36.h"
#include "weights/w40.h"
#include "weights/w45.h"
#include "weights/w49.h"

#include <thread>
#include <sstream>

#define NBUFFER 1
#define NUM_CU 1

#define STRINGIFY2(var) #var
#define STRINGIFY(var) STRINGIFY2(var)

void print_nanoseconds(std::string prefix, std::chrono::time_point<std::chrono::system_clock> now, int ik, std::ofstream& fout) {
    auto duration = now.time_since_epoch();
    
    typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<8>
    >::type> Days; /* UTC: +8:00 */
    
    Days days = std::chrono::duration_cast<Days>(duration);
        duration -= days;
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
        duration -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
        duration -= minutes;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        duration -= seconds;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        duration -= milliseconds;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);
        duration -= microseconds;
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);

    std::cout << "input set" << ik << ", "
          << prefix << hours.count() << ":"
          << minutes.count() << ":"
          << seconds.count() << ":"
          << milliseconds.count() << ":"
          << microseconds.count() << ":"
          << nanoseconds.count() << std::endl;
	
	fout << "input set" << ik << ", "
          << prefix << hours.count() << ":"
          << minutes.count() << ":"
          << seconds.count() << ":"
          << milliseconds.count() << ":"
          << microseconds.count() << ":"
          << nanoseconds.count() << std::endl;
}

void print_nanoseconds(std::string prefix, std::chrono::time_point<std::chrono::system_clock> now, int ik, std::stringstream &ss) {
    auto duration = now.time_since_epoch();
    
    typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<8>
    >::type> Days; /* UTC: +8:00 */
    
    Days days = std::chrono::duration_cast<Days>(duration);
        duration -= days;
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
        duration -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
        duration -= minutes;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        duration -= seconds;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        duration -= milliseconds;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);
        duration -= microseconds;
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);

    ss << "input set" << ik << ", " << prefix 
          << hours.count() << ":"
          << minutes.count() << ":"
          << seconds.count() << ":"
          << milliseconds.count() << ":"
          << microseconds.count() << ":"
          << nanoseconds.count() << "\n";
}

// An event callback function that prints the operations performed by the OpenCL
// runtime.
void event_cb(cl_event event1, cl_int cmd_status, void *data) {
    cl_int err;
    cl_command_type command;
    cl::Event event(event1, true);
    OCL_CHECK(err, err = event.getInfo(CL_EVENT_COMMAND_TYPE, &command));
    cl_int status;
    OCL_CHECK(err,
              err = event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status));
    const char *command_str;
    const char *status_str;
    switch (command) {
    case CL_COMMAND_READ_BUFFER:
        command_str = "buffer read";
        break;
    case CL_COMMAND_WRITE_BUFFER:
        command_str = "buffer write";
        break;
    case CL_COMMAND_NDRANGE_KERNEL:
        command_str = "kernel";
        break;
    case CL_COMMAND_MAP_BUFFER:
        command_str = "kernel";
        break;
    case CL_COMMAND_COPY_BUFFER:
        command_str = "kernel";
        break;
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
        command_str = "buffer migrate";
        break;
    default:
        command_str = "unknown";
    }
    switch (status) {
    case CL_QUEUED:
        status_str = "Queued";
        break;
    case CL_SUBMITTED:
        status_str = "Submitted";
        break;
    case CL_RUNNING:
        status_str = "Executing";
        break;
    case CL_COMPLETE:
        status_str = "Completed";
        break;
    }
    printf("[%s]: %s %s\n",
           reinterpret_cast<char *>(data),
           status_str,
           command_str);
    fflush(stdout);
}

// Sets the callback for a particular event
void set_callback(cl::Event event, const char *queue_name) {
    cl_int err;
    OCL_CHECK(err,
              err =
                  event.setCallback(CL_COMPLETE, event_cb, (void *)queue_name));
}

class fpgaObj {
  public:
    std::stringstream ss;
    int ithr;
    int nevents;
    int ikern;
    std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_in;
    std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_hw_results;
    std::vector<model_default_t,aligned_allocator<model_default_t>> source_w27_in;
    std::vector<model_default_t,aligned_allocator<model_default_t>> source_w31_in;
    std::vector<model_default_t,aligned_allocator<model_default_t>> source_w36_in;
    std::vector<model_default_t,aligned_allocator<model_default_t>> source_w40_in;
    std::vector<model_default_t,aligned_allocator<model_default_t>> source_w45_in;
    std::vector<model_default_t,aligned_allocator<model_default_t>> source_w49_in;
    cl::Program program;
    std::vector<cl::CommandQueue> q;
    std::vector<cl::Kernel> krnl_xil;
    std::vector<std::vector<cl::Event>>   writeList;
    std::vector<std::vector<cl::Event>>   kernList;
    std::vector<std::vector<cl::Event>>   readList;
    std::vector<cl::Buffer> buffer_in;
    std::vector<cl::Buffer> buffer_out;
    std::vector<cl::Buffer> buffer_wvec_in;
    std::vector<cl::Event>   write_event;
    std::vector<cl::Event>   kern_event;
    //std::vector<cl::Event>   read_event;
    std::vector<bool> isFirstRun;
    cl_int err;

    std::pair<int,bool> get_info_lock() {
      int i;
      bool first;
      mtx.lock();
      //i = rand() % 1;
      i = ikern++;
      if (ikern==NUM_CU*NBUFFER) ikern = 0;
      first = isFirstRun[i];
      if (first) isFirstRun[i]=false;
      mtx.unlock();
      return std::make_pair(i,first);
    }
    void get_ilock(int ik) {
      mtxi[ik].lock();
    }
    void release_ilock(int ik) {
      mtxi[ik].unlock();
    }
    void write_ss_safe(std::string newss) {
      smtx.lock();
      ss << "Thread " << ithr << "\n" << newss << "\n";
      ithr++;
      smtx.unlock();
    }

    std::stringstream runFPGA() {
        auto t0 = Clock::now();
        auto t1 = Clock::now();
        auto t1a = Clock::now();
        auto t1b = Clock::now();
        auto t2 = Clock::now();
        auto t3 = Clock::now();
        std::stringstream ss;
        for (int i = 0 ; i < nevents ; i++){
            t0 = Clock::now();
            auto ikf = get_info_lock();
            int ikb = ikf.first;
            int ik = ikb%NUM_CU;
            bool firstRun = ikf.second;

            t1 = Clock::now();
            auto ts1 = SClock::now();
            //print_nanoseconds("        start:  ",ts1, ik, ss);
            std::string queuename = "ooo_queue "+std::to_string(ikb);
        
            get_ilock(ikb);
            //Copy input data to device global memory
            if (!firstRun) {
                OCL_CHECK(err, err = kern_event[ikb].wait());
            }
            OCL_CHECK(err,
                      err =
            q[ik].enqueueMigrateMemObjects({buffer_in[ikb],buffer_wvec_in[0],buffer_wvec_in[1],buffer_wvec_in[2],buffer_wvec_in[3],buffer_wvec_in[4],buffer_wvec_in[5]},
                                                     0 /* 0 means from host*/,
                                                     NULL,
                                                     &(write_event[ikb])));
            t1a = Clock::now();
            writeList[ikb].clear();
            writeList[ikb].push_back(write_event[ikb]);
            //Launch the kernel
            OCL_CHECK(err,
                      err = q[ik].enqueueNDRangeKernel(
                          krnl_xil[ikb], 0, 1, 1, &(writeList[ikb]), &(kern_event[ikb])));
            t1b = Clock::now();
            kernList[ikb].clear();
            kernList[ikb].push_back(kern_event[ikb]);
            cl::Event read_event;
            OCL_CHECK(err,
                      err = q[ik].enqueueMigrateMemObjects({buffer_out[ikb]},
                                                       CL_MIGRATE_MEM_OBJECT_HOST,
                                                       &(kernList[ikb]),
                                                       &(read_event)));

            release_ilock(ikb);
        
            OCL_CHECK(err, err = kern_event[ikb].wait());
            OCL_CHECK(err, err = read_event.wait());
            auto ts2 = SClock::now();
            //print_nanoseconds("       finish:  ",ts2, ik, ss);
            t2 = Clock::now();
            t3 = Clock::now();
            std::cout << "Prep time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << " ns" << std::endl;
            std::cout << "FPGA time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns" << std::endl;
            std::cout << "inputs: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1a - t1).count() << " ns" << std::endl;
            std::cout << "kernel: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1b - t1a).count() << " ns" << std::endl;
            std::cout << "outputs: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1b).count() << " ns" << std::endl;
            std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t0).count() << " ns" << std::endl;
        }
        return ss;
    }

  private:
    mutable std::mutex mtx;
    mutable std::mutex mtxi[NUM_CU*NBUFFER];
    mutable std::mutex smtx;
};

void FPGA(fpgaObj& theFPGA) {
    std::stringstream ss;
    ss << (theFPGA.runFPGA()).str();
    theFPGA.write_ss_safe(ss.str());
}

int main(int argc, char** argv)
{
	int nevents = 1;
	std::string datadir = STRINGIFY(HLS4ML_DATA_DIR);
	std::string xclbinFilename = "";
	if (argc > 1) xclbinFilename = argv[1];
	if (argc > 2) nevents = atoi(argv[2]);
	if (argc > 3) datadir = argv[3];
	std::cout << "Will run " << nevents << " time(s), using " << datadir << " to get input features and output predictions (tb_input_features.dat and tb_output_predictions.dat)" << std::endl;
	
		
	size_t vector_size_in_bytes = sizeof(bigdata_t) *IN_STREAM_LEN*DATA_SIZE_IN;
	size_t vector_size_out_bytes = sizeof(bigdata_t) * BIGSTREAMSIZE_OUT;
	size_t vector_size_in_w27_bytes = sizeof(model_default_t) * NW1;
	size_t vector_size_in_w31_bytes = sizeof(model_default_t) * NW2;
	size_t vector_size_in_w36_bytes = sizeof(model_default_t) * NW3;
	size_t vector_size_in_w40_bytes = sizeof(model_default_t) * NW4;
	size_t vector_size_in_w45_bytes = sizeof(model_default_t) * NW5;
	size_t vector_size_in_w49_bytes = sizeof(model_default_t) * NW6;
	fpgaObj fpga;
	fpga.nevents = nevents;
	fpga.ikern = 0;
	fpga.source_in.reserve(IN_STREAM_LEN*DATA_SIZE_IN);
	fpga.source_hw_results.reserve(DATA_SIZE_OUT);
	fpga.source_w27_in.reserve(NW1);
	fpga.source_w31_in.reserve(NW2);
	fpga.source_w36_in.reserve(NW3);
	fpga.source_w40_in.reserve(NW4);
	fpga.source_w45_in.reserve(NW5);
	fpga.source_w49_in.reserve(NW6);

	//initialize
	for(int j = 0 ; j < IN_STREAM_LEN*DATA_SIZE_IN ; j++){
		fpga.source_in[j] = 0;
	}
	for(int j = 0 ; j < DATA_SIZE_OUT ; j++){
		fpga.source_hw_results[j] = 0;
	}
	for(int j = 0; j < NW1; j++){
		fpga.source_w27_in[j] = w27[j];
	}
	for(int j = 0; j < NW2; j++){
		fpga.source_w31_in[j] = w31[j];
	}
	for(int j = 0; j < NW3; j++){
		fpga.source_w36_in[j] = w36[j];
	}
	for(int j = 0; j < NW4; j++){
		fpga.source_w40_in[j] = w40[j];
	}
	for(int j = 0; j < NW5; j++){
		fpga.source_w45_in[j] = w45[j];
	}
	for(int j = 0; j < NW6; j++){
		fpga.source_w49_in[j] = w49[j];
	}

	// OPENCL HOST CODE AREA START
	// get_xil_devices() is a utility API which will find the xilinx
	// platforms and will return list of devices connected to Xilinx platform
	std::vector<cl::Device> devices = xcl::get_xil_devices();
	cl::Device device = devices[0];
	cl::Context context(device);
	for (int i = 0; i < NUM_CU; i++) {
		cl::CommandQueue q_tmp(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
		fpga.q.push_back(q_tmp);
	}
	std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 
	std::cout << "Found Device=" << device_name.c_str() << std::endl;

	cl::Program::Binaries bins;
	// Load xclbin
	std::cout << "Loading: '" << xclbinFilename << "'\n";
	std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
	bin_file.seekg (0, bin_file.end);
	unsigned nb = bin_file.tellg();
	bin_file.seekg (0, bin_file.beg);
	char *buf = new char [nb];
	bin_file.read(buf, nb);

	// Creating Program from Binary File
	bins.push_back({buf,nb});

	devices.resize(1);
	cl::Program tmp_program(context, devices, bins);
	fpga.program = tmp_program;

	for (int ib = 0; ib < NBUFFER; ib++) {
		for (int i = 0; i < NUM_CU; i++) {
			std::string cu_id = std::to_string(NUM_CU>1 ? i : 1);
			std::string krnl_name_full =
			"alveo_hls4ml:{alveo_hls4ml_" + cu_id + "}";
			printf("Creating a kernel [%s] for CU(%d)\n",
					krnl_name_full.c_str(),
					i);
			//Here Kernel object is created by specifying kernel name along with compute unit.
			//For such case, this kernel object can only access the specific Compute unit
			cl::Kernel krnl_tmp = cl::Kernel(
				fpga.program, krnl_name_full.c_str(), &fpga.err);
			fpga.krnl_xil.push_back(krnl_tmp);
		}
	}
	// Allocate Buffer in Global Memory
	// Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
	// Device-to-host communication
	cl::Buffer buffer_in_w27_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_w27_bytes, fpga.source_w27_in.data());
	cl::Buffer buffer_in_w31_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_w31_bytes, fpga.source_w31_in.data());
	cl::Buffer buffer_in_w36_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_w36_bytes, fpga.source_w36_in.data());
	cl::Buffer buffer_in_w40_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_w40_bytes, fpga.source_w40_in.data());
	cl::Buffer buffer_in_w45_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_w45_bytes, fpga.source_w45_in.data());
	cl::Buffer buffer_in_w49_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_w49_bytes, fpga.source_w49_in.data());
	fpga.buffer_wvec_in.push_back(buffer_in_w27_tmp);
	fpga.buffer_wvec_in.push_back(buffer_in_w31_tmp);
	fpga.buffer_wvec_in.push_back(buffer_in_w36_tmp);
	fpga.buffer_wvec_in.push_back(buffer_in_w40_tmp);
	fpga.buffer_wvec_in.push_back(buffer_in_w45_tmp);
	fpga.buffer_wvec_in.push_back(buffer_in_w49_tmp);
    
	fpga.writeList.reserve(NUM_CU*NBUFFER);
	fpga.kernList.reserve(NUM_CU*NBUFFER);
	fpga.readList.reserve(NUM_CU*NBUFFER);
		
	for (int ib = 0; ib < NBUFFER; ib++) {
		for (int ik = 0; ik < NUM_CU; ik++) {
			cl::Buffer buffer_in_tmp    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,   vector_size_in_bytes, fpga.source_in.data());
			cl::Buffer buffer_out_tmp(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector_size_out_bytes, fpga.source_hw_results.data());
			fpga.buffer_in.push_back(buffer_in_tmp);
			fpga.buffer_out.push_back(buffer_out_tmp);
	
			cl::Event tmp_write = cl::Event();
			cl::Event tmp_kern = cl::Event();
			cl::Event tmp_read = cl::Event();
			fpga.write_event.push_back(tmp_write);
			fpga.kern_event.push_back(tmp_kern);
			//fpga.read_event.push_back(tmp_read);

			int narg = 0;
			fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_in[ib*NUM_CU+ik]);
			fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_wvec_in[0]);
			fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_wvec_in[1]);
			fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_wvec_in[2]);
			fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_wvec_in[3]);
			fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_wvec_in[4]);
			fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_wvec_in[5]);

			fpga.krnl_xil[ib*NUM_CU+ik].setArg(narg++, fpga.buffer_out[ib*NUM_CU+ik]);
			fpga.isFirstRun.push_back(true);
			std::vector<cl::Event> tmp_write_vec(1);
			std::vector<cl::Event> tmp_kern_vec(1);
			std::vector<cl::Event> tmp_read_vec(1);
			fpga.writeList.push_back(tmp_write_vec);
			fpga.kernList.push_back(tmp_kern_vec);
			fpga.readList.push_back(tmp_read_vec);
		}
	}
//=======================
//load input&output file
//=======================
	//load input data from text file
	std::ifstream fin(datadir+"./tb_input_features.dat");
	//load predictions from text file
	std::ifstream fpr(datadir+"./tb_output_predictions.dat");
	
	std::string iline;
	std::string pline;
	int e = 0;
	if (!(fin.is_open()) || !(fpr.is_open())) {
		std::cout << "Unable to open input/predictions file, using random input" << std::endl;
		//flag for success/failure of finding data files
	}
	std::ofstream fout;
	fout.open("./tb_output_data.dat");
//=====================
//input
//=====================
	int input_set = 0;
	
	std::cout << "first time (just load the weight file)" << std::endl;
	//Send into FPGA's DRAM
	for(int i0 = 0; i0 < IN_STREAM_LEN*DATA_SIZE_IN; i0++) { 
		fpga.source_in[i0] = 0;
	}
	//Reset the output result
	for(int j = 0 ; j < DATA_SIZE_OUT ; j++){
		fpga.source_hw_results[j] = 0;
	}
	fpga.ithr = 0;
	std::cout <<"run FPGA"<<std::endl;
	FPGA(std::ref(fpga));
	
	//start to run
	if(fin.is_open() && fpr.is_open()){
	// If files are valid and their end has not been reached yet, get inputs/predictions from files
	while(std::getline(fin,iline) && std::getline(fpr,pline)) {
		
	if (e%1000==0) std::cout << "Processing event " << e << std::endl;
	e++;
	
		//Here is input.
	char* cstr=const_cast<char*>(iline.c_str());
	char* current;
	std::vector<float> in;
	current=strtok(cstr," ");
	while(current!=NULL){
		in.push_back(atof(current));
		current=strtok(NULL," ");
	}
	
	//Here is the corresponding output(correct one)
	cstr=const_cast<char*>(pline.c_str());
	std::vector<float> pr;
	current=strtok(cstr," ");
	while(current!=NULL){
		pr.push_back(atof(current));
		current=strtok(NULL," ");
	}
	//Send into FPGA's DRAM
	for(int i0 = 0; i0 < IN_STREAM_LEN*DATA_SIZE_IN; i0++) { 
		fpga.source_in[i0] = (bigdata_t)in[i0];
		//std::cout<<(double)fpga.source_in[i0]<<std::endl;
	}
	//Reset the output result
	for(int j = 0 ; j < DATA_SIZE_OUT ; j++){
		fpga.source_hw_results[j] = 0;
	}
//=====================
//run the FPGA
//=====================

	auto ts0 = Clock::now();
	fpga.ithr = 0;
	std::cout <<"run FPGA"<<std::endl;
	FPGA(std::ref(fpga));
	auto ts4 = Clock::now();
	for (int i = 0 ; i < NUM_CU ; i++){
		OCL_CHECK(fpga.err, fpga.err = fpga.q[i].flush());
		OCL_CHECK(fpga.err, fpga.err = fpga.q[i].finish());
	}
	//OPENCL HOST CODE AREA END
	auto ts5 = Clock::now();
	//print the time
	print_nanoseconds("      begin:  ",ts0, input_set, fout);
	print_nanoseconds("       done:  ",ts4, input_set, fout);
	print_nanoseconds("       end:   ",ts5, input_set, fout);
	//std::cout << fpga.ss.str();


//=====================
//output result
//=====================
	std::cout<<"Predictions: \n";
	fout <<"Predictions:  ";
	std::cout << pr[0] << " \t";
	fout << pr[0] << "\n";
	std::cout << std::endl;

	std::cout<<"Quantized predictions: \n";
	fout <<"Quantized predictions:  ";
	std::cout << fpga.source_hw_results[0] << " \t";
	fout << fpga.source_hw_results[0] << "\n\n"; 
	std::cout << std::endl;
	std::cout<<"---- END EVENT "<<" ----"<<std::endl;
	input_set++;
	}
}
	fin.close();
	fpr.close();
	fout.close();
	return EXIT_SUCCESS;
}

