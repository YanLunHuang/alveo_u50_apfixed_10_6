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

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce latency and 
    device resource utilization of the resulting RTL code
    This is a wrapper to be used with an hls4ml project to enable proper handling by SDAccel
*******************************************************************************/
#include <iostream>
#define PROJ_HDR <MYPROJ.h>

#include PROJ_HDR
#include "kernel_params.h"

template<unsigned N> 
void fillWeights(const model_default_t iWeightsIn[N], model_default_t weights[N]) { 
	for(int i0 = 0; i0 < N; i0++) { 
		weights[i0] = iWeightsIn[i0];
	}
}

extern "C" {

void alveo_hls4ml(
	const bigdata_t *in, // Read-Only Vector
	const model_default_t *in_weights1, 
	const model_default_t *in_weights2, 
	const model_default_t *in_weights3,
	const model_default_t *in_weights4,
	const model_default_t *in_weights5,
	const model_default_t *in_weights6,
	bigdata_t *out       // Output Result
	)
{
	// SDAccel kernel must have one and only one s_axilite interface which will be used by host application to configure the kernel.
	// Here bundle control is defined which is s_axilite interface and associated with all the arguments (in and out),
	// control interface must also be associated with "return".
	// All the global memory access arguments must be associated to one m_axi(AXI Master Interface). Here all two arguments(in, out) are 
	// associated to bundle gmem which means that a AXI master interface named "gmem" will be created in Kernel and all these variables will be 
	// accessing global memory through this interface.
	// Multiple interfaces can also be created based on the requirements. For example when multiple memory accessing arguments need access to
	// global memory simultaneously, user can create multiple master interfaces and can connect to different arguments.
	#pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=in_weights1  offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=in_weights2  offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=in_weights3  offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=in_weights4  offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=in_weights5  offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=in_weights6  offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem
	#pragma HLS INTERFACE s_axilite port=in   bundle=control
	#pragma HLS INTERFACE s_axilite port=in_weights1   bundle=control
	#pragma HLS INTERFACE s_axilite port=in_weights2   bundle=control
	#pragma HLS INTERFACE s_axilite port=in_weights3   bundle=control
	#pragma HLS INTERFACE s_axilite port=in_weights4   bundle=control
	#pragma HLS INTERFACE s_axilite port=in_weights5   bundle=control
	#pragma HLS INTERFACE s_axilite port=in_weights6   bundle=control
	#pragma HLS INTERFACE s_axilite port=out  bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	//necessary for hls4ml kernel, not used
	#pragma HLS DATAFLOW


	static model_default_t w27[NW1];
	static model_default_t w31[NW2];
	static model_default_t w36[NW3];
	static model_default_t w40[NW4];
	static model_default_t w45[NW5];
	static model_default_t w49[NW6];
	#pragma HLS RESOURCE variable=w27  core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=w31  core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=w36  core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=w40  core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=w45  core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=w49  core=XPM_MEMORY uram
	
	//load for the first time
	static bool fillWeights_ = false;
	if(!fillWeights_) {
		fillWeights<NW1>(in_weights1,w27);
		fillWeights<NW2>(in_weights2,w31);
		fillWeights<NW3>(in_weights3,w36);
		fillWeights<NW4>(in_weights4,w40);
		fillWeights<NW5>(in_weights5,w45);
		fillWeights<NW6>(in_weights6,w49);
		fillWeights_ = true;
	}
	else {
	bigdata_t in_bigbuf[DATA_SIZE_IN*IN_STREAM_LEN];
	bigdata_t out_bigbuf;
	
	hls::stream<input_t> in_buf[DATA_SIZE_IN];
	hls::stream<result_t> out_buf[DATA_SIZE_OUT];
	//these will get partitioned properly in the hls4ml code

	#pragma HLS ARRAY_PARTITION   variable=in_buf  complete dim=0
	#pragma HLS ARRAY_PARTITION   variable=out_buf complete dim=0
	#pragma HLS STREAM   variable=in_buf  depth=1000
	#pragma HLS STREAM   variable=out_buf depth=1
	
	//getting data from DRAM
	for (int i = 0; i < DATA_SIZE_IN*IN_STREAM_LEN; i++) {
		#pragma HLS LOOP UNROLL
		in_bigbuf[i] = in[i];
	}
	
	std::cout<<"------------------"<<std::endl;
	//=============================================
	//input
	//=============================================
	input_t tmp;
	for(int i0 = 0; i0 < IN_STREAM_LEN; i0++) { 
		for(int i1 = 0; i1 < DATA_SIZE_IN; i1++) { 
			#pragma HLS UNROLL
			tmp = in_bigbuf[i0*4+i1];
			in_buf[i1].write(tmp);
			std::cout<<double(tmp)<<std::endl;
		}
	}
	std::cout<<"inf start"<<std::endl;
	hls4ml: MYPROJ(in_buf,out_buf,w27,w31,w36,w40,w45,w49);
	std::cout<<"inf end"<<std::endl;

	//=============================================
	//output
	//=============================================
	for(int i1 = 0; i1 < DATA_SIZE_OUT; i1++) {
		//#pragma HLS UNROLL
		std::cout<<"reading from ["<<i1<<"]"<<std::endl;
		result_t tmp_small = out_buf[i1].read();
		out_bigbuf = tmp_small;
	}

	out[0] = out_bigbuf;
	std::cout <<(double)out_bigbuf<<std::endl;
	}
}
}
