//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"
//hls-fpga-machine-learning insert weights
#include "weights/s3.h"
#include "weights/b3.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w13.h"
#include "weights/b13.h"
#include "weights/w18.h"
#include "weights/b18.h"
#include "weights/w22.h"
#include "weights/b22.h"
#include "weights/b27.h"
#include "weights/b31.h"
#include "weights/b36.h"
#include "weights/b40.h"
#include "weights/b45.h"
#include "weights/b49.h"
#include "weights/w53.h"
#include "weights/b53.h"


void myproject(
	       hls::stream<input_t>   em_barrel[N_INPUT_3_1],
	       hls::stream<result_t>  layer55_out[N_LAYER_53],
	       //hls::stream<result_t>  layer54_out[N_LAYER_52],
	       model_default_t w27[73728],
	       model_default_t w31[147456],
	       model_default_t w36[294912],
	       model_default_t w40[589824],
	       model_default_t w45[589824],
	       model_default_t w49[65536]
	       ) { 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 4>(s3, "s3.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(b3, "b3.txt");
        nnet::load_weights_from_txt<model_default_t, 1600>(w4, "w4.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 4608>(w9, "w9.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b9, "b9.txt");
        nnet::load_weights_from_txt<model_default_t, 9216>(w13, "w13.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b13, "b13.txt");
        nnet::load_weights_from_txt<model_default_t, 18432>(w18, "w18.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b18, "b18.txt");
        nnet::load_weights_from_txt<model_default_t, 36864>(w22, "w22.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b22, "b22.txt");
        nnet::load_weights_from_txt<model_default_t, 73728>(w27, "w27.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b27, "b27.txt");
        nnet::load_weights_from_txt<model_default_t, 147456>(w31, "w31.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b31, "b31.txt");
        nnet::load_weights_from_txt<model_default_t, 294912>(w36, "w36.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b36, "b36.txt");
        nnet::load_weights_from_txt<model_default_t, 589824>(w40, "w40.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b40, "b40.txt");
        nnet::load_weights_from_txt<model_default_t, 589824>(w45, "w45.txt");
        nnet::load_weights_from_txt<bias45_t, 256>(b45, "b45.txt");
        nnet::load_weights_from_txt<model_default_t, 65536>(w49, "w49.txt");
        nnet::load_weights_from_txt<bias49_t, 256>(b49, "b49.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(w53, "w53.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b53, "b53.txt");
        loaded_weights = true;
    }
#endif

  // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=3080
    nnet::resize_nearest_me<input_t, config2>(em_barrel, layer2_out); // up_sampling2d

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=3080
    nnet::normalize_me<layer2_t, layer3_t, config3>(layer2_out, layer3_out, s3, b3); // batch_normalization

    hls::stream<layer56_t> layer56_out("layer56_out");
    #pragma HLS STREAM variable=layer56_out depth=3540
    nnet::zeropad2d_cl_me<layer3_t, layer56_t, config56>(layer3_out, layer56_out); // zp2d_conv2d

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=3080
    nnet::conv_2d_cl_me<layer56_t, layer4_t, config4>(layer56_out, layer4_out, w4, b4); // conv2d

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=3080
    nnet::leaky_relu_me<layer4_t, layer7_t, LeakyReLU_config7>(layer4_out, 0.30000001192092896, layer7_out); // leaky_re_lu

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=756
    nnet::pooling2d_large_cl_nopad_pad_me<layer7_t, layer8_t, config8>(layer7_out, layer8_out); // max_pooling2d

    hls::stream<layer57_t> layer57_out("layer57_out");
    #pragma HLS STREAM variable=layer57_out depth=870
    nnet::zeropad2d_cl_me<layer8_t, layer57_t, config57>(layer8_out, layer57_out); // zp2d_conv2d_1

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=756
    nnet::conv_2d_cl_me<layer57_t, layer9_t, config9>(layer57_out, layer9_out, w9, b9); // conv2d_1

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=756
    nnet::leaky_relu_me<layer9_t, layer12_t, LeakyReLU_config12>(layer9_out, 0.30000001192092896, layer12_out); // leaky_re_lu_1

    hls::stream<layer58_t> layer58_out("layer58_out");
    #pragma HLS STREAM variable=layer58_out depth=870
    nnet::zeropad2d_cl_me<layer12_t, layer58_t, config58>(layer12_out, layer58_out); // zp2d_conv2d_2

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=756
    nnet::conv_2d_cl_me<layer58_t, layer13_t, config13>(layer58_out, layer13_out, w13, b13); // conv2d_2

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=756
    nnet::leaky_relu_me<layer13_t, layer16_t, LeakyReLU_config16>(layer13_out, 0.30000001192092896, layer16_out); // leaky_re_lu_2

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=182
    nnet::pooling2d_large_cl_nopad_pad_me<layer16_t, layer17_t, config17>(layer16_out, layer17_out); // max_pooling2d_1

    hls::stream<layer59_t> layer59_out("layer59_out");
    #pragma HLS STREAM variable=layer59_out depth=240
    nnet::zeropad2d_cl_me<layer17_t, layer59_t, config59>(layer17_out, layer59_out); // zp2d_conv2d_3

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=182
    nnet::conv_2d_cl_me<layer59_t, layer18_t, config18>(layer59_out, layer18_out, w18, b18); // conv2d_3

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=182
    nnet::leaky_relu_me<layer18_t, layer21_t, LeakyReLU_config21>(layer18_out, 0.30000001192092896, layer21_out); // leaky_re_lu_3

    hls::stream<layer60_t> layer60_out("layer60_out");
    #pragma HLS STREAM variable=layer60_out depth=240
    nnet::zeropad2d_cl_me<layer21_t, layer60_t, config60>(layer21_out, layer60_out); // zp2d_conv2d_4

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=182
    nnet::conv_2d_cl_me<layer60_t, layer22_t, config22>(layer60_out, layer22_out, w22, b22); // conv2d_4

    hls::stream<layer25_t> layer25_out("layer25_out");
    #pragma HLS STREAM variable=layer25_out depth=182
    nnet::leaky_relu_me<layer22_t, layer25_t, LeakyReLU_config25>(layer22_out, 0.30000001192092896, layer25_out); // leaky_re_lu_4

    hls::stream<layer26_t> layer26_out("layer26_out");
    #pragma HLS STREAM variable=layer26_out depth=42
    nnet::pooling2d_large_cl_nopad_pad_me<layer25_t, layer26_t, config26>(layer25_out, layer26_out); // max_pooling2d_2

    hls::stream<layer61_t> layer61_out("layer61_out");
    #pragma HLS STREAM variable=layer61_out depth=72
    nnet::zeropad2d_cl_me<layer26_t, layer61_t, config61>(layer26_out, layer61_out); // zp2d_conv2d_5

    hls::stream<layer27_t> layer27_out("layer27_out");
    #pragma HLS STREAM variable=layer27_out depth=42
    nnet::conv_2d_cl_me<layer61_t, layer27_t, config27>(layer61_out, layer27_out, w27, b27); // conv2d_5

    hls::stream<layer30_t> layer30_out("layer30_out");
    #pragma HLS STREAM variable=layer30_out depth=42
    nnet::leaky_relu_me<layer27_t, layer30_t, LeakyReLU_config30>(layer27_out, 0.30000001192092896, layer30_out); // leaky_re_lu_5

    hls::stream<layer62_t> layer62_out("layer62_out");
    #pragma HLS STREAM variable=layer62_out depth=72
    nnet::zeropad2d_cl_me<layer30_t, layer62_t, config62>(layer30_out, layer62_out); // zp2d_conv2d_6

    hls::stream<layer31_t> layer31_out("layer31_out");
    #pragma HLS STREAM variable=layer31_out depth=42
    nnet::conv_2d_cl_me<layer62_t, layer31_t, config31>(layer62_out, layer31_out, w31, b31); // conv2d_6

    hls::stream<layer34_t> layer34_out("layer34_out");
    #pragma HLS STREAM variable=layer34_out depth=42
    nnet::leaky_relu_me<layer31_t, layer34_t, LeakyReLU_config34>(layer31_out, 0.30000001192092896, layer34_out); // leaky_re_lu_6

    hls::stream<layer35_t> layer35_out("layer35_out");
    #pragma HLS STREAM variable=layer35_out depth=9
    nnet::pooling2d_large_cl_nopad_pad_me<layer34_t, layer35_t, config35>(layer34_out, layer35_out); // max_pooling2d_3

    hls::stream<layer63_t> layer63_out("layer63_out");
    #pragma HLS STREAM variable=layer63_out depth=25
    nnet::zeropad2d_cl_me<layer35_t, layer63_t, config63>(layer35_out, layer63_out); // zp2d_conv2d_7

    hls::stream<layer36_t> layer36_out("layer36_out");
    #pragma HLS STREAM variable=layer36_out depth=9
    nnet::conv_2d_cl_me<layer63_t, layer36_t, config36>(layer63_out, layer36_out, w36, b36); // conv2d_7

    hls::stream<layer39_t> layer39_out("layer39_out");
    #pragma HLS STREAM variable=layer39_out depth=9
    nnet::leaky_relu_me<layer36_t, layer39_t, LeakyReLU_config39>(layer36_out, 0.30000001192092896, layer39_out); // leaky_re_lu_7

    hls::stream<layer64_t> layer64_out("layer64_out");
    #pragma HLS STREAM variable=layer64_out depth=25
    nnet::zeropad2d_cl_me<layer39_t, layer64_t, config64>(layer39_out, layer64_out); // zp2d_conv2d_8

    hls::stream<layer40_t> layer40_out("layer40_out");
    #pragma HLS STREAM variable=layer40_out depth=9
    nnet::conv_2d_cl_me<layer64_t, layer40_t, config40>(layer64_out, layer40_out, w40, b40); // conv2d_8

    hls::stream<layer43_t> layer43_out("layer43_out");
    #pragma HLS STREAM variable=layer43_out depth=9
    nnet::leaky_relu_me<layer40_t, layer43_t, LeakyReLU_config43>(layer40_out, 0.30000001192092896, layer43_out); // leaky_re_lu_8

    hls::stream<layer45_t> layer45_out("layer45_out");
    #pragma HLS STREAM variable=layer45_out depth=1
    nnet::dense_large_stream_me<layer43_t, layer45_t, config45>(layer43_out, layer45_out, w45, b45); // dense

    hls::stream<layer48_t> layer48_out("layer48_out");
    #pragma HLS STREAM variable=layer48_out depth=1
    nnet::leaky_relu_me<layer45_t, layer48_t, LeakyReLU_config48>(layer45_out, 0.30000001192092896, layer48_out); // leaky_re_lu_9

    hls::stream<layer49_t> layer49_out("layer49_out");
    #pragma HLS STREAM variable=layer49_out depth=1
    nnet::dense_large_stream_me<layer48_t, layer49_t, config49>(layer48_out, layer49_out, w49, b49); // dense_1

    hls::stream<layer52_t> layer52_out("layer52_out");
    #pragma HLS STREAM variable=layer52_out depth=1
    nnet::leaky_relu_me<layer49_t, layer52_t, LeakyReLU_config52>(layer49_out, 0.30000001192092896, layer52_out); // leaky_re_lu_10

    hls::stream<layer53_t> layer53_out("layer53_out");
    #pragma HLS STREAM variable=layer53_out depth=1
    nnet::dense_large_stream_me<layer52_t, layer53_t, config53>(layer52_out, layer53_out, w53, b53); // dense_2

    nnet::relu_me<layer53_t, result_t, relu_config55>(layer53_out, layer55_out); // activation
}

