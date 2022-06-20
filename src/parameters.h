#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_image.h"
#include "nnet_utils/nnet_image_stream.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
 

//hls-fpga-machine-learning insert layer-config
// up_sampling2d
struct config2 : nnet::resize_config {
    static const unsigned height = 56;
    static const unsigned width = 11;
    static const unsigned n_chan = 4;
    static const unsigned new_height = 56;
    static const unsigned new_width = 55;
};

// batch_normalization
struct config3 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_2*OUT_WIDTH_2*N_CHAN_2;
    static const unsigned n_filt = 4;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// zp2d_conv2d
struct config56 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_2;
    static const unsigned in_width = OUT_WIDTH_2;
    static const unsigned n_chan = N_CHAN_2;
    static const unsigned out_height = OUT_HEIGHT_56;
    static const unsigned out_width = OUT_WIDTH_56;
    static const unsigned pad_top = 2;
    static const unsigned pad_bottom = 2;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 2;
};

// conv2d
struct config4_mult : nnet::dense_config {
    static const unsigned n_in = 100;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 100;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config4 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_56;
    static const unsigned in_width = OUT_WIDTH_56;
    static const unsigned n_chan = N_CHAN_56;
    static const unsigned filt_height = 5;
    static const unsigned filt_width = 5;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_4;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_4;
    static const unsigned out_width = OUT_WIDTH_4;
    static const unsigned reuse_factor = 1600;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 9;
    static const unsigned min_width = 9;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config4_mult mult_config;
};
const ap_uint<config4::filt_height * config4::filt_width> config4::pixels[] = {1,3,7,15,31,30,28,24,16,33,99,231,495,1023,990,924,792,528,1057,3171,7399,15855,32767,31710,29596,25368,16912,33825,101475,236775,507375,1048575,1014750,947100,811800,541200,1082401,3247203,7576807,16236015,33554431,32472030,30307228,25977624,17318416,1082400,3247200,7576800,16236000,33554400,32472000,30307200,25977600,17318400,1082368,3247104,7576576,16235520,33553408,32471040,30306304,25976832,17317888,1081344,3244032,7569408,16220160,33521664,32440320,30277632,25952256,17301504,1048576,3145728,7340032,15728640,32505856,31457280,29360128,25165824,16777216};

// leaky_re_lu
struct LeakyReLU_config7 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef ap_fixed<18,8> table_t;
};

// max_pooling2d
struct config8 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_4;
    static const unsigned in_width = OUT_WIDTH_4;
    static const unsigned n_filt = N_FILT_8;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;
    static const unsigned n_chan = N_FILT_8;

    static const unsigned out_height = OUT_HEIGHT_8;
    static const unsigned out_width = OUT_WIDTH_8;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse = 2000;
    typedef ap_fixed<10,6> accum_t;
};

// zp2d_conv2d_1
struct config57 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_8;
    static const unsigned in_width = OUT_WIDTH_8;
    static const unsigned n_chan = N_FILT_8;
    static const unsigned out_height = OUT_HEIGHT_57;
    static const unsigned out_width = OUT_WIDTH_57;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_1
struct config9_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 144;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config9 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_57;
    static const unsigned in_width = OUT_WIDTH_57;
    static const unsigned n_chan = N_CHAN_57;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_9;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_9;
    static const unsigned out_width = OUT_WIDTH_9;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config9_mult mult_config;
};
const ap_uint<config9::filt_height * config9::filt_width> config9::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_1
struct LeakyReLU_config12 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_9*OUT_WIDTH_9*N_FILT_9;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_conv2d_2
struct config58 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_9;
    static const unsigned in_width = OUT_WIDTH_9;
    static const unsigned n_chan = N_FILT_9;
    static const unsigned out_height = OUT_HEIGHT_58;
    static const unsigned out_width = OUT_WIDTH_58;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_2
struct config13_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 288;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config13 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_58;
    static const unsigned in_width = OUT_WIDTH_58;
    static const unsigned n_chan = N_CHAN_58;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_13;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_13;
    static const unsigned out_width = OUT_WIDTH_13;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config13_mult mult_config;
};
const ap_uint<config13::filt_height * config13::filt_width> config13::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_2
struct LeakyReLU_config16 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_13*OUT_WIDTH_13*N_FILT_13;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef ap_fixed<18,8> table_t;
};

// max_pooling2d_1
struct config17 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_13;
    static const unsigned in_width = OUT_WIDTH_13;
    static const unsigned n_filt = N_FILT_17;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;
    static const unsigned n_chan = N_FILT_17;

    static const unsigned out_height = OUT_HEIGHT_17;
    static const unsigned out_width = OUT_WIDTH_17;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse = 2000;
    typedef ap_fixed<10,6> accum_t;
};

// zp2d_conv2d_3
struct config59 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_17;
    static const unsigned in_width = OUT_WIDTH_17;
    static const unsigned n_chan = N_FILT_17;
    static const unsigned out_height = OUT_HEIGHT_59;
    static const unsigned out_width = OUT_WIDTH_59;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_3
struct config18_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 288;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config18 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_59;
    static const unsigned in_width = OUT_WIDTH_59;
    static const unsigned n_chan = N_CHAN_59;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_18;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_18;
    static const unsigned out_width = OUT_WIDTH_18;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config18_mult mult_config;
};
const ap_uint<config18::filt_height * config18::filt_width> config18::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_3
struct LeakyReLU_config21 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_18*OUT_WIDTH_18*N_FILT_18;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_conv2d_4
struct config60 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_18;
    static const unsigned in_width = OUT_WIDTH_18;
    static const unsigned n_chan = N_FILT_18;
    static const unsigned out_height = OUT_HEIGHT_60;
    static const unsigned out_width = OUT_WIDTH_60;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_4
struct config22_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 576;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config22 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_60;
    static const unsigned in_width = OUT_WIDTH_60;
    static const unsigned n_chan = N_CHAN_60;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_22;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_22;
    static const unsigned out_width = OUT_WIDTH_22;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config22_mult mult_config;
};
const ap_uint<config22::filt_height * config22::filt_width> config22::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_4
struct LeakyReLU_config25 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_22*OUT_WIDTH_22*N_FILT_22;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef ap_fixed<18,8> table_t;
};

// max_pooling2d_2
struct config26 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_22;
    static const unsigned in_width = OUT_WIDTH_22;
    static const unsigned n_filt = N_FILT_26;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;
    static const unsigned n_chan = N_FILT_26;

    static const unsigned out_height = OUT_HEIGHT_26;
    static const unsigned out_width = OUT_WIDTH_26;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse = 2000;
    typedef ap_fixed<10,6> accum_t;
};

// zp2d_conv2d_5
struct config61 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_26;
    static const unsigned in_width = OUT_WIDTH_26;
    static const unsigned n_chan = N_FILT_26;
    static const unsigned out_height = OUT_HEIGHT_61;
    static const unsigned out_width = OUT_WIDTH_61;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_5
struct config27_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 576;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config27 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_61;
    static const unsigned in_width = OUT_WIDTH_61;
    static const unsigned n_chan = N_CHAN_61;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_27;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_27;
    static const unsigned out_width = OUT_WIDTH_27;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config27_mult mult_config;
};
const ap_uint<config27::filt_height * config27::filt_width> config27::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_5
struct LeakyReLU_config30 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_27*OUT_WIDTH_27*N_FILT_27;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_conv2d_6
struct config62 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_27;
    static const unsigned in_width = OUT_WIDTH_27;
    static const unsigned n_chan = N_FILT_27;
    static const unsigned out_height = OUT_HEIGHT_62;
    static const unsigned out_width = OUT_WIDTH_62;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_6
struct config31_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 1152;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config31 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_62;
    static const unsigned in_width = OUT_WIDTH_62;
    static const unsigned n_chan = N_CHAN_62;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_31;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_31;
    static const unsigned out_width = OUT_WIDTH_31;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config31_mult mult_config;
};
const ap_uint<config31::filt_height * config31::filt_width> config31::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_6
struct LeakyReLU_config34 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_31*OUT_WIDTH_31*N_FILT_31;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef ap_fixed<18,8> table_t;
};

// max_pooling2d_3
struct config35 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_31;
    static const unsigned in_width = OUT_WIDTH_31;
    static const unsigned n_filt = N_FILT_35;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;
    static const unsigned n_chan = N_FILT_35;

    static const unsigned out_height = OUT_HEIGHT_35;
    static const unsigned out_width = OUT_WIDTH_35;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse = 2000;
    typedef ap_fixed<10,6> accum_t;
};

// zp2d_conv2d_7
struct config63 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_35;
    static const unsigned in_width = OUT_WIDTH_35;
    static const unsigned n_chan = N_FILT_35;
    static const unsigned out_height = OUT_HEIGHT_63;
    static const unsigned out_width = OUT_WIDTH_63;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_7
struct config36_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 1152;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config36 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_63;
    static const unsigned in_width = OUT_WIDTH_63;
    static const unsigned n_chan = N_CHAN_63;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_36;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_36;
    static const unsigned out_width = OUT_WIDTH_36;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config36_mult mult_config;
};
const ap_uint<config36::filt_height * config36::filt_width> config36::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_7
struct LeakyReLU_config39 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_36*OUT_WIDTH_36*N_FILT_36;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_conv2d_8
struct config64 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_36;
    static const unsigned in_width = OUT_WIDTH_36;
    static const unsigned n_chan = N_FILT_36;
    static const unsigned out_height = OUT_HEIGHT_64;
    static const unsigned out_width = OUT_WIDTH_64;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_8
struct config40_mult : nnet::dense_config {
    static const unsigned n_in = 2304;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 2304;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config40 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_64;
    static const unsigned in_width = OUT_WIDTH_64;
    static const unsigned n_chan = N_CHAN_64;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_40;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_40;
    static const unsigned out_width = OUT_WIDTH_40;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config40_mult mult_config;
};
const ap_uint<config40::filt_height * config40::filt_width> config40::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_8
struct LeakyReLU_config43 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_40*OUT_WIDTH_40*N_FILT_40;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef ap_fixed<18,8> table_t;
};

// dense
struct config45 : nnet::dense_config {
    static const unsigned n_in = N_SIZE_1_44;
    static const unsigned n_out = N_LAYER_45;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 589824;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<10,6> accum_t;
    typedef bias45_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// leaky_re_lu_9
struct LeakyReLU_config48 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_45;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef ap_fixed<18,8> table_t;
};

// dense_1
struct config49 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_45;
    static const unsigned n_out = N_LAYER_49;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 256;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 65536;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<10,6> accum_t;
    typedef bias49_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// leaky_re_lu_10
struct LeakyReLU_config52 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_49;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef ap_fixed<18,8> table_t;
};

// dense_2
struct config53 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_49;
    static const unsigned n_out = N_LAYER_53;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 256;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 256;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<10,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// activation
struct relu_config55 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_53;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef ap_fixed<18,8> table_t;
};


#endif
