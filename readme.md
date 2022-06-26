# hls4ml on Alveo U50 (HLS C/C++ Kernel)

## Compile SDAccel project

make check TARGET=sw_emu DEVICE=xilinx_u50_xdma_201920_1 all # software emulation
make check TARGET=hw_emu DEVICE=xilinx_u50_xdma_201920_1 all # hardware emulation
make TARGET=hw DEVICE=xilinx_u50_xdma_201920_1 all # build

##Run project

./host 
