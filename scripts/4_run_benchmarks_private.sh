#!/bin/bash

configs=("with_sector_cache_260428")

# benchmarks=("convolution2d" "fastwalshtransform" "gups" "jacobi1d" "jacobi2d" "kmeans" "matrixtranspose" "mis" "pagerank" "simpleconvolution" "shoc-reduction" "spmv" "stencil2d" "syrk" "syr2k")
benchmarks=("adi" "aes" "atax" "bfs" "bicg" "bitonicsort" "color" "conv2d" "convolution3d" "correlation" "covariance" "doitgen" "fastwalshtransform" "fdtd2d" "fft" "fir" "floydwarshall" "gemm" "gemver" "gesummv" "gramschmidt" "gups" "im2col" "jacobi1d" "jacobi2d" "kmeans" "lenet" "lu" "matrixmultiplication" "matrixtranspose" "minerva" "mis" "mm2" "mm3" "mvt" "nbody" "pagerank" "relu" "simpleconvolution" "shocreduction" "spmv" "sssp" "stencil2d" "syrk" "syr2k" "vgg16" "xor")


# benchmarks=("diffusion")
# benchmarks=("hit")
# benchmarks=("als")
# benchmarks=("hotspot")
# benchmarks=("backprop")
# benchmarks=("eqwp")
# benchmarks=("lulesh")
# benchmarks=("pennant")
# benchmarks=("mst")
# benchmarks=("snap")
# benchmarks=("tartan-pagerank")
# benchmarks=("nekbone10")
# benchmarks=("lud")
# benchmarks=("gaussian")
# benchmarks=("srad")
# benchmarks=("diffusion" "hit" "als" "hotspot" "backprop" "eqwp" "lud" "lulesh" "pennant" "mst" "snap" "tartan-pagerank" "nekbone10" "gaussian" "srad")
# benchmarks=("mis")

for config in ${configs[@]}; 
do
  for benchmark in ${benchmarks[@]}; 
  do
    echo $config $benchmark
    cd $config
    pwd
        
    bash ${benchmark}.sh > /dev/null 2>&1 &
    # bash ${benchmark}.sh | head -c 80M > ${benchmark}.csv &
    
    cd ..
  done
done
        

