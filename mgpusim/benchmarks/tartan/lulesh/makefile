all: kernels.hsaco

kernels.hsaco: native/kernels.cl
	/opt/rocm/bin/clang-ocl -mcpu=gfx803 -mcode-object-version=2 native/kernels.cl -o kernels.hsaco

kernels.asm: native/kernels.cl
	/opt/rocm/bin/clang-ocl -mcpu=gfx803 -mcode-object-version=2 native/kernels.cl -S -o kernels.asm

.PHONY: all
