all: kernels.hsaco

kernels.hsaco: native/kernels.cl
	/opt/rocm/bin/clang-ocl -mcpu=gfx803 native/kernels.cl -o kernels.hsaco

kernels.asm: native/kernels.cl
	/opt/rocm/bin/clang-ocl -mcpu=gfx803 native/kernels.cl -S -o kernels.asm

kernels.disasm: kernels.hsaco
	/opt/rocm/hcc/bin/llvm-objdump -mcpu=gfx803 -disassemble $^ > $@

.PHONY: all
