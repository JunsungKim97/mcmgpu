# HSACO for Akita GCN3 / mgpusim — see:
# https://github.com/sarchlab/akita_docs/blob/main/docs/mgpusim/03_prepare_new_benchmarks.md
#
# Recommended: ROCm in Docker (e.g. rocm/dev-ubuntu-20.04:3.9). clang-ocl lives under
# /opt/rocm/bin. ISA must be gfx803:
#   clang-ocl -mcpu=gfx803 kernels.cl -o kernels.hsaco

all: kernels.hsaco

kernels.hsaco: native/kernels.cl
	/opt/rocm/bin/clang-ocl -mcpu=gfx803 native/kernels.cl -o kernels.hsaco

# Optional: human-readable assembly (same doc)
kernels.asm: native/kernels.cl
	/opt/rocm/bin/clang-ocl -mcpu=gfx803 native/kernels.cl -S -o kernels.asm

kernels.disasm: kernels.hsaco
	/opt/rocm/hcc/bin/llvm-objdump -mcpu=gfx803 -disassemble $^ > $@

.PHONY: all
