package nekbone10

import (
	_ "embed"
	"log"
	"math"

	"gitlab.com/akita/mgpusim/driver"
	"gitlab.com/akita/mgpusim/insts"
	"gitlab.com/akita/mgpusim/kernels"
)

//go:embed kernels.hsaco
var hsacoBytes []byte

type KernelArgs struct {
	X                   driver.GPUPtr
	D                   driver.GPUPtr
	Y                   driver.GPUPtr
	N                   int32
	Elements            int32
	Lambda              float32
	Pad0                int32
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
	HiddenNone0         int64
	HiddenNone1         int64
	HiddenNone2         int64
	HiddenMultiGridSync int64
}

type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	gpus    []int
	kernel  *insts.HsaCo

	Order, Elements, Iterations int
	Lambda                      float32

	dmat      []float32
	x0, x1, y []float32
	gpuOut    []float32
	dX0, dX1  driver.GPUPtr
	dY, dD    driver.GPUPtr
	useUM     bool
	useLASP   bool
}

func NewBenchmark(d *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:     d,
		context:    d.Init(),
		Order:      10,
		Elements:   64,
		Iterations: 3,
		Lambda:     0.1,
	}
	b.kernel = kernels.LoadProgramFromMemory(hsacoBytes, "nekbone10_apply")
	if b.kernel == nil {
		log.Panic("cannot load nekbone10_apply")
	}
	return b
}

func (b *Benchmark) SelectGPU(g []int) {
	if len(g) > 1 {
		panic("nekbone10 single GPU only")
	}
	b.gpus = g
}
func (b *Benchmark) SetUnifiedMemory()   { b.useUM = true }
func (b *Benchmark) SetLASPMemoryAlloc() { b.useLASP = true }

func (b *Benchmark) Run() {
	b.initMem()
	b.exec()
}

func (b *Benchmark) alloc(sz uint64) driver.GPUPtr {
	if b.useUM {
		return b.driver.AllocateUnifiedMemory(b.context, sz)
	}
	if b.useLASP {
		return b.driver.AllocateMemoryLASP(b.context, sz, "div4")
	}
	return b.driver.AllocateMemory(b.context, sz)
}

func (b *Benchmark) initMem() {
	n := b.Order
	total := b.Elements * n * n * n
	b.x0 = make([]float32, total)
	b.x1 = make([]float32, total)
	b.y = make([]float32, total)
	b.gpuOut = make([]float32, total)
	b.dmat = make([]float32, n*n)

	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			if r == c {
				b.dmat[r*n+c] = 2.0
			} else if c == r+1 {
				b.dmat[r*n+c] = -1.0
			} else if r == c+1 {
				b.dmat[r*n+c] = 1.0
			}
		}
	}
	for i := range b.x0 {
		b.x0[i] = float32((i%13)+1) * 0.03125
	}

	b.dX0 = b.alloc(uint64(len(b.x0) * 4))
	b.dX1 = b.alloc(uint64(len(b.x1) * 4))
	b.dY = b.alloc(uint64(len(b.y) * 4))
	b.dD = b.alloc(uint64(len(b.dmat) * 4))
}

func (b *Benchmark) exec() {
	b.driver.MemCopyH2D(b.context, b.dX0, b.x0)
	b.driver.MemCopyH2D(b.context, b.dX1, b.x1)
	b.driver.MemCopyH2D(b.context, b.dD, b.dmat)

	n := uint32(b.Order)
	local := [3]uint16{8, 8, 1}
	global := [3]uint32{
		alignUp(n, uint32(local[0])),
		alignUp(n*n, uint32(local[1])),
		alignUp(uint32(b.Elements), uint32(local[2])),
	}
	src := b.dX0
	dst := b.dX1
	for it := 0; it < b.Iterations; it++ {
		args := KernelArgs{
			X:                   src,
			D:                   b.dD,
			Y:                   dst,
			N:                   int32(b.Order),
			Elements:            int32(b.Elements),
			Lambda:              b.Lambda,
			Pad0:                0,
			HiddenGlobalOffsetX: 0,
			HiddenGlobalOffsetY: 0,
			HiddenGlobalOffsetZ: 0,
			HiddenNone0:         0,
			HiddenNone1:         0,
			HiddenNone2:         0,
			HiddenMultiGridSync: 0,
		}
		b.driver.LaunchKernel(b.context, b.kernel, global, local, &args)
		src, dst = dst, src
	}
	b.driver.MemCopyD2H(b.context, b.gpuOut, src)
}

func (b *Benchmark) Verify() {
	n := b.Order
	ref0 := append([]float32(nil), b.x0...)
	ref1 := make([]float32, len(ref0))
	src := ref0
	dst := ref1

	idx4 := func(e, i, j, k int) int { return ((e*n+k)*n+j)*n + i }

	for it := 0; it < b.Iterations; it++ {
		for e := 0; e < b.Elements; e++ {
			for k := 0; k < n; k++ {
				for j := 0; j < n; j++ {
					for i := 0; i < n; i++ {
						var sx, sy, sz float32
						for p := 0; p < n; p++ {
							sx += b.dmat[i*n+p] * src[idx4(e, p, j, k)]
							sy += b.dmat[j*n+p] * src[idx4(e, i, p, k)]
							sz += b.dmat[k*n+p] * src[idx4(e, i, j, p)]
						}
						out := idx4(e, i, j, k)
						dst[out] = sx + sy + sz + b.Lambda*src[out]
					}
				}
			}
		}
		src, dst = dst, src
	}

	for i := range src {
		if math.Abs(float64(src[i]-b.gpuOut[i])) > 5e-5 {
			log.Panicf("nekbone10 mismatch at %d: want %f got %f", i, src[i], b.gpuOut[i])
		}
	}
	log.Printf("Passed!\n")
}

func alignUp(v, a uint32) uint32 {
	if a == 0 {
		return v
	}
	return ((v + a - 1) / a) * a
}
