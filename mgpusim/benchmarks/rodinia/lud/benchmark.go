package lud

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
	A                   driver.GPUPtr
	N                   int32
	K                   int32
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

	Dim int

	aInit   []float32
	aCPU    []float32
	aGPU    []float32
	dA      driver.GPUPtr
	useUM   bool
	useLASP bool
}

func NewBenchmark(d *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:  d,
		context: d.Init(),
		Dim:     128,
	}
	b.kernel = kernels.LoadProgramFromMemory(hsacoBytes, "lud_trailing_update")
	if b.kernel == nil {
		log.Panic("failed to load lud_trailing_update")
	}
	return b
}

func (b *Benchmark) SelectGPU(g []int) {
	if len(g) > 1 {
		panic("lud single GPU only")
	}
	b.gpus = g
}
func (b *Benchmark) SetUnifiedMemory()   { b.useUM = true }
func (b *Benchmark) SetLASPMemoryAlloc() { b.useLASP = true }

func (b *Benchmark) alloc(sz uint64) driver.GPUPtr {
	if b.useUM {
		return b.driver.AllocateUnifiedMemory(b.context, sz)
	}
	if b.useLASP {
		return b.driver.AllocateMemoryLASP(b.context, sz, "div4")
	}
	return b.driver.AllocateMemory(b.context, sz)
}

func (b *Benchmark) Run() {
	b.initMem()
	b.exec()
}

func (b *Benchmark) initMem() {
	n := b.Dim
	b.aInit = make([]float32, n*n)
	b.aCPU = make([]float32, n*n)
	b.aGPU = make([]float32, n*n)

	// Diagonal-dominant matrix to keep no-pivot LU numerically stable.
	for i := 0; i < n; i++ {
		var rowSum float32
		for j := 0; j < n; j++ {
			v := float32(((i*17+j*13)%23)+1) * 0.01
			b.aInit[i*n+j] = v
			rowSum += float32(math.Abs(float64(v)))
		}
		b.aInit[i*n+i] += rowSum + 1.0
	}
	copy(b.aCPU, b.aInit)

	b.dA = b.alloc(uint64(n * n * 4))
}

func (b *Benchmark) exec() {
	n := b.Dim
	b.driver.MemCopyH2D(b.context, b.dA, b.aInit)

	local := [3]uint16{8, 8, 1}

	for k := 0; k < n; k++ {
		b.driver.MemCopyD2H(b.context, b.aGPU, b.dA)

		pivot := b.aGPU[k*n+k]
		for j := k; j < n; j++ {
			b.aGPU[k*n+j] /= pivot
		}
		b.aGPU[k*n+k] = 1.0

		for i := k + 1; i < n; i++ {
			sum := b.aGPU[i*n+k]
			for p := 0; p < k; p++ {
				sum -= b.aGPU[i*n+p] * b.aGPU[p*n+k]
			}
			b.aGPU[i*n+k] = sum
		}

		b.driver.MemCopyH2D(b.context, b.dA, b.aGPU)

		if k+1 < n {
			span := uint32(n - (k + 1))
			global := [3]uint32{
				alignUp(span, uint32(local[0])),
				alignUp(span, uint32(local[1])),
				1,
			}
			args := KernelArgs{
				A:                   b.dA,
				N:                   int32(n),
				K:                   int32(k),
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
		}
	}

	b.driver.MemCopyD2H(b.context, b.aGPU, b.dA)
}

func (b *Benchmark) Verify() {
	n := b.Dim
	a := b.aCPU

	for k := 0; k < n; k++ {
		pivot := a[k*n+k]
		for j := k; j < n; j++ {
			a[k*n+j] /= pivot
		}
		a[k*n+k] = 1.0

		for i := k + 1; i < n; i++ {
			sum := a[i*n+k]
			for p := 0; p < k; p++ {
				sum -= a[i*n+p] * a[p*n+k]
			}
			a[i*n+k] = sum
		}

		for i := k + 1; i < n; i++ {
			for j := k + 1; j < n; j++ {
				a[i*n+j] -= a[i*n+k] * a[k*n+j]
			}
		}
	}

	for i := range a {
		if math.Abs(float64(a[i]-b.aGPU[i])) > 3e-4 {
			log.Panicf("lud mismatch at %d: want %f got %f", i, a[i], b.aGPU[i])
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
