package gaussian

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

type NormalizeArgs struct {
	A                   driver.GPUPtr
	B                   driver.GPUPtr
	N                   int32
	T                   int32
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
	HiddenNone0         int64
	HiddenNone1         int64
	HiddenNone2         int64
	HiddenMultiGridSync int64
}

type EliminateArgs struct {
	A                   driver.GPUPtr
	B                   driver.GPUPtr
	N                   int32
	T                   int32
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

	normKernel *insts.HsaCo
	elimKernel *insts.HsaCo

	Dim int

	aInit []float32
	bInit []float32
	aCPU  []float32
	bCPU  []float32
	xCPU  []float32
	aGPU  []float32
	bGPU  []float32
	xGPU  []float32

	dA, dB  driver.GPUPtr
	useUM   bool
	useLASP bool
}

func NewBenchmark(d *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:  d,
		context: d.Init(),
		Dim:     128,
	}
	b.normKernel = kernels.LoadProgramFromMemory(hsacoBytes, "gaussian_normalize")
	b.elimKernel = kernels.LoadProgramFromMemory(hsacoBytes, "gaussian_eliminate")
	if b.normKernel == nil || b.elimKernel == nil {
		log.Panic("failed to load gaussian kernels")
	}
	return b
}

func (b *Benchmark) SelectGPU(g []int) {
	if len(g) > 1 {
		panic("gaussian single GPU only")
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
	b.bInit = make([]float32, n)
	b.aCPU = make([]float32, n*n)
	b.bCPU = make([]float32, n)
	b.xCPU = make([]float32, n)
	b.aGPU = make([]float32, n*n)
	b.bGPU = make([]float32, n)
	b.xGPU = make([]float32, n)

	for i := 0; i < n; i++ {
		var rowSum float32
		for j := 0; j < n; j++ {
			v := float32(((i*11+j*7)%31)+1) * 0.02
			b.aInit[i*n+j] = v
			rowSum += float32(math.Abs(float64(v)))
		}
		b.aInit[i*n+i] += rowSum + 1.0
		b.bInit[i] = float32((i%17)+1) * 0.1
	}
	copy(b.aCPU, b.aInit)
	copy(b.bCPU, b.bInit)

	b.dA = b.alloc(uint64(n * n * 4))
	b.dB = b.alloc(uint64(n * 4))
}

func (b *Benchmark) exec() {
	n := b.Dim
	b.driver.MemCopyH2D(b.context, b.dA, b.aInit)
	b.driver.MemCopyH2D(b.context, b.dB, b.bInit)

	local1 := [3]uint16{64, 1, 1}
	local2 := [3]uint16{8, 8, 1}

	for t := 0; t < n; t++ {
		span1 := uint32(n - t)
		normArgs := NormalizeArgs{
			A:                   b.dA,
			B:                   b.dB,
			N:                   int32(n),
			T:                   int32(t),
			HiddenGlobalOffsetX: 0,
			HiddenGlobalOffsetY: 0,
			HiddenGlobalOffsetZ: 0,
			HiddenNone0:         0,
			HiddenNone1:         0,
			HiddenNone2:         0,
			HiddenMultiGridSync: 0,
		}
		b.driver.LaunchKernel(
			b.context, b.normKernel,
			[3]uint32{alignUp(span1, uint32(local1[0])), 1, 1},
			local1, &normArgs,
		)

		if t+1 < n {
			span2 := uint32(n - (t + 1))
			elimArgs := EliminateArgs{
				A:                   b.dA,
				B:                   b.dB,
				N:                   int32(n),
				T:                   int32(t),
				HiddenGlobalOffsetX: 0,
				HiddenGlobalOffsetY: 0,
				HiddenGlobalOffsetZ: 0,
				HiddenNone0:         0,
				HiddenNone1:         0,
				HiddenNone2:         0,
				HiddenMultiGridSync: 0,
			}
			b.driver.LaunchKernel(
				b.context, b.elimKernel,
				[3]uint32{
					alignUp(span2, uint32(local2[0])),
					alignUp(span2, uint32(local2[1])),
					1,
				},
				local2, &elimArgs,
			)
		}
	}

	b.driver.MemCopyD2H(b.context, b.aGPU, b.dA)
	b.driver.MemCopyD2H(b.context, b.bGPU, b.dB)
	b.backSubstituteGPU()
}

func (b *Benchmark) backSubstituteGPU() {
	n := b.Dim
	for i := n - 1; i >= 0; i-- {
		sum := b.bGPU[i]
		for j := i + 1; j < n; j++ {
			sum -= b.aGPU[i*n+j] * b.xGPU[j]
		}
		b.xGPU[i] = sum / b.aGPU[i*n+i]
	}
}

func (b *Benchmark) Verify() {
	n := b.Dim
	a := b.aCPU
	rhs := b.bCPU

	for t := 0; t < n; t++ {
		piv := a[t*n+t]
		for j := t; j < n; j++ {
			a[t*n+j] /= piv
		}
		rhs[t] /= piv

		for i := t + 1; i < n; i++ {
			f := a[i*n+t]
			for j := t + 1; j < n; j++ {
				a[i*n+j] -= f * a[t*n+j]
			}
			rhs[i] -= f * rhs[t]
			a[i*n+t] = 0
		}
	}

	for i := n - 1; i >= 0; i-- {
		sum := rhs[i]
		for j := i + 1; j < n; j++ {
			sum -= a[i*n+j] * b.xCPU[j]
		}
		b.xCPU[i] = sum / a[i*n+i]
	}

	for i := 0; i < n; i++ {
		if math.Abs(float64(b.xCPU[i]-b.xGPU[i])) > 5e-4 {
			log.Panicf("gaussian mismatch at %d: want %f got %f", i, b.xCPU[i], b.xGPU[i])
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
