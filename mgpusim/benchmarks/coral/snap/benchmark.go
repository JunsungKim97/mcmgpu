package snap

import (
	_ "embed"
	"log"
	"math"
	"math/rand"

	"gitlab.com/akita/mgpusim/driver"
	"gitlab.com/akita/mgpusim/insts"
	"gitlab.com/akita/mgpusim/kernels"
)

//go:embed kernels.hsaco
var hsacoBytes []byte

type KernelArgs struct {
	PsiIn               driver.GPUPtr
	Source              driver.GPUPtr
	PsiOut              driver.GPUPtr
	NX                  int32
	NY                  int32
	NZ                  int32
	Ax                  float32
	Ay                  float32
	Az                  float32
	SigmaT              float32
	PadBeforeHidden     int32
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
	queues  []*driver.CommandQueue
	kernel  *insts.HsaCo

	NX, NY, NZ int
	Iterations int
	Ax, Ay, Az float32
	SigmaT     float32

	psi0, psi1, src []float32
	gpuOut          []float32
	dPsi0, dPsi1    driver.GPUPtr
	dSrc            driver.GPUPtr
	useUM, useLASP  bool
}

func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:     driver,
		context:    driver.Init(),
		NX:         64,
		NY:         32,
		NZ:         32,
		Iterations: 4,
		Ax:         0.9,
		Ay:         0.8,
		Az:         0.7,
		SigmaT:     1.2,
	}
	b.loadProgram()
	return b
}
func (b *Benchmark) loadProgram() {
	b.kernel = kernels.LoadProgramFromMemory(hsacoBytes, "snap_sweep_step")
	if b.kernel == nil {
		log.Panic("failed to load snap_sweep_step")
	}
}
func (b *Benchmark) SelectGPU(gpus []int) {
	if len(gpus) > 1 {
		panic("snap benchmark single GPU only")
	}
	b.gpus = gpus
}
func (b *Benchmark) SetUnifiedMemory()   { b.useUM = true }
func (b *Benchmark) SetLASPMemoryAlloc() { b.useLASP = true }

func (b *Benchmark) Run() {
	for _, gpu := range b.gpus {
		b.driver.SelectGPU(b.context, gpu)
		b.queues = append(b.queues, b.driver.CreateCommandQueue(b.context))
	}
	b.initMem()
	b.exec()
}
func (b *Benchmark) initMem() {
	n := b.NX * b.NY * b.NZ
	b.psi0 = make([]float32, n)
	b.psi1 = make([]float32, n)
	b.src = make([]float32, n)
	b.gpuOut = make([]float32, n)
	rand.Seed(1)
	for i := 0; i < n; i++ {
		b.psi0[i] = rand.Float32()
		b.src[i] = 0.5 + 0.5*rand.Float32()
	}
	alloc := func(sz uint64) driver.GPUPtr {
		if b.useUM {
			return b.driver.AllocateUnifiedMemory(b.context, sz)
		}
		if b.useLASP {
			return b.driver.AllocateMemoryLASP(b.context, sz, "div4")
		}
		return b.driver.AllocateMemory(b.context, sz)
	}
	bytes := uint64(n * 4)
	b.dPsi0 = alloc(bytes)
	b.dPsi1 = alloc(bytes)
	b.dSrc = alloc(bytes)
}
func (b *Benchmark) exec() {
	b.driver.MemCopyH2D(b.context, b.dPsi0, b.psi0)
	b.driver.MemCopyH2D(b.context, b.dPsi1, b.psi1)
	b.driver.MemCopyH2D(b.context, b.dSrc, b.src)
	local := [3]uint16{8, 8, 1}
	gx := alignUp(uint32(b.NX), uint32(local[0]))
	gy := alignUp(uint32(b.NY*b.NZ), uint32(local[1]))
	global := [3]uint32{gx, gy, 1}
	src := b.dPsi0
	dst := b.dPsi1
	for i := 0; i < b.Iterations; i++ {
		args := KernelArgs{
			PsiIn:               src,
			Source:              b.dSrc,
			PsiOut:              dst,
			NX:                  int32(b.NX),
			NY:                  int32(b.NY),
			NZ:                  int32(b.NZ),
			Ax:                  b.Ax,
			Ay:                  b.Ay,
			Az:                  b.Az,
			SigmaT:              b.SigmaT,
			PadBeforeHidden:     0,
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
	ref0 := append([]float32(nil), b.psi0...)
	ref1 := make([]float32, len(ref0))
	src := ref0
	dst := ref1
	for it := 0; it < b.Iterations; it++ {
		for z := 0; z < b.NZ; z++ {
			for y := 0; y < b.NY; y++ {
				for x := 0; x < b.NX; x++ {
					i := z*b.NX*b.NY + y*b.NX + x
					px, py, pz := float32(0), float32(0), float32(0)
					if x > 0 {
						px = src[i-1]
					}
					if y > 0 {
						py = src[i-b.NX]
					}
					if z > 0 {
						pz = src[i-b.NX*b.NY]
					}
					dst[i] = (b.src[i] + b.Ax*px + b.Ay*py + b.Az*pz) / (b.SigmaT + b.Ax + b.Ay + b.Az)
				}
			}
		}
		src, dst = dst, src
	}
	for i := range src {
		if math.Abs(float64(src[i]-b.gpuOut[i])) > 2e-5 {
			log.Panicf("snap mismatch at %d: want %f got %f", i, src[i], b.gpuOut[i])
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
