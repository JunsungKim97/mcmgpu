// Package diffusion implements a single-GPU diffusion benchmark ported from Tartan's diffusion suite.
package diffusion

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

// KernelArgs defines kernel arguments for diffusion_step.
type KernelArgs struct {
	In                  driver.GPUPtr
	Out                 driver.GPUPtr
	NX                  int32
	NY                  int32
	NZ                  int32
	Alpha               float32
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
	HiddenNone0         int64
	HiddenNone1         int64
	HiddenNone2         int64
	HiddenMultiGridSync int64
}

// Benchmark defines the diffusion benchmark.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	gpus    []int
	queues  []*driver.CommandQueue

	kernel *insts.HsaCo

	NX, NY, NZ int
	Iterations int
	Alpha      float32

	u       []float32
	uGPU    []float32
	temp    []float32
	dU0     driver.GPUPtr
	dU1     driver.GPUPtr
	useUM   bool
	useLASP bool
}

// NewBenchmark creates a benchmark instance.
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:     driver,
		context:    driver.Init(),
		NX:         64,
		NY:         64,
		NZ:         64,
		Iterations: 10,
		Alpha:      0.1,
	}
	b.loadProgram()
	return b
}

func (b *Benchmark) loadProgram() {
	b.kernel = kernels.LoadProgramFromMemory(hsacoBytes, "diffusion_step")
	if b.kernel == nil {
		log.Panic("failed to load diffusion_step from kernels.hsaco")
	}
}

// SelectGPU selects the GPU IDs for execution.
func (b *Benchmark) SelectGPU(gpus []int) {
	if len(gpus) > 1 {
		panic("diffusion benchmark currently supports single-GPU only")
	}
	b.gpus = gpus
}

// SetUnifiedMemory enables unified memory allocation.
func (b *Benchmark) SetUnifiedMemory() {
	b.useUM = true
}

// SetLASPMemoryAlloc enables LASP allocation.
func (b *Benchmark) SetLASPMemoryAlloc() {
	b.useLASP = true
}

// Run runs the benchmark.
func (b *Benchmark) Run() {
	for _, gpu := range b.gpus {
		b.driver.SelectGPU(b.context, gpu)
		b.queues = append(b.queues, b.driver.CreateCommandQueue(b.context))
	}
	b.initMem()
	b.exec()
}

func (b *Benchmark) initMem() {
	if b.NX < 3 || b.NY < 3 || b.NZ < 3 {
		panic("NX, NY, NZ must be >= 3")
	}

	total := b.NX * b.NY * b.NZ
	b.u = make([]float32, total)
	b.temp = make([]float32, total)
	b.uGPU = make([]float32, total)

	rand.Seed(1)
	for z := 0; z < b.NZ; z++ {
		for y := 0; y < b.NY; y++ {
			for x := 0; x < b.NX; x++ {
				idx := b.index(x, y, z)
				// Smooth-ish random initial condition in [0, 1].
				b.u[idx] = rand.Float32()
			}
		}
	}

	byteSize := uint64(total * 4)
	if b.useUM {
		b.dU0 = b.driver.AllocateUnifiedMemory(b.context, byteSize)
		b.dU1 = b.driver.AllocateUnifiedMemory(b.context, byteSize)
	} else if b.useLASP {
		b.dU0 = b.driver.AllocateMemoryLASP(b.context, byteSize, "div4")
		b.dU1 = b.driver.AllocateMemoryLASP(b.context, byteSize, "div4")
	} else {
		b.dU0 = b.driver.AllocateMemory(b.context, byteSize)
		b.dU1 = b.driver.AllocateMemory(b.context, byteSize)
	}
}

func (b *Benchmark) exec() {
	b.driver.MemCopyH2D(b.context, b.dU0, b.u)
	b.driver.MemCopyH2D(b.context, b.dU1, b.u)

	localSize := [3]uint16{8, 8, 1}
	globalX := uint32(((b.NX-1)/int(localSize[0]) + 1) * int(localSize[0]))
	flatYZ := b.NY * b.NZ
	globalY := uint32(((flatYZ-1)/int(localSize[1]) + 1) * int(localSize[1]))
	globalZ := uint32(1)
	globalSize := [3]uint32{globalX, globalY, globalZ}

	src := b.dU0
	dst := b.dU1
	for i := 0; i < b.Iterations; i++ {
		args := KernelArgs{
			In:                  src,
			Out:                 dst,
			NX:                  int32(b.NX),
			NY:                  int32(b.NY),
			NZ:                  int32(b.NZ),
			Alpha:               b.Alpha,
			HiddenGlobalOffsetX: 0,
			HiddenGlobalOffsetY: 0,
			HiddenGlobalOffsetZ: 0,
			HiddenNone0:         0,
			HiddenNone1:         0,
			HiddenNone2:         0,
			HiddenMultiGridSync: 0,
		}
		b.driver.LaunchKernel(b.context, b.kernel, globalSize, localSize, &args)
		src, dst = dst, src
	}

	b.driver.MemCopyD2H(b.context, b.uGPU, src)
}

// Verify checks GPU output against CPU reference.
func (b *Benchmark) Verify() {
	b.cpuReference()

	const eps = 1e-4
	for i := range b.u {
		if math.Abs(float64(b.temp[i]-b.uGPU[i])) > eps {
			log.Panicf("diffusion mismatch at %d: expected %f, got %f",
				i, b.temp[i], b.uGPU[i])
		}
	}
	log.Printf("Passed!\n")
}

func (b *Benchmark) cpuReference() {
	copy(b.temp, b.u)
	next := make([]float32, len(b.temp))
	copy(next, b.temp)

	for iter := 0; iter < b.Iterations; iter++ {
		for z := 0; z < b.NZ; z++ {
			for y := 0; y < b.NY; y++ {
				for x := 0; x < b.NX; x++ {
					idx := b.index(x, y, z)
					if x == 0 || y == 0 || z == 0 || x == b.NX-1 || y == b.NY-1 || z == b.NZ-1 {
						next[idx] = b.temp[idx]
						continue
					}

					center := b.temp[idx]
					lap := b.temp[b.index(x-1, y, z)] +
						b.temp[b.index(x+1, y, z)] +
						b.temp[b.index(x, y-1, z)] +
						b.temp[b.index(x, y+1, z)] +
						b.temp[b.index(x, y, z-1)] +
						b.temp[b.index(x, y, z+1)] -
						6*center
					next[idx] = center + b.Alpha*lap
				}
			}
		}
		b.temp, next = next, b.temp
	}
}

func (b *Benchmark) index(x, y, z int) int {
	return z*b.NX*b.NY + y*b.NX + x
}
