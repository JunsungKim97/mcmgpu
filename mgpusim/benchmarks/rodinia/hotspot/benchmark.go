// Package hotspot implements the Rodinia Hotspot benchmark.
package hotspot

import (
	_ "embed"
	"log"
	"math"
	"math/rand"

	"gitlab.com/akita/mgpusim/driver"
	"gitlab.com/akita/mgpusim/insts"
	"gitlab.com/akita/mgpusim/kernels"
)

const (
	maxPD      = 3.0e6
	precision  = 0.001
	specHeatSI = 1.75e6
	kSI        = 100
	factorChip = 0.5
	tChip      = 0.0005
	chipHeight = 0.016
	chipWidth  = 0.016
	ambient    = 80.0
)

//go:embed kernels.hsaco
var hsacoBytes []byte

// KernelArgs defines kernel arguments.
// Must match HSACO KernargSegmentByteSize (120 for clang-ocl -mcpu=gfx803).
// Driver uses encoding/binary.Write, which packs fields with no implicit padding; the OpenCL
// ABI still aligns the hidden args to 8 bytes after the last scalar, so we need an explicit
// 4-byte field before HiddenGlobalOffsetX (see akita_docs mgpusim/03_prepare_new_benchmarks.md).
// Remaining slots match clang-ocl HiddenNone / MultiGridSync metadata in .note.
type KernelArgs struct {
	Iteration int32
	Pad0      int32
	Power     driver.GPUPtr
	TempSrc   driver.GPUPtr
	TempDst   driver.GPUPtr
	GridCols  int32
	GridRows  int32
	Cap       float32
	Rx        float32
	Ry        float32
	Rz        float32
	Step      float32
	PadBeforeHidden     int32
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
	HiddenNone0         int64
	HiddenNone1         int64
	HiddenNone2         int64
	HiddenMultiGridSync int64
}

// Benchmark defines a benchmark.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	gpus    []int
	queues  []*driver.CommandQueue

	kernel *insts.HsaCo

	GridRows      int
	GridCols      int
	Iterations    int
	PyramidHeight int

	power   []float32
	tempIn  []float32
	tempOut []float32
	tempCPU []float32

	dPower   driver.GPUPtr
	dTempA   driver.GPUPtr
	dTempB   driver.GPUPtr
	finalGPU []float32

	useUnifiedMemory      bool
	useLASPMemoryAlloc    bool
	useLASPHSLMemoryAlloc bool
}

// NewBenchmark creates a new benchmark.
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:        driver,
		context:       driver.Init(),
		GridRows:      256,
		GridCols:      256,
		Iterations:    60,
		PyramidHeight: 1,
	}
	b.loadProgram()
	return b
}

func (b *Benchmark) loadProgram() {
	b.kernel = kernels.LoadProgramFromMemory(hsacoBytes, "hotspot")
	if b.kernel == nil {
		log.Panic("failed to load hotspot kernel binary; regenerate kernels.hsaco")
	}
}

// SelectGPU configures which GPU to use.
func (b *Benchmark) SelectGPU(gpus []int) {
	if len(gpus) > 1 {
		panic("hotspot does not support multi-GPU mode")
	}
	b.gpus = gpus
}

// SetUnifiedMemory enables unified memory.
func (b *Benchmark) SetUnifiedMemory() {
	b.useUnifiedMemory = true
}

// SetLASPMemoryAlloc enables LASP allocation.
func (b *Benchmark) SetLASPMemoryAlloc() {
	b.useLASPMemoryAlloc = true
}

// SetLASPHSLMemoryAlloc enables LASP HSL allocation.
func (b *Benchmark) SetLASPHSLMemoryAlloc() {
	b.useLASPHSLMemoryAlloc = true
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
	size := b.GridRows * b.GridCols
	rand.Seed(1)

	b.power = make([]float32, size)
	b.tempIn = make([]float32, size)
	b.tempOut = make([]float32, size)
	b.tempCPU = make([]float32, size)
	b.finalGPU = make([]float32, size)

	for i := 0; i < size; i++ {
		b.power[i] = rand.Float32()*10.0 + 1.0
		b.tempIn[i] = rand.Float32()*15.0 + 65.0
		b.tempCPU[i] = b.tempIn[i]
	}

	byteSize := uint64(size * 4)
	b.dPower = b.allocate(byteSize)
	b.dTempA = b.allocate(byteSize)
	b.dTempB = b.allocate(byteSize)
}

func (b *Benchmark) allocate(byteSize uint64) driver.GPUPtr {
	if b.useUnifiedMemory {
		return b.driver.AllocateUnifiedMemory(b.context, byteSize)
	}
	if b.useLASPMemoryAlloc || b.useLASPHSLMemoryAlloc {
		return b.driver.AllocateMemoryLASP(b.context, byteSize, "div4")
	}
	return b.driver.AllocateMemory(b.context, byteSize)
}

func (b *Benchmark) physicalModel() (cap, rx, ry, rz, step float32) {
	gridHeight := float32(chipHeight) / float32(b.GridRows)
	gridWidth := float32(chipWidth) / float32(b.GridCols)
	cap = float32(factorChip*specHeatSI*tChip) * gridWidth * gridHeight
	rx = gridWidth / (2.0 * float32(kSI) * float32(tChip) * gridHeight)
	ry = gridHeight / (2.0 * float32(kSI) * float32(tChip) * gridWidth)
	rz = float32(tChip) / (float32(kSI) * gridHeight * gridWidth)
	maxSlope := float32(maxPD / (factorChip * tChip * specHeatSI))
	step = float32(precision) / maxSlope
	return
}

func (b *Benchmark) exec() {
	b.driver.MemCopyH2D(b.context, b.dPower, b.power)
	b.driver.MemCopyH2D(b.context, b.dTempA, b.tempIn)

	localSize := [3]uint16{16, 16, 1}
	globalX := uint32(((b.GridCols - 1) / int(localSize[0]) + 1) * int(localSize[0]))
	globalY := uint32(((b.GridRows - 1) / int(localSize[1]) + 1) * int(localSize[1]))
	globalSize := [3]uint32{globalX, globalY, 1}

	cap, rx, ry, rz, step := b.physicalModel()
	remaining := b.Iterations
	src := b.dTempA
	dst := b.dTempB
	for remaining > 0 {
		iter := b.PyramidHeight
		if iter > remaining {
			iter = remaining
		}

		args := KernelArgs{
			Iteration: int32(iter),
			Pad0:      0,
			Power:     b.dPower,
			TempSrc:   src,
			TempDst:   dst,
			GridCols:  int32(b.GridCols),
			GridRows:  int32(b.GridRows),
			Cap:       cap,
			Rx:        rx,
			Ry:        ry,
			Rz:        rz,
			Step:      step,
			PadBeforeHidden:     0,
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
		remaining -= iter
	}

	b.driver.MemCopyD2H(b.context, b.finalGPU, src)
}

// Verify validates GPU result with CPU result.
func (b *Benchmark) Verify() {
	b.runCPU()

	const epsilon = 0.05
	for i := range b.finalGPU {
		if math.Abs(float64(b.finalGPU[i]-b.tempCPU[i])) > epsilon {
			log.Panicf("hotspot mismatch at %d, expected %f, got %f",
				i, b.tempCPU[i], b.finalGPU[i])
		}
	}
	log.Printf("Passed!\n")
}

func (b *Benchmark) runCPU() {
	cap, rx, ry, rz, step := b.physicalModel()
	stepDivCap := step / cap
	curr := make([]float32, len(b.tempCPU))
	next := make([]float32, len(b.tempCPU))
	copy(curr, b.tempCPU)

	for iter := 0; iter < b.Iterations; iter++ {
		for r := 0; r < b.GridRows; r++ {
			for c := 0; c < b.GridCols; c++ {
				idx := r*b.GridCols + c
				w := idx
				e := idx
				n := idx
				s := idx
				if c > 0 {
					w = idx - 1
				}
				if c < b.GridCols-1 {
					e = idx + 1
				}
				if r > 0 {
					n = idx - b.GridCols
				}
				if r < b.GridRows-1 {
					s = idx + b.GridCols
				}

				temp := curr[idx]
				delta := b.power[idx] +
					(curr[w]+curr[e]-2*temp)/rx +
					(curr[n]+curr[s]-2*temp)/ry +
					(float32(ambient)-temp)/rz
				next[idx] = temp + stepDivCap*delta
			}
		}
		curr, next = next, curr
	}

	copy(b.tempCPU, curr)
}
