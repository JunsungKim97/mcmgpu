// Package backprop implements a Rodinia-style backpropagation training step.
package backprop

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
	width    = 16
	height   = 16
	eta      = float32(0.3)
	momentum = float32(0.3)
)

//go:embed kernels.hsaco
var hsacoBytes []byte

type LayerForwardArgs struct {
	InputCuda           driver.GPUPtr
	OutputHiddenCuda    driver.GPUPtr
	InputHiddenCuda     driver.GPUPtr
	HiddenPartialSum    driver.GPUPtr
	In                  int32
	Hid                 int32
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
	HiddenNone0         int64
	HiddenNone1         int64
	HiddenNone2         int64
	HiddenMultiGridSync int64
}

type AdjustWeightArgs struct {
	Delta               driver.GPUPtr
	Hid                 int32
	PadAfterHid         int32
	Ly                  driver.GPUPtr
	In                  int32
	PadBeforeW          int32
	W                   driver.GPUPtr
	OldW                driver.GPUPtr
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

	kernelForward *insts.HsaCo
	kernelAdjust  *insts.HsaCo

	InputN     int
	HiddenN    int
	OutputN    int
	Iterations int

	inputUnits        []float32
	hiddenUnits       []float32
	outputUnits       []float32
	hiddenDelta       []float32
	outputDelta       []float32
	target            []float32
	inputWeights      []float32
	inputPrevWeights  []float32
	hiddenWeights     []float32
	hiddenPrevWeights []float32
	hiddenPartialSum  []float32

	gpuInputWeights []float32
	initInputUnits  []float32
	initTarget      []float32
	initInputW      []float32
	initInputPrevW  []float32
	initHiddenW     []float32
	initHiddenPrevW []float32

	dInputUnits       driver.GPUPtr
	dOutputHidden     driver.GPUPtr
	dInputWeights     driver.GPUPtr
	dHiddenPartialSum driver.GPUPtr
	dHiddenDelta      driver.GPUPtr
	dInputPrevWeights driver.GPUPtr

	useUM, useLASP bool
}

func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:     driver,
		context:    driver.Init(),
		InputN:     1024,
		HiddenN:    16,
		OutputN:    1,
		Iterations: 2,
	}
	b.loadProgram()
	return b
}

func (b *Benchmark) loadProgram() {
	b.kernelForward = kernels.LoadProgramFromMemory(hsacoBytes, "bpnn_layerforward_ocl")
	b.kernelAdjust = kernels.LoadProgramFromMemory(hsacoBytes, "bpnn_adjust_weights_ocl")
	if b.kernelForward == nil || b.kernelAdjust == nil {
		log.Panic("failed to load backprop kernels from kernels.hsaco")
	}
}

func (b *Benchmark) SelectGPU(gpus []int) {
	if len(gpus) > 1 {
		panic("backprop currently supports single-GPU only")
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
	if b.HiddenN != 16 {
		panic("HiddenN must be 16 for this Rodinia backprop port")
	}
	if b.InputN%width != 0 {
		panic("InputN must be a multiple of 16")
	}

	rand.Seed(1)
	in := b.InputN
	hid := b.HiddenN
	out := b.OutputN
	numBlocks := in / width

	b.inputUnits = make([]float32, in+1)
	b.hiddenUnits = make([]float32, hid+1)
	b.outputUnits = make([]float32, out+1)
	b.hiddenDelta = make([]float32, hid+1)
	b.outputDelta = make([]float32, out+1)
	b.target = make([]float32, out+1)
	b.inputWeights = make([]float32, (in+1)*(hid+1))
	b.inputPrevWeights = make([]float32, (in+1)*(hid+1))
	b.hiddenWeights = make([]float32, (hid+1)*(out+1))
	b.hiddenPrevWeights = make([]float32, (hid+1)*(out+1))
	b.hiddenPartialSum = make([]float32, numBlocks*hid)
	b.gpuInputWeights = make([]float32, (in+1)*(hid+1))

	for i := 0; i <= in; i++ {
		b.inputUnits[i] = rand.Float32()
	}
	for i := 0; i < len(b.inputWeights); i++ {
		b.inputWeights[i] = rand.Float32()
	}
	for i := 0; i < len(b.hiddenWeights); i++ {
		b.hiddenWeights[i] = rand.Float32()
	}
	for i := 0; i <= out; i++ {
		b.target[i] = 0.1
	}
	b.initInputUnits = append([]float32(nil), b.inputUnits...)
	b.initTarget = append([]float32(nil), b.target...)
	b.initInputW = append([]float32(nil), b.inputWeights...)
	b.initInputPrevW = append([]float32(nil), b.inputPrevWeights...)
	b.initHiddenW = append([]float32(nil), b.hiddenWeights...)
	b.initHiddenPrevW = append([]float32(nil), b.hiddenPrevWeights...)

	alloc := func(byteSize uint64) driver.GPUPtr {
		if b.useUM {
			return b.driver.AllocateUnifiedMemory(b.context, byteSize)
		}
		if b.useLASP {
			return b.driver.AllocateMemoryLASP(b.context, byteSize, "div4")
		}
		return b.driver.AllocateMemory(b.context, byteSize)
	}

	b.dInputUnits = alloc(uint64((in + 1) * 4))
	b.dOutputHidden = alloc(uint64((hid + 1) * 4))
	b.dInputWeights = alloc(uint64((in + 1) * (hid + 1) * 4))
	b.dHiddenPartialSum = alloc(uint64(numBlocks * hid * 4))
	b.dHiddenDelta = alloc(uint64((hid + 1) * 4))
	b.dInputPrevWeights = alloc(uint64((in + 1) * (hid + 1) * 4))
}

func (b *Benchmark) exec() {
	in := b.InputN
	hid := b.HiddenN
	numBlocks := in / width

	b.driver.MemCopyH2D(b.context, b.dInputUnits, b.inputUnits)

	forwardGlobal := [3]uint32{uint32(hid), uint32(numBlocks), 1}
	adjustGlobal := [3]uint32{uint32(hid), uint32(in), 1}
	forwardLocal := [3]uint16{16, 1, 1}
	adjustLocal := [3]uint16{16, 1, 1}

	for it := 0; it < b.Iterations; it++ {
		b.driver.MemCopyH2D(b.context, b.dInputWeights, b.inputWeights)
		b.driver.MemCopyH2D(b.context, b.dInputPrevWeights, b.inputPrevWeights)

		fargs := LayerForwardArgs{
			InputCuda:           b.dInputUnits,
			OutputHiddenCuda:    b.dOutputHidden,
			InputHiddenCuda:     b.dInputWeights,
			HiddenPartialSum:    b.dHiddenPartialSum,
			In:                  int32(in),
			Hid:                 int32(hid),
			HiddenGlobalOffsetX: 0,
			HiddenGlobalOffsetY: 0,
			HiddenGlobalOffsetZ: 0,
			HiddenNone0:         0,
			HiddenNone1:         0,
			HiddenNone2:         0,
			HiddenMultiGridSync: 0,
		}
		b.driver.LaunchKernel(b.context, b.kernelForward, forwardGlobal, forwardLocal, &fargs)
		b.driver.MemCopyD2H(b.context, b.hiddenPartialSum, b.dHiddenPartialSum)

		b.layerForwardAndErrorOnCPU()

		b.driver.MemCopyH2D(b.context, b.dHiddenDelta, b.hiddenDelta)
		aargs := AdjustWeightArgs{
			Delta:               b.dHiddenDelta,
			Hid:                 int32(hid),
			PadAfterHid:         0,
			Ly:                  b.dInputUnits,
			In:                  int32(in),
			PadBeforeW:          0,
			W:                   b.dInputWeights,
			OldW:                b.dInputPrevWeights,
			HiddenGlobalOffsetX: 0,
			HiddenGlobalOffsetY: 0,
			HiddenGlobalOffsetZ: 0,
			HiddenNone0:         0,
			HiddenNone1:         0,
			HiddenNone2:         0,
			HiddenMultiGridSync: 0,
		}
		b.driver.LaunchKernel(b.context, b.kernelAdjust, adjustGlobal, adjustLocal, &aargs)

		b.driver.MemCopyD2H(b.context, b.inputWeights, b.dInputWeights)
		b.driver.MemCopyD2H(b.context, b.inputPrevWeights, b.dInputPrevWeights)
	}

	copy(b.gpuInputWeights, b.inputWeights)
}

func (b *Benchmark) layerForwardAndErrorOnCPU() {
	in := b.InputN
	hid := b.HiddenN
	out := b.OutputN
	numBlocks := in / width

	b.hiddenUnits[0] = 1.0
	for j := 1; j <= hid; j++ {
		sum := float32(0)
		for blk := 0; blk < numBlocks; blk++ {
			sum += b.hiddenPartialSum[blk*hid+(j-1)]
		}
		sum += b.inputWeights[j] // bias row k=0
		b.hiddenUnits[j] = squash(sum)
	}

	b.hiddenUnits[0] = 1.0
	for j := 1; j <= out; j++ {
		sum := float32(0)
		for k := 0; k <= hid; k++ {
			sum += b.hiddenWeights[k*(out+1)+j] * b.hiddenUnits[k]
		}
		b.outputUnits[j] = squash(sum)
	}

	for j := 1; j <= out; j++ {
		o := b.outputUnits[j]
		t := b.target[j]
		b.outputDelta[j] = o * (1 - o) * (t - o)
	}

	for j := 1; j <= hid; j++ {
		h := b.hiddenUnits[j]
		sum := float32(0)
		for k := 1; k <= out; k++ {
			sum += b.outputDelta[k] * b.hiddenWeights[j*(out+1)+k]
		}
		b.hiddenDelta[j] = h * (1 - h) * sum
	}

	// Adjust hidden->output weights on CPU (same as Rodinia flow).
	for j := 1; j <= out; j++ {
		for k := 0; k <= hid; k++ {
			newDW := eta*b.outputDelta[j]*b.hiddenUnits[k] + momentum*b.hiddenPrevWeights[k*(out+1)+j]
			b.hiddenWeights[k*(out+1)+j] += newDW
			b.hiddenPrevWeights[k*(out+1)+j] = newDW
		}
	}
}

func (b *Benchmark) Verify() {
	cpu := b.cloneForCPU()
	cpu.runCPUReference()

	const eps = float64(2e-4)
	for i := range b.gpuInputWeights {
		if math.Abs(float64(cpu.inputWeights[i]-b.gpuInputWeights[i])) > eps {
			log.Panicf("backprop mismatch at %d: want %f got %f",
				i, cpu.inputWeights[i], b.gpuInputWeights[i])
		}
	}
	log.Printf("Passed!\n")
}

func (b *Benchmark) cloneForCPU() *Benchmark {
	c := *b
	c.inputUnits = append([]float32(nil), b.initInputUnits...)
	c.hiddenUnits = make([]float32, len(b.hiddenUnits))
	c.outputUnits = make([]float32, len(b.outputUnits))
	c.hiddenDelta = make([]float32, len(b.hiddenDelta))
	c.outputDelta = make([]float32, len(b.outputDelta))
	c.target = append([]float32(nil), b.initTarget...)
	c.inputWeights = append([]float32(nil), b.initInputW...)
	c.inputPrevWeights = append([]float32(nil), b.initInputPrevW...)
	c.hiddenWeights = append([]float32(nil), b.initHiddenW...)
	c.hiddenPrevWeights = append([]float32(nil), b.initHiddenPrevW...)
	c.hiddenPartialSum = make([]float32, len(b.hiddenPartialSum))
	return &c
}

func (b *Benchmark) runCPUReference() {
	in := b.InputN
	hid := b.HiddenN
	numBlocks := in / width

	for it := 0; it < b.Iterations; it++ {
		// Recompute equivalent partial sums from input_units and input_weights.
		for blk := 0; blk < numBlocks; blk++ {
			for j := 0; j < hid; j++ {
				acc := float32(0)
				for k := 0; k < width; k++ {
					inIdx := blk*height + k + 1
					wIdx := inIdx*(hid+1) + (j + 1)
					acc += b.inputWeights[wIdx] * b.inputUnits[inIdx]
				}
				b.hiddenPartialSum[blk*hid+j] = acc
			}
		}

		b.layerForwardAndErrorOnCPU()

		// Equivalent of bpnn_adjust_weights_ocl.
		for blk := 0; blk < numBlocks; blk++ {
			for ty := 0; ty < height; ty++ {
				indexY := blk*height + ty + 1
				for tx := 0; tx < width; tx++ {
					indexX := tx + 1
					wIdx := indexY*(hid+1) + indexX
					newDW := eta*b.hiddenDelta[indexX]*b.inputUnits[indexY] + momentum*b.inputPrevWeights[wIdx]
					b.inputWeights[wIdx] += newDW
					b.inputPrevWeights[wIdx] = newDW
				}
			}
		}
		for indexX := 1; indexX <= width; indexX++ {
			wIdx := indexX
			newDW := eta*b.hiddenDelta[indexX] + momentum*b.inputPrevWeights[wIdx]
			b.inputWeights[wIdx] += newDW
			b.inputPrevWeights[wIdx] = newDW
		}
	}
}

func squash(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}
