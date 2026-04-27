// Package lulesh ports a nodal-update slice of Tartan LULESH.
//
// Upstream is CUDA + MPI and much larger; this benchmark focuses on one
// single-GPU kernel equivalent to:
// - CalcAccelerationForNodes_kernel
// - CalcPositionAndVelocityForNodes_kernel
//
// Reference: https://github.com/uuudown/Tartan/tree/master/scale-out/scale-out/lulesh
package lulesh

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

// KernelArgs must match the clang-ocl kernarg ABI.
type KernelArgs struct {
	Fx                  driver.GPUPtr
	Fy                  driver.GPUPtr
	Fz                  driver.GPUPtr
	NodalMass           driver.GPUPtr
	X                   driver.GPUPtr
	Y                   driver.GPUPtr
	Z                   driver.GPUPtr
	Xd                  driver.GPUPtr
	Yd                  driver.GPUPtr
	Zd                  driver.GPUPtr
	NumNode             int32
	Deltatime           float32
	UCut                float32
	PadBeforeHidden     int32
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
	HiddenNone0         int64
	HiddenNone1         int64
	HiddenNone2         int64
	HiddenMultiGridSync int64
}

// Benchmark runs repeated nodal updates.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	gpus    []int
	queues  []*driver.CommandQueue
	kernel  *insts.HsaCo

	NumNode    int
	Iterations int
	Deltatime  float32
	UCut       float32

	hFx, hFy, hFz       []float32
	hMass               []float32
	hX, hY, hZ          []float32
	hXd, hYd, hZd       []float32
	gpuX, gpuY, gpuZ    []float32
	gpuXd, gpuYd, gpuZd []float32

	dFx, dFy, dFz  driver.GPUPtr
	dMass          driver.GPUPtr
	dX, dY, dZ     driver.GPUPtr
	dXd, dYd, dZd  driver.GPUPtr
	useUM, useLASP bool
}

// NewBenchmark creates LULESH nodal benchmark defaults.
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:     driver,
		context:    driver.Init(),
		NumNode:    1 << 14,
		Iterations: 4,
		Deltatime:  1.0e-7,
		UCut:       1.0e-7,
	}
	b.loadProgram()
	return b
}

func (b *Benchmark) loadProgram() {
	b.kernel = kernels.LoadProgramFromMemory(hsacoBytes, "lulesh_lagrange_nodal")
	if b.kernel == nil {
		log.Panic("failed to load lulesh_lagrange_nodal from kernels.hsaco")
	}
}

// SelectGPU selects GPUs (single-GPU only).
func (b *Benchmark) SelectGPU(gpus []int) {
	if len(gpus) > 1 {
		panic("lulesh benchmark currently supports single-GPU only")
	}
	b.gpus = gpus
}

// SetUnifiedMemory enables unified memory.
func (b *Benchmark) SetUnifiedMemory() { b.useUM = true }

// SetLASPMemoryAlloc enables LASP allocation.
func (b *Benchmark) SetLASPMemoryAlloc() { b.useLASP = true }

// Run executes the benchmark.
func (b *Benchmark) Run() {
	for _, gpu := range b.gpus {
		b.driver.SelectGPU(b.context, gpu)
		b.queues = append(b.queues, b.driver.CreateCommandQueue(b.context))
	}
	b.initMem()
	b.exec()
}

func (b *Benchmark) initMem() {
	if b.NumNode < 1 {
		panic("NumNode must be >= 1")
	}

	n := b.NumNode
	b.hFx = make([]float32, n)
	b.hFy = make([]float32, n)
	b.hFz = make([]float32, n)
	b.hMass = make([]float32, n)
	b.hX = make([]float32, n)
	b.hY = make([]float32, n)
	b.hZ = make([]float32, n)
	b.hXd = make([]float32, n)
	b.hYd = make([]float32, n)
	b.hZd = make([]float32, n)
	b.gpuX = make([]float32, n)
	b.gpuY = make([]float32, n)
	b.gpuZ = make([]float32, n)
	b.gpuXd = make([]float32, n)
	b.gpuYd = make([]float32, n)
	b.gpuZd = make([]float32, n)

	rand.Seed(1)
	for i := 0; i < n; i++ {
		b.hFx[i] = (rand.Float32() - 0.5) * 4.0
		b.hFy[i] = (rand.Float32() - 0.5) * 4.0
		b.hFz[i] = (rand.Float32() - 0.5) * 4.0
		b.hMass[i] = 0.5 + rand.Float32()*2.0
		b.hX[i] = rand.Float32() * 10.0
		b.hY[i] = rand.Float32() * 10.0
		b.hZ[i] = rand.Float32() * 10.0
		b.hXd[i] = (rand.Float32() - 0.5) * 0.1
		b.hYd[i] = (rand.Float32() - 0.5) * 0.1
		b.hZd[i] = (rand.Float32() - 0.5) * 0.1
	}

	byteSize := uint64(n * 4)
	alloc := func() driver.GPUPtr {
		if b.useUM {
			return b.driver.AllocateUnifiedMemory(b.context, byteSize)
		}
		if b.useLASP {
			return b.driver.AllocateMemoryLASP(b.context, byteSize, "div4")
		}
		return b.driver.AllocateMemory(b.context, byteSize)
	}

	b.dFx = alloc()
	b.dFy = alloc()
	b.dFz = alloc()
	b.dMass = alloc()
	b.dX = alloc()
	b.dY = alloc()
	b.dZ = alloc()
	b.dXd = alloc()
	b.dYd = alloc()
	b.dZd = alloc()
}

func (b *Benchmark) exec() {
	b.driver.MemCopyH2D(b.context, b.dFx, b.hFx)
	b.driver.MemCopyH2D(b.context, b.dFy, b.hFy)
	b.driver.MemCopyH2D(b.context, b.dFz, b.hFz)
	b.driver.MemCopyH2D(b.context, b.dMass, b.hMass)
	b.driver.MemCopyH2D(b.context, b.dX, b.hX)
	b.driver.MemCopyH2D(b.context, b.dY, b.hY)
	b.driver.MemCopyH2D(b.context, b.dZ, b.hZ)
	b.driver.MemCopyH2D(b.context, b.dXd, b.hXd)
	b.driver.MemCopyH2D(b.context, b.dYd, b.hYd)
	b.driver.MemCopyH2D(b.context, b.dZd, b.hZd)

	localSize := [3]uint16{256, 1, 1}
	globalX := uint32(((b.NumNode-1)/int(localSize[0]) + 1) * int(localSize[0]))
	globalSize := [3]uint32{globalX, 1, 1}

	for iter := 0; iter < b.Iterations; iter++ {
		args := KernelArgs{
			Fx:                  b.dFx,
			Fy:                  b.dFy,
			Fz:                  b.dFz,
			NodalMass:           b.dMass,
			X:                   b.dX,
			Y:                   b.dY,
			Z:                   b.dZ,
			Xd:                  b.dXd,
			Yd:                  b.dYd,
			Zd:                  b.dZd,
			NumNode:             int32(b.NumNode),
			Deltatime:           b.Deltatime,
			UCut:                b.UCut,
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
	}

	b.driver.MemCopyD2H(b.context, b.gpuX, b.dX)
	b.driver.MemCopyD2H(b.context, b.gpuY, b.dY)
	b.driver.MemCopyD2H(b.context, b.gpuZ, b.dZ)
	b.driver.MemCopyD2H(b.context, b.gpuXd, b.dXd)
	b.driver.MemCopyD2H(b.context, b.gpuYd, b.dYd)
	b.driver.MemCopyD2H(b.context, b.gpuZd, b.dZd)
}

// Verify compares GPU output with CPU reference.
func (b *Benchmark) Verify() {
	x := append([]float32(nil), b.hX...)
	y := append([]float32(nil), b.hY...)
	z := append([]float32(nil), b.hZ...)
	xd := append([]float32(nil), b.hXd...)
	yd := append([]float32(nil), b.hYd...)
	zd := append([]float32(nil), b.hZd...)

	for iter := 0; iter < b.Iterations; iter++ {
		for i := 0; i < b.NumNode; i++ {
			invm := 1.0 / b.hMass[i]
			xdd := b.hFx[i] * invm
			ydd := b.hFy[i] * invm
			zdd := b.hFz[i] * invm

			xdtmp := xd[i] + float32(xdd)*b.Deltatime
			ydtmp := yd[i] + float32(ydd)*b.Deltatime
			zdtmp := zd[i] + float32(zdd)*b.Deltatime

			if math.Abs(float64(xdtmp)) < float64(b.UCut) {
				xdtmp = 0
			}
			if math.Abs(float64(ydtmp)) < float64(b.UCut) {
				ydtmp = 0
			}
			if math.Abs(float64(zdtmp)) < float64(b.UCut) {
				zdtmp = 0
			}

			x[i] += xdtmp * b.Deltatime
			y[i] += ydtmp * b.Deltatime
			z[i] += zdtmp * b.Deltatime
			xd[i] = xdtmp
			yd[i] = ydtmp
			zd[i] = zdtmp
		}
	}

	const eps = float64(1e-6)
	check := func(name string, want, got []float32) {
		for i := range want {
			if math.Abs(float64(want[i]-got[i])) > eps {
				log.Panicf("lulesh %s mismatch at %d: want %f got %f", name, i, want[i], got[i])
			}
		}
	}
	check("x", x, b.gpuX)
	check("y", y, b.gpuY)
	check("z", z, b.gpuZ)
	check("xd", xd, b.gpuXd)
	check("yd", yd, b.gpuYd)
	check("zd", zd, b.gpuZd)
	log.Printf("Passed!\n")
}
