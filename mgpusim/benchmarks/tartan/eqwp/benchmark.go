// Package eqwp implements a single-GPU stress-update kernel from the Tartan
// b2reqwp earthquake wave propagation (EQWP) model (eqwp_fd4_stress in eqwp.cu).
// MPI/B2R halo exchange is omitted; periodic boundaries are used so the stencil
// matches everywhere on the grid.
//
// Reference: https://github.com/uuudown/Tartan/tree/master/scale-out/scale-out/b2reqwp
package eqwp

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

// Physical / numerical constants (match native/kernels.cl and Tartan eqwp.cu).
const (
	dt    = float32(0.5)
	dx    = float32(10.0)
	dy    = float32(10.0)
	dz    = float32(10.0)
	lame1 = float32(32.00)
	lame2 = float32(174.87)
	coefA = float32(-1.0 / 12.0)
	coefB = float32(2.0 / 3.0)
)

// KernelArgs layout must match HSACO (clang-ocl hidden tail).
type KernelArgs struct {
	Vx                  driver.GPUPtr
	Vy                  driver.GPUPtr
	Vz                  driver.GPUPtr
	SigmaXX             driver.GPUPtr
	SigmaXY             driver.GPUPtr
	SigmaXZ             driver.GPUPtr
	SigmaYY             driver.GPUPtr
	SigmaYZ             driver.GPUPtr
	SigmaZZ             driver.GPUPtr
	NX                  int32
	NY                  int32
	NZ                  int32
	PadBeforeHidden     int32
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
	HiddenNone0         int64
	HiddenNone1         int64
	HiddenNone2         int64
	HiddenMultiGridSync int64
}

// Benchmark runs repeated stress updates on a 3D periodic grid.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	gpus    []int
	queues  []*driver.CommandQueue
	kernel  *insts.HsaCo

	NX, NY, NZ int
	Iterations int

	hVx, hVy, hVz                                  []float32
	hSxx, hSxy, hSxz, hSyy, hSyz, hSzz             []float32
	gpuSxx, gpuSxy, gpuSxz, gpuSyy, gpuSyz, gpuSzz []float32

	dVx, dVy, dVz                      driver.GPUPtr
	dSxx, dSxy, dSxz, dSyy, dSyz, dSzz driver.GPUPtr
	useUM, useLASP                     bool
}

// NewBenchmark creates EQWP benchmark with modest defaults.
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:     driver,
		context:    driver.Init(),
		NX:         24,
		NY:         24,
		NZ:         24,
		Iterations: 3,
	}
	b.loadProgram()
	return b
}

func (b *Benchmark) loadProgram() {
	b.kernel = kernels.LoadProgramFromMemory(hsacoBytes, "eqwp_stress_fd4")
	if b.kernel == nil {
		log.Panic("failed to load eqwp_stress_fd4 from kernels.hsaco")
	}
}

// SelectGPU selects GPUs (single-GPU only).
func (b *Benchmark) SelectGPU(gpus []int) {
	if len(gpus) > 1 {
		panic("eqwp benchmark currently supports single-GPU only")
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
	if b.NX < 3 || b.NY < 3 || b.NZ < 3 {
		panic("NX, NY, NZ must be >= 3")
	}
	n := b.NX * b.NY * b.NZ
	b.hVx = make([]float32, n)
	b.hVy = make([]float32, n)
	b.hVz = make([]float32, n)
	b.hSxx = make([]float32, n)
	b.hSxy = make([]float32, n)
	b.hSxz = make([]float32, n)
	b.hSyy = make([]float32, n)
	b.hSyz = make([]float32, n)
	b.hSzz = make([]float32, n)
	b.gpuSxx = make([]float32, n)
	b.gpuSxy = make([]float32, n)
	b.gpuSxz = make([]float32, n)
	b.gpuSyy = make([]float32, n)
	b.gpuSyz = make([]float32, n)
	b.gpuSzz = make([]float32, n)

	rand.Seed(1)
	for z := 0; z < b.NZ; z++ {
		for y := 0; y < b.NY; y++ {
			for x := 0; x < b.NX; x++ {
				i := b.index(x, y, z)
				b.hVx[i] = float32(math.Sin(0.11*float64(x))) * float32(math.Cos(0.07*float64(y)))
				b.hVy[i] = float32(math.Cos(0.09*float64(z))) * float32(math.Sin(0.05*float64(x)))
				b.hVz[i] = float32(math.Sin(0.08*float64(y))) * float32(math.Cos(0.06*float64(z)))
			}
		}
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
	b.dVx = alloc()
	b.dVy = alloc()
	b.dVz = alloc()
	b.dSxx = alloc()
	b.dSxy = alloc()
	b.dSxz = alloc()
	b.dSyy = alloc()
	b.dSyz = alloc()
	b.dSzz = alloc()
}

func (b *Benchmark) exec() {
	b.driver.MemCopyH2D(b.context, b.dVx, b.hVx)
	b.driver.MemCopyH2D(b.context, b.dVy, b.hVy)
	b.driver.MemCopyH2D(b.context, b.dVz, b.hVz)
	b.driver.MemCopyH2D(b.context, b.dSxx, b.hSxx)
	b.driver.MemCopyH2D(b.context, b.dSxy, b.hSxy)
	b.driver.MemCopyH2D(b.context, b.dSxz, b.hSxz)
	b.driver.MemCopyH2D(b.context, b.dSyy, b.hSyy)
	b.driver.MemCopyH2D(b.context, b.dSyz, b.hSyz)
	b.driver.MemCopyH2D(b.context, b.dSzz, b.hSzz)

	localSize := [3]uint16{8, 8, 1}
	globalX := uint32(((b.NX-1)/int(localSize[0]) + 1) * int(localSize[0]))
	flatYZ := b.NY * b.NZ
	globalY := uint32(((flatYZ-1)/int(localSize[1]) + 1) * int(localSize[1]))
	globalZ := uint32(1)
	globalSize := [3]uint32{globalX, globalY, globalZ}

	for iter := 0; iter < b.Iterations; iter++ {
		args := KernelArgs{
			Vx:                  b.dVx,
			Vy:                  b.dVy,
			Vz:                  b.dVz,
			SigmaXX:             b.dSxx,
			SigmaXY:             b.dSxy,
			SigmaXZ:             b.dSxz,
			SigmaYY:             b.dSyy,
			SigmaYZ:             b.dSyz,
			SigmaZZ:             b.dSzz,
			NX:                  int32(b.NX),
			NY:                  int32(b.NY),
			NZ:                  int32(b.NZ),
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

	b.driver.MemCopyD2H(b.context, b.gpuSxx, b.dSxx)
	b.driver.MemCopyD2H(b.context, b.gpuSxy, b.dSxy)
	b.driver.MemCopyD2H(b.context, b.gpuSxz, b.dSxz)
	b.driver.MemCopyD2H(b.context, b.gpuSyy, b.dSyy)
	b.driver.MemCopyD2H(b.context, b.gpuSyz, b.dSyz)
	b.driver.MemCopyD2H(b.context, b.gpuSzz, b.dSzz)
}

// Verify compares GPU stress tensors to CPU reference.
func (b *Benchmark) Verify() {
	b.cpuReference()

	const eps = float64(5e-3)
	check := func(name string, want, got []float32) {
		for i := range want {
			if math.Abs(float64(want[i]-got[i])) > eps {
				log.Panicf("eqwp %s mismatch at %d: want %f got %f", name, i, want[i], got[i])
			}
		}
	}
	check("sxx", b.hSxx, b.gpuSxx)
	check("sxy", b.hSxy, b.gpuSxy)
	check("sxz", b.hSxz, b.gpuSxz)
	check("syy", b.hSyy, b.gpuSyy)
	check("syz", b.hSyz, b.gpuSyz)
	check("szz", b.hSzz, b.gpuSzz)
	log.Printf("Passed!\n")
}

func (b *Benchmark) wrap(a, n int) int {
	m := a % n
	if m < 0 {
		m += n
	}
	return m
}

func (b *Benchmark) readv(v []float32, x, y, z int) float32 {
	return v[b.index(b.wrap(x, b.NX), b.wrap(y, b.NY), b.wrap(z, b.NZ))]
}

func (b *Benchmark) cpuReference() {
	n := b.NX * b.NY * b.NZ
	// Velocities fixed; only stress evolves (same as GPU).
	vx := append([]float32(nil), b.hVx...)
	vy := append([]float32(nil), b.hVy...)
	vz := append([]float32(nil), b.hVz...)
	for iter := 0; iter < b.Iterations; iter++ {
		nextSxx := make([]float32, n)
		copy(nextSxx, b.hSxx)
		nextSxy := make([]float32, n)
		copy(nextSxy, b.hSxy)
		nextSxz := make([]float32, n)
		copy(nextSxz, b.hSxz)
		nextSyy := make([]float32, n)
		copy(nextSyy, b.hSyy)
		nextSyz := make([]float32, n)
		copy(nextSyz, b.hSyz)
		nextSzz := make([]float32, n)
		copy(nextSzz, b.hSzz)

		for z := 0; z < b.NZ; z++ {
			for y := 0; y < b.NY; y++ {
				for x := 0; x < b.NX; x++ {
					i := b.index(x, y, z)
					vxBh2 := b.readv(vx, x, y, z-2)
					vxBh1 := b.readv(vx, x, y, z-1)
					vxIf1 := b.readv(vx, x, y, z+1)
					vxIf2 := b.readv(vx, x, y, z+2)
					vyBh2 := b.readv(vy, x, y, z-2)
					vyBh1 := b.readv(vy, x, y, z-1)
					vyIf1 := b.readv(vy, x, y, z+1)
					vyIf2 := b.readv(vy, x, y, z+2)
					vzBh2 := b.readv(vz, x, y, z-2)
					vzBh1 := b.readv(vz, x, y, z-1)
					vzIf1 := b.readv(vz, x, y, z+1)
					vzIf2 := b.readv(vz, x, y, z+2)

					dvxx := (coefA*(b.readv(vx, x+2, y, z)-b.readv(vx, x-2, y, z)) +
						coefB*(b.readv(vx, x+1, y, z)-b.readv(vx, x-1, y, z))) / dx
					dvyy := (coefA*(b.readv(vy, x, y+2, z)-b.readv(vy, x, y-2, z)) +
						coefB*(b.readv(vy, x, y+1, z)-b.readv(vy, x, y-1, z))) / dy
					dvzz := (coefA*(vzIf2-vzBh2) + coefB*(vzIf1-vzBh1)) / dz
					dvxy := (coefA*(b.readv(vx, x, y+2, z)-b.readv(vx, x, y-2, z)) +
						coefB*(b.readv(vx, x, y+1, z)-b.readv(vx, x, y-1, z))) / dy
					dvyx := (coefA*(b.readv(vy, x+2, y, z)-b.readv(vy, x-2, y, z)) +
						coefB*(b.readv(vy, x+1, y, z)-b.readv(vy, x-1, y, z))) / dx
					dvxz := (coefA*(vxIf2-vxBh2) + coefB*(vxIf1-vxBh1)) / dz
					dvzx := (coefA*(b.readv(vz, x+2, y, z)-b.readv(vz, x-2, y, z)) +
						coefB*(b.readv(vz, x+1, y, z)-b.readv(vz, x-1, y, z))) / dx
					dvyz := (coefA*(vyIf2-vyBh2) + coefB*(vyIf1-vyBh1)) / dz
					dvzy := (coefA*(b.readv(vz, x, y+2, z)-b.readv(vz, x, y-2, z)) +
						coefB*(b.readv(vz, x, y+1, z)-b.readv(vz, x, y-1, z))) / dy

					dsxx := (lame1+2*lame2)*dvxx + lame1*(dvyy+dvzz)
					dsyy := (lame1+2*lame2)*dvyy + lame1*(dvxx+dvzz)
					dszz := (lame1+2*lame2)*dvzz + lame1*(dvxx+dvyy)
					dsxy := lame2 * (dvxy + dvyx)
					dsxz := lame2 * (dvxz + dvzx)
					dsyz := lame2 * (dvyz + dvzy)

					nextSxx[i] += dt * dsxx
					nextSxy[i] += dt * dsxy
					nextSxz[i] += dt * dsxz
					nextSyy[i] += dt * dsyy
					nextSyz[i] += dt * dsyz
					nextSzz[i] += dt * dszz
				}
			}
		}
		b.hSxx, b.hSxy, b.hSxz, b.hSyy, b.hSyz, b.hSzz = nextSxx, nextSxy, nextSxz, nextSyy, nextSyz, nextSzz
	}
}

func (b *Benchmark) index(x, y, z int) int {
	return z*b.NX*b.NY + y*b.NX + x
}
