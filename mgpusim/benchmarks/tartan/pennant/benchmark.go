// Package pennant ports a PENNANT-like unstructured mesh zone update.
//
// Reference: https://github.com/lanl/PENNANT
package pennant

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
	ZonePointOffsets    driver.GPUPtr
	ZonePointIndices    driver.GPUPtr
	PointVelocity       driver.GPUPtr
	ZoneEnergy          driver.GPUPtr
	ZonePressure        driver.GPUPtr
	NumZones            int32
	Dt                  float32
	Gamma               float32
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

	NumZones      int
	NumPoints     int
	MinPtsPerZone int
	MaxPtsPerZone int
	Iterations    int
	Dt            float32
	Gamma         float32

	zoneOffsets []int32
	zonePoints  []int32
	pointVel    []float32
	zoneEnergy  []float32
	zonePress   []float32

	gpuZoneEnergy []float32
	gpuZonePress  []float32

	dZoneOffsets driver.GPUPtr
	dZonePoints  driver.GPUPtr
	dPointVel    driver.GPUPtr
	dZoneEnergy  driver.GPUPtr
	dZonePress   driver.GPUPtr

	useUM, useLASP bool
}

func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:        driver,
		context:       driver.Init(),
		NumZones:      4096,
		NumPoints:     8192,
		MinPtsPerZone: 3,
		MaxPtsPerZone: 6,
		Iterations:    6,
		Dt:            0.01,
		Gamma:         1.4,
	}
	b.loadProgram()
	return b
}

func (b *Benchmark) loadProgram() {
	b.kernel = kernels.LoadProgramFromMemory(hsacoBytes, "pennant_zone_update")
	if b.kernel == nil {
		log.Panic("failed to load pennant_zone_update from kernels.hsaco")
	}
}

func (b *Benchmark) SelectGPU(gpus []int) {
	if len(gpus) > 1 {
		panic("pennant benchmark currently supports single-GPU only")
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
	if b.NumZones < 1 || b.NumPoints < 1 || b.MinPtsPerZone < 1 || b.MaxPtsPerZone < b.MinPtsPerZone {
		panic("invalid pennant benchmark sizing")
	}
	b.buildSyntheticMesh()

	b.gpuZoneEnergy = make([]float32, len(b.zoneEnergy))
	b.gpuZonePress = make([]float32, len(b.zonePress))

	alloc := func(size uint64) driver.GPUPtr {
		if b.useUM {
			return b.driver.AllocateUnifiedMemory(b.context, size)
		}
		if b.useLASP {
			return b.driver.AllocateMemoryLASP(b.context, size, "div4")
		}
		return b.driver.AllocateMemory(b.context, size)
	}

	b.dZoneOffsets = alloc(uint64(len(b.zoneOffsets) * 4))
	b.dZonePoints = alloc(uint64(len(b.zonePoints) * 4))
	b.dPointVel = alloc(uint64(len(b.pointVel) * 4))
	b.dZoneEnergy = alloc(uint64(len(b.zoneEnergy) * 4))
	b.dZonePress = alloc(uint64(len(b.zonePress) * 4))
}

func (b *Benchmark) buildSyntheticMesh() {
	rand.Seed(1)

	b.pointVel = make([]float32, b.NumPoints)
	for i := 0; i < b.NumPoints; i++ {
		b.pointVel[i] = 0.1 + 1.9*rand.Float32()
	}

	b.zoneEnergy = make([]float32, b.NumZones)
	b.zonePress = make([]float32, b.NumZones)
	for z := 0; z < b.NumZones; z++ {
		b.zoneEnergy[z] = 0.5 + 0.5*rand.Float32()
		b.zonePress[z] = b.Gamma * b.zoneEnergy[z]
	}

	b.zoneOffsets = make([]int32, b.NumZones+1)
	total := 0
	for z := 0; z < b.NumZones; z++ {
		b.zoneOffsets[z] = int32(total)
		n := b.MinPtsPerZone + (z % (b.MaxPtsPerZone - b.MinPtsPerZone + 1))
		total += n
	}
	b.zoneOffsets[b.NumZones] = int32(total)

	b.zonePoints = make([]int32, total)
	idx := 0
	for z := 0; z < b.NumZones; z++ {
		begin := int(b.zoneOffsets[z])
		end := int(b.zoneOffsets[z+1])
		for p := begin; p < end; p++ {
			// deterministic pseudo-random unstructured connectivity
			pt := (z*37 + p*17 + idx*13) % b.NumPoints
			b.zonePoints[idx] = int32(pt)
			idx++
		}
	}
}

func (b *Benchmark) exec() {
	b.driver.MemCopyH2D(b.context, b.dZoneOffsets, b.zoneOffsets)
	b.driver.MemCopyH2D(b.context, b.dZonePoints, b.zonePoints)
	b.driver.MemCopyH2D(b.context, b.dPointVel, b.pointVel)
	b.driver.MemCopyH2D(b.context, b.dZoneEnergy, b.zoneEnergy)
	b.driver.MemCopyH2D(b.context, b.dZonePress, b.zonePress)

	local := [3]uint16{256, 1, 1}
	global := [3]uint32{alignUp(uint32(b.NumZones), uint32(local[0])), 1, 1}

	for it := 0; it < b.Iterations; it++ {
		args := KernelArgs{
			ZonePointOffsets:    b.dZoneOffsets,
			ZonePointIndices:    b.dZonePoints,
			PointVelocity:       b.dPointVel,
			ZoneEnergy:          b.dZoneEnergy,
			ZonePressure:        b.dZonePress,
			NumZones:            int32(b.NumZones),
			Dt:                  b.Dt,
			Gamma:               b.Gamma,
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
	}

	b.driver.MemCopyD2H(b.context, b.gpuZoneEnergy, b.dZoneEnergy)
	b.driver.MemCopyD2H(b.context, b.gpuZonePress, b.dZonePress)
}

func (b *Benchmark) Verify() {
	refE := append([]float32(nil), b.zoneEnergy...)
	refP := append([]float32(nil), b.zonePress...)

	for it := 0; it < b.Iterations; it++ {
		for z := 0; z < b.NumZones; z++ {
			begin := int(b.zoneOffsets[z])
			end := int(b.zoneOffsets[z+1])
			count := end - begin
			vsum := float32(0)
			for p := begin; p < end; p++ {
				vsum += b.pointVel[b.zonePoints[p]]
			}
			avgV := vsum / float32(count)
			e := refE[z]
			p := refP[z]
			e = e + b.Dt*(p*avgV-0.1*e)
			p = b.Gamma * e
			refE[z] = e
			refP[z] = p
		}
	}

	const eps = 2e-5
	for i := range refE {
		if math.Abs(float64(refE[i]-b.gpuZoneEnergy[i])) > eps {
			log.Panicf("pennant energy mismatch at %d: want %f got %f",
				i, refE[i], b.gpuZoneEnergy[i])
		}
		if math.Abs(float64(refP[i]-b.gpuZonePress[i])) > eps {
			log.Panicf("pennant pressure mismatch at %d: want %f got %f",
				i, refP[i], b.gpuZonePress[i])
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
