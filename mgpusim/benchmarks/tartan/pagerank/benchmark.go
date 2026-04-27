// Package pagerank implements synchronous iterative PageRank.
package pagerank

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
	InPtr               driver.GPUPtr
	InSrc               driver.GPUPtr
	OutDeg              driver.GPUPtr
	PRIn                driver.GPUPtr
	PROut               driver.GPUPtr
	NumNodes            int32
	Damping             float32
	Base                float32
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

	NumNodes     int
	EdgesPerNode int
	Iterations   int
	Damping      float32

	inPtr  []int32
	inSrc  []int32
	outDeg []int32

	prInit   []float32
	prGPU    []float32
	prCPURef []float32

	dInPtr  driver.GPUPtr
	dInSrc  driver.GPUPtr
	dOutDeg driver.GPUPtr
	dPR0    driver.GPUPtr
	dPR1    driver.GPUPtr

	useUM, useLASP bool
}

func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:       driver,
		context:      driver.Init(),
		NumNodes:     4096,
		EdgesPerNode: 8,
		Iterations:   10,
		Damping:      0.85,
	}
	b.loadProgram()
	return b
}

func (b *Benchmark) loadProgram() {
	b.kernel = kernels.LoadProgramFromMemory(hsacoBytes, "pagerank_update_sync")
	if b.kernel == nil {
		log.Panic("failed to load pagerank_update_sync from kernels.hsaco")
	}
}

func (b *Benchmark) SelectGPU(gpus []int) {
	if len(gpus) > 1 {
		panic("tartan pagerank benchmark currently supports single-GPU only")
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
	if b.NumNodes < 1 || b.EdgesPerNode < 1 {
		panic("NumNodes and EdgesPerNode must be >= 1")
	}

	b.buildPeerToPeerLikeGraph()

	n := b.NumNodes
	initVal := float32(1.0 / float64(n))
	b.prInit = make([]float32, n)
	for i := range b.prInit {
		b.prInit[i] = initVal
	}
	b.prGPU = make([]float32, n)
	b.prCPURef = make([]float32, n)

	alloc := func(size uint64) driver.GPUPtr {
		if b.useUM {
			return b.driver.AllocateUnifiedMemory(b.context, size)
		}
		if b.useLASP {
			return b.driver.AllocateMemoryLASP(b.context, size, "div4")
		}
		return b.driver.AllocateMemory(b.context, size)
	}

	b.dInPtr = alloc(uint64(len(b.inPtr) * 4))
	b.dInSrc = alloc(uint64(len(b.inSrc) * 4))
	b.dOutDeg = alloc(uint64(len(b.outDeg) * 4))
	b.dPR0 = alloc(uint64(n * 4))
	b.dPR1 = alloc(uint64(n * 4))
}

func (b *Benchmark) buildPeerToPeerLikeGraph() {
	n := b.NumNodes
	k := b.EdgesPerNode

	incoming := make([][]int32, n)
	outDeg := make([]int32, n)

	// Deterministic, distributed source pattern to emulate peer-to-peer traffic.
	for dst := 0; dst < n; dst++ {
		for e := 0; e < k; e++ {
			src := (dst*131 + e*17 + 7) % n
			incoming[dst] = append(incoming[dst], int32(src))
			outDeg[src]++
		}
	}

	b.inPtr = make([]int32, n+1)
	total := 0
	for i := 0; i < n; i++ {
		b.inPtr[i] = int32(total)
		total += len(incoming[i])
	}
	b.inPtr[n] = int32(total)

	b.inSrc = make([]int32, total)
	pos := 0
	for i := 0; i < n; i++ {
		copy(b.inSrc[pos:pos+len(incoming[i])], incoming[i])
		pos += len(incoming[i])
	}

	b.outDeg = outDeg
}

func (b *Benchmark) exec() {
	b.driver.MemCopyH2D(b.context, b.dInPtr, b.inPtr)
	b.driver.MemCopyH2D(b.context, b.dInSrc, b.inSrc)
	b.driver.MemCopyH2D(b.context, b.dOutDeg, b.outDeg)
	b.driver.MemCopyH2D(b.context, b.dPR0, b.prInit)
	b.driver.MemCopyH2D(b.context, b.dPR1, b.prInit)

	local := [3]uint16{256, 1, 1}
	global := [3]uint32{alignUp(uint32(b.NumNodes), uint32(local[0])), 1, 1}
	base := float32((1.0 - float64(b.Damping)) / float64(b.NumNodes))

	src := b.dPR0
	dst := b.dPR1
	for it := 0; it < b.Iterations; it++ {
		args := KernelArgs{
			InPtr:               b.dInPtr,
			InSrc:               b.dInSrc,
			OutDeg:              b.dOutDeg,
			PRIn:                src,
			PROut:               dst,
			NumNodes:            int32(b.NumNodes),
			Damping:             b.Damping,
			Base:                base,
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

	b.driver.MemCopyD2H(b.context, b.prGPU, src)
}

func (b *Benchmark) Verify() {
	copy(b.prCPURef, b.prInit)
	next := make([]float32, b.NumNodes)
	base := float32((1.0 - float64(b.Damping)) / float64(b.NumNodes))

	for it := 0; it < b.Iterations; it++ {
		for v := 0; v < b.NumNodes; v++ {
			begin := int(b.inPtr[v])
			end := int(b.inPtr[v+1])
			acc := float32(0)
			for p := begin; p < end; p++ {
				u := b.inSrc[p]
				acc += b.prCPURef[u] / float32(b.outDeg[u])
			}
			next[v] = base + b.Damping*acc
		}
		b.prCPURef, next = next, b.prCPURef
	}

	const eps = float64(2e-5)
	for i := range b.prGPU {
		if math.Abs(float64(b.prGPU[i]-b.prCPURef[i])) > eps {
			log.Panicf("pagerank mismatch at %d: want %f got %f", i, b.prCPURef[i], b.prGPU[i])
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
