package mst

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
	RowOffsets          driver.GPUPtr
	ColIndices          driver.GPUPtr
	EdgeWeights         driver.GPUPtr
	CompLabel           driver.GPUPtr
	BestDst             driver.GPUPtr
	BestW               driver.GPUPtr
	NumNodes            int32
	Pad0                int32
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
	HiddenNone0         int64
	HiddenNone1         int64
	HiddenNone2         int64
	HiddenMultiGridSync int64
}

type edge struct {
	u, v int
	w    float32
}

type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	gpus    []int
	queues  []*driver.CommandQueue
	kernel  *insts.HsaCo

	NumNodes       int
	ExtraEdges     int
	MaxBoruvkaIter int

	rowOffsets []int32
	colIndices []int32
	weights    []float32
	comp       []int32
	bestDst    []int32
	bestW      []float32

	mstWeightGPU float64
	mstWeightCPU float64

	dRowOffsets driver.GPUPtr
	dColIndices driver.GPUPtr
	dWeights    driver.GPUPtr
	dComp       driver.GPUPtr
	dBestDst    driver.GPUPtr
	dBestW      driver.GPUPtr

	useUM, useLASP bool
}

func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:         driver,
		context:        driver.Init(),
		NumNodes:       4096,
		ExtraEdges:     16384,
		MaxBoruvkaIter: 20,
	}
	b.loadProgram()
	return b
}

func (b *Benchmark) loadProgram() {
	b.kernel = kernels.LoadProgramFromMemory(hsacoBytes, "mst_find_min_edge")
	if b.kernel == nil {
		log.Panic("failed to load mst_find_min_edge from kernels.hsaco")
	}
}

func (b *Benchmark) SelectGPU(gpus []int) {
	if len(gpus) > 1 {
		panic("mst benchmark currently supports single-GPU only")
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
	b.buildSyntheticGraph()
	n := b.NumNodes
	b.comp = make([]int32, n)
	b.bestDst = make([]int32, n)
	b.bestW = make([]float32, n)
	for i := 0; i < n; i++ {
		b.comp[i] = int32(i)
	}

	alloc := func(size uint64) driver.GPUPtr {
		if b.useUM {
			return b.driver.AllocateUnifiedMemory(b.context, size)
		}
		if b.useLASP {
			return b.driver.AllocateMemoryLASP(b.context, size, "div4")
		}
		return b.driver.AllocateMemory(b.context, size)
	}

	b.dRowOffsets = alloc(uint64(len(b.rowOffsets) * 4))
	b.dColIndices = alloc(uint64(len(b.colIndices) * 4))
	b.dWeights = alloc(uint64(len(b.weights) * 4))
	b.dComp = alloc(uint64(n * 4))
	b.dBestDst = alloc(uint64(n * 4))
	b.dBestW = alloc(uint64(n * 4))
}

func (b *Benchmark) buildSyntheticGraph() {
	n := b.NumNodes
	extra := b.ExtraEdges
	rand.Seed(1)
	adj := make([][]edge, n)

	// Ring edges ensure connectivity.
	for u := 0; u < n; u++ {
		v := (u + 1) % n
		w := 1.0 + float32((u*13)%97)/100.0
		adj[u] = append(adj[u], edge{u, v, w})
		adj[v] = append(adj[v], edge{v, u, w})
	}
	// Extra undirected edges.
	for i := 0; i < extra; i++ {
		u := rand.Intn(n)
		v := rand.Intn(n)
		if u == v {
			continue
		}
		w := 0.5 + 9.5*rand.Float32()
		adj[u] = append(adj[u], edge{u, v, w})
		adj[v] = append(adj[v], edge{v, u, w})
	}

	b.rowOffsets = make([]int32, n+1)
	total := 0
	for u := 0; u < n; u++ {
		b.rowOffsets[u] = int32(total)
		total += len(adj[u])
	}
	b.rowOffsets[n] = int32(total)
	b.colIndices = make([]int32, total)
	b.weights = make([]float32, total)
	k := 0
	for u := 0; u < n; u++ {
		for _, e := range adj[u] {
			b.colIndices[k] = int32(e.v)
			b.weights[k] = e.w
			k++
		}
	}
}

func (b *Benchmark) exec() {
	b.driver.MemCopyH2D(b.context, b.dRowOffsets, b.rowOffsets)
	b.driver.MemCopyH2D(b.context, b.dColIndices, b.colIndices)
	b.driver.MemCopyH2D(b.context, b.dWeights, b.weights)
	b.driver.MemCopyH2D(b.context, b.dComp, b.comp)

	local := [3]uint16{256, 1, 1}
	global := [3]uint32{alignUp(uint32(b.NumNodes), uint32(local[0])), 1, 1}

	weightSum := float64(0)
	uf := newUnionFind(b.NumNodes)
	for iter := 0; iter < b.MaxBoruvkaIter; iter++ {
		args := KernelArgs{
			RowOffsets:          b.dRowOffsets,
			ColIndices:          b.dColIndices,
			EdgeWeights:         b.dWeights,
			CompLabel:           b.dComp,
			BestDst:             b.dBestDst,
			BestW:               b.dBestW,
			NumNodes:            int32(b.NumNodes),
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
		b.driver.MemCopyD2H(b.context, b.bestDst, b.dBestDst)
		b.driver.MemCopyD2H(b.context, b.bestW, b.dBestW)

		merged := 0
		for u := 0; u < b.NumNodes; u++ {
			v := int(b.bestDst[u])
			if v < 0 {
				continue
			}
			if uf.union(u, v) {
				weightSum += float64(b.bestW[u])
				merged++
			}
		}
		for i := 0; i < b.NumNodes; i++ {
			b.comp[i] = int32(uf.find(i))
		}
		b.driver.MemCopyH2D(b.context, b.dComp, b.comp)

		if merged == 0 || uf.count == 1 {
			break
		}
	}
	b.mstWeightGPU = weightSum
}

func (b *Benchmark) Verify() {
	// Run same boruvka-like loop on CPU.
	comp := make([]int32, len(b.comp))
	for i := range comp {
		comp[i] = int32(i)
	}
	bestDst := make([]int32, b.NumNodes)
	bestW := make([]float32, b.NumNodes)
	uf := newUnionFind(b.NumNodes)
	wSum := float64(0)
	for iter := 0; iter < b.MaxBoruvkaIter; iter++ {
		for u := 0; u < b.NumNodes; u++ {
			cu := comp[u]
			start := int(b.rowOffsets[u])
			end := int(b.rowOffsets[u+1])
			bestDst[u] = -1
			bestW[u] = float32(1e30)
			for e := start; e < end; e++ {
				v := int(b.colIndices[e])
				if comp[v] == cu {
					continue
				}
				w := b.weights[e]
				if w < bestW[u] {
					bestW[u] = w
					bestDst[u] = int32(v)
				}
			}
		}
		merged := 0
		for u := 0; u < b.NumNodes; u++ {
			v := int(bestDst[u])
			if v < 0 {
				continue
			}
			if uf.union(u, v) {
				wSum += float64(bestW[u])
				merged++
			}
		}
		for i := 0; i < b.NumNodes; i++ {
			comp[i] = int32(uf.find(i))
		}
		if merged == 0 || uf.count == 1 {
			break
		}
	}
	b.mstWeightCPU = wSum
	if math.Abs(b.mstWeightCPU-b.mstWeightGPU) > 1e-4 {
		log.Panicf("mst weight mismatch: cpu=%f gpu=%f", b.mstWeightCPU, b.mstWeightGPU)
	}
	log.Printf("Passed!\n")
}

type unionFind struct {
	parent []int
	rank   []int
	count  int
}

func newUnionFind(n int) *unionFind {
	p := make([]int, n)
	r := make([]int, n)
	for i := 0; i < n; i++ {
		p[i] = i
	}
	return &unionFind{parent: p, rank: r, count: n}
}
func (u *unionFind) find(x int) int {
	if u.parent[x] != x {
		u.parent[x] = u.find(u.parent[x])
	}
	return u.parent[x]
}
func (u *unionFind) union(a, b int) bool {
	ra := u.find(a)
	rb := u.find(b)
	if ra == rb {
		return false
	}
	if u.rank[ra] < u.rank[rb] {
		ra, rb = rb, ra
	}
	u.parent[rb] = ra
	if u.rank[ra] == u.rank[rb] {
		u.rank[ra]++
	}
	u.count--
	return true
}

func alignUp(v, a uint32) uint32 {
	if a == 0 {
		return v
	}
	return ((v + a - 1) / a) * a
}
