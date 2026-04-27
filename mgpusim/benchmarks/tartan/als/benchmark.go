// Package als implements alternating least-squares style matrix factorization
// for sparse ratings.
package als

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

type UserKernelArgs struct {
	UserPtr             driver.GPUPtr
	UserItemIdx         driver.GPUPtr
	UserRating          driver.GPUPtr
	ItemFactors         driver.GPUPtr
	UserFactors         driver.GPUPtr
	NumUsers            int32
	Rank                int32
	LambdaReg           float32
	PadBeforeHidden     int32
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
	HiddenNone0         int64
	HiddenNone1         int64
	HiddenNone2         int64
	HiddenMultiGridSync int64
}

type ItemKernelArgs struct {
	ItemPtr             driver.GPUPtr
	ItemUserIdx         driver.GPUPtr
	ItemRating          driver.GPUPtr
	UserFactors         driver.GPUPtr
	ItemFactors         driver.GPUPtr
	NumItems            int32
	Rank                int32
	LambdaReg           float32
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

	userKernel *insts.HsaCo
	itemKernel *insts.HsaCo

	NumUsers   int
	NumItems   int
	Rank       int
	NNZPerUser int
	Iterations int
	LambdaReg  float32

	userPtr     []int32
	userItemIdx []int32
	userRating  []float32
	itemPtr     []int32
	itemUserIdx []int32
	itemRating  []float32

	userFactors []float32
	itemFactors []float32
	gpuUserF    []float32
	gpuItemF    []float32

	dUserPtr     driver.GPUPtr
	dUserItemIdx driver.GPUPtr
	dUserRating  driver.GPUPtr
	dItemPtr     driver.GPUPtr
	dItemUserIdx driver.GPUPtr
	dItemRating  driver.GPUPtr
	dUserFactors driver.GPUPtr
	dItemFactors driver.GPUPtr

	useUM, useLASP bool
}

func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:     driver,
		context:    driver.Init(),
		NumUsers:   1024,
		NumItems:   1024,
		Rank:       16,
		NNZPerUser: 16,
		Iterations: 4,
		LambdaReg:  0.05,
	}
	b.loadProgram()
	return b
}

func (b *Benchmark) loadProgram() {
	b.userKernel = kernels.LoadProgramFromMemory(hsacoBytes, "als_update_users")
	b.itemKernel = kernels.LoadProgramFromMemory(hsacoBytes, "als_update_items")
	if b.userKernel == nil || b.itemKernel == nil {
		log.Panic("failed to load als kernels from kernels.hsaco")
	}
}

func (b *Benchmark) SelectGPU(gpus []int) {
	if len(gpus) > 1 {
		panic("als benchmark currently supports single-GPU only")
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
	if b.NumUsers < 1 || b.NumItems < 1 || b.Rank < 1 || b.NNZPerUser < 1 {
		panic("NumUsers/NumItems/Rank/NNZPerUser must be >= 1")
	}

	b.buildSyntheticDataset()

	b.gpuUserF = make([]float32, len(b.userFactors))
	b.gpuItemF = make([]float32, len(b.itemFactors))

	alloc := func(byteSize uint64) driver.GPUPtr {
		if b.useUM {
			return b.driver.AllocateUnifiedMemory(b.context, byteSize)
		}
		if b.useLASP {
			return b.driver.AllocateMemoryLASP(b.context, byteSize, "div4")
		}
		return b.driver.AllocateMemory(b.context, byteSize)
	}

	b.dUserPtr = alloc(uint64(len(b.userPtr) * 4))
	b.dUserItemIdx = alloc(uint64(len(b.userItemIdx) * 4))
	b.dUserRating = alloc(uint64(len(b.userRating) * 4))
	b.dItemPtr = alloc(uint64(len(b.itemPtr) * 4))
	b.dItemUserIdx = alloc(uint64(len(b.itemUserIdx) * 4))
	b.dItemRating = alloc(uint64(len(b.itemRating) * 4))
	b.dUserFactors = alloc(uint64(len(b.userFactors) * 4))
	b.dItemFactors = alloc(uint64(len(b.itemFactors) * 4))
}

func (b *Benchmark) buildSyntheticDataset() {
	rand.Seed(1)

	nnz := b.NumUsers * b.NNZPerUser
	b.userPtr = make([]int32, b.NumUsers+1)
	b.userItemIdx = make([]int32, nnz)
	b.userRating = make([]float32, nnz)

	pos := 0
	for u := 0; u < b.NumUsers; u++ {
		b.userPtr[u] = int32(pos)
		for j := 0; j < b.NNZPerUser; j++ {
			it := (u*17 + j*13) % b.NumItems
			b.userItemIdx[pos] = int32(it)
			// rating in [1, 5]
			b.userRating[pos] = 1.0 + 4.0*rand.Float32()
			pos++
		}
	}
	b.userPtr[b.NumUsers] = int32(pos)

	countItem := make([]int, b.NumItems)
	for _, it := range b.userItemIdx {
		countItem[it]++
	}
	b.itemPtr = make([]int32, b.NumItems+1)
	total := 0
	for i := 0; i < b.NumItems; i++ {
		b.itemPtr[i] = int32(total)
		total += countItem[i]
	}
	b.itemPtr[b.NumItems] = int32(total)

	b.itemUserIdx = make([]int32, total)
	b.itemRating = make([]float32, total)
	writePos := make([]int, b.NumItems)
	for i := 0; i < b.NumItems; i++ {
		writePos[i] = int(b.itemPtr[i])
	}
	for u := 0; u < b.NumUsers; u++ {
		start := int(b.userPtr[u])
		end := int(b.userPtr[u+1])
		for p := start; p < end; p++ {
			it := int(b.userItemIdx[p])
			w := writePos[it]
			b.itemUserIdx[w] = int32(u)
			b.itemRating[w] = b.userRating[p]
			writePos[it]++
		}
	}

	b.userFactors = make([]float32, b.NumUsers*b.Rank)
	b.itemFactors = make([]float32, b.NumItems*b.Rank)
	for i := range b.userFactors {
		b.userFactors[i] = 0.05 + 0.1*rand.Float32()
	}
	for i := range b.itemFactors {
		b.itemFactors[i] = 0.05 + 0.1*rand.Float32()
	}
}

func (b *Benchmark) exec() {
	b.driver.MemCopyH2D(b.context, b.dUserPtr, b.userPtr)
	b.driver.MemCopyH2D(b.context, b.dUserItemIdx, b.userItemIdx)
	b.driver.MemCopyH2D(b.context, b.dUserRating, b.userRating)
	b.driver.MemCopyH2D(b.context, b.dItemPtr, b.itemPtr)
	b.driver.MemCopyH2D(b.context, b.dItemUserIdx, b.itemUserIdx)
	b.driver.MemCopyH2D(b.context, b.dItemRating, b.itemRating)
	b.driver.MemCopyH2D(b.context, b.dUserFactors, b.userFactors)
	b.driver.MemCopyH2D(b.context, b.dItemFactors, b.itemFactors)

	local := [3]uint16{128, 1, 1}
	userGlobal := [3]uint32{alignUp(uint32(b.NumUsers), uint32(local[0])), 1, 1}
	itemGlobal := [3]uint32{alignUp(uint32(b.NumItems), uint32(local[0])), 1, 1}

	for it := 0; it < b.Iterations; it++ {
		uargs := UserKernelArgs{
			UserPtr:             b.dUserPtr,
			UserItemIdx:         b.dUserItemIdx,
			UserRating:          b.dUserRating,
			ItemFactors:         b.dItemFactors,
			UserFactors:         b.dUserFactors,
			NumUsers:            int32(b.NumUsers),
			Rank:                int32(b.Rank),
			LambdaReg:           b.LambdaReg,
			PadBeforeHidden:     0,
			HiddenGlobalOffsetX: 0,
			HiddenGlobalOffsetY: 0,
			HiddenGlobalOffsetZ: 0,
			HiddenNone0:         0,
			HiddenNone1:         0,
			HiddenNone2:         0,
			HiddenMultiGridSync: 0,
		}
		b.driver.LaunchKernel(b.context, b.userKernel, userGlobal, local, &uargs)

		iargs := ItemKernelArgs{
			ItemPtr:             b.dItemPtr,
			ItemUserIdx:         b.dItemUserIdx,
			ItemRating:          b.dItemRating,
			UserFactors:         b.dUserFactors,
			ItemFactors:         b.dItemFactors,
			NumItems:            int32(b.NumItems),
			Rank:                int32(b.Rank),
			LambdaReg:           b.LambdaReg,
			PadBeforeHidden:     0,
			HiddenGlobalOffsetX: 0,
			HiddenGlobalOffsetY: 0,
			HiddenGlobalOffsetZ: 0,
			HiddenNone0:         0,
			HiddenNone1:         0,
			HiddenNone2:         0,
			HiddenMultiGridSync: 0,
		}
		b.driver.LaunchKernel(b.context, b.itemKernel, itemGlobal, local, &iargs)
	}

	b.driver.MemCopyD2H(b.context, b.gpuUserF, b.dUserFactors)
	b.driver.MemCopyD2H(b.context, b.gpuItemF, b.dItemFactors)
}

func (b *Benchmark) Verify() {
	cpuUser := append([]float32(nil), b.userFactors...)
	cpuItem := append([]float32(nil), b.itemFactors...)

	for it := 0; it < b.Iterations; it++ {
		updateUsersCPU(
			b.userPtr, b.userItemIdx, b.userRating,
			cpuItem, cpuUser, b.NumUsers, b.Rank, b.LambdaReg,
		)
		updateItemsCPU(
			b.itemPtr, b.itemUserIdx, b.itemRating,
			cpuUser, cpuItem, b.NumItems, b.Rank, b.LambdaReg,
		)
	}

	const eps = float64(2e-5)
	check := func(name string, want, got []float32) {
		for i := range want {
			if math.Abs(float64(want[i]-got[i])) > eps {
				log.Panicf("als %s mismatch at %d: want %f got %f", name, i, want[i], got[i])
			}
		}
	}
	check("user_factors", cpuUser, b.gpuUserF)
	check("item_factors", cpuItem, b.gpuItemF)
	log.Printf("Passed!\n")
}

func updateUsersCPU(
	userPtr, userItemIdx []int32,
	userRating []float32,
	itemFactors []float32,
	userFactors []float32,
	numUsers, rank int,
	lambdaReg float32,
) {
	for u := 0; u < numUsers; u++ {
		start := int(userPtr[u])
		end := int(userPtr[u+1])
		for f := 0; f < rank; f++ {
			num := float32(0)
			den := lambdaReg
			for p := start; p < end; p++ {
				it := int(userItemIdx[p])
				y := itemFactors[it*rank+f]
				r := userRating[p]
				num += r * y
				den += y * y
			}
			if den > 0 {
				userFactors[u*rank+f] = num / den
			} else {
				userFactors[u*rank+f] = 0
			}
		}
	}
}

func updateItemsCPU(
	itemPtr, itemUserIdx []int32,
	itemRating []float32,
	userFactors []float32,
	itemFactors []float32,
	numItems, rank int,
	lambdaReg float32,
) {
	for it := 0; it < numItems; it++ {
		start := int(itemPtr[it])
		end := int(itemPtr[it+1])
		for f := 0; f < rank; f++ {
			num := float32(0)
			den := lambdaReg
			for p := start; p < end; p++ {
				u := int(itemUserIdx[p])
				x := userFactors[u*rank+f]
				r := itemRating[p]
				num += r * x
				den += x * x
			}
			if den > 0 {
				itemFactors[it*rank+f] = num / den
			} else {
				itemFactors[it*rank+f] = 0
			}
		}
	}
}

func alignUp(v, a uint32) uint32 {
	if a == 0 {
		return v
	}
	return ((v + a - 1) / a) * a
}
