package srad

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

type CoeffArgs struct {
	Img                 driver.GPUPtr
	Coeff               driver.GPUPtr
	Rows                int32
	Cols                int32
	Q0sqr               float32
	Pad0                int32
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
	HiddenNone0         int64
	HiddenNone1         int64
	HiddenNone2         int64
	HiddenMultiGridSync int64
}

type UpdateArgs struct {
	Img                 driver.GPUPtr
	Coeff               driver.GPUPtr
	Rows                int32
	Cols                int32
	Lambda              float32
	Pad0                int32
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

	coeffKernel  *insts.HsaCo
	updateKernel *insts.HsaCo

	Rows, Cols, Iterations int
	Lambda                 float32

	imgInit []float32
	imgCPU  []float32
	imgGPU  []float32
	coeff   []float32
	dImg    driver.GPUPtr
	dCoeff  driver.GPUPtr

	useUM   bool
	useLASP bool
}

func NewBenchmark(d *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:     d,
		context:    d.Init(),
		Rows:       128,
		Cols:       128,
		Iterations: 10,
		Lambda:     0.5,
	}
	b.coeffKernel = kernels.LoadProgramFromMemory(hsacoBytes, "srad_compute_coeff")
	b.updateKernel = kernels.LoadProgramFromMemory(hsacoBytes, "srad_update")
	if b.coeffKernel == nil || b.updateKernel == nil {
		log.Panic("failed to load srad kernels")
	}
	return b
}

func (b *Benchmark) SelectGPU(g []int) {
	if len(g) > 1 {
		panic("srad single GPU only")
	}
	b.gpus = g
}
func (b *Benchmark) SetUnifiedMemory()   { b.useUM = true }
func (b *Benchmark) SetLASPMemoryAlloc() { b.useLASP = true }

func (b *Benchmark) alloc(sz uint64) driver.GPUPtr {
	if b.useUM {
		return b.driver.AllocateUnifiedMemory(b.context, sz)
	}
	if b.useLASP {
		return b.driver.AllocateMemoryLASP(b.context, sz, "div4")
	}
	return b.driver.AllocateMemory(b.context, sz)
}

func (b *Benchmark) Run() {
	b.initMem()
	b.exec()
}

func (b *Benchmark) initMem() {
	n := b.Rows * b.Cols
	b.imgInit = make([]float32, n)
	b.imgCPU = make([]float32, n)
	b.imgGPU = make([]float32, n)
	b.coeff = make([]float32, n)
	for r := 0; r < b.Rows; r++ {
		for c := 0; c < b.Cols; c++ {
			// Deterministic smooth + textured pattern.
			v := 1.0 + 0.2*math.Sin(float64(r)*0.05) + 0.15*math.Cos(float64(c)*0.04)
			b.imgInit[r*b.Cols+c] = float32(v)
		}
	}
	copy(b.imgCPU, b.imgInit)

	b.dImg = b.alloc(uint64(n * 4))
	b.dCoeff = b.alloc(uint64(n * 4))
}

func (b *Benchmark) exec() {
	b.driver.MemCopyH2D(b.context, b.dImg, b.imgInit)
	local := [3]uint16{16, 16, 1}
	global := [3]uint32{
		alignUp(uint32(b.Cols), uint32(local[0])),
		alignUp(uint32(b.Rows), uint32(local[1])),
		1,
	}

	for it := 0; it < b.Iterations; it++ {
		q0 := b.computeQ0sqrFromGPU()

		carg := CoeffArgs{
			Img:                 b.dImg,
			Coeff:               b.dCoeff,
			Rows:                int32(b.Rows),
			Cols:                int32(b.Cols),
			Q0sqr:               q0,
			Pad0:                0,
			HiddenGlobalOffsetX: 0,
			HiddenGlobalOffsetY: 0,
			HiddenGlobalOffsetZ: 0,
			HiddenNone0:         0,
			HiddenNone1:         0,
			HiddenNone2:         0,
			HiddenMultiGridSync: 0,
		}
		b.driver.LaunchKernel(b.context, b.coeffKernel, global, local, &carg)

		uarg := UpdateArgs{
			Img:                 b.dImg,
			Coeff:               b.dCoeff,
			Rows:                int32(b.Rows),
			Cols:                int32(b.Cols),
			Lambda:              b.Lambda,
			Pad0:                0,
			HiddenGlobalOffsetX: 0,
			HiddenGlobalOffsetY: 0,
			HiddenGlobalOffsetZ: 0,
			HiddenNone0:         0,
			HiddenNone1:         0,
			HiddenNone2:         0,
			HiddenMultiGridSync: 0,
		}
		b.driver.LaunchKernel(b.context, b.updateKernel, global, local, &uarg)
	}
	b.driver.MemCopyD2H(b.context, b.imgGPU, b.dImg)
}

func (b *Benchmark) computeQ0sqrFromGPU() float32 {
	b.driver.MemCopyD2H(b.context, b.imgGPU, b.dImg)
	return computeQ0sqr(b.imgGPU)
}

func computeQ0sqr(img []float32) float32 {
	var sum, sum2 float64
	for _, v := range img {
		f := float64(v)
		sum += f
		sum2 += f * f
	}
	n := float64(len(img))
	mean := sum / n
	variance := sum2/n - mean*mean
	if mean == 0 {
		return 0
	}
	return float32(variance / (mean * mean))
}

func (b *Benchmark) Verify() {
	n := b.Rows * b.Cols
	img := make([]float32, n)
	copy(img, b.imgCPU)
	coeff := make([]float32, n)

	idx := func(r, c int) int { return r*b.Cols + c }

	for it := 0; it < b.Iterations; it++ {
		q0 := computeQ0sqr(img)
		for r := 0; r < b.Rows; r++ {
			nr := r - 1
			sr := r + 1
			if nr < 0 {
				nr = 0
			}
			if sr >= b.Rows {
				sr = b.Rows - 1
			}
			for c := 0; c < b.Cols; c++ {
				wc := c - 1
				ec := c + 1
				if wc < 0 {
					wc = 0
				}
				if ec >= b.Cols {
					ec = b.Cols - 1
				}
				jc := img[idx(r, c)]
				dN := img[idx(nr, c)] - jc
				dS := img[idx(sr, c)] - jc
				dW := img[idx(r, wc)] - jc
				dE := img[idx(r, ec)] - jc
				g2 := (dN*dN + dS*dS + dW*dW + dE*dE) / (jc*jc + 1e-12)
				l := (dN + dS + dW + dE) / (jc + 1e-12)
				num := 0.5*g2 - 0.0625*(l*l)
				den := 1.0 + 0.25*l
				qsqr := num / (den*den + 1e-12)
				den2 := (qsqr - q0) / (q0*(1.0+q0) + 1e-12)
				cval := 1.0 / (1.0 + den2)
				if cval < 0 {
					cval = 0
				}
				if cval > 1 {
					cval = 1
				}
				coeff[idx(r, c)] = cval
			}
		}

		next := make([]float32, n)
		for r := 0; r < b.Rows; r++ {
			nr := r - 1
			sr := r + 1
			if nr < 0 {
				nr = 0
			}
			if sr >= b.Rows {
				sr = b.Rows - 1
			}
			for c := 0; c < b.Cols; c++ {
				wc := c - 1
				ec := c + 1
				if wc < 0 {
					wc = 0
				}
				if ec >= b.Cols {
					ec = b.Cols - 1
				}
				jc := img[idx(r, c)]
				dN := img[idx(nr, c)] - jc
				dS := img[idx(sr, c)] - jc
				dW := img[idx(r, wc)] - jc
				dE := img[idx(r, ec)] - jc
				div := coeff[idx(nr, c)]*dN + coeff[idx(sr, c)]*dS + coeff[idx(r, wc)]*dW + coeff[idx(r, ec)]*dE
				next[idx(r, c)] = jc + 0.25*b.Lambda*div
			}
		}
		img = next
	}

	for i := 0; i < n; i++ {
		if math.Abs(float64(img[i]-b.imgGPU[i])) > 2e-4 {
			log.Panicf("srad mismatch at %d: want %f got %f", i, img[i], b.imgGPU[i])
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
