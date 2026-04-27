package platform

import (
	"fmt"

	"gitlab.com/akita/akita"
	"gitlab.com/akita/mem"
	"gitlab.com/akita/mgpusim/builders"
	"gitlab.com/akita/mgpusim/driver"
)

// MI300GPUPlatformBuilder assembles an engine, driver, and MI300-style GPU
// (IdealVM-2 class: IdealVM2 shader arrays, L1I/L1S TLB nil, MCM/IF-style chiplet count).
// Default topology targets MI300X-like scale: 8 chiplets (XCDs), 16 HBM banks per chiplet, 16GB.
type MI300GPUPlatformBuilder struct {
	CommonPlatformBuilder
}

// MakeMI300GPUPlatformBuilder returns a platform builder with MI300-oriented defaults.
func MakeMI300GPUPlatformBuilder() MI300GPUPlatformBuilder {
	return MI300GPUPlatformBuilder{
		CommonPlatformBuilder: CommonPlatformBuilder{
			numGPU:                   1,
			log2PageSize:             uint64(12),
			log2CacheLineSize:        uint64(7), //junsung "2026-04-27" default 64B cache line
			numCUPerShaderArray:      uint64(4),
			numShaderArrayPerChiplet: uint64(8),
			numMemoryBankPerChiplet:  uint64(16),
			numChiplets:              uint64(8), // junsung chiplet num
			memGroupSize:             uint64(2), // junsung
			totalMem:                 32 * mem.GB,
			// totalMem:                 16 * mem.GB,
			bankSize: 256 * mem.MB,
			lowAddr:  4 * mem.GB,
		},
	}
}

// Build instantiates the simulation platform.
func (b MI300GPUPlatformBuilder) Build() (akita.Engine, *driver.Driver) {
	engine := b.createEngine()

	gpuDriver := driver.NewDriver(engine, b.log2PageSize, b.memAllocatorType)
	gpuBuilder := b.createGPUBuilder(engine, gpuDriver)
	pcieConnector, rootComplexID := b.createConnection(engine, gpuDriver)

	rdmaAddressTable := b.createRDMAAddrTable()
	pmcAddressTable := b.createPMCPageTable()

	b.createGPUs(
		rootComplexID, pcieConnector,
		gpuBuilder, gpuDriver,
		rdmaAddressTable, pmcAddressTable,
	)

	return engine, gpuDriver
}

func (b *MI300GPUPlatformBuilder) createGPUBuilder(
	engine akita.Engine,
	gpuDriver *driver.Driver,
) builders.Builder {
	//junsung "2026-04-27" mem-group-size validation for grouped local memory dies.
	if b.memGroupSize == 0 {
		panic("mi300 mem-group-size must be >= 1")
	}
	if b.numChiplets%b.memGroupSize != 0 {
		panic(fmt.Sprintf("mi300 mem-group-size (%d) must divide num-chiplets (%d)", b.memGroupSize, b.numChiplets))
	}

	gpuBuilder := builders.MakeMI300GPUBuilder()
	gpuBuilder.WithEngine(engine)
	gpuBuilder.WithNumCUPerShaderArray(int(b.numCUPerShaderArray))
	gpuBuilder.WithNumShaderArrayPerChiplet(int(b.numShaderArrayPerChiplet))
	gpuBuilder.WithNumMemoryBankPerChiplet(int(b.numMemoryBankPerChiplet))
	gpuBuilder.WithNumChiplet(int(b.numChiplets))
	//junsung "2026-04-27" pass mem-group-size into MI300 builder for local/remote routing.
	gpuBuilder.WithMemGroupSize(int(b.memGroupSize))
	gpuBuilder.WithTotalMem(b.totalMem)
	gpuBuilder.CalculateMemoryParameters()
	gpuBuilder.WithLog2PageSize(b.log2PageSize)
	//junsung "2026-04-27" propagate platform cacheline size to MI300 builder/cache hierarchy
	gpuBuilder.WithLog2CacheLineSize(b.log2CacheLineSize)
	gpuBuilder.WithPageTable(gpuDriver.PageTable)
	gpuBuilder.WithAlg(b.alg)
	gpuBuilder.WithSchedulingPartition(b.partition)

	b.setVisTracer(gpuDriver, gpuBuilder)
	b.setMemTracer(gpuBuilder)
	b.setISADebugger(gpuBuilder)

	if b.disableProgressBar {
		gpuBuilder.WithoutProgressBar()
	}

	return gpuBuilder
}
