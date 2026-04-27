package builders

import (
	"fmt"

	"gitlab.com/akita/akita"
	"gitlab.com/akita/mem/cache"
	"gitlab.com/akita/mgpusim"
	"gitlab.com/akita/noc/networking/chipnetwork"
)

// MI300GPUBuilder builds a GPU modeled after Instinct MI300 / IdealVM-2 style:
// no L1I/L1S TLB path (nil'd), MCM chiplets, chip RDMA, same SA path as IdealVM2.
type MI300GPUBuilder struct {
	*CommonBuilder
}

// MakeMI300GPUBuilder returns a builder that mirrors IdealVM2GPUBuilder behavior
// (interconnect, CP wiring, mem banks, L1I/STLB nil) for MI300 family experiments.
func MakeMI300GPUBuilder() MI300GPUBuilder {
	cbp := CommonBuilder{}
	b := MI300GPUBuilder{CommonBuilder: &cbp}
	b.SetDefaultCommonBuilderParams()
	return b
}

func (b MI300GPUBuilder) Build(name string, id uint64) *mgpusim.GPU {
	b.createGPU(name, id)

	b.buildCP()

	chipRdmaAddressTable := b.createChipRDMAAddrTable()
	rdmaResponsePorts := make([]akita.Port, b.numChiplet)
	// rdmaResponsePorts := make([]akita.Port, 4)

	//junsung "2026-04-27" pass 1: build all chiplets/components first so group wiring can reference peer L2s.
	for i := 0; i < b.numChiplet; i++ {
		chipletName := fmt.Sprintf("%s.chiplet_%02d", b.gpuName, i)
		chiplet := NewChiplet(chipletName, uint64(i))

		b.BuildSAs(chiplet)
		b.buildXCDCache(chiplet)
		b.buildMemBanks(chiplet)

		b.configChipRDMAEngine(chiplet, chipRdmaAddressTable, rdmaResponsePorts)

		b.chiplets = append(b.chiplets, chiplet)
	}

	//junsung "2026-04-27" pass 2: connect each chiplet L1 to its XCD private cache.
	for _, chiplet := range b.chiplets {
		b.connectL1ToXCDCache(chiplet)
	}

	//junsung "2026-04-27" pass 3: connect grouped XCD caches to grouped L2 banks (mem-group-size locality).
	if b.memGroupSize <= 0 {
		panic("mem-group-size must be >= 1")
	}
	for groupStart := 0; groupStart < b.numChiplet; groupStart += b.memGroupSize {
		groupEnd := groupStart + b.memGroupSize - 1
		b.connectXCDCachesToGroupedL2(groupStart, groupEnd)
	}

	//junsung "2026-04-27" pass 4: each chiplet still connects its L2 to its own DRAM controllers.
	for _, chiplet := range b.chiplets {
		b.connectL2ToDRAM(chiplet)
	}

	b.gpu.L1STLBs = nil
	b.gpu.L1ITLBs = nil

	b.buildPageMigrationController()
	b.setupDMA()

	b.connectCP()
	b.setupInterchipNetwork()
	return b.gpu
}

func (b *MI300GPUBuilder) connectCP() {
	b.internalConn = akita.NewDirectConnection(
		b.gpuName+"InternalConn", b.engine, b.freq)
	b.gpu.InternalConnection = b.internalConn

	b.internalConn.PlugIn(b.cp.ToDriver, 1)
	b.internalConn.PlugIn(b.cp.ToDMA, 128)
	b.internalConn.PlugIn(b.cp.ToCaches, 128)
	b.internalConn.PlugIn(b.cp.ToCUs, 128)
	b.internalConn.PlugIn(b.cp.ToTLBs, 128)
	b.internalConn.PlugIn(b.cp.ToAddressTranslators, 128)
	b.internalConn.PlugIn(b.cp.ToRDMA, 4)
	b.internalConn.PlugIn(b.cp.ToPMC, 4)

	b.internalConn.PlugIn(b.cp.ToRTU, 4)
	b.internalConn.PlugIn(b.cp.ToMMUs, 4)

	b.cp.RDMA = b.rdmaEngine.CtrlPort
	b.internalConn.PlugIn(b.cp.RDMA, 1)

	b.cp.DMAEngine = b.dmaEngine.ToCP
	b.internalConn.PlugIn(b.dmaEngine.ToCP, 1)

	b.cp.PMC = b.pageMigrationController.CtrlPort
	b.internalConn.PlugIn(b.pageMigrationController.CtrlPort, 1)

	b.connectCPWithCUs()
	b.connectCPWithAddressTranslators()
	b.connectCPWithCaches()
}

func (b *MI300GPUBuilder) setupInterchipNetwork() {
	chipConnector := chipnetwork.NewInterChipletConnector().
		WithEngine(b.engine).
		WithSwitchLatency(32). // junsung 32ns
		// WithSwitchLatency(360).
		WithFreq(1 * akita.GHz).
		WithFlitByteSize(64).
		WithNumReqPerCycle(12).
		WithNetworkName("ICN")
	chipConnector.CreateNetwork(b.numChiplet) //junsung chiplet num
	// chipConnector.CreateNetwork()
	for _, chiplet := range b.chiplets {
		chipConnector.PlugInChip(b.InterChipletPorts(chiplet))
	}
	chipConnector.MakeNetwork()
}

func (b *MI300GPUBuilder) InterChipletPorts(c *Chiplet) []akita.Port {
	return []akita.Port{
		c.chipRdmaEngine.RequestPort,
		c.chipRdmaEngine.ResponsePort,
	}
}

func (b *MI300GPUBuilder) connectXCDCachesToGroupedL2(groupStart, groupEnd int) {
	//junsung "2026-04-27" Group-level direct connection lets all XCDs in the group access all group memory dies locally.
	groupConn := akita.NewDirectConnection(
		fmt.Sprintf("%s.group_%02d_to_%02d.XCD-L2", b.gpuName, groupStart, groupEnd),
		b.engine, b.freq)

	groupLowModules := make([]akita.Port, 0, b.memGroupSize*b.numMemoryBankPerChiplet)

	for die := groupStart; die <= groupEnd; die++ {
		chiplet := b.chiplets[die]
		for _, l2 := range chiplet.L2Caches {
			groupLowModules = append(groupLowModules, l2.TopPort)
			groupConn.PlugIn(l2.TopPort, 64)
		}
	}

	localStartBank := uint64(groupStart * b.numMemoryBankPerChiplet)
	localEndBank := uint64((groupEnd+1)*b.numMemoryBankPerChiplet - 1)

	for die := groupStart; die <= groupEnd; die++ {
		chiplet := b.chiplets[die]

		lowModuleFinder := cache.NewStripedLocalVRemoteLowModuleFinder(
			b.memAddrOffset,
			uint64(b.numChiplet*b.numMemoryBankPerChiplet),
			1<<b.log2MemoryBankInterleavingSize,
			localStartBank,
			localEndBank,
		)
		lowModuleFinder.ModuleForOtherAddresses = chiplet.chipRdmaEngine.ToL1
		lowModuleFinder.LowModules = append(lowModuleFinder.LowModules, groupLowModules...)

		chiplet.chipRdmaEngine.SetLocalModuleFinder(lowModuleFinder)
		groupConn.PlugIn(chiplet.chipRdmaEngine.ToL1, 64)
		groupConn.PlugIn(chiplet.chipRdmaEngine.ToL2, 64)

		for _, xcd := range chiplet.XCDCaches {
			xcd.SetLowModuleFinder(lowModuleFinder)
			groupConn.PlugIn(xcd.BottomPort, 64)
		}

		chiplet.lowModuleFinderForL1 = lowModuleFinder
	}
}

// BuildSAs reuses the IdealVM-2 shader-array path (L1V TLB, same CU/L1 graph).
func (b *MI300GPUBuilder) BuildSAs(chiplet *Chiplet) {
	saBuilder := makeIdealVM2ShaderArrayBuilder()
	saBuilder.withEngine(b.engine)
	saBuilder.withFreq(b.freq)
	saBuilder.withGPUID(b.gpu.GPUID)
	saBuilder.withLog2CachelineSize(b.log2CacheLineSize)
	saBuilder.withLog2PageSize(b.log2PageSize)
	saBuilder.withNumCU(b.numCUPerShaderArray)
	saBuilder.withPageTable(b.pageTable)

	if b.enableVisTracing {
		saBuilder.withVisTracer(b.visTracer)
	}

	for i := 0; i < b.numShaderArrayPerChiplet; i++ {
		saName := fmt.Sprintf("%s.SA_%02d", chiplet.name, i)
		sa := saBuilder.Build(saName)
		b.collectSAComponents(sa, chiplet)
	}
	chiplet.L1STLBs = nil
	chiplet.L1ITLBs = nil
}
