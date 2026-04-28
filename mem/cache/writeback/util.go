package writeback

import (
	"gitlab.com/akita/akita"
	"gitlab.com/akita/mem/cache"
)

func getCacheLineID(
	addr uint64,
	blockSizeAsPowerOf2 uint64,
) (cacheLineID, offset uint64) {
	mask := uint64(0xffffffffffffffff << blockSizeAsPowerOf2)
	cacheLineID = addr & mask
	offset = addr & ^mask
	return
}

func bankID(block *cache.Block, wayAssocitivity, numBanks int) int {
	return (block.SetID*wayAssocitivity + block.WayID) % numBanks
}

func sectorSize(log2SectorSize uint64) uint64 { // junsung sector cache
	return uint64(1) << log2SectorSize // junsung sector cache
} // junsung sector cache

func ensureSectorMetadata(block *cache.Block, log2BlockSize, log2SectorSize uint64) { // junsung sector cache
	sectorCount := 1 << (log2BlockSize - log2SectorSize) // junsung sector cache
	if sectorCount <= 0 { // junsung sector cache
		sectorCount = 1 // junsung sector cache
	} // junsung sector cache
	if len(block.SectorValid) != sectorCount { // junsung sector cache
		block.SectorValid = make([]bool, sectorCount) // junsung sector cache
		if block.IsValid { // junsung sector cache
			for i := range block.SectorValid { // junsung sector cache
				block.SectorValid[i] = true // junsung sector cache
			} // junsung sector cache
		} // junsung sector cache
	} // junsung sector cache
	if len(block.SectorDirty) != sectorCount { // junsung sector cache
		block.SectorDirty = make([]bool, sectorCount) // junsung sector cache
		if block.IsDirty { // junsung sector cache
			for i := range block.SectorDirty { // junsung sector cache
				block.SectorDirty[i] = true // junsung sector cache
			} // junsung sector cache
		} // junsung sector cache
	} // junsung sector cache
} // junsung sector cache

func markAllSectorsValid(block *cache.Block, log2BlockSize, log2SectorSize uint64) { // junsung sector cache
	ensureSectorMetadata(block, log2BlockSize, log2SectorSize) // junsung sector cache
	for i := range block.SectorValid { // junsung sector cache
		block.SectorValid[i] = true // junsung sector cache
	} // junsung sector cache
} // junsung sector cache

func touchedSectorRange(offset, size uint64, log2SectorSize uint64) (int, int) { // junsung sector cache
	secSize := sectorSize(log2SectorSize) // junsung sector cache
	start := int(offset / secSize) // junsung sector cache
	end := int((offset + size - 1) / secSize) // junsung sector cache
	return start, end // junsung sector cache
} // junsung sector cache

func sectorsValidForAccess(block *cache.Block, offset, size, log2BlockSize, log2SectorSize uint64) bool { // junsung sector cache
	ensureSectorMetadata(block, log2BlockSize, log2SectorSize) // junsung sector cache
	start, end := touchedSectorRange(offset, size, log2SectorSize) // junsung sector cache
	for i := start; i <= end; i++ { // junsung sector cache
		if i < 0 || i >= len(block.SectorValid) || !block.SectorValid[i] { // junsung sector cache
			return false // junsung sector cache
		} // junsung sector cache
	} // junsung sector cache
	return true // junsung sector cache
} // junsung sector cache

func markSectorsValid(block *cache.Block, offset, size, log2BlockSize, log2SectorSize uint64) { // junsung sector cache
	ensureSectorMetadata(block, log2BlockSize, log2SectorSize) // junsung sector cache
	start, end := touchedSectorRange(offset, size, log2SectorSize) // junsung sector cache
	for i := start; i <= end; i++ { // junsung sector cache
		if i >= 0 && i < len(block.SectorValid) { // junsung sector cache
			block.SectorValid[i] = true // junsung sector cache
		} // junsung sector cache
	} // junsung sector cache
} // junsung sector cache

func markSectorsDirty(block *cache.Block, offset, size, log2BlockSize, log2SectorSize uint64) { // junsung sector cache
	ensureSectorMetadata(block, log2BlockSize, log2SectorSize) // junsung sector cache
	start, end := touchedSectorRange(offset, size, log2SectorSize) // junsung sector cache
	for i := start; i <= end; i++ { // junsung sector cache
		if i >= 0 && i < len(block.SectorDirty) { // junsung sector cache
			block.SectorDirty[i] = true // junsung sector cache
		} // junsung sector cache
	} // junsung sector cache
} // junsung sector cache

func refreshLineDirtyFromSectors(block *cache.Block) { // junsung sector cache
	block.IsDirty = false // junsung sector cache
	for _, dirty := range block.SectorDirty { // junsung sector cache
		if dirty { // junsung sector cache
			block.IsDirty = true // junsung sector cache
			return // junsung sector cache
		} // junsung sector cache
	} // junsung sector cache
} // junsung sector cache

func clearPort(p akita.Port, now akita.VTimeInSec) {
	for {
		item := p.Retrieve(now)
		if item == nil {
			return
		}
	}
}
