package main

import (
	"flag"

	tartanpagerank "gitlab.com/akita/mgpusim/benchmarks/tartan/pagerank"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var nodeFlag = flag.Int("node", 16384, "Number of nodes")
var edgesPerNodeFlag = flag.Int("edges-per-node", 8, "Incoming edges per node")
var iterFlag = flag.Int("iterations", 10, "Number of synchronous iterations")
var dampingFlag = flag.Float64("damping", 0.85, "PageRank damping factor")

func main() {
	flag.Parse()

	r := new(runner.Runner).ParseFlag().Init()
	bench := tartanpagerank.NewBenchmark(r.GPUDriver)
	bench.NumNodes = *nodeFlag
	bench.EdgesPerNode = *edgesPerNodeFlag
	bench.Iterations = *iterFlag
	bench.Damping = float32(*dampingFlag)

	r.AddBenchmark(bench)
	r.Run()
}
