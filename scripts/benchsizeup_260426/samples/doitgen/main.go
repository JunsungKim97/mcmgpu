package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/polybench/doitgen"
	"gitlab.com/akita/mgpusim/samples/runner"
)

// 1GB
// var rFlag = flag.Int("r", 512, "Dunno")
// var qFlag = flag.Int("q", 512, "Dunno")
// var pFlag = flag.Int("p", 512, "Dunno")

var rFlag = flag.Int("r", 1024, "Dunno")
var qFlag = flag.Int("q", 1024, "Dunno")
var pFlag = flag.Int("p", 1024, "Dunno")

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()

	benchmark := doitgen.NewBenchmark(runner.GPUDriver)
	benchmark.NR = *rFlag
	benchmark.NQ = *qFlag
	benchmark.NP = *pFlag

	runner.AddBenchmark(benchmark)

	runner.Run()
}
