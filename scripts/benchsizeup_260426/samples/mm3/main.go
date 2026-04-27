package main

import (
	"flag"

	_ "net/http/pprof"

	"gitlab.com/akita/mgpusim/benchmarks/polybench/mm3"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var niFlag = flag.Uint("ni", 1024, "The height of the first matrix.")
var njFlag = flag.Uint("nj", 1024, "The height of the first matrix.")
var nkFlag = flag.Uint("nk", 1024, "The height of the first matrix.")
var nlFlag = flag.Uint("nl", 1024, "The height of the first matrix.")
var nmFlag = flag.Uint("nm", 1024, "The height of the first matrix.")

// var niFlag = flag.Uint("ni", 512, "The height of the first matrix.")
// var njFlag = flag.Uint("nj", 512, "The height of the first matrix.")
// var nkFlag = flag.Uint("nk", 512, "The height of the first matrix.")
// var nlFlag = flag.Uint("nl", 512, "The height of the first matrix.")
// var nmFlag = flag.Uint("nm", 512, "The height of the first matrix.")

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()

	benchmark := mm3.NewBenchmark(runner.GPUDriver)
	benchmark.NI = int(*niFlag)
	benchmark.NJ = int(*njFlag)
	benchmark.NK = int(*nkFlag)
	benchmark.NL = int(*nlFlag)
	benchmark.NM = int(*nmFlag)

	runner.AddBenchmark(benchmark)

	runner.Run()
}
