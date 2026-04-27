package main

import (
	"flag"

	"gitlab.com/akita/mgpusim/benchmarks/tartan/pennant"
	"gitlab.com/akita/mgpusim/samples/runner"
)

var zoneFlag = flag.Int("zones", 65536, "Number of zones")
var pointFlag = flag.Int("points", 131072, "Number of points")

// var zoneFlag = flag.Int("zones", 4096, "Number of zones")
// var pointFlag = flag.Int("points", 8192, "Number of points")
var minPtsFlag = flag.Int("min-pts", 3, "Min points per zone")
var maxPtsFlag = flag.Int("max-pts", 6, "Max points per zone")
var iterFlag = flag.Int("iterations", 6, "Number of update iterations")
var dtFlag = flag.Float64("dt", 0.01, "Time step")
var gammaFlag = flag.Float64("gamma", 1.4, "EOS gamma")

func main() {
	flag.Parse()

	r := new(runner.Runner).ParseFlag().Init()
	b := pennant.NewBenchmark(r.GPUDriver)
	b.NumZones = *zoneFlag
	b.NumPoints = *pointFlag
	b.MinPtsPerZone = *minPtsFlag
	b.MaxPtsPerZone = *maxPtsFlag
	b.Iterations = *iterFlag
	b.Dt = float32(*dtFlag)
	b.Gamma = float32(*gammaFlag)

	r.AddBenchmark(b)
	r.Run()
}
