#!/bin/bash
cd samples
cd simpleconvolution
./simpleconvolution -timing -no-progress-bar -report-all -scheduling round-robin -platform-type mi300 -mem-group-size 2 -log2-cacheline-size 7 -sched-partition Xdiv -width=8190 -height=8190 