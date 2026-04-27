#!/bin/bash
cd samples
cd stencil2d
./stencil2d -timing -no-progress-bar -report-all -scheduling round-robin -platform-type mi300 -mem-group-size 2 -log2-cacheline-size 7 -sched-partition Xdiv -row=2048 -col=2048 