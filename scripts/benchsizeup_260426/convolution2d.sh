#!/bin/bash
cd samples
cd convolution2d
./convolution2d -timing -no-progress-bar -report-all -scheduling round-robin -platform-type mi300 -mem-group-size 2 -log2-cacheline-size 7 -sched-partition Ydiv -ni=8192 -nj=8192 