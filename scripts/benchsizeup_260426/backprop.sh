#!/bin/bash
cd samples
cd backprop
./backprop -timing -no-progress-bar -report-all -scheduling round-robin -platform-type mi300 -mem-group-size 2 -log2-cacheline-size 7 