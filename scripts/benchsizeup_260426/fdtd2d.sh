#!/bin/bash
cd samples
cd fdtd2d
./fdtd2d -timing -no-progress-bar -report-all -scheduling round-robin -platform-type mi300 -mem-group-size 2 -log2-cacheline-size 7 