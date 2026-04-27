#!/bin/bash
cd samples
cd jacobi1d
./jacobi1d -timing -no-progress-bar -report-all -scheduling round-robin -platform-type i2 -sched-partition Xdiv -n=67108864 -steps=1