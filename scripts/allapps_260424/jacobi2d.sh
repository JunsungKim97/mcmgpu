#!/bin/bash
cd samples
cd jacobi2d
./jacobi2d -timing -no-progress-bar -report-all -scheduling round-robin -platform-type i2 -sched-partition Ydiv -n=4096 -steps=1