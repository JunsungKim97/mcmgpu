#!/bin/bash
cd samples
cd stencil2d
./stencil2d -timing -no-progress-bar -report-all -scheduling round-robin -platform-type mi300 -sched-partition Xdiv -row=2048 -col=2048 