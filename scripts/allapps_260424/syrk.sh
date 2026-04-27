#!/bin/bash
cd samples
cd syrk
./syrk -timing -no-progress-bar -report-all -scheduling round-robin -platform-type i2 -sched-partition Xdiv -max-inst 10000000 -ni=2048 -nj=2048 