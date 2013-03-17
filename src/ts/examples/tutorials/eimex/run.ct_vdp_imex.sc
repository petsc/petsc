#!/bin/bash

for ((j=1;j<=4;j++))
do
  for((i=$j;i<=4;i++))
  do
    for dt in 0.005 0.001 0.0005
    do
      ftime=0.5
      msteps=`echo $ftime / $dt | bc`
      ./ct_vdp_imex -ts_type eimex  -ts_adapt_type none -fp_trap -pc_type lu -ts_dt $dt -ts_max_steps $msteps -ts_eimex_row_col $i,$j
    done
  done
done
