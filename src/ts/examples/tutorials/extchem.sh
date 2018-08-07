#! /bin/sh -f
#
#  Runs several extchem mechanisms with various integrators
#
for mech in h2_10sp.inp gasoline.inp meth_ethanol.inp;
do
    for type in arkimex "bdf -ts_adapt_type basic -ts_adapt_dt_max 1.e-5" sundials radau5;
    do
        echo ${mech} ${type}
        ./extchem -options_file ${mech} -ts_max_steps 10000 -ts_type ${type}  -ts_view -log_view -ts_monitor_lg_timestep -ts_monitor_lg_solution -lg_use_markers true -draw_pause -2 -draw_size .5,.5 > extchem_output_${mech}_${type} 2>&1
    done
done
