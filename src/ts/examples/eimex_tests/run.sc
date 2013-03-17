#!/bin/bash
./allen_cahn -ts_monitor_solution -fp_trap -ts_dt 0.005 -pc_type lu -ksp_error_if_not_converged TRUE
