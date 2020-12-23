-include ../../../../../petscdir.mk

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/utils/dmplexlandau/tutorials/
EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       = ex1f90.F90
MANSEC          = TS
SUBMANSECi      = LANDAU

runex2_cuda:
	${MPIEXEC} -n 1 ./ex2 -ex2_test_type spitzer -ex2_connor_e_field_units -dm_landau_Ez 0 -petscspace_degree 3 -dm_preallocate_only  -dm_landau_type p4est -dm_landau_ion_masses 2,182,182,182,182,182,182,182,2 -dm_landau_ion_charges 1,1,1,1,1,1,1,1,1 -dm_landau_thermal_temps 4,4,4,4,4,4,4,4,.05,4 -dm_landau_n 1.0000008,1,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7 -ts_monitor -ts_adapt_monitor -snes_rtol 1.e-9 -snes_stol 1.e-9 -snes_converged_reason -snes_max_it 25 -ts_type arkimex -ts_exact_final_time stepover -ts_arkimex_type 1bee -ts_max_snes_failures -1 -ts_rtol 1e-2 -ts_dt 1 -ts_adapt_clip .5,1.1 -ts_adapt_dt_max .5 -ts_adapt_dt_min 1e-3 -ts_max_time 150 -ts_max_steps 1 -ts_adapt_scale_solve_failed 0.5 -ts_adapt_time_step_increase_delay 5 -pc_type lu -ksp_type preonly -dm_landau_amr_levels_max 15 -dm_landau_amr_post_refine 0 -dm_landau_domain_radius 6 -ex2_plot_dt .5 -ex2_impurity_source_type pulse -ex2_pulse_start_time 1000 -ex2_pulse_width_time 15 -ex2_pulse_rate 2 -ex2_t_cold .05 -dm_landau_device_type cuda -dm_mat_type aijcusparse -dm_vec_type cuda -log_view

runex2_kokkos:
	${MPIEXEC} -n 1 ./ex2 -ex2_test_type spitzer -ex2_connor_e_field_units -dm_landau_Ez 0 -petscspace_degree 3 -dm_preallocate_only  -dm_landau_type p4est -dm_landau_ion_masses 2,182,182,182,182,182,182,182,2 -dm_landau_ion_charges 1,1,1,1,1,1,1,1,1 -dm_landau_thermal_temps 4,4,4,4,4,4,4,4,.05,4 -dm_landau_n 1.0000008,1,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7 -ts_monitor -ts_adapt_monitor -snes_rtol 1.e-9 -snes_stol 1.e-9 -snes_converged_reason -snes_max_it 25 -ts_type arkimex -ts_exact_final_time stepover -ts_arkimex_type 1bee -ts_max_snes_failures -1 -ts_rtol 1e-2 -ts_dt 1 -ts_adapt_clip .5,1.1 -ts_adapt_dt_max .5 -ts_adapt_dt_min 1e-3 -ts_max_time 150 -ts_max_steps 1 -ts_adapt_scale_solve_failed 0.5 -ts_adapt_time_step_increase_delay 5 -pc_type lu -ksp_type preonly -dm_landau_amr_levels_max 15 -dm_landau_amr_post_refine 0 -dm_landau_domain_radius 6 -ex2_plot_dt .5 -ex2_impurity_source_type pulse -ex2_pulse_start_time 1000 -ex2_pulse_width_time 15 -ex2_pulse_rate 2 -ex2_t_cold .05 -dm_landau_sub_thread_block_size 8 -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -log_view

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
