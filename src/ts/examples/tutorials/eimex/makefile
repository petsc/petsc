
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/eimex/
EXAMPLESC       = allen_cahn.c ct_vdp_imex.c
MANSEC          = TS
DIRS            =
CLEANFILES      = ct_vdp_imex allen_cahn eimex_nonstiff_vdp.txt

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ct_vdp_imex: ct_vdp_imex.o  chkopts
	-${CLINKER} -o ct_vdp_imex ct_vdp_imex.o ${PETSC_TS_LIB}
	${RM} ct_vdp_imex.o

allen_cahn: allen_cahn.o  chkopts
	-${CLINKER} -o allen_cahn allen_cahn.o ${PETSC_TS_LIB}
	${RM} allen_cahn.o

runct_vdp_imex:
	-@${MPIEXEC} -n 1 ./ct_vdp_imex -ts_type eimex -ts_adapt_type none -fp_trap -pc_type lu -ts_dt 0.01 -ts_final_time 10 -ts_eimex_row_col 3,3 -ts_monitor_lg_solution > runct_vdp_imex.tmp 2>&1; \
	   ${DIFF} output/runct_vdp_imex.out runct_vdp_imex.tmp || printf "${PWD}\nPossible problem with runct_vdp_imex diffs above\n=========================================\n"; \
        ${RM} -f runct_vdp_imex.tmp

runct_vdp_imex_adapt:
	-@${MPIEXEC} -n 1 ./ct_vdp_imex -ts_type eimex -ts_adapt_type none -fp_trap -pc_type lu -ts_dt 0.01 -ts_final_time 10 -ts_eimex_order_adapt -ts_eimex_max_rows 7 -ts_monitor_lg_solution > runct_vdp_imex_adapt.tmp 2>&1; \
	   ${DIFF} output/runct_vdp_imex_adapt.out runct_vdp_imex_adapt.tmp || printf "${PWD}\nPossible problem with runct_vdp_imex_adapt diffs above\n=========================================\n"; \
        ${RM} -f runct_vdp_imex_adapt.tmp

runct_vdp_imex_conv:
	-@${RM} -f eimex_nonstiff_vdp.txt runct_vdp_imex_conv.tmp; touch runct_vdp_imex_conv.tmp ; \
          for j in 1 2 3; do \
            for i in $$(seq $$j 3); do \
              for dt in 0.005 0.001 0.0005; do \
                 ftime=0.5; \
                 msteps=`echo $$ftime / $$dt | bc`; \
                 ${MPIEXEC} -n 1 ./ct_vdp_imex -ts_type eimex  -ts_adapt_type none -fp_trap -pc_type lu -ts_dt $$dt -ts_max_steps $$msteps -ts_eimex_row_col $$i,$$j >>  runct_vdp_imex_conv.tmp; \
              done; \
            done; \
          done; ${DIFF} output/runct_vdp_imex_conv.out runct_vdp_imex_conv.tmp || printf "${PWD}\nPossible problem with runct_vdp_imex_conv, diffs above\n=========================================\n"; \
        ${RM} -f runct_vdp_imex_conv.tmp

runallen_cahn:
	-@${MPIEXEC} -n 1 ./allen_cahn  -fp_trap -ts_rtol 1e-04 -ts_dt 0.025 -pc_type lu -ksp_error_if_not_converged TRUE  -ts_type eimex -ts_adapt_type none -ts_eimex_order_adapt -ts_eimex_max_rows 7 -ts_monitor_draw_solution > runallen_cahn.tmp 2>&1; \
	   ${DIFF} output/runallen_cahn.out runallen_cahn.tmp || printf "${PWD}\nPossible problem with runallen_cahn diffs above\n=========================================\n"; \
        ${RM} -f runallen_cahn.tmp

TESTEXAMPLES_C_X = ct_vdp_imex.PETSc runct_vdp_imex runct_vdp_imex_adapt runct_vdp_imex_conv ct_vdp_imex.rm allen_cahn.PETSc runallen_cahn allen_cahn.rm

include ${PETSC_DIR}/lib/petsc/conf/test
