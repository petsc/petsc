
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/eimex/
EXAMPLESC       = allen_cahn.c ct_vdp_imex.c
MANSEC          = TS
DIRS            =

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ct_vdp_imex: ct_vdp_imex.o  chkopts
	-${CLINKER} -o ct_vdp_imex ct_vdp_imex.o ${PETSC_TS_LIB}
	${RM} ct_vdp_imex.o

allen_cahn: allen_cahn.o  chkopts
	-${CLINKER} -o allen_cahn allen_cahn.o ${PETSC_TS_LIB}
	${RM} allen_cahn.o

runct_vdp_imex:
	-@for j in 1 2 3; do \
            for i in $$(seq $$j 3); do \
              for dt in 0.005 0.001 0.0005; do \
                 ftime=0.5; \
                 msteps=`echo $$ftime / $$dt | bc`; \
                 ${MPIEXEC} -n 1 ./ct_vdp_imex -ts_type eimex  -ts_adapt_type none -fp_trap -pc_type lu -ts_dt $dt -ts_max_steps $$msteps -ts_eimex_row_col $$i,$$j; \
              done; \
            done; \
          done

runallen_cahn:
	-@${MPIEXEC} -n 1 ./allen_cahn -ts_monitor -fp_trap -ts_dt 0.005 -pc_type lu -ksp_error_if_not_converged TRUE

include ${PETSC_DIR}/conf/test
