
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



TESTEXAMPLES_C_X = ct_vdp_imex.PETSc runct_vdp_imex_conv ct_vdp_imex.rm 

include ${PETSC_DIR}/lib/petsc/conf/test
