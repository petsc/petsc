# $Id: makefile,v 1.70 2000/03/04 15:48:31 bsmith Exp bsmith $ 

ALL: ex1

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/snes/examples/tutorials/

TESTEXAMPLES_1	 = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2  runex2_3 ex2.rm \
		   ex3.PETSc runex3 runex3_2 runex3_3 ex3.rm ex6.PETSc runex6 runex6_2 \
		   ex6.rm ex4.PETSc runex4 runex4_2 runex4_3 ex4.rm ex5.PETSc  \
                   ex5.rm ex5d.PETSc ex5d.rm ex8.PETSc runex8 ex8.rm ex12.PETSc ex12.rm \
                   ex14.PETSc runex14 runex14_2 ex14.rm ex15.PETSc ex15.rm \
                   ex9.PETSc ex9.rm clean
TESTEXAMPLES_2	 = ex5.PETSc runex5_8 runex5_9 \
		   runex5_10 ex5.rm ex5d.PETSc runex5d_8 runex5d_9 \
		   runex5d_10 ex5d.rm ex8.PETSc runex8_2 ex8.rm clean
TESTEXAMPLES_3	 = ex1f.PETSc runex1f ex1f.rm ex4f.PETSc runex4f_2 runex4f_3 ex4f.rm clean
TESTEXAMPLES_6   = ex16.PETSc ex16.rm clean
TESTEXAMPLES_8	 = ex5f.PETSc runex5f runex5f_2 runex5f_3 ex5f.rm clean
TESTEXAMPLES_9	 = ex1f.PETSc runex1f ex1f.rm ex4f.PETSc runex4f ex4f.rm clean
TESTEXAMPLES_4	 = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc \
		   runex3 ex3.rm ex6.PETSc runex6 runex6_2 ex6.rm \
		   ex4.PETSc runex4 runex4_2 runex4_3 ex4.rm ex5.PETSc ex5.rm \
                   ex14.PETSc ex14.rm ex15.PETSc ex15.rm clean
TESTEXAMPLES_6   = ex16.PETSc runex16 runex16_2 ex16.rm 
TESTEXAMPLES_12	 = ex5f90.PETSc runex5f90 runex5f90_2 runex5f90_3 ex5f90.rm clean
TESTEXAMPLES_13  = ex7.PETSc ex7.rm ex13.PETSc ex13.rm clean
TESTEXAMPLES_15  = ex12.PETSc runex12 runex12_2 ex12.rm clean
BUILDEXAMPLES_1	 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm ex6.PETSc \
		   ex6.rm ex4.PETSc ex4.rm ex5.PETSc ex5.rm ex8.PETSc ex8.rm ex16.PETSc ex16.rm clean \
		   ex12.PETSc ex12.rm clean
BUILDEXAMPLES_2	 = ex3.PETSc ex3.rm clean
BUILDEXAMPLES_3	 = ex1f.PETSc ex1f.rm clean
BUILDEXAMPLES_4	 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm ex6.PETSc \
		   ex6.rm ex4.PETSc ex4.rm ex5.PETSc ex5.rm ex8.PETSc ex8.rm clean
BUILDEXAMPLES_8	 = ex5f.PETSc ex5f.rm clean
BUILDEXAMPLES_12 = ex5f90.PETSc ex5f90.rm clean
EXAMPLESC	 = ex1.c ex2.c ex3.c ex4.c ex5.c ex5c.c ex6.c ex7.c ex8.c ex5s.c ex9.c \
                   common8and9.c ex12.c ex13.c ex14.c ex15.c ex16.c
EXAMPLESF	 = ex1f.F ex4f.F ex5f.F ex5f90.F ex5fs.F 
EXAMPLESCH	 = ex8and9.h 
EXAMPLESFH       = ex5f.h
DOCS		 = readme
DIRS		 = ex10d

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_SNES_LIB}
	${RM} ex1.o
ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o ${PETSC_FORTRAN_LIB} ${PETSC_SNES_LIB}
	${RM} ex1f.o
ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_SNES_LIB}
	${RM} ex2.o
ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_SNES_LIB}
	${RM} ex3.o
ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_SNES_LIB}
	${RM} ex4.o
ex4f: ex4f.o  chkopts
	-${FLINKER} -o ex4f ex4f.o ${PETSC_FORTRAN_LIB} ${PETSC_SNES_LIB}
	${RM} ex4f.o
ex5: ex5.o chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_SNES_LIB}
	${RM} ex5.o
ex5c: ex5c.o chkopts
	-${CLINKER} -o ex5c ex5c.o ${PETSC_SNES_LIB}
	${RM} ex5c.o
ex5d: ex5d.o chkopts
	-${CLINKER} -o ex5d ex5d.o ${PETSC_SNES_LIB}
	${RM} ex5d.o
ex5f: ex5f.o  chkopts
	-${FLINKER} -o ex5f ex5f.o ${PETSC_FORTRAN_LIB} ${PETSC_SNES_LIB}
	${RM} ex5f.o
#
#  The SGI parallelizing compiler generates incorrect code by treating 
#  the math functions (such as sqrt and exp) as local variables. The 
#  sed below patches this.
#
ex5s: chkopts
	-${CC} -pca keep  -WK,-lo=l ${FCONF} ${CFLAGS} -c ex5s.c
	sed "s/, sqrt/ /g"   ex5s.M | sed "s/, exp/ /g"  > ex5s_tmp.c
	-${CC} -mp ${COPTFLAGS} ${CFLAGS} ${CCPPFLAGS} -c ex5s_tmp.c
	-${FC} -pfa keep -mp -64 ${FOPTFLAGS} ${FFLAGS} ${FCPPFLAGS} -c ex5fs.F
	-${CLINKER} -mp -o ex5s ex5s_tmp.o ex5fs.o ${PETSC_SNES_LIB}
	${RM} ex5s.o
ex5f90: ex5f90.o  chkopts
	-${FLINKER} -o ex5f90 ex5f90.o ${PETSC_FORTRAN_LIB} ${PETSC_SNES_LIB}
	${RM} ex5f90.o
ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_SNES_LIB}
	${RM} ex6.o
ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o ${PETSC_SNES_LIB}
	${RM} ex7.o
ex8: ex8.o common8and9.o chkopts
	-${CLINKER} -o ex8 ex8.o common8and9.o ${PETSC_SNES_LIB}
	${RM} ex8.o common8and9.o
ex9: ex9.o common8and9.o chkopts
	-${CLINKER} -o ex9 ex9.o common8and9.o ${PETSC_SNES_LIB}
	${RM} ex9.o common8and9.o
ex12: ex12.o chkopts
	-${CLINKER} -o ex12 ex12.o ${PETSC_SNES_LIB}
	${RM} ex12.o
ex13: ex13.o chkopts
	-${CLINKER} -o ex13 ex13.o ${PETSC_SNES_LIB}
	${RM} ex13.o
ex14: ex14.o chkopts
	-${CLINKER} -o ex14 ex14.o ${PETSC_SNES_LIB}
	${RM} ex14.o
ex15: ex15.o chkopts
	-${CLINKER} -o ex15 ex15.o ${PETSC_SNES_LIB}
	${RM} ex15.o
ex16: ex16.o chkopts
	-${CLINKER} -o ex16 ex16.o ${PETSC_SNES_LIB}
	${RM} ex16.o
runex1:
	-@${MPIRUN} -np 1 ex1 -ksp_gmres_irorthog -snes_smonitor > ex1_1.tmp 2>&1;   \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp
runex1f:
	-@${MPIRUN} -np 1 ex1f -ksp_gmres_irorthog -snes_smonitor > ex1f_1.tmp 2>&1;   \
	   if (diff output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   ${RM} -f ex1f_1.tmp
runex2:
	-@${MPIRUN} -np 1 ex2 -nox -snes_cancelmonitors -snes_smonitor \
		 -snes_view -pc_type jacobi -ksp_gmres_irorthog >ex2_1.tmp 2>&1;	\
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp
runex2_2:
	-@${MPIRUN} -np 1 ex2 -nox -snes_cancelmonitors -snes_smonitor \
		-snes_type tr -snes_view \
		-pc_type jacobi -ksp_gmres_irorthog > ex2_2.tmp 2>&1; \
	   if (diff output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi; \
	   ${RM} -f ex2_2.tmp
runex2_3:
	-@${MPIRUN} -np 1 ex2 -nox -snes_cancelmonitors -snes_smonitor -trmalloc_off \
		 -snes_view -pc_type jacobi -ksp_gmres_irorthog >ex2_3.tmp 2>&1;	\
	   if (diff output/ex2_1.out ex2_3.tmp) then true; \
	   else echo "Possible problem with ex2_3, diffs above"; fi; \
	   ${RM} -f ex2_3.tmp
runex3:
	-@${MPIRUN} -np 1 ex3 -nox -snes_cancelmonitors -snes_smonitor -ksp_gmres_irorthog > ex3_1.tmp 2>&1;	  \
	   if (diff output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIRUN} -np 3 ex3 -nox -pc_type asm -mat_mpiaij -mat_aij_oneindex \
	   -snes_cancelmonitors -snes_smonitor -ksp_gmres_irorthog > ex3_2.tmp 2>&1; \
	   if (diff output/ex3_2.out ex3_2.tmp) then true; \
	   else echo "Possible problem with ex3_2, diffs above"; fi; \
	   ${RM} -f ex3_2.tmp
runex3_3:
	-@${MPIRUN} -np 2 ex3 -nox -snes_cancelmonitors -snes_smonitor -ksp_gmres_irorthog > ex3_3.tmp 2>&1; \
	   if (diff output/ex3_3.out ex3_3.tmp) then true; \
	   else echo "Possible problem with ex3_3, diffs above"; fi; \
	   ${RM} -f ex3_3.tmp
runex4:
	-@${MPIRUN} -np 1 ex4 -nox -ksp_gmres_irorthog > ex4_1.tmp 2>&1;	 \
	   if (diff output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   ${RM} -f ex4_1.tmp
runex4_2:
	-@${MPIRUN} -np 1 ex4 -snes_smonitor -nox -snes_type tr -ksp_gmres_irorthog > ex4_2.tmp 2>&1;\
	   if (diff output/ex4_2.out ex4_2.tmp) then true; \
	   else echo "Possible problem with ex4_2, diffs above"; fi; \
	   ${RM} -f ex4_2.tmp
runex4_3:
	-@${MPIRUN} -np 1 ex4 -snes_smonitor -mat_coloring_type sl -nox -snes_fd_coloring -mx 8 -my 11 \
                -ksp_gmres_irorthog > ex4_3.tmp 2>&1;\
	   if (diff output/ex4_3.out ex4_3.tmp) then true; \
	   else echo "Possible problem with ex4_3, diffs above"; fi; \
	   ${RM} -f ex4_3.tmp
runex4f:
	-@${MPIRUN} -np 1 ex4f -snes_smonitor -nox -snes_type tr -ksp_gmres_irorthog > ex4f_1.tmp 2>&1;\
	   if (diff output/ex4f_1.out ex4f_1.tmp) then true; \
	   else echo "Possible problem with ex4f_1, diffs above"; fi; \
	   ${RM} -f ex4f_1.tmp
runex4f_2:
	-@${MPIRUN} -np 1 ex4f -snes_smonitor -nox -snes_fd -ksp_gmres_irorthog > ex4f_2.tmp 2>&1;\
	   if (diff output/ex4f_2.out ex4f_2.tmp) then true; \
	   else echo "Possible problem with ex4f_2, diffs above"; fi; \
	   ${RM} -f ex4f_2.tmp
runex4f_3:
	-@${MPIRUN} -np 1 ex4f -snes_smonitor -nox -snes_fd_coloring -ksp_gmres_irorthog > ex4f_3.tmp 2>&1;\
	   if (diff output/ex4f_2.out ex4f_3.tmp) then true; \
	   else echo "Possible problem with ex4f_3, diffs above"; fi; \
	   ${RM} -f ex4f_3.tmp
runex5:
	-@${MPIRUN} -np 4 ex5 -snes_mf -Nx 4 -Ny 1 -snes_smonitor -ksp_gmres_irorthog > ex5_1.tmp 2>&1; \
	   if (diff output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp
runex5_2:
	-@${MPIRUN} -np 4 ex5 -Nx 2 -Ny 2 -snes_smonitor -ksp_gmres_irorthog > ex5_2.tmp 2>&1; \
	   if (diff output/ex5_2.out ex5_2.tmp) then true; \
	   else echo "Possible problem with ex5_2, diffs above"; fi; \
	   ${RM} -f ex5_2.tmp
runex5_3:
	-@${MPIRUN} -np 4 ex5 -snes_mf -Nx 2 -Ny 2 -snes_smonitor-ksp_gmres_irorthog > ex5_3.tmp 2>&1;\
	   if (diff output/ex5_1.out ex5_3.tmp) then true; \
	   else echo "Possible problem with ex5_3, diffs above"; fi; \
	   ${RM} -f ex5_3.tmp
runex5_4:
	-@${MPIRUN} -np 4 ex5 -Nx 2 -Ny 2 -snes_smonitor -pc_type asm \
	   -pc_asm_blocks 4 -pc_asm_overlap 1 -ksp_gmres_irorthog > ex5_4.tmp 2>&1; \
	   if (diff output/ex5_4.out ex5_4.tmp) then true; \
	   else echo "Possible problem with ex5_4, diffs above"; fi; \
	   ${RM} -f ex5_4.tmp
runex5_5:
	-@${MPIRUN} -np 4 ex5 -Nx 2 -Ny 2 -snes_smonitor -pc_type bjacobi \
	   -snes_view -ksp_gmres_irorthog > ex5_5.tmp 2>&1; \
	   if (diff output/ex5_5.out ex5_5.tmp) then true; \
	   else echo "Possible problem with ex5_5, diffs above"; fi; \
	   ${RM} -f ex5_5.tmp
runex5_6:
	-@${MPIRUN} -np 4 ex5 -snes_fd -Nx 2 -Ny 2 -snes_smonitor -ksp_gmres_irorthog > ex5_6.tmp 2>&1; \
	   if (diff output/ex5_6.out ex5_6.tmp) then true; \
	   else echo "Possible problem with ex5_6, diffs above"; fi; \
	   ${RM} -f ex5_6.tmp
runex5_7:
	-@${MPIRUN} -np 3 ex5 -snes_fd  -snes_smonitor -ksp_gmres_irorthog > ex5_7.tmp 2>&1;\
	   if (diff output/ex5_7.out ex5_7.tmp) then true; \
	   else echo "Possible problem with ex5_7, diffs above"; fi; \
	   ${RM} -f ex5_7.tmp
runex5_8:
	-@${MPIRUN} -np 2 ex5 -snes_vecmonitor -ksp_gmres_irorthog > /dev/null 2>&1
runex5_9:
	-@${MPIRUN} -np 4 ex5 -snes_mf -mx 30 -my 30 -snes_smonitor>ex5_9.tmp 2>&1;\
	   if (diff output/ex5_9.out ex5_9.tmp) then true; \
	   else echo "Possible problem with ex5_9, diffs above"; fi; \
	   ${RM} -f ex5_9.tmp
runex5_10:
	-@${MPIRUN} -np 2 ex5 -sles_diagonal_scale -ksp_smonitor -snes_smonitor -mx 12 -my 11-ksp_gmres_irorthog > ex5_10.tmp 2>&1;\
	   if (diff output/ex5_10.out ex5_10.tmp) then true; \
	   else echo "Possible problem with ex5_10, diffs above"; fi; \
	   ${RM} -f ex5_10.tmp
runex5d_8:
	-@${MPIRUN} -np 2 ex5d -snes_vecmonitor -ksp_gmres_irorthog > /dev/null 2>&1
runex5d_9:
	-@${MPIRUN} -np 4 ex5d -snes_mf -mx 30 -my 30 -snes_smonitor -ksp_gmres_irorthog > ex5d_9.tmp 2>&1;\
	   if (diff output/ex5_9.out ex5d_9.tmp) then true; \
	   else echo "Possible problem with ex5d_9, diffs above"; fi; \
	   ${RM} -f ex5d_9.tmp
runex5d_10:
	-@${MPIRUN} -np 2 ex5d -sles_diagonal_scale -ksp_smonitor -snes_smonitor -mx 12 -my 11 -ksp_gmres_irorthog > ex5d_10.tmp 2>&1;\
	   if (diff output/ex5_10.out ex5d_10.tmp) then true; \
	   else echo "Possible problem with ex5d_10, diffs above"; fi; \
	   ${RM} -f ex5d_10.tmp
runex5f:
	-@${MPIRUN} -np 4 ex5f -snes_mf -Nx 4 -Ny 1 -snes_smonitor -ksp_gmres_irorthog > ex5f_1.tmp 2>&1; \
	   if (diff output/ex5f_1.out ex5f_1.tmp) then true; \
	   else echo "Possible problem with ex5f_1, diffs above"; fi; \
	   ${RM} -f ex5f_1.tmp
runex5f_2:
	-@${MPIRUN} -np 4 ex5f  -Nx 2 -Ny 2 -snes_smonitor -ksp_gmres_irorthog > ex5f_2.tmp 2>&1; \
	   if (diff output/ex5f_2.out ex5f_2.tmp) then true; \
	   else echo "Possible problem with ex5f_2, diffs above"; fi; \
	   ${RM} -f ex5f_2.tmp
runex5f_3:
	-@${MPIRUN} -np 3 ex5f -snes_fd  -snes_smonitor -ksp_gmres_irorthog > ex5f_3.tmp 2>&1;\
	   if (diff output/ex5f_3.out ex5f_3.tmp) then true; \
	   else echo "Possible problem with ex5f_3, diffs above"; fi; \
	   ${RM} -f ex5f_3.tmp
runex5f90:
	-@${MPIRUN} -np 4 ex5f90 -snes_mf -Nx 4 -Ny 1 -snes_smonitor -ksp_gmres_irorthog > ex5f90_1.tmp 2>&1; \
	   if (diff output/ex5f_1.out ex5f90_1.tmp) then true; \
	   else echo "Possible problem with ex5f90_1, diffs above"; fi; \
	   ${RM} -f ex5f90_1.tmp
runex5f90_2:
	-@${MPIRUN} -np 4 ex5f90 -Nx 2 -Ny 2 -snes_smonitor -ksp_gmres_irorthog > ex5f90_2.tmp 2>&1; \
	   if (diff output/ex5f_2.out ex5f90_2.tmp) then true; \
	   else echo "Possible problem with ex5f90_2, diffs above"; fi; \
	   ${RM} -f ex5f90_2.tmp
runex5f90_3:
	-@${MPIRUN} -np 3 ex5f90 -snes_fd  -snes_smonitor -ksp_gmres_irorthog > ex5f90_3.tmp 2>&1;\
	   if (diff output/ex5f_3.out ex5f90_3.tmp) then true; \
	   else echo "Possible problem with ex5f90_3, diffs above"; fi; \
	   ${RM} -f ex5f90_3.tmp
runex6:
	-@${MPIRUN} -np 1 ex6 -snes_smonitor -ksp_gmres_irorthog > ex6_1.tmp 2>&1;   \
	   if (diff output/ex6_1.out ex6_1.tmp) then true; \
	   else echo "Possible problem with ex6_1, diffs above"; fi; \
	   ${RM} -f ex6_1.tmp
runex6_2:
	-@${MPIRUN} -np 1 ex6 -snes_mf -snes_smonitor -ksp_gmres_irorthog > ex6_2.tmp 2>&1;   \
	   if (diff output/ex6_2.out ex6_2.tmp) then true; \
	   else echo "Possible problem with ex6_2, diffs above"; fi; \
	   ${RM} -f ex6_2.tmp
runex8:
	-@${MPIRUN} -np 2 ex8  -mx 16 -my 16 -snes_smonitor -ksp_gmres_irorthog \
                    -lidvelocity 0. -grashof 1000. > ex8_1.tmp 2>&1; \
           if (diff output/ex8_1.out ex8_1.tmp) then true; \
           else echo "Possible problem with ex8_1, diffs above"; fi; \
           ${RM} -f ex8_1.tmp
runex8_2:
	-@${MPIRUN} -np 2 ex8 -snes_vecmonitor -ksp_gmres_irorthog > /dev/null 2>&1 
runex9:
	-@${MPIRUN} -np 4 ex9 -nox -mx 4 -my 4 -nlevels 4 -snes_smonitor \
                    -lidvelocity 0. -grashof 1000. -ksp_gmres_irorthog > ex9_1.tmp 2>&1; \
           if (diff output/ex9_1.out ex9_1.tmp) then true; \
           else echo "Possible problem with ex9_1, diffs above"; fi; \
           ${RM} -f ex9_1.tmp
runex9_2:
	-@${MPIRUN} -np 6 ex9 -nox -Nx 3 -Ny 2 -mx 7 -my 7 -nlevels 3 -snes_smonitor \
                    -lidvelocity 10. -grashof 1000. -ksp_gmres_irorthog > ex9_2.tmp 2>&1; \
           if (diff output/ex9_2.out ex9_2.tmp) then true; \
           else echo "Possible problem with ex9_2, diffs above"; fi; \
           ${RM} -f ex9_2.tmp
runex12:
	-@echo "path(path,['${PETSC_DIR}/src/sys/src/viewer/impls/socket/matlab']); path(path,['${AMS_HOME}/matlab']); ex12(1) ; exit" | ${MATLABCOMMAND} -display maverick:0.0 > ex12_1.tmp 2>&1; \
                egrep '(Unable To Connect|Warning)' ex12_1.tmp ; \
                if [ "$$?" != 1 ]; then cat ex12_1.tmp; echo "Possible problem with AMS MEMORY SNOOPER ex12_1"; fi; \
	        ${RM} -f ex12_1.tmp
runex12_2:
	-@echo "path(path,['${PETSC_DIR}/src/sys/src/viewer/impls/socket/matlab']); path(path,['${AMS_HOME}/matlab']); ex12(2) ; exit" | ${MATLABCOMMAND} -display maverick:0.0 > ex12_2.tmp 2>&1; \
            egrep '(Unable To Connect|Warning)' ex12_2.tmp ; \
              if [ "$$?" != 1 ]; then cat ex12_2.tmp; echo "Possible problem with AMS MEMORY SNOOPER ex12_2"; fi; \
         ${RM} -f ex12_2.tmp ex12_2.tmp2
runex14:
	-@${MPIRUN} -np 4 ex14 -Nx 2 -Ny 2 -Nz 1 -mx 8 -my 8 -mz 8 -snes_smonitor -ksp_gmres_irorthog > ex14_1.tmp 2>&1; \
	   if (diff output/ex14_1.out ex14_1.tmp) then true; \
	   else echo "Possible problem with ex14_1, diffs above"; fi; \
	   ${RM} -f ex14_1.tmp
runex14_2:
	-@${MPIRUN} -np 4 ex14 -Nx 2 -Ny 1 -Nz 2 -mx 8 -my 8 -mz 8 -snes_smonitor -ksp_gmres_irorthog > ex14_2.tmp 2>&1; \
	   if (diff output/ex14_2.out ex14_2.tmp) then true; \
	   else echo "Possible problem with ex14_2, diffs above"; fi; \
	   ${RM} -f ex14_2.tmp
runex16:
	-@${MPIRUN} -np 1 ex16 -options_file ex16options -snes_smonitor -ksp_smonitor -ksp_gmres_irorthog > ex16_1.tmp 2>&1; \
	   if (diff output/ex16_1.out ex16_1.tmp) then true; \
	   else echo "Possible problem with ex16_1, diffs above"; fi; \
	   ${RM} -f ex16_1.tmp
runex16_2:
	-@${MPIRUN} -np 4 ex16 -options_file ex16options_2 -snes_smonitor -ksp_smonitor -ksp_gmres_irorthog > ex16_2.tmp 2>&1; \
	   if (diff output/ex16_2.out ex16_2.tmp) then true; \
	   else echo "Possible problem with ex16_2, diffs above"; fi; \
	   ${RM} -f ex16_2.tmp


