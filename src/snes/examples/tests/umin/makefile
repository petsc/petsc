# $Id: makefile,v 1.63 1999/03/01 04:57:33 bsmith Exp balay $ 

ALL: ex1

CFLAGS          = 
FFLAGS          = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/snes/examples/umin/

TESTEXAMPLES_1  = ex2.PETSc runex2 runex2_2 runex2_3 ex2.rm ex3.PETSc\
		  runex3 ex3.rm clean
TESTEXAMPLES_2  = ex3.PETSc runex3_2 ex3.rm clean
TESTEXAMPLES_8  = ex1.PETSc  runex1 runex1_2 ex1.rm clean
TESTEXAMPLES_13 = ex4.PETSc ex4.rm clean
BUILDEXAMPLES_1 = ex2.PETSc ex2.rm ex3.PETSc ex3.rm clean
BUILDEXAMPLES_2 = ex3.PETSc ex3.rm clean
BUILDEXAMPLES_8 = ex1.PETSc ex1.rm clean

EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c
EXAMPLESF       = deptfg.f dgl1fg.f dgl2fg.f dmsafg.f dodcfg.f dpjbfg.f \
		  dsscfg.f dgl1hs.f dgl2hs.f dmsahs.f dodchs.f dpjbhs.f \
		  dsschs.f dmsabc.f dodcps.f dgl2co.f mpinter.f depths.f \
		  pdgl2fg.f
DOCS            = readme
DIRS            = 
FOBJ1 = deptfg.o dgl1fg.o dgl2fg.o dmsafg.o dodcfg.o dpjbfg.o dsscfg.o \
        depths.o dgl1hs.o dgl2hs.o dmsahs.o dodchs.o dpjbhs.o dsschs.o \
        dmsabc.o dodcps.o dgl2co.o mpinter.o

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o ${FOBJ1} chkopts
	-${CLINKER} -o ex1 ex1.o ${FOBJ1}  ${PETSC_LIB}
	${RM} ex1.o ${FOBJ1}

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_LIB}
	${RM} ex2.o

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_LIB}
	${RM} ex3.o

FOBJ4 = pdgl2fg.o
ex4: ex4.o ${FOBJ4} chkopts
	-${CLINKER} -o ex4 ex4.o ${FOBJ4}  ${PETSC_LIB}
	${RM} ex4.o ${FOBJ4}

runex1:
	-@${MPIRUN} -np 1 ex1 -snes_smonitor > ex1_1.tmp 2>&1;   \
           if (diff output/ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIRUN} -np 1 ex1 -snes_smonitor -snes_type umls > ex1_2.tmp 2>&1;\
           if (diff output/ex1_2.out ex1_2.tmp) then true; \
           else echo "Possible problem with ex1_2, diffs above"; fi; \
           ${RM} -f ex1_2.tmp
runex2:
	-@${MPIRUN} -np 1 ex2 -p 1 -snes_smonitor > ex2_1.tmp 2>&1;   \
           if (diff output/ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi; \
           ${RM} -f ex2_1.tmp
runex2_2:
	-@${MPIRUN} -np 1 ex2 -p 1 -snes_smonitor -snes_type umls > ex2_2.tmp 2>&1;\
           if (diff output/ex2_2.out ex2_2.tmp) then true; \
           else echo "Possible problem with ex2_2, diffs above"; fi; \
           ${RM} -f ex2_2.tmp
runex2_3:
	-@${MPIRUN} -np 1 ex2 -p 1 -snes_smonitor -snes_type umls -snes_mf > ex2_3.tmp 2>&1;\
           if (diff output/ex2_3.out ex2_3.tmp) then true; \
           else echo "Possible problem with ex2_3, diffs above"; fi; \
           ${RM} -f ex2_3.tmp
runex3:
	-@${MPIRUN} -np 2 ex3 -snes_smonitor > ex3_1.tmp 2>&1;\
           if (diff output/ex3_1.out ex3_1.tmp) then true; \
           else echo "Possible problem with ex3_1, diffs above"; fi;\
	   ${RM} -f ex3_1.tmp

runex3_2:
	-@${MPIRUN} -np 2 ex3 -Ny 2 -snes_type umls -snes_smonitor > ex3_2.tmp 2>&1;\
           if (diff output/ex3_2.out ex3_2.tmp) then true; \
           else echo "Possible problem with ex3_2, diffs above"; fi;\
	   ${RM} -f ex3_2.tmp

runex3_3:
	-@${MPIRUN} -np 4 ex3 -Nx 2 -Ny 2 -snes_fd -snes_smonitor > ex3_3.tmp 2>&1;\
           if (diff output/ex3_3.out ex3_3.tmp) then true; \
           else echo "Possible problem with ex3_3, diffs above"; fi;\
	   ${RM} -f ex3_3.tmp

runex4:
	-@${MPIRUN} -np 2 ex4 -snes_smonitor > ex4_1.tmp 2>&1;\
           if (diff output/ex4_1.out ex4_1.tmp) then true; \
           else echo "Possible problem with ex4_1, diffs above"; fi;\
	   ${RM} -f ex4_1.tmp

