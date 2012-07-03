#DIM              = 2
#NUM_TERMS        = 17
#NUM_COEFFICIENTS = ${NUM_TERMS}
DIM              = 3
NUM_TERMS        = 10
NUM_COEFFICIENTS = ${NUM_TERMS}*${NUM_TERMS}*3

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        = -I${PETFMM_DIR}/src -I. -DDIMENSION=${DIM} -DNUM_TERMS=${NUM_TERMS} -DNUM_COEFFICIENTS=${NUM_COEFFICIENTS}
FPPFLAGS        = 
LOCDIR          = src/dm/mesh/examples/tutorials/
EXAMPLESC       = ex1.c ex10.cxx ex12.cxx
EXAMPLESF       = ex12f90.F90 mex12f90.F90
MANSEC          = DM
NP              = 1
MATTSDISPLAY         = :0.0
ORDER           = 2
EXTRA_ARGS      = -mesh_view_vtk -vec_view_vtk -mat_type umfpack -pc_type lu

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_DM_LIB}
	${RM} -f ex1.o

closuretest: closuretest.o   chkopts
	-${CLINKER} -o closuretest closuretest.o  ${PETSC_DM_LIB}
	${RM} -f closuretest.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_DM_LIB}
	${RM} -f ex2.o

ex2f: ex2f.o   chkopts
	-${CLINKER} -o ex2f ex2f.o  ${PETSC_DM_LIB}
	${RM} -f ex2.o

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_DM_LIB}
	${RM} -f ex3.o

ex5: ex5.o   chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_DM_LIB}
	${RM} -f ex5.o

ex6: ex6.o   chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_DM_LIB}
	${RM} -f ex6.o

ex7: ex7.o   chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_DM_LIB}
	${RM} -f ex7.o

ex10: ex10.o   chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_DM_LIB}
	${RM} -f ex10.o

ex11: ex11.o   chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_DM_LIB}
	${RM} -f ex11.o

ex12: ex12.o mex12f90.o ex12f90.o chkopts
	-${CLINKER} -o ex12 ex12.o ex12f90.o mex12f90.o ${PETSC_SNES_LIB}
	${RM} -f ex12.o mex12f90.mod mex12f90.o ex12f90.o

ex_coarsen: ex_coarsen.o   chkopts
	-${CLINKER} -o ex_coarsen ex_coarsen.o  ${PETSC_DM_LIB}
	${RM} -f ex_coarsen.o

mesh_surgery: mesh_surgery.o   chkopts
	-${CLINKER} -o mesh_surgery mesh_surgery.o  ${PETSC_DM_LIB}
	${RM} -f mesh_surgery.o

ex_coarsen_3: ex_coarsen_3.o   chkopts
	-${CLINKER} -o ex_coarsen_3 ex_coarsen_3.o  ${PETSC_DM_LIB}
	${RM} -f ex_coarsen_3.o

ex_interpolate_1: ex_interpolate_1.o   chkopts
	-${CLINKER} -o ex_interpolate_1 ex_interpolate_1.o  ${PETSC_DM_LIB}
	${RM} -f ex_interpolate_1.o

ionChannel: ionChannel.o chkopts
	-${CLINKER} -o ionChannel ionChannel.o ${PETSC_DM_LIB}
	${RM} -f ionChannel.o

mixedpoisson: mixedpoisson_quadrature.h mixedpoisson.o chkopts
	-${CLINKER} -o mixedpoisson mixedpoisson.o ${PETSC_SNES_LIB}
	${RM} -f mixedpoisson.o

stokes: stokes_quadrature.h stokes.o chkopts
	-${CLINKER} -o stokes stokes.o ${PETSC_SNES_LIB}
	${RM} -f stokes.o

stokesPeriodic: stokesPeriodic_quadrature.h stokesPeriodic.o chkopts
	-${CLINKER} -o stokesPeriodic stokesPeriodic.o ${PETSC_SNES_LIB}
	${RM} -f stokesPeriodic.o

stokesVector: stokesVector_quadrature.h stokesVector.o chkopts
	-${CLINKER} -o stokesVector stokesVector.o ${PETSC_SNES_LIB}
	${RM} -f stokesVector.o

navierStokes: navierStokes_quadrature.h navierStokes.o chkopts
	-${CLINKER} -o navierStokes navierStokes.o ${PETSC_SNES_LIB}
	${RM} -f navierStokes.o

grade2test: stokes_quadrature.h grade2test.o chkopts
	-${CLINKER} -o grade2test grade2test.o ${PETSC_SNES_LIB}
	${RM} -f grade2test.o

grade2: grade2_quadrature.h grade2.o chkopts
	-${CLINKER} -o grade2 grade2.o ${PETSC_SNES_LIB}
	${RM} -f grade2.o

bratu: bratu_quadrature.h GMVFile.o GMVFileAscii.o BinaryIO.o GMVFileBinary.o bratu.o chkopts
	-${CLINKER} -o bratu  GMVFile.o GMVFileAscii.o BinaryIO.o GMVFileBinary.o bratu.o ${PETSC_SNES_LIB}
	${RM} -f bratu.o

bratu_1d.h: bratu_1d.form
	ffc bratu_1d.form

bratu_2d.h: bratu_2d.form
	ffc bratu_2d.form

bratu_3d.h: bratu_3d.form
	ffc bratu_3d.form

vector_poisson_2d.h: vector_poisson_2d.form
	ffc vector_poisson_2d.form

vector_poisson_3d.h: vector_poisson_3d.form
	ffc vector_poisson_3d.form

stokes_2d.h: stokes_2d.form
	ffc stokes_2d.form

stokes_3d.h: stokes_3d.form
	ffc stokes_3d.form

bratu_ufc: bratu_1d.h bratu_2d.h bratu_3d.h vector_poisson_2d.h vector_poisson_3d.h GMVFile.o GMVFileAscii.o BinaryIO.o GMVFileBinary.o bratu_ufc.o chkopts
	-${CLINKER} -o bratu_ufc  GMVFile.o GMVFileAscii.o BinaryIO.o GMVFileBinary.o bratu_ufc.o ${PETSC_SNES_LIB}
	${RM} -f bratu_ufc.o

ex_ufc: stokes_2d.h stokes_3d.h vector_poisson_2d.h vector_poisson_3d.h bratu_1d.h bratu_2d.h bratu_3d.h GMVFile.o GMVFileAscii.o BinaryIO.o GMVFileBinary.o ex_ufc.o chkopts
	-${CLINKER} -o ex_ufc  GMVFile.o GMVFileAscii.o BinaryIO.o GMVFileBinary.o ex_ufc.o ${PETSC_SNES_LIB}
	${RM} -f ex_ufc.o

bratu2: bratu2_quadrature.h bratu2.o chkopts
	-${CLINKER} -o bratu2 bratu2.o ${PETSC_SNES_LIB}
	${RM} -f bratu2.o

electrostatic: electrostatic_quadrature.h electrostatic.o chkopts
	-${CLINKER} -o electrostatic electrostatic.o ${PETSC_SNES_LIB}
	${RM} -f electrostatic.o

subduction: subduction_quadrature.h subduction.o chkopts
	-${CLINKER} -o subduction subduction.o ${PETSC_SNES_LIB}
	${RM} -f subduction.o

dolfin: dolfin.o chkopts
	-${CLINKER} -o dolfin dolfin.o ${PETSC_DM_LIB}
	${RM} -f dolfin.o

lagrit: CohesiveTopology.o lagrit.o chkopts
	-${CLINKER} -o lagrit CohesiveTopology.o lagrit.o ${PETSC_DM_LIB}
	${RM} -f lagrit.o

rt: rt_quadrature.h rt.o chkopts
	-${CLINKER} -o rt rt.o ${PETSC_SNES_LIB}
	${RM} -f rt.o

dirk: dirk.o chkopts
	-${CLINKER} -o dirk dirk.o ${PETSC_SNES_LIB}
	${RM} -f dirk.o

fetidp: fetidp_quadrature.h fetidp.o chkopts
	-${CLINKER} -o fetidp fetidp.o ${PETSC_SNES_LIB}
	${RM} -f fetidp.o

refinement: GMVFile.o GMVFileAscii.o BinaryIO.o GMVFileBinary.o refinement.o chkopts
	-${CLINKER} -o refinement GMVFile.o GMVFileAscii.o BinaryIO.o GMVFileBinary.o refinement.o ${PETSC_SNES_LIB}
	${RM} -f refinement.o

fvTest: fvTest.o chkopts
	-${CLINKER} -o fvTest fvTest.o ${PETSC_SNES_LIB}
	${RM} -f fvTest.o

problemTests: problemTests.o chkopts
	-${CLINKER} -o problemTests problemTests.o ${PETSC_SNES_LIB} -lcppunit
	${RM} -f problemTests.o

watson: watson.o chkopts
	-${CLINKER} -o watson watson.o ${PETSC_SNES_LIB}
	${RM} -f watson.o

#-----------------------------------------------------------------
runmixedpoisson:
	${MPIEXEC} -n ${NP} ./mixedpoisson ${EXTRA_ARGS}

runstokes:
	-@${MPIEXEC} -n ${NP} ./stokes ${EXTRA_ARGS}

rungrade2test:
	${MPIEXEC} -n ${NP} ./grade2test ${EXTRA_ARGS}

rungrade2:
	${MPIEXEC} -n ${NP} ./grade2 ${EXTRA_ARGS}

debuggrade2:
	-@${MPIEXEC} -n ${NP} ./grade2 ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY}

runex1:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 1 -use_zero_base 1 -base_file data/ex1_1d ${EXTRA_ARGS}

debugex1:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 1 -use_zero_base 1 -base_file data/ex1_1d ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY}

runex1_pylith:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 3 -file_type pylith -use_zero_base 0 -interpolate 0 -output_local 1 -base_file data/bm1b/bm1b ${EXTRA_ARGS}

debugex1_pylith:
	-${MPIEXEC} -n ${NP} ./ex1 -dim 3 -file_type pylith -use_zero_base 0 -interpolate 0 -output_local 1 -base_file data/bm1b/bm1b ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY}
runex1_pylith2:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 3 -file_type pylith -use_zero_base 0 -interpolate 0 -output_local 1 -base_file data/bm1m/bm1m ${EXTRA_ARGS}

debugex1_pylith2:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 3 -file_type pylith -use_zero_base 0 -interpolate 0 -output_local 1 -base_file data/bm1m/bm1m ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY}

runex1_pylith_split:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 3 -file_type pylith -use_zero_base 0 -interpolate 0 -output_local 1 -base_file data/splitFaultTest/splittest ${EXTRA_ARGS}

runex1_2:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 2 -use_zero_base 1 -base_file data/ex1_2d ${EXTRA_ARGS}

runex1_2_hex:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 2 -use_zero_base 1 -base_file data/ex1_2d_hex ${EXTRA_ARGS}

runex1_3:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 3 -use_zero_base 1 -base_file data/ex1_3d ${EXTRA_ARGS}

runex1_3_hex:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 3 -use_zero_base 1 -base_file data/ex1_3d_hex ${EXTRA_ARGS}

runex2:
	-@${MPIEXEC} -n ${NP} ./ex2 -dim 2 ${EXTRA_ARGS}

debugex2:
	-@${MPIEXEC} -n ${NP} ./ex2 -dim 2 ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY}

runex2_3:
	-@${MPIEXEC} -n ${NP} ./ex2 -dim 3 ${EXTRA_ARGS}

runex3:
	-@${MPIEXEC} -n ${NP} ./ex3 -dim 2 ${EXTRA_ARGS}

debugex3:
	-@${MPIEXEC} -n ${NP} ./ex3 -dim 2 ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY}

runex4:
	-@${MPIEXEC} -n ${NP} ./ex4 -dim 2 ${EXTRA_ARGS}

runex5:
	-@${MPIEXEC} -n ${NP} ./ex5 -dim 1 -use_zero_base 1 -base_file data/ex1_1d ${EXTRA_ARGS}

runex5_2:
	-@${MPIEXEC} -n ${NP} ./ex5 -dim 2 -use_zero_base 1 -base_file data/ex1_2d ${EXTRA_ARGS}

runex5_3:
	-@${MPIEXEC} -n ${NP} ./ex5 -dim 3 -use_zero_base 1 -base_file data/ex1_3d ${EXTRA_ARGS}

runex6:
	-@${MPIEXEC} -n ${NP} ./ex6 -dim 2 ${EXTRA_ARGS}

testex1_1:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 1 -use_zero_base 1 -base_file data/ex1_1d > /dev/null 2>&1; \
    if (${DIFF} data/ex1_1d.lcon testMesh.lcon) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_1, diffs above \n========================================="; fi; \
    if (${DIFF} data/ex1_1d.nodes testMesh.nodes) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_1, diffs above \n========================================="; fi

testex1_2:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 2 -use_zero_base 1 -base_file data/ex1_2d > /dev/null 2>&1; \
    if (${DIFF} data/ex1_2d.lcon testMesh.lcon) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_2, diffs above \n========================================="; fi; \
    if (${DIFF} data/ex1_2d.nodes testMesh.nodes) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_2, diffs above \n========================================="; fi

testex1_3:
	-@${MPIEXEC} -n ${NP} ./ex1 -dim 3 -use_zero_base 1 -base_file data/ex1_3d > /dev/null 2>&1; \
    if (${DIFF} data/ex1_3d.lcon testMesh.lcon) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_3, diffs above \n========================================="; fi; \
    if (${DIFF} data/ex1_3d.nodes testMesh.nodes) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_3, diffs above \n========================================="; fi

testex1_4:
	-@${MPIEXEC} -n 5 ./ex1 -dim 3 -file_type pylith -output_local 1 -base_file data/bm1b/bm1b > /dev/null 2>&1; \
    if (${DIFF} data/bm1b/bm1b.0.connect testMesh.0.connect) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_4, diffs above \n========================================="; fi; \
    if (${DIFF} data/bm1b/bm1b.0.coord testMesh.0.coord) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_4, diffs above \n========================================="; fi; \
    if (${DIFF} data/bm1b/bm1b.1.connect testMesh.1.connect) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_4, diffs above \n========================================="; fi; \
    if (${DIFF} data/bm1b/bm1b.1.coord testMesh.1.coord) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_4, diffs above \n========================================="; fi; \
    if (${DIFF} data/bm1b/bm1b.2.connect testMesh.2.connect) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_4, diffs above \n========================================="; fi; \
    if (${DIFF} data/bm1b/bm1b.2.coord testMesh.2.coord) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_4, diffs above \n========================================="; fi; \
    if (${DIFF} data/bm1b/bm1b.3.connect testMesh.3.connect) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_4, diffs above \n========================================="; fi; \
    if (${DIFF} data/bm1b/bm1b.3.coord testMesh.3.coord) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_4, diffs above \n========================================="; fi; \
    if (${DIFF} data/bm1b/bm1b.4.connect testMesh.4.connect) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_4, diffs above \n========================================="; fi; \
    if (${DIFF} data/bm1b/bm1b.4.coord testMesh.4.coord) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex1_4, diffs above \n========================================="; fi

testex2_1:
	-@${MPIEXEC} -n 1 ./ex2 -dim 2 -refinement_limit 0.0625 > /dev/null 2>&1; \
    if (${DIFF} data/ex2_1.vtk testMesh.vtk) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex2_1, diffs above \n========================================="; fi

testex2_2:
	-@${MPIEXEC} -n 2 ./ex2 -dim 2 -refinement_limit 0.0625 > /dev/null 2>&1; \
    if (${DIFF} data/ex2_1.vtk testMesh.vtk) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex2_1, diffs above \n========================================="; fi

testex2_3:
	-@${MPIEXEC} -n 3 ./ex2 -dim 2 -refinement_limit 0.0625 > /dev/null 2>&1; \
    if (${DIFF} data/ex2_1.vtk testMesh.vtk) then true; \
	else echo ${PWD} ; echo "Possible problem with with ex2_1, diffs above \n========================================="; fi

debugex4:
	-@${MPIEXEC} -n ${NP} ./ex4 ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY}

runex10:
	-@${MPIEXEC} -n 2 ./ex10 -partitioner zoltan -test 1 > ex10_1.tmp 2>&1; \
    if (${DIFF} output/ex10_1.out ex10_1.tmp) then true; \
    else echo ${PWD} ; echo "Possible problem with with ex10_1, diffs above \n========================================="; fi; \
    ${RM} -f ex10_1.tmp


TESTEXAMPLES_SIEVE 		  = ex1.PETSc testex1_1 testex1_2 testex1_3 testex1_4 ex1.rm ex2.PETSc testex2_1 testex2_2 testex2_3 ex2.rm ex10.PETSc runex10 ex10.rm
TESTEXAMPLES_C  		  = 
TESTEXAMPLES_C_X  	  = 
TESTEXAMPLES_FORTRAN	  = 
TESTEXAMPLES_F90          = 
TESTEXAMPLES_C_X_MPIUNI = 
TESTEXAMPLES_C_COMPLEX	  = 
TESTEXAMPLES_C_NOCOMPLEX  = 

include ${PETSC_DIR}/conf/test
