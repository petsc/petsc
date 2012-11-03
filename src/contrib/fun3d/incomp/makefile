#requiresscalar real
#requiresprecision double

ALL: flow

USERCPPFLAGS     =   -DINTERLACING -DBLOCKING 

# To run the benchmark case on the mesh called BigSample, uncomment
# the following line and run configure to use matsingle
#USERCPPFLAGS     =  -DCFL3D_AXIS -DINTERLACING -DBLOCKING 
CFLAGS	         =
FFLAGS	         =	 
#/real_size:64 /integer_size:32
CPPFLAGS         = $(USERCPPFLAGS) $(FCONF)
FPPFLAGS         = $(USERCPPFLAGS)
SOURCEC	         = 
SOURCEF	         =
SOURCEH	         =
CLEANFILES       = flow.exe flow.o flow history.out
LOCDIR           = src/contrib/fun3d/incomp/
EXAMPLESC        = flow.c user.h
EXAMPLESF        = user.F userJac.F
FLOW_OBJSF       = $(EXAMPLESF:.F=.o)
#
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include $(PETSC_DIR)/conf/test

flow: flow.o $(FLOW_OBJSF) chkopts
	-$(CLINKER) -o flow flow.o $(FLOW_OBJSF) $(PETSC_SNES_LIB)

puns3d: puns3d.a.o
	$(FLINKER) -o puns3d puns3d.a.o $(PETSC_DM_LIB)

# unfortunately we do not have permission from NASA to release the data files used in the tests below
runflow1:
	-@${MPIEXEC} -n 1 ./flow -options_file ${DATAFILESPATH}/fun3dgrid/petsc.opt -mesh ${DATAFILESPATH}/fun3dgrid/uns3d.msh -partition ${DATAFILESPATH}/fun3dgrid/part_vec.part.1 | grep clift | tail -1 > runflow1.tmp 2>&1;   \
	   if (${DIFF} ${DATAFILESPATH}/fun3dgrid/coef.dat runflow1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with runflow1, diffs above \n========================================="; fi; \
	   ${RM} -f runflow1.tmp
runflow4:
	-@${MPIEXEC} -n 4 ./flow -options_file ${DATAFILESPATH}/fun3dgrid/petsc.opt -mesh ${DATAFILESPATH}/fun3dgrid/uns3d.msh -partition ${DATAFILESPATH}/fun3dgrid/part_vec.part.4 | grep clift | tail -1 > runflow4.tmp 2>&1;   \
	   if (${DIFF} ${DATAFILESPATH}/fun3dgrid/coef.dat runflow4.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with runflow4, diffs above \n========================================="; fi; \
	   ${RM} -f runflow4.tmp
runflow64:
	-@${MPIEXEC} -n 64 ./flow -options_file 1Grid/petsc.opt -mesh 1Grid/uns3d.msh -partition 1Grid/Pmetis/part_vec.part.64 

TESTEXAMPLES_DATAFILESPATH = flow.PETSc runflow1 runflow4 flow.rm

DATAFILESPATH    = /homes/kaushik/FUN3DGrids/BigSample

runbm1:
	-@echo "Using datafiles from" ${DATAFILESPATH}
	-@${MPIEXEC} -n 1 ./flow -options_file ${DATAFILESPATH}/petsc.opt -mesh ${DATAFILESPATH}/uns3d.msh >  out.1.${HOST}

runbm2:
	-@echo "Using datafiles from" ${DATAFILESPATH}
	-@${MPIEXEC} -n 2 ./flow -options_file ${DATAFILESPATH}/petsc.opt -mesh ${DATAFILESPATH}/uns3d.msh -partition ${DATAFILESPATH}/part_vec.part.2 > out.2.${HOST}

runbm4:
	-@${MPIEXEC} -n 4 ./flow -options_file ${DATAFILESPATH}/petsc.opt -mesh ${DATAFILESPATH}/uns3d.msh -partition ${DATAFILESPATH}/part_vec.part.4 > out.4.${HOST}
