
#requiresscalar real

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
OBJSC	         = 
OBJSF            =
CLEANFILES       = flow.exe flow.o flow history.out
LOCDIR           = src/contrib/fun3d/comp/
EXAMPLESC        = flow.c user.h
EXAMPLESF        = user.F userJac.F
FLOW_OBJSF       = user.o userJac.o
#
include $(PETSC_DIR)/bmake/common/base
include $(PETSC_DIR)/bmake/common/test

flow: flow.o $(FLOW_OBJSF) chkopts
	-$(CLINKER) -o flow flow.o $(FLOW_OBJSF) $(PETSC_FORTRAN_LIB) $(PETSC_SNES_LIB)

puns3d: puns3d.a.o
	$(FLINKER) -o puns3d puns3d.a.o $(PETSC_FORTRAN_LIB) $(PETSC_DM_LIB)

runflow1:
	-@${MPIRUN} -np 1 ./flow -options_file ${DATAFILESPATH}/fun3dgrid/petsc.opt -mesh ${DATAFILESPATH}/fun3dgrid/uns3d.msh -partition ${DATAFILESPATH}/fun3dgrid/part_vec.part.1 | grep clift | tail -1 > runflow1.tmp 2>&1;   \
	   if (${DIFF} ${DATAFILESPATH}/fun3dgrid/coef.dat runflow1.tmp) then true; \
	   else echo "Possible problem with runflow1, diffs above"; fi; \
	   ${RM} -f runflow1.tmp
runflow4:
	-@${MPIRUN} -np 4 ./flow -options_file ${DATAFILESPATH}/fun3dgrid/petsc.opt -mesh ${DATAFILESPATH}/fun3dgrid/uns3d.msh -partition ${DATAFILESPATH}/fun3dgrid/part_vec.part.4 | grep clift | tail -1 > runflow4.tmp 2>&1;   \
	   if (${DIFF} ${DATAFILESPATH}/fun3dgrid/coef.dat runflow4.tmp) then true; \
	   else echo "Possible problem with runflow4, diffs above"; fi; \
	   ${RM} -f runflow4.tmp
runflow64:
	-@${MPIRUN} -np 64 ./flow -options_file 1Grid/petsc.opt -mesh 1Grid/uns3d.msh -partition 1Grid/Pmetis/part_vec.part.64 

TESTEXAMPLES_FORTRAN_NOCOMPLEX	 = flow.PETSc runflow1 runflow4 flow.rm

DATAFILESPATH    = /homes/kaushik/FUN3DGrids/BigSample

runbm1:
	-@echo "Using datafiles from" ${DATAFILESPATH}
	-@${MPIRUN} -np 1 ./flow -options_file ${DATAFILESPATH}/petsc.opt -mesh ${DATAFILESPATH}/uns3d.msh >  out.1.${HOST}

runbm2:
	-@echo "Using datafiles from" ${DATAFILESPATH}
	-@${MPIRUN} -np 2 ./flow -options_file ${DATAFILESPATH}/petsc.opt -mesh ${DATAFILESPATH}/uns3d.msh -partition ${DATAFILESPATH}/part_vec.part.2 > out.2.${HOST}

runbm4:
	-@${MPIRUN} -np 4 ./flow -options_file ${DATAFILESPATH}/petsc.opt -mesh ${DATAFILESPATH}/uns3d.msh -partition ${DATAFILESPATH}/part_vec.part.4 > out.4.${HOST}
