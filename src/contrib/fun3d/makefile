# $Id: makefile,v 1.21 2001/08/24 15:51:53 bsmith Exp $ 

#requiresscalar real

ALL: flow

USERCPPFLAGS     =   -DINTERLACING -DBLOCKING 
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
OBJSF	         = user.o userJac.o
CLEANFILES       = flow.exe flow.o flow history.out
LOCDIR           = src/contrib/fun3d/
EXAMPLESC        = flow.c user.h
EXAMPLESF        = user.F userJac.F
#
include $(PETSC_DIR)/bmake/common/base

flow: flow.o $(OBJSF) chkopts
	-$(CLINKER) -o flow flow.o $(OBJSF) $(PETSC_FORTRAN_LIB) $(PETSC_SNES_LIB)

puns3d: puns3d.a.o
	$(FLINKER) -o puns3d puns3d.a.o $(PETSC_FORTRAN_LIB) $(PETSC_DM_LIB)

runflow1:
	-@${MPIRUN} -np 1 flow -options_file /home/petsc/datafiles/fun3dgrid/petsc.opt -mesh /home/petsc/datafiles/fun3dgrid/uns3d.msh -partition /home/petsc/datafiles/fun3dgrid/part_vec.part.1 | grep clift | tail -1 > runflow1.tmp 2>&1;   \
	   if (${DIFF} /home/petsc/datafiles/fun3dgrid/coef.dat runflow1.tmp) then true; \
	   else echo "Possible problem with runflow1, diffs above"; fi; \
	   ${RM} -f runflow1.tmp
runflow4:
	-@${MPIRUN} -np 4 flow -options_file /home/petsc/datafiles/fun3dgrid/petsc.opt -mesh /home/petsc/datafiles/fun3dgrid/uns3d.msh -partition /home/petsc/datafiles/fun3dgrid/part_vec.part.4 | grep clift | tail -1 > runflow4.tmp 2>&1;   \
	   if (${DIFF} /home/petsc/datafiles/fun3dgrid/coef.dat runflow4.tmp) then true; \
	   else echo "Possible problem with runflow4, diffs above"; fi; \
	   ${RM} -f runflow4.tmp
runflow64:
	-@${MPIRUN} -np 64 flow -options_file 1Grid/petsc.opt -mesh 1Grid/uns3d.msh -partition 1Grid/Pmetis/part_vec.part.64 

TESTEXAMPLES_3	 = flow.PETSc runflow1 runflow4 flow.rm

include $(PETSC_DIR)/bmake/common/test





