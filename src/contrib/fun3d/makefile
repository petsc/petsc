# $Id: makefile,v 1.9 2000/01/18 17:22:41 bsmith Exp kaushik $ 

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
CLEANFILES       = flow.exe flow.o flow
DIRS             = 
LOCDIR           = src/contrib/fun3d
TESTEXAMPLES_3	 = flow.PETSc runflow1 runflow4 flow.rm
BUILDEXAMPLES_3  = flow.PETSc flow.rm clean
EXAMPLESC        = flow.c user.h
EXAMPLESF        = user.F userJac.F
#
include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

flow: flow.o $(OBJSF) chkopts
	-$(FLINKER) -o flow flow.o $(OBJSF) $(PETSC_FORTRAN_LIB) $(PETSC_LIB)

user.o:
#f77 -c -r8 -i4 -O4 -xs user.f
#	$(FC) -c $(FFLAGS) $(CFLAGS) $(FOPTFLAGS) user.F

userJac.o:
#	$(FC) -c $(FFLAGS) $(CFLAGS) $(FOPTFLAGS) userJac.F

puns3d: puns3d.a.o
	$(FLINKER) -o puns3d puns3d.a.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
puns3d.a.o:

runflow1:
	-@${MPIRUN} -np 1 flow | grep clift | tail -1 > runflow1.tmp 2>&1;   \
	   if (diff testgrid/coef.dat runflow1.tmp) then true; \
	   else echo "Possible problem with runflow1, diffs above"; fi; \
	   ${RM} -f runflow1.tmp
runflow4:
	-@${MPIRUN} -np 4 flow | grep clift | tail -1 > runflow4.tmp 2>&1;   \
	   if (diff testgrid/coef.dat runflow4.tmp) then true; \
	   else echo "Possible problem with runflow4, diffs above"; fi; \
	   ${RM} -f runflow4.tmp
tags:
	etags -f TAGS *.F *.c *.h

#.F.o:
#	-${FC} -c ${FOPTFLAGS} ${FFLAGS} ${CPPFLAGS} ${USERCPPFLAGS} $*.F











