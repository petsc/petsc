# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib

CFLAGS	 = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC	 = mpiaij.c mmaij.c mpiaijpc.c aijpc.c mpiov.c fdmpiaij.c
SOURCEF	 =
SOURCEH	 = mpiaij.h
OBJSC	 = mpiaij.o mmaij.o mpiaijpc.o aijpc.o mpiov.o fdmpiaij.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = 2
LOCDIR	 = src/mat/impls/aij/mpi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
