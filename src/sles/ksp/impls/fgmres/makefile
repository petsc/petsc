
# $Id: makefile,v 1.37 1999/02/02 21:29:21 balay Exp $ 

# Makefile for my library with fgmres and the pcfamily preconditioner

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = fgmres.c fborthog.c fborthog2.c fborthog3.c fgmres2.c fgmreig.c \
           fgmpre.c fgmcnv.c pcfamily.c modpcf.c
SOURCEH  = fgmresp.h
SOURCEF  = 
OBJSC    = fgmres.o fborthog.o fborthog2.o fborthog3.o fgmres2.o fgmreig.o fgmpre.o fgmcnv.o pcfamily.o modpcf.o
#OBJSC    =  pcfamily.o
OBJSF    = 
LIBBASE  = liballison
# LIBBASE  = libpetscsles
MANSEC   = KSP
LOCDIR   = /home/baker/working
# LOCDIR   = src/sles/ksp/impls/fgmres/


include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

PDIR = /home/bsmith/petsc/src/contrib/fgmres/lib/lib${BOPT}/${PETSC_ARCH}

# LDIR_ALLISON = /home/baker/working/lib/lib${BOPT}/${PETSC_ARCH}
# ALLISON_LIB  = -L${LDIR_ALLISON} -lallison
