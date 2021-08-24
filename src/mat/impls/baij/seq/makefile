-include ../../../../../petscdir.mk
ALL: lib

CFLAGS   =
FFLAGS   =
CPPFLAGS =
SOURCEC  = baij.c baij2.c baijfact.c baijfact2.c dgefa.c dgedi.c dgefa3.c dgefa4.c dgefa5.c dgefa2.c dgefa6.c dgefa7.c aijbaij.c baijfact3.c baijfact4.c baijfact5.c baijfact7.c baijfact9.c baijfact11.c baijfact13.c baijfact81.c baijsolv.c baijsolvtrannat1.c baijsolvtrannat2.c baijsolvtrannat3.c baijsolvtrannat4.c baijsolvtrannat5.c baijsolvtrannat6.c baijsolvtrannat7.c baijsolvtran1.c baijsolvtran2.c baijsolvtran3.c baijsolvtran4.c baijsolvtran5.c baijsolvtran6.c baijsolvtran7.c baijsolvtrann.c baijsolvnat1.c baijsolvnat2.c baijsolvnat3.c baijsolvnat4.c baijsolvnat5.c baijsolvnat6.c baijsolvnat7.c baijsolvnat11.c baijsolvnat14.c baijsolvnat15.c
SOURCEF  =
SOURCEH  = baij.h
LIBBASE  = libpetscmat
DIRS     = ftn-kernels baijmkl
MANSEC   = Mat
LOCDIR   = src/mat/impls/baij/seq/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
