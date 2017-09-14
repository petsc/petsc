
ALL: lib

CFLAGS   =
FFLAGS   =
CPPFLAGS =
SOURCEC  = baij.c baij2.c baijfact.c baijfact2.c dgefa.c dgedi.c dgefa3.c \
	   dgefa4.c dgefa5.c dgefa2.c dgefa6.c dgefa7.c aijbaij.c baijfact3.c baijfact4.c \
           baijfact5.c baijfact7.c baijfact9.c baijfact11.c baijfact13.c \
           baijsolvtrannat.c baijsolvtran.c baijsolv.c baijsolvnat.c
SOURCEF  =
SOURCEH  = baij.h
LIBBASE  = libpetscmat
DIRS     = ftn-kernels baijmkl
MANSEC   = Mat
LOCDIR   = src/mat/impls/baij/seq/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
