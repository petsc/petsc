-include ../petscdir.mk
ALL: lib

CFLAGS	 =
FFLAGS	 =
CPPFLAGS =
FPPFLAGS =
SOURCEC	 = sbaij.c sbaij2.c sbaijfact.c sbaijfact2.c sro.c sbaijfact3.c \
           sbaijfact4.c sbaijfact5.c sbaijfact6.c sbaijfact7.c sbaijfact8.c sbaijfact9.c \
           sbaijfact10.c sbaijfact11.c sbaijfact12.c aijsbaij.c
SOURCEF	 =
SOURCEH	 = sbaij.h relax.h
LIBBASE	 = libpetscmat
DIRS     = cholmod
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/sbaij/seq/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test





