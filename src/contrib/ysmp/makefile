# $Id: makefile,v 1.13 1998/11/24 04:17:25 bsmith Exp $ 

ALL: ex1

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
FPPFLAGS  =
LOCDIR    = src/contrib/ysmp/

EXAMPLES  = ex1
EXAMPLESC = ex1.c
LINCLUDE  = ${SOURCEH}

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: chkopts ex1.o ${PETSC_ARCH}ysmp.a
	-${CLINKER} -o ex1 ex1.o ${LIBNAME} ${PETSC_LIB} ${PETSC_ARCH}ysmp.a
	${RM} ex1.o


#
#   Makefile for Yale sparse matrix package
#

OBJ = cdrv.o md.o mdi.o mdm.o mdp.o mdu.o ndrv.o nnf.o \
      nnfc.o nns.o nnsc.o nnt.o nntc.o nroc.o nsf.o \
      nsfc.o odrv.o sdrv.o snf.o sns.o sro.o ssf.o \
      tdrv.o trk.o r1mach.o xsetf.o xsetun.o d1mach.o \
      i1mach.o xermax.o xerror.o xerrwv.o j4save.o  \
      xerprt.o xerabt.o fdump.o xerctl.o xgetua.o xersav.o

SRC = cdrv.f md.f mdi.f mdm.f mdp.f mdu.f ndrv.f nnf.f \
      nnfc.f nns.f nnsc.f nnt.f nntc.f nroc.f nsf.f \
      nsfc.f odrv.f sdrv.f snf.f sns.f sro.f ssf.f \
      tdrv.f trk.f r1mach.f xsetf.f xsetun.f d1mach.f \
      i1mach.f xermax.f xerror.f xerrwv.f j4save.f  \
      xerprt.f xerabt.f fdump.f xerctl.f xgetua.f xersav.f

test1:  ysmp.o ${PETSC_ARCH}ysmp.a
	f77 -o test1 ysmp.o ${PETSC_ARCH}ysmp.a
	rm -f ysmp.o

${PETSC_ARCH}ysmp.a  : ${SRC}
	${FC} -c ${FFLAGS}${BASEOPTF} ${SRC} 
	ar r ${PETSC_ARCH}ysmp.a ${OBJ}
	ranlib ${PETSC_ARCH} ysmp.a
	rm -f  ${OBJ}



