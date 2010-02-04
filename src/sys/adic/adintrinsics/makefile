
ALL: lib

CFLAGS   = -DDISABLE_FORTRAN -I${PETSC_DIR}/include/adic
FFLAGS   = 
SOURCEC  = adintr_acos.c adintr_exp.c adintr_max.c adintr_acosh.c \
	adintr_fabs.c adintr_min.c adintr_asin.c adintr_floor.c \
	adintr_modf.c adintr_asinh.c adintr_fmax.c adintr_pow.c \
	adintr_atan.c adintr_fmin.c adintr_sin.c adintr_atan2.c \
	adintr_fmod.c adintr_sinh.c adintr_atanh.c adintr_frexp.c \
	adintr_sqrt.c adintr_ceil.c adintr_ldexp.c adintr_tan.c \
	adintr_cos.c adintr_log.c adintr_tanh.c adintr_cosh.c \
	adintr_log10.c adintrinsics.c report-once.c

SOURCEF  =
SOURCEH  = ../../../../include/adic/run-map.h ../../../../include/adic/run-alloc.h report-once.h
OBJSC    = adintr_acos.o adintr_exp.o adintr_max.o adintr_acosh.o \
	adintr_fabs.o adintr_min.o adintr_asin.o adintr_floor.o \
	adintr_modf.o adintr_asinh.o adintr_fmax.o adintr_pow.o \
	adintr_atan.o adintr_fmin.o adintr_sin.o adintr_atan2.o \
	adintr_fmod.o adintr_sinh.o adintr_atanh.o adintr_frexp.o \
	adintr_sqrt.o adintr_ceil.o adintr_ldexp.o adintr_tan.o \
	adintr_cos.o adintr_log.o adintr_tanh.o adintr_cosh.o \
	adintr_log10.o adintrinsics.o report-once.o
OBJSF    =
LIBBASE  = libpetscsys
DIRS     = 
LOCDIR   = src/sys/adic/adintrinsics/
MANSEC   = Sys

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

