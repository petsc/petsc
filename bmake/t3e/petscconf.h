#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.26 2001/03/26 19:23:56 balay Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_t3d
#define PETSC_ARCH_NAME "t3d"

#define HAVE_POPEN
#define HAVE_LIMITS_H
#define HAVE_PWD_H 
#define HAVE_STRING_H 
#define HAVE_MALLOC_H 
#define HAVE_DRAND48 
#define HAVE_UNISTD_H
#define HAVE_STDLIB_H
#define HAVE_SYS_TIME_H 
#define HAVE_UNAME
#define HAVE_GETCWD
#define HAVE_SYS_PARAM_H
#define HAVE_SYS_STAT_H

#define PETSC_HAVE_FORTRAN_CAPS 
#define PETSC_USES_CPTOFCD  
#define PETSC_USES_FORTRAN_SINGLE

#define HAVE_READLINK
#define HAVE_MEMMOVE

#define PETSC_CANNOT_START_DEBUGGER

#define HAVE_DOUBLES_ALIGNED
#define HAVE_DOUBLE_ALIGN_MALLOC
#define PETSC_HAVE_FAST_MPI_WTIME
#define SIZEOF_VOID_P 8
#define SIZEOF_INT 8
#define SIZEOF_SHORT 4
#define SIZEOF_DOUBLE 8
#define BITS_PER_BYTE 8

#define HAVE_PXFGETARG
#define HAVE_SYS_RESOURCE_H
#define HAVE_CLOCK

#define WORDS_BIGENDIAN 1

#define HAVE_SLEEP
#define PETSC_CAN_SLEEP_AFTER_ERROR
#define PETSC_USE_CTABLE
#define PETSC_USE_SBREAK_FOR_SIZE

#define PETSC_HAVE_F90_H "f90impl/f90_t3e.h"
#define PETSC_HAVE_F90_C "src/sys/src/f90/f90_t3e.c"

#endif
