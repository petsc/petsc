#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.17 1999/03/24 15:12:23 bsmith Exp balay $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_t3d

#define HAVE_PWD_H 
#define HAVE_STRING_H 
#define HAVE_MALLOC_H 
#define HAVE_DRAND48 
#define HAVE_UNISTD_H
#define HAVE_STDLIB_H
#define HAVE_SYS_TIME_H 
#define HAVE_UNAME
#define HAVE_X11

#define HAVE_FORTRAN_CAPS 
#define USES_CPTOFCD  
#define USES_FORTRAN_SINGLE

#define HAVE_READLINK
#define HAVE_MEMMOVE

#define CANNOT_START_DEBUGGER

#define HAVE_DOUBLE_ALIGN
#define HAVE_DOUBLE_ALIGN_MALLOC

#define HAVE_FAST_MPI_WTIME
#define HAVE_T3EF90

#define SIZEOF_VOIDP 8
#define SIZEOF_INT 8
#define SIZEOF_SHORT 4
#define SIZEOF_DOUBLE 8

#define HAVE_MISSING_DGESVD
#define HAVE_PXFGETARG
#define HAVE_SYS_RESOURCE_H
#define HAVE_CLOCK

#define WORDS_BIGENDIAN 1

#define CAN_SLEEP_AFTER_ERROR
#define USE_CTABLE

#endif
