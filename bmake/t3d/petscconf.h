#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.9 1998/04/26 21:09:58 balay Exp bsmith $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_t3d 

#define HAVE_PWD_H 
#define HAVE_STRING_H 
#define HAVE_MALLOC_H
#define HAVE_64BIT_INT 
#define HAVE_DRAND48
#define USES_CPTOFCD  
#define HAVE_UNISTD_H
#define HAVE_STDLIB_H
#define HAVE_SYS_TIME_H 
#define HAVE_UNAME

#define HAVE_FORTRAN_CAPS 
#define USES_FORTRAN_SINGLE

#define HAVE_READLINK
#define HAVE_MEMMOVE

#define CANNOT_START_DEBUGGER

#define HAVE_DOUBLE_ALIGN
#define HAVE_DOUBLE_ALIGN_MALLOC

#define HAVE_FAST_MPI_WTIME

#define HAVE_64BITS
#define HAVE_MISSING_DGESVD

#endif
