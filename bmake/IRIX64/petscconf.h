#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.23 1999/02/15 18:42:04 balay Exp bsmith $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H
 
#define PARCH_IRIX64 

#define HAVE_PWD_H 
#define HAVE_STRING_H 
#define HAVE_STROPTS_H 
#define HAVE_MALLOC_H 
#define HAVE_X11
#define HAVE_DRAND48 
#define HAVE_GETDOMAINNAME 
#define HAVE_UNAME 
#define HAVE_UNISTD_H 
#define HAVE_STDLIB_H
#define HAVE_SYS_TIME_H 
#define USE_SHARED_MEMORY

#define HAVE_FORTRAN_UNDERSCORE 
#define SIZEOF_VOIDP 8
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8

#define HAVE_IRIXF90

#define HAVE_IRIXF90

#define WORDS_BIGENDIAN 1

#define HAVE_MEMMOVE
#define NEEDS_GETTIMEOFDAY_PROTO

#define HAVE_DOUBLE_ALIGN
#define HAVE_DOUBLE_ALIGN_MALLOC

#define HAVE_MEMALIGN

#define HAVE_FAST_MPI_WTIME

#define USE_DBX_DEBUGGER
#define HAVE_SYS_RESOURCE_H

#define USE_DYNAMIC_LIBRARIES 1
#define HAVE_RTLD_GLOBAL 1

#define CAN_SLEEP_AFTER_ERROR
#endif
