#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.9 1998/04/25 23:22:55 balay Exp bsmith $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H
 
#define PARCH_IRIX

#define HAVE_PWD_H 
#define HAVE_STRING_H 
#define HAVE_STROPTS_H
#define HAVE_MALLOC_H 
#define HAVE_X11 
#define HAVE_FORTRAN_UNDERSCORE 
#define HAVE_DRAND48  
#define HAVE_GETDOMAINNAME 
#define HAVE_UNAME 
#define HAVE_UNISTD_H 
#define HAVE_SYS_TIME_H

#define HAVE_FORTRAN_UNDERSCORE 

#define NEEDS_GETTIMEOFDAY_PROTO

#define HAVE_DOUBLE_ALIGN
#define HAVE_DOUBLE_ALIGN_MALLOC

#define HAVE_MEMALIGN

#define USE_DBX_DEBUGGER

#endif
