#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.6 1998/04/25 23:29:24 balay Exp bsmith $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_solaris 

#define HAVE_STROPTS_H 
#define HAVE_SEARCH_H 
#define HAVE_PWD_H 
#define HAVE_STRING_H 
#define HAVE_MALLOC_H
#define HAVE_STDLIB_H
#define HAVE_X11 
#define HAVE_UNISTD_H 
#define HAVE_DRAND48 
#define HAVE_SYS_TIME_H
#define HAVE_SYS_SYSTEMINFO_H
#define HAVE_SYSINFO
#define HAVE_SUNMATH_H
#define HAVE_RESTRICT
#define HAVE_SUNMATHPRO

#define HAVE_FORTRAN_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE

#define HAVE_DOUBLE_ALIGN
#define HAVE_DOUBLE_ALIGN_MALLOC
#define HAVE_MEMALIGN

#define USE_DBX_DEBUGGER

#endif
