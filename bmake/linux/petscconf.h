#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.14 1998/10/20 15:30:51 bsmith Exp bsmith $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_linux

#define HAVE_PWD_H 
#define HAVE_MALLOC_H 
#define HAVE_STRING_H 
#define HAVE_X11
#define HAVE_GETDOMAINNAME
#define HAVE_DRAND48 
#define HAVE_UNAME 
#define HAVE_UNISTD_H 
#define HAVE_SYS_TIME_H 
#define HAVE_STDLIB_H
#define HAVE_UNISTD_H

#define HAVE_FORTRAN_UNDERSCORE 
#define HAVE_FORTRAN_UNDERSCORE_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE

#define HAVE_DOUBLE_ALIGN_MALLOC
#define HAVE_MEMALIGN
#define HAVE_SYS_RESOURCE_H
#define SIZEOF_VOIDP 4
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8

#if defined(fixedsobug)
#define USE_DYNAMIC_LIBRARIES 1
#define HAVE_RTLD_GLOBAL 1
#endif

#endif
