#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.2 1998/10/19 22:14:14 bsmith Exp bsmith $"
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
#define HAVE_SYS_RESOURCE_H

#define HAVE_SYS_PROCFS_H
#define HAVE_FCNTL_H
#define SIZEOF_VOIDP 4
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8

#define WORDS_BIGENDIAN 1

#define USE_DYNAMIC_LIBRARIES 1
#define HAVE_RTLD_GLOBAL 1

#endif
