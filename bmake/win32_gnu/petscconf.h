#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.17 1999/02/08 22:22:33 bsmith Exp balay $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_win32_gnu

#define HAVE_SLOW_NRM2
#define HAVE_SEARCH_H 
#define HAVE_PWD_H 
#define HAVE_STRING_H
#define HAVE_X11 
#define HAVE_GETDOMAINNAME  
#define HAVE_UNISTD_H
#define HAVE_SYS_TIME_H 
#define HAVE_UNAME
#define HAVE_MALLOC_H
#define HAVE_STDLIB_H
#define HAVE_UNISTD_H
#define HAVE_SYS_TIME_H

#define HAVE_FORTRAN_UNDERSCORE 
#define HAVE_FORTRAN_UNDERSCORE_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE
#define HAVE_RAND
#define HAVE_DOUBLE_ALIGN_MALLOC

#define CANNOT_START_DEBUGGER
#define HAVE_SYS_RESOURCE_H

#define HAVE_GET_USER_NAME
#define SIZEOF_VOIDP 4
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8

#define USE_NT_TIME
#endif
