#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.2 1999/02/08 22:22:33 bsmith Exp balay $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_alpha
#define USE_GETCLOCK

#define HAVE_PWD_H 
#define HAVE_STRING_H 
#define HAVE_MALLOC_H 
#define HAVE_STDLIB_H 
#define HAVE_X11 
#define HAVE_DRAND48  
#define HAVE_GETDOMAINNAME  
#define HAVE_UNISTD_H 
#define HAVE_SYS_TIME_H 
#define HAVE_UNAME  
#define HAVE_FORTRAN_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE
#define NEED_UTYPE_TYPEDEFS
#define USE_DBX_DEBUGGER
#define HAVE_SYS_RESOURCE_H

#define SIZEOF_VOIDP 8
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8
 
#define USE_DYNAMIC_LIBRARIES 1
#define USE_NONEXECUTABLE_SO 1

#endif
