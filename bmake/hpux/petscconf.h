#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.7 1998/05/05 19:58:25 bsmith Exp bsmith $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_hpux 

#define HAVE_STDLIB_H 
#define HAVE_PWD_H 
#define HAVE_MALLOC_H 
#define HAVE_STRING_H 
#define HAVE_X11 
#define _POSIX_SOURCE 
#define HAVE_DRAND48 
#define _INCLUDE_XOPEN_SOURCE 
#define _INCLUDE_XOPEN_SOURCE_EXTENDED 
#define _INCLUDE_HPUX_SOURCE 
#define HAVE_GETDOMAINNAME 
#define HAVE_SYS_TIME_H
#define HAVE_UNISTD_H 
#define HAVE_UNAME

#define HAVE_READLINK
#define HAVE_MEMMOVE

#define USE_XDB_DEBUGGER

#define HAVE_BROKEN_RECURSIVE_MACRO
#define HAVE_SYS_RESOURCE_H

#define HAVE_CLOCK

#endif
