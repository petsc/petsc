#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.12 1998/11/10 20:52:07 balay Exp bsmith $"
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
#define _INCLUDE_POSIX_SOURCE
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

#define HAVE_SYS_RESOURCE_H

#define HAVE_CLOCK
#define SIZEOF_VOIDP 4
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8

#define WORDS_BIGENDIAN 1
#define HAVE_HPUXF90

#endif
