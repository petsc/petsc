#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.7 1998/10/05 19:00:55 balay Exp bsmith $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_freebsd

#define HAVE_PWD_H 
#define HAVE_STDLIB_H 
#define HAVE_STRING_H 
#define HAVE_X11 
#define HAVE_DRAND48  
#define HAVE_GETDOMAINNAME 
#define HAVE_UNISTD_H  
#define HAVE_UNAME 
#define HAVE_SYS_TIME_H

#define HAVE_FORTRAN_UNDERSCORE_UNDERSCORE
#define HAVE_FORTRAN_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE

#if (__GNUC__ == 2 && __GNUC_MINOR__ >= 7)
#define HAVE_VPRINTF_CHAR
#endif
#define HAVE_SYS_RESOURCE_H
#define SIZEOF_VOIDP 4
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8

#endif
