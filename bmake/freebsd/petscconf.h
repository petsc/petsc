#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.4 1998/04/25 23:23:55 balay Exp bsmith $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_freebsd

#define HAVE_PWD_H 
#define HAVE_STDLIB_H 
#define HAVE_STRING_H 
#define HAVE_SWAPPED_BYTES 
#define HAVE_X11 
#define HAVE_DRAND48  
#define HAVE_GETDOMAINNAME 
#define HAVE_UNISTD_H  
#define HAVE_UNAME 
#define HAVE_SYS_TIME_H

#define HAVE_FORTRAN_UNDERSCORE_UNDERSCOR
#define HAVE_FORTRAN_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE

#if (__GNUC__ == 2 && __GNUC_MINOR__ >= 7)
#define HAVE_VPRINTF_CHAR
#endif
#define HAVE_SYS_RESOURCE_H

#endif
