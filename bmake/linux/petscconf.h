/* $Id: petscconf.h,v 1.3 1998/04/14 02:44:34 bsmith Exp balay $ */

/*
    Defines the configuration for this machine
*/

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
#define HAVE_SWAPPED_BYTES 

#define HAVE_FORTRAN_UNDERSCORE 
#define HAVE_FORTRAN_UNDERSCORE_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE

#define HAVE_MEMALIGN

#endif
