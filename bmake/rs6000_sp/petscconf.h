/* $Id: petscconf.h,v 1.7 1998/04/20 19:27:19 bsmith Exp balay $ */

/*
    Defines the configuration for this machine
*/
#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_rs6000

#define HAVE_STROPTS_H 
#define HAVE_SEARCH_H 
#define HAVE_PWD_H 
#define HAVE_STDLIB_H
#define HAVE_STRING_H 
#define HAVE_STRINGS_H 
#define HAVE_MALLOC_H 
#define HAVE_X11 
#define _POSIX_SOURCE 
#define HAVE_DRAND48  
#define HAVE_GETDOMAINNAME 
#define _XOPEN_SOURCE 
#define HAVE_UNISTD_H 
#define HAVE_SYS_TIME_H 
#define HAVE_UNAME 
#define _XOPEN_SOURCE_EXTENDED  1
#define _ALL_SOURCE   
#define HAVE_BROKEN_REQUEST_FREE 
#define HAVE_STRINGS_H
#define HAVE_DOUBLE_ALIGN_MALLOC

#if !defined(HAVE_XLF90)
#define HAVE_XLF90
#endif

#define PREFER_BZERO

#define HAVE_READLINK
#define HAVE_MEMMOVE

#define HAVE_PRAGMA_DISJOINT


#endif
