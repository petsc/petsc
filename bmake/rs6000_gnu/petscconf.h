#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.11 1999/02/08 22:22:33 bsmith Exp balay $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_rs6000 
#define USE_IBM_ASM_CLOCK

#define HAVE_STROPTS_H 
#define HAVE_SEARCH_H 
#define HAVE_PWD_H 
#define HAVE_STDLIB_H
#define HAVE_STRING_H 
#define HAVE_STRINGS_H 
#define HAVE_MALLOC_H 
#define _POSIX_SOURCE
#define HAVE_DRAND48  
#define HAVE_GETDOMAINNAME  
#if !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 
#endif
#define HAVE_UNISTD_H 
#define HAVE_SYS_TIME_H 
#define NEED_UTYPE_TYPEDEFS 
#define _XOPEN_SOURCE_EXTENDED 1
#define HAVE_UNAME  
#define HAVE_BROKEN_REQUEST_FREE 
#define NEEDS_GETTIMEOFDAY_PROTO
#define USES_TEMPLATED_COMPLEX
#define HAVE_DOUBLE_ALIGN_MALLOC

#define HAVE_FORTRAN_UNDERSCORE 
#define HAVE_FORTRAN_UNDERSCORE_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE
#define HAVE_SYS_RESOURCE_H

#define SIZEOF_VOIDP 4
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8

#define WORDS_BIGENDIAN 1

#endif
