#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.1 2000/12/15 15:57:44 bsmith Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_UXPV
#define PETSC_ARCH_NAME "uxpv"

#define HAVE_PWD_H 
#define HAVE_STRING_H 
#define HAVE_MALLOC_H 
#define HAVE_STDLIB_H 
#define HAVE_LIMITS_H
#define HAVE_GETCWD
#define HAVE_SLEEP
#define HAVE_SYS_PARAM_H
#define HAVE_SYS_STAT_H

#define HAVE_DRAND48  
#define HAVE_UNISTD_H 
#define HAVE_SYS_TIME_H 
#define HAVE_UNAME  
#define PETSC_HAVE_FORTRAN_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE
#define PETSC_NEED_UTYPE_TYPEDEFS
#define HAVE_SYS_RESOURCE_H

#define SIZEOF_VOID_P 8
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8
#define BITS_PER_BYTE 8

#define WORDS_BIGENDIAN 1
#endif
