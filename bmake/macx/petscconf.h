#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.7 2001/03/31 01:16:54 balay Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H


#define PARCH_macx
#define PETSC_ARCH_NAME "macx"

#define HAVE_POPEN
#define HAVE_SLEEP

#define HAVE_SYS_WAIT_H 1
#define RETSIGTYPE void
#define STDC_HEADERS 1
#define TIME_WITH_SYS_TIME 1
#define WORDS_BIGENDIAN 1

#define SIZEOF_INT 4
#define SIZEOF_VOID_P 4
#define BITS_PER_BYTE 8
#define HAVE_GETCWD 1
#define HAVE_GETHOSTNAME 1
#define HAVE_GETTIMEOFDAY 1
#define HAVE_GETWD 1
#define HAVE_MEMMOVE 1
#define HAVE_RAND 1
#define HAVE_READLINK 1
#define HAVE_SIGACTION 1
#define HAVE_SIGNAL 1
#define HAVE_SOCKET 1
#define HAVE_STRSTR 1
#define HAVE_UNAME 1
#define HAVE_FCNTL_H 1
#define HAVE_LIMITS_H 1
#define HAVE_PWD_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_STRINGS_H 1
#define HAVE_SYS_RESOURCE_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_UNISTD_H 1
#define PETSC_HAVE_TEMPLATED_COMPLEX


#define PETSC_HAVE_FORTRAN_UNDERSCORE 
#define PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE

#define HAVE_DOUBLE_ALIGN_MALLOC
#define PETSC_HAVE_NO_GETRUSAGE


#endif

