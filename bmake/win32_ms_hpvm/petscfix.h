/*$Id: petscfix.h,v 1.1 2001/03/27 22:16:33 balay Exp $*/

/*
    This fixes various things in system files that are incomplete, for 
  instance many systems don't properly prototype all system functions.
  It is not intended to DUPLICATE anything in the system include files;
  if the compiler reports a conflict between a prototye in a system file
  and this file then the prototype in this file should be removed.

    This is included by files in src/sys/src
*/

#if !defined(_PETSCFIX_H)
#define _PETSCFIX_H

#include "petsc.h"

/*
  This prototype lets us resolve the datastructure 'rusage' only in
  the source files using getrusage, and not in other source files.
*/
typedef struct rusage* s_rusage;

/* -----------------------Windows NT with MS Visual C++ ---------------------*/

/* Fixes from sys/types.h */
typedef int uid_t;
typedef int gid_t;

/* Fixes from sys/stat.h */
#if !defined(R_OK)
#define R_OK 04
#endif
#if !defined(W_OK)
#define W_OK 02
#endif
#if !defined(X_OK)
#define X_OK 01
#endif

#ifndef S_IFMT
#define S_IFMT _S_IFMT
#endif
#ifndef S_IFDIR
#define S_IFDIR _S_IFDIR
#endif
#ifndef S_IFCHR
#define S_IFCHR _S_IFCHR
#endif
#ifndef S_IFIFO
#define S_IFIFO _S_IFIFO
#endif
#ifndef S_IFREG
#define S_IFREG _S_IFREG
#endif
#ifndef S_IREAD
#define S_IREAD _S_IREAD
#endif
#ifndef S_IWRITE
#define S_IWRITE _S_IWRITE
#endif
#define S_IEXEC  _S_IEXEC                                                       
#ifndef S_IXUSR
#define S_IXUSR _S_IEXEC
#endif
#ifndef S_IXGRP
#define S_IXGRP _S_IEXEC
#endif
#ifndef S_IXOTH
#define S_IXOTH _S_IEXEC
#endif
#ifndef S_IRUSR
#define S_IRUSR _S_IREAD
#endif
#ifndef S_IWUSR
#define S_IWUSR _S_IWRITE
#endif
#ifndef S_IROTH
#define S_IROTH _S_IREAD
#endif
#ifndef S_IWOTH
#define S_IWOTH _S_IWRITE
#endif
#ifndef S_IRGRP
#define S_IRGRP _S_IREAD
#endif
#ifndef S_IWGRP
#define S_IWGRP _S_IWRITE
#endif
#ifndef O_RDWR
#define O_RDWR _O_RDWR
#endif
#ifndef O_CREAT
#define O_CREAT _O_CREAT
#endif
#ifndef O_TRUNC
#define O_TRUNC _O_TRUNC
#endif
#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#endif
#ifndef O_WRONLY
#define O_WRONLY _O_WRONLY
#endif
#ifndef O_APPEND
#define O_APPEND _O_APPEND
#endif
#ifndef O_TEXT
#define O_TEXT _O_TEXT
#endif
#ifndef O_BINARY
#define O_BINARY _O_BINARY
#endif
#ifndef O_EXCL
#define O_EXCL _O_EXCL
#endif

/* Test for each symbol individually and define the ones necessary (some
   systems claiming Posix compatibility define some but not all). */
 
#if defined (S_IFBLK) && !defined (S_ISBLK)
#define        S_ISBLK(m)      (((m)&S_IFMT) == S_IFBLK)       /* block device */
#endif
 
#if defined (S_IFCHR) && !defined (S_ISCHR)
#define        S_ISCHR(m)      (((m)&S_IFMT) == S_IFCHR)       /* character device */
#endif
 
#if defined (S_IFDIR) && !defined (S_ISDIR)
#define        S_ISDIR(m)      (((m)&S_IFMT) == S_IFDIR)       /* directory */
#endif
 
#if defined (S_IFREG) && !defined (S_ISREG)
#define        S_ISREG(m)      (((m)&S_IFMT) == S_IFREG)       /* file */
#endif                                                                          

#if defined (S_IFIFO) && !defined (S_ISFIFO)
#define        S_ISFIFO(m)     (((m)&S_IFMT) == S_IFIFO)       /* fifo - named pipe */
#endif
 
#if defined (S_IFLNK) && !defined (S_ISLNK)
#define        S_ISLNK(m)      (((m)&S_IFMT) == S_IFLNK)       /* symbolic link */
#endif
 
#if defined (S_IFSOCK) && !defined (S_ISSOCK)
#define        S_ISSOCK(m)     (((m)&S_IFMT) == S_IFSOCK)      /* socket */
#endif

#endif /* _PETSCFIX_H */
