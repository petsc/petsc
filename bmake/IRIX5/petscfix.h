/*$Id: rich.c,v 1.85 1999/11/05 14:46:46 bsmith Exp bsmith $*/

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



/* -------------------------Sun Sparc Adjustments  ----------------------*/
#if defined(PARCH_sun4)
#if defined(__cplusplus)
extern "C" {
extern int     getdomainname(char *,int);
extern char   *getwd(char *);
extern int    getrusage(int,s_rusage);
extern char   *mktemp(char *);
extern char   *realpath(char *,char *);
extern void   *memalign(int,int);
extern int    getpagesize();
/*
   On some machines with older versions of the gnu compiler and 
   system libraries these prototypes may be needed; they are now
   prototyped in the GNU version of stdlib.h
   
   extern char   *getenv( char *);
   extern int    atoi(char*);
   extern char   *getcwd(char *,long unsigned int);
   extern void   *malloc(long unsigned int );
   extern void   perror(const char *);
   extern double atof(const char *);
   extern void    free(void *);
   extern int    strcasecmp(const char *,const char *);
   extern void   exit(int);
   extern int    abort(); 
*/
}

#else
extern char   *getwd(char *);
extern char   *mktemp(char *);
extern int     getdomainname(char *,int);
extern char   *realpath(char *,char *);
extern int    getrusage(int,s_rusage);
extern int    vfprintf (FILE *, const char *, char * );
extern int    getpagesize();
/*
   On some machines with older versions of the gnu compiler and 
   system libraries these prototypes may be needed
   
   extern char   *getenv( char *);
   extern int    atoi(char*);
   extern double atof(const char*);
   extern int    fclose(FILE *);
   extern void   perror(const char *);
   extern int    strcasecmp(const char *,const char *);
   extern int vsprintf(char *, const char *, char * ); or
   extern char   *vsprintf (char *, const char *, char * ); 
*/
#endif
#endif


/* -----------------------Sun Sparc running Solaris ------------------------*/
#if defined(PARCH_solaris)
#include <sys/utsname.h>
#include <sys/systeminfo.h>
#if defined(__cplusplus)
extern "C" {
extern char     *mktemp(char *);
/*
   Older OS versions may require
   
   extern double   atof(const char*);
*/
}
#else

extern char   *mktemp(char *); 
/*
   Older OS versions may require
   
   extern double atof(const char*);
*/
#endif

#endif

/* ----------------------IBM RS6000 ----------------------------------------*/
#if defined(PARCH_rs6000)
/* Some of the following prototypes are present in AIX 4.2 but not in AIX 3.X */
#if defined(__cplusplus)
extern "C" {
extern char   *mktemp(char *);
extern char   *getwd(char *);
extern int    getdomainname(char *,int);
extern int    strcasecmp(const char *, const char *);
extern int    getrusage(int,s_rusage);
}
#else
extern char   *mktemp(char *);
extern int    strcasecmp(const char *, const char *);
extern int    getrusage(int,s_rusage);
#endif
#endif

/* -----------------------freeBSD ------------------------------------------*/
#if defined(PARCH_freebsd)
#if defined(__cplusplus)
extern "C" {
extern char   *mktemp(char *);
extern char   *getwd(char *);
extern int    getdomainname(char *,int);
extern void   perror(const char *);
extern double atof(const char *);
extern int    getrusage(int,s_rusage);
extern int    getpagesize();
/*
    These where added to freeBSD recently, thus no longer are needed.
    If you have an old installation of freeBSD you may need the 
    prototypes below.
    
    extern int    free(void *);
    extern void   *malloc(long unsigned int );
    extern char   *getenv( char *);
    extern int    atoi(char*);
    extern int    exit(int);
    extern int    abort();
*/
}

#else
extern int    getdomainname(char *,int);
extern int    getrusage(int,s_rusage);
extern int    getpagesize();
/* 
    These were added to the latest freeBSD release, thus no longer needed.
    If you have an old installation of freeBSD you may need the 
    prototypes below.

    extern char   *getenv( char *);
    extern double atof(char *);
    extern int    atoi(char*);
*/
#endif
#endif

/* -----------------------SGI IRIX -----------------------------------------*/
#if defined(PARCH_IRIX) || defined(PARCH_IRIX64) || defined(PARCH_IRIX5)
#if defined(__cplusplus)
extern "C" {
/* 
    Variation needed on older versions of the OS

    extern int gettimeofday(struct timeval *, struct timezone *);
*/
#include <sys/resource.h>
extern int gettimeofday(struct timeval *,...);
}
#else
#endif
#endif

/* -----------------------DEC alpha ----------------------------------------*/

#if defined(PARCH_alpha)
#if defined(__cplusplus)
extern "C" {
extern int    getdomainname(char *,int);
extern unsigned int sleep (unsigned int );

}
#else
#endif
#endif

/* -------------------- HP UX --------------------------------*/
#if defined(PARCH_hpux)
#if defined(__cplusplus)
extern "C" {
extern int    getdomainname(char *,int);
extern void   exit(int);
extern void   abort();
extern int    readlink(const char *, char *, size_t);
}
#else
extern char   *mktemp(char*);
#define SIGBUS  _SIGBUS
#define SIGSYS  _SIGSYS
#define SIGQUIT _SIGQUIT
#endif
#endif

/* ------------------ Cray t3d --------------------------------*/
#if defined(PARCH_t3d)

#if defined(__cplusplus)
extern "C" {
extern unsigned int sleep(unsigned int);
extern int          close(int);
}
#else
#endif
#endif

/* ------------------ Cray t3d --------------------------------*/
#if defined(PARCH_ascired)

#if defined(__cplusplus)
extern "C" {
extern char *getwd(char *_name);
}
#else
extern char *getwd(char *_name);
#endif
#endif

/* -------------------------------------------------------------------------*/
#if defined(PARCH_paragon)

#if defined(__cplusplus)

#else
extern char   *mktemp(char *);
extern char   *getenv( char *);
extern void   *malloc(long unsigned int );
/*
  Earlier versions of the Paragon use
  extern int    free(void *);
*/
extern void   free(void *);
extern double atof(char *);
extern int    getpagesize();
#endif
#endif

/* -----------------------linux ------------------------------------------*/
#if defined(PARCH_linux)
#if defined(__cplusplus)
extern "C" {
extern void* memalign (int,int);
}
#endif
#endif

/* -----------------------Windows NT with gcc --------------------------*/
#if defined(PARCH_win32_gnu)

#if defined(__cplusplus)
extern "C" {
#include <unistd.h>
/* The following are suspicious. Not sure if they really exist */
extern int    readlink(const char *, char *, int);
extern int    getdomainname(char *,int);
#if !defined (htons)
#define htons __htons
#endif
}

#else

#include <unistd.h>
/* The following are suspicious. Not sure if they really exist */
extern int    readlink(const char *, char *, int);
extern int    getdomainname(char *,int);
#endif
#endif

/* -----------------------Windows NT with MS Visual C++ ---------------------*/
#if defined(PARCH_win32)
#endif

#endif









