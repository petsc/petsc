
/*
    This fixes various things in system files that are incomplete.
   For instance many systems don't properly prototype all system functions.

    This is included by files in src/sys/src and src/viewer/impls/.
*/

#if !defined(_PETSFIX_H)
#define _PETSCFIX_H

#ifndef MAXHOSTNAMELEN
#define MAXHOSTNAMELEN 64
#endif

/* -------------------------Sun Sparc Adjustments  ----------------------*/
#if defined(PARCH_sun4)

#if defined(__cplusplus)
extern "C" {
extern char   *mktemp(char *);
extern char   *getcwd(char *,long unsigned int);
extern char   *getwd(char *);
extern int     gethostname(char *,int);
extern int     getdomainname(char *,int);
extern char   *realpath(char *,char *);
extern char   *getenv( char *);
extern void   *malloc(long unsigned int );
#include <search.h>
extern char   *tsearch(char *,char **, int (*)(void*,void*));
extern void   twalk(char *,void (*)(void*,VISIT,int));
extern int    atoi(char*);
extern void   perror(const char *);
extern double atof(const char *);
extern void    free(void *);
extern void   *malloc(long unsigned int );
#include <sys/time.h>
extern int    gettimeofday(struct timeval *,struct timezone *);
extern int    abort();
extern int    exit(int);
};

#else
extern char   *getwd(char *);
extern char   *mktemp(char *);
extern int     gethostname(char *,int);
extern int     getdomainname(char *,int);
extern char   *realpath(char *,char *);
extern char   *getenv( char *);
extern int    atoi(char*);
extern double atof(const char*);
#include <search.h>
extern char   *tsearch(char *,char **, int (*)(void*,void*));
extern void   twalk(char *,void (*)(void*,VISIT,int));
extern int    fclose(FILE *);
extern void   perror(const char *);
extern int    vfprintf (FILE *, const char *, char * );
extern int    vsprintf (char *, const char *, char * );
#endif
#endif


/* -----------------------Sun Sparc running Solaris ------------------------*/
#if defined(PARCH_solaris)
#include <sys/utsname.h>
#include <sys/systeminfo.h>
extern char   *mktemp(char *);
extern double atof(const char*);
#endif

/* ----------------------IBM RS6000 ----------------------------------------*/
#if defined(PARCH_rs6000)

#if defined(__cplusplus)
extern "C" {
extern char   *mktemp(char *);
extern char   *getcwd(char *,long unsigned int);
extern char   *getwd(char *);
extern int     gethostname(char *,int);
extern int     getdomainname(char *,int);
extern void   abort(void);
extern int    atoi(const char*);
extern void   exit(int);
extern void   perror(const char *);
extern double atof(const char *);
extern void   free(void *);
extern void   *malloc(long unsigned int );
};

#else
extern char   *mktemp(char *);
#endif
#endif

/* -----------------------freeBSD ------------------------------------------*/
#if defined(PARCH_freebsd)

#if defined(__cplusplus)
extern "C" {
extern char   *mktemp(char *);
extern char   *getwd(char *);
extern int     gethostname(char *,int);
extern int     getdomainname(char *,int);
extern char   *getenv( char *);
extern int    atoi(char*);
extern int    exit(int);
extern void   perror(const char *);
extern int    abort();
extern double atof(const char *);
extern int    free(void *);
extern void   *malloc(long unsigned int );
};

#else
extern int    getdomainname(char *,int);
extern char   *getenv( char *);
extern double atof(char *);
extern int    atoi(char*);
#endif
#endif

/* -----------------------SGI IRIX -----------------------------------------*/
#if defined(PARCH_IRIX)

#if defined(__cplusplus)
extern "C" {
extern char   *mktemp(char *);
extern char   *getwd(char *);
extern int     gethostname(char *,int);
extern int     getdomainname(char *,int);
extern char   *getenv( char *);
extern int    atoi(char*);
extern void   perror(const char *);
extern int    abort();
extern double atof(const char *);
extern int    free(void *);
extern void   *malloc(long unsigned int );
extern int    abort();
extern double rint(double);
};

#else
extern char   *getenv( char *);
extern int    atoi(char*);
#endif
#endif

/* -----------------------DEC alpha ----------------------------------------*/

#if defined(PARCH_alpha)

#if defined(__cplusplus)
extern "C" {
extern char   *mktemp(char *);
extern char   *getcwd(char *,long unsigned int);
extern char   *getwd(char *);
extern int    gethostname(char *,int);
extern int    getdomainname(char *,int);
extern void   perror(const char *);
extern double atof(const char *);
extern void   *malloc(long unsigned int );
extern int    readlink(const char *,char *,int);
extern void   usleep(unsigned int);
extern unsigned int sleep(unsigned int);
};

#else
extern char   *mktemp(char *);
extern void   *malloc(long unsigned int);
extern char   *getenv( char *);
extern void   perror(char *);
extern double atof(char *);
extern int    atoi(char*);
#endif
#endif

/* -------------------- HP UX --------------------------------*/
#if defined(PARCH_hpux)

#if defined(__cplusplus)
extern int     getdomainname(char *,int);
extern int    exit(int);
extern int    abort();

#else
extern char *mktemp(char*);
#define SIGBUS _SIGBUS
#define SIGSYS _SIGSYS
#endif
#endif

/* --------------------NeXT -----------------------------------*/
#if defined(PARCH_NeXT)

#if defined(__cplusplus)

#else
extern char   *mktemp(char *);
#endif
#endif

/* ------------------ Cray t3d --------------------------------*/
#if defined(PARCH_t3d)

#if defined(__cplusplus)

#else
extern char   *getenv( char *);
extern char   *mktemp(char *);
#endif
#endif

/* -------------------------------------------------------------------------*/
#if defined(PARCH_paragon)

#if defined(__cplusplus)

#else
extern char   *mktemp(char *);
extern char   *getenv( char *);
extern void   *malloc(long unsigned int );
extern int    free(void *);
extern double atof(char *);
#endif
#endif

/* -----------------------linux ------------------------------------------*/
#if defined(PARCH_linux)

#if defined(__cplusplus)
extern "C" {
extern char   *mktemp(char *);
extern char   *getwd(char *);
extern int     gethostname(char *,int);
extern int     getdomainname(char *,int);
extern char   *getenv( char *);
extern int    atoi(char*);
extern int    exit(int);
extern void   perror(const char *);
extern int    abort();
extern double atof(const char *);
extern int    free(void *);
extern void   *malloc(long unsigned int );
};

#else
extern char   *getenv( char *);
extern double atof(char *);
extern int    atoi(char*);
#endif
#endif

#endif










