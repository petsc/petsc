
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

/* -----------------------freeBSD ------------------------------------------*/
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
    extern void   *malloc(long unsigned int);
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
