/*$Id: petscfix.h,v 1.99 1999/11/23 18:12:07 bsmith Exp bsmith $*/

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
   extern void   *malloc(long unsigned int);
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
extern int    vfprintf (FILE *, const char *, char *);
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
   extern int vsprintf(char *, const char *, char *); or
   extern char   *vsprintf (char *, const char *, char *); 
*/
#endif
#endif
