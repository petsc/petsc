/* $Id: sys.h,v 1.11 1996/01/22 01:13:28 curfman Exp curfman $ */
/*
    Provides access to a small number of system related and general utility routines.
*/
#if !defined(__SYS_PACKAGE)
#define __SYS_PACKAGE

#include "petsc.h"

extern int  SYGetArchType(char*,int);
extern int  SYIsort(int,int*);
extern int  SYIsortperm(int,int*,int*);
extern int  SYDsort(int,double*);
extern char *SYGetDate();
extern int  TrDebugLevel(int);
extern int  TrValid();

#define SYRANDOM_COOKIE PETSC_COOKIE+19

typedef enum { RANDOM_DEFAULT } SYRandomType;

typedef struct _SYRandom*   SYRandom;

extern int  SYRandomCreate(SYRandomType,SYRandom*);
extern int  SYRandomGetValue(SYRandom,Scalar*);
extern int  SYRandomDestroy(SYRandom);

#endif      

