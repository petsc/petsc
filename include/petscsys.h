/* $Id: sys.h,v 1.27 1997/05/23 18:33:50 balay Exp bsmith $ */
/*
    Provides access to system related and general utility routines.
*/
#if !defined(__SYS_PACKAGE)
#define __SYS_PACKAGE

#include "petsc.h"

extern int  PetscGetArchType(char *,int);
extern int  PetscGetHostName(char *,int);
extern int  PetscGetUserName(char *,int);

extern char *PetscGetDate();

extern int  PetscSortInt(int,int*);
extern int  PetscSortIntWithPermutation(int,int*,int*);
extern int  PetscSortDouble(int,double*);
extern int  PetscSortDoubleWithPermutation(int,double*,int*);

extern int  PetscSetDisplay();
extern int  PetscGetDisplay(char *,int);

#define PETSCRANDOM_COOKIE PETSC_COOKIE+19

typedef enum { RANDOM_DEFAULT, RANDOM_DEFAULT_REAL, 
               RANDOM_DEFAULT_IMAGINARY } PetscRandomType;

typedef struct _p_PetscRandom*   PetscRandom;

extern int PetscRandomCreate(MPI_Comm,PetscRandomType,PetscRandom*);
extern int PetscRandomGetValue(PetscRandom,Scalar*);
extern int PetscRandomSetInterval(PetscRandom,Scalar,Scalar);
extern int PetscRandomDestroy(PetscRandom);

extern int PetscGetFullPath(char*,char*,int);
extern int PetscGetRelativePath(char*,char*,int);
extern int PetscGetWorkingDirectory(char *, int);
extern int PetscGetRealPath(char *,char*);
extern int PetscGetHomeDirectory(int,char*);

typedef enum { BINARY_INT, BINARY_DOUBLE, BINARY_SHORT, BINARY_FLOAT,
               BINARY_CHAR } PetscBinaryType;
#define BINARY_SCALAR BINARY_DOUBLE /* not correct if compiled with complex */
extern int PetscBinaryRead(int,void*,int,PetscBinaryType);
extern int PetscBinaryWrite(int,void*,int,PetscBinaryType,int);
extern int PetscBinaryOpen(char *,int,int *);
extern int PetscBinaryClose(int);

extern int PetscSetDebugger(char *,int,char *);
extern int PetscAttachDebugger();

#endif      

