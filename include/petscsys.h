/* $Id: sys.h,v 1.45 2000/02/03 22:03:49 bsmith Exp bsmith $ */
/*
    Provides access to system related and general utility routines.
*/
#if !defined(__SYS_H)
#define __SYS_H

#include "petsc.h"
#include <errno.h> 

extern int  PetscGetArchType(char[],int);
extern int  PetscGetHostName(char[],int);
extern int  PetscGetUserName(char[],int);
extern int  PetscGetProgramName(char[],int);
extern int  PetscSetProgramName(const char[]);
extern int  PetscGetDate(char[],int);
extern int  PetscSetInitialDate(void);
extern int  PetscGetInitialDate(char[],int);

extern int  PetscSortInt(int,int[]);
extern int  PetscSortIntWithPermutation(int,const int[],int[]);
extern int  PetscSortDouble(int,double[]);
extern int  PetscSortDoubleWithPermutation(int,const double[],int[]);

extern int  PetscSetDisplay(void);
extern int  PetscGetDisplay(char[],int);

#define PETSCRANDOM_COOKIE PETSC_COOKIE+19

typedef enum { RANDOM_DEFAULT,RANDOM_DEFAULT_REAL,
               RANDOM_DEFAULT_IMAGINARY } PetscRandomType;

typedef struct _p_PetscRandom*   PetscRandom;

extern int PetscRandomCreate(MPI_Comm,PetscRandomType,PetscRandom*);
extern int PetscRandomGetValue(PetscRandom,Scalar*);
extern int PetscRandomSetInterval(PetscRandom,Scalar,Scalar);
extern int PetscRandomDestroy(PetscRandom);

extern int PetscGetFullPath(const char[],char[],int);
extern int PetscGetRelativePath(const char[],char[],int);
extern int PetscGetWorkingDirectory(char[],int);
extern int PetscGetRealPath(char[],char[]);
extern int PetscGetHomeDirectory(char[],int);
extern int PetscTestFile(const char[],char,PetscTruth*);
extern int PetscBinaryRead(int,void*,int,PetscDataType);
extern int PetscBinaryWrite(int,void*,int,PetscDataType,int);
extern int PetscBinaryOpen(const char[],int,int *);
extern int PetscBinaryClose(int);
extern int PetscSharedTmp(MPI_Comm,PetscTruth *);
extern int PetscSharedWorkingDirectory(MPI_Comm,PetscTruth *);
extern int PetscGetTmp(MPI_Comm,char *,int);
extern int PetscFileRetrieve(MPI_Comm,const char *,char *,int,PetscTruth*);

/*
   In binary files variables are stored using the following lengths,
  regardless of how they are stored in memory on any one particular
  machine. Use these rather then sizeof() in computing sizes for 
  PetscBinarySeek().
*/
#define BINARY_INT_SIZE    (32/8)
#define BINARY_FLOAT_SIZE  (32/8)
#define BINARY_CHAR_SIZE    (8/8)
#define BINARY_SHORT_SIZE  (16/8)
#define BINARY_DOUBLE_SIZE (64/8)
#define BINARY_SCALAR_SIZE sizeof(Scalar)

typedef enum {BINARY_SEEK_SET = 0,BINARY_SEEK_CUR = 1,BINARY_SEEK_END = 2} PetscBinarySeekType;
extern int PetscBinarySeek(int,int,PetscBinarySeekType,int*);

extern int PetscSetDebugger(const char[],int);
extern int PetscAttachDebugger(void);
extern int PetscStopForDebugger(void);

#endif      

