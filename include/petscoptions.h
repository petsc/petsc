/* $Id: petscoptions.h,v 1.42 2000/09/22 20:47:47 bsmith Exp bsmith $ */
/*
   Routines to determine options set in the options database.
*/
#if !defined(__PETSCOPTIONS_H)
#define __PETSCOPTIONS_H
#include "petsc.h"

EXTERN int  PetscOptionsHasName(const char[],const char[],PetscTruth*);
EXTERN int  PetscOptionsGetInt(const char[],const char [],int *,PetscTruth*);
EXTERN int  PetscOptionsGetLogical(const char[],const char [],PetscTruth *,PetscTruth*);
EXTERN int  PetscOptionsGetDouble(const char[],const char[],double *,PetscTruth*);
EXTERN int  PetscOptionsGetScalar(const char[],const char[],Scalar *,PetscTruth*);
EXTERN int  PetscOptionsGetIntArray(const char[],const char[],int[],int *,PetscTruth*);
EXTERN int  PetscOptionsGetDoubleArray(const char[],const char[],double[],int *,PetscTruth*);
EXTERN int  PetscOptionsGetString(const char[],const char[],char[],int,PetscTruth*);
EXTERN int  PetscOptionsGetStringArray(const char[],const char[],char**,int*,PetscTruth*);

EXTERN int  PetscOptionsSetAlias(const char[],const char[]);
EXTERN int  PetscOptionsSetValue(const char[],const char[]);
EXTERN int  PetscOptionsClearValue(const char[]);

EXTERN int  PetscOptionsAllUsed(int *);
EXTERN int  PetscOptionsLeft(void);
EXTERN int  PetscOptionsPrint(FILE *);

EXTERN int  PetscOptionsCreate(void);
EXTERN int  PetscOptionsInsert(int *,char ***,const char[]);
EXTERN int  PetscOptionsInsertFile(const char[]);
EXTERN int  PetscOptionsDestroy(void);

EXTERN int  PetscOptionsReject(const char[],const char[]);
EXTERN int  PetscOptionsGetAll(char*[]);

EXTERN int  PetscOptionsGetenv(MPI_Comm,const char *,char[],int,PetscTruth *);
EXTERN int  PetscOptionsAtoi(const char[],int*);
EXTERN int  PetscOptionsAtod(const char[],double*);

extern PetscTruth PetscOptionsPublish;
extern int        PetscOptionsPublishCount;
#define    PetscOptionsBegin(comm,prefix,mess,sec) 0; {\
             for (PetscOptionsPublishCount=(PetscOptionsPublish?-1:1); PetscOptionsPublishCount<2; PetscOptionsPublishCount++) {\
             int __ierr = PetscOptionsBegin_Private(comm,prefix,mess,sec);CHKERRQ(__ierr);
#define    PetscOptionsEnd() __ierr = PetscOptionsEnd_Private();CHKERRQ(__ierr);}}
EXTERN int PetscOptionsBegin_Private(MPI_Comm,char*,char*,char*);
EXTERN int PetscOptionsEnd_Private(void);
EXTERN int PetscOptionsHead(char*);
#define    PetscOptionsTail() 0; {if (PetscOptionsPublishCount != 1) PetscFunctionReturn(0);}

EXTERN int PetscOptionsInt(char*,char*,char*,int,int*,PetscTruth*);
EXTERN int PetscOptionsDouble(char*,char*,char*,double,double*,PetscTruth*);
EXTERN int PetscOptionsScalar(char*,char*,char*,Scalar,Scalar*,PetscTruth*);
EXTERN int PetscOptionsName(char*,char*,char*,PetscTruth*);
EXTERN int PetscOptionsString(char*,char*,char*,char*,char*,int,PetscTruth*);
EXTERN int PetscOptionsLogical(char*,char*,char*,PetscTruth,PetscTruth*,PetscTruth*);
EXTERN int PetscOptionsLogicalGroupBegin(char*,char*,char*,PetscTruth*);
EXTERN int PetscOptionsLogicalGroup(char*,char*,char*,PetscTruth*);
EXTERN int PetscOptionsLogicalGroupEnd(char*,char*,char*,PetscTruth*);
EXTERN int PetscOptionsList(char*,char*,char*,PetscFList,char*,char*,int,PetscTruth*);
EXTERN int PetscOptionsEList(char*,char*,char*,char**,int,char*,char *,int,PetscTruth*);
EXTERN int PetscOptionsDoubleArray(char*,char*,char*,double[],int*,PetscTruth*);
EXTERN int PetscOptionsStringArray(char*,char*,char*,char**,int*,PetscTruth*);
#endif


