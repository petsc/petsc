/* $Id: petscoptions.h,v 1.41 2000/09/02 02:50:55 bsmith Exp bsmith $ */
/*
   Routines to determine options set in the options database.
*/
#if !defined(__PETSCOPTIONS_H)
#define __PETSCOPTIONS_H
#include "petsc.h"

EXTERN int  OptionsHasName(const char[],const char[],PetscTruth*);
EXTERN int  OptionsGetInt(const char[],const char [],int *,PetscTruth*);
EXTERN int  OptionsGetLogical(const char[],const char [],PetscTruth *,PetscTruth*);
EXTERN int  OptionsGetDouble(const char[],const char[],double *,PetscTruth*);
EXTERN int  OptionsGetScalar(const char[],const char[],Scalar *,PetscTruth*);
EXTERN int  OptionsGetIntArray(const char[],const char[],int[],int *,PetscTruth*);
EXTERN int  OptionsGetDoubleArray(const char[],const char[],double[],int *,PetscTruth*);
EXTERN int  OptionsGetString(const char[],const char[],char[],int,PetscTruth*);
EXTERN int  OptionsGetStringArray(const char[],const char[],char**,int*,PetscTruth*);

EXTERN int  OptionsSetAlias(const char[],const char[]);
EXTERN int  OptionsSetValue(const char[],const char[]);
EXTERN int  OptionsClearValue(const char[]);

EXTERN int  OptionsAllUsed(int *);
EXTERN int  OptionsLeft(void);
EXTERN int  OptionsPrint(FILE *);

EXTERN int  OptionsCreate(void);
EXTERN int  OptionsInsert(int *,char ***,const char[]);
EXTERN int  OptionsInsertFile(const char[]);
EXTERN int  OptionsDestroy(void);

EXTERN int  OptionsReject(const char[],const char[]);
EXTERN int  OptionsGetAll(char*[]);

EXTERN int  OptionsGetenv(MPI_Comm,const char *,char[],int,PetscTruth *);
EXTERN int  OptionsAtoi(const char[],int*);
EXTERN int  OptionsAtod(const char[],double*);

extern PetscTruth OptionsPublish;
extern int        OptionsPublishCount;
#define    OptionsBegin(comm,prefix,mess,sec) 0; {\
             for (OptionsPublishCount=(OptionsPublish?-1:1); OptionsPublishCount<2; OptionsPublishCount++) {\
             int __ierr = OptionsBegin_Private(comm,prefix,mess,sec);CHKERRQ(__ierr);
#define    OptionsEnd() __ierr = OptionsEnd_Private();CHKERRQ(__ierr);}}
EXTERN int OptionsBegin_Private(MPI_Comm,char*,char*,char*);
EXTERN int OptionsEnd_Private(void);
EXTERN int OptionsHead(char*);
#define    OptionsTail() 0; {if (OptionsPublishCount != 1) PetscFunctionReturn(0);}

EXTERN int OptionsInt(char*,char*,char*,int,int*,PetscTruth*);
EXTERN int OptionsDouble(char*,char*,char*,double,double*,PetscTruth*);
EXTERN int OptionsScalar(char*,char*,char*,Scalar,Scalar*,PetscTruth*);
EXTERN int OptionsName(char*,char*,char*,PetscTruth*);
EXTERN int OptionsString(char*,char*,char*,char*,char*,int,PetscTruth*);
EXTERN int OptionsLogical(char*,char*,char*,PetscTruth,PetscTruth*,PetscTruth*);
EXTERN int OptionsLogicalGroupBegin(char*,char*,char*,PetscTruth*);
EXTERN int OptionsLogicalGroup(char*,char*,char*,PetscTruth*);
EXTERN int OptionsLogicalGroupEnd(char*,char*,char*,PetscTruth*);
EXTERN int OptionsList(char*,char*,char*,FList,char*,char*,int,PetscTruth*);
EXTERN int OptionsEList(char*,char*,char*,char**,int,char*,char *,int,PetscTruth*);
EXTERN int OptionsDoubleArray(char*,char*,char*,double[],int*,PetscTruth*);
EXTERN int OptionsStringArray(char*,char*,char*,char**,int*,PetscTruth*);
#endif


