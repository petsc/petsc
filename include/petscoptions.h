/* $Id: petscoptions.h,v 1.38 2000/05/08 15:09:50 balay Exp bsmith $ */
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

extern PetscTruth PetscPublishOptions;
EXTERN int        OptionsSelectBegin(MPI_Comm,char*,char*);
EXTERN int        OptionsSelectInt(MPI_Comm,char*,char*,int);
EXTERN int        OptionsSelectDouble(MPI_Comm,char*,char*,double);
EXTERN int        OptionsSelectName(MPI_Comm,char*,char*);
EXTERN int        OptionsSelectList(MPI_Comm,char*,char*,char**,int,char*);
EXTERN int        OptionsSelectEnd(MPI_Comm);
#endif
