/* $Id: options.h,v 1.32 1998/11/20 15:31:26 bsmith Exp bsmith $ */
/*
   Routines to determine options set in the options database.
*/
#if !defined(__OPTIONS_H)
#define __OPTIONS_H
#include "petsc.h"

extern int  OptionsHasName(const char[],const char[],int*);
extern int  OptionsGetInt(const char[],const char [],int *,int*);
extern int  OptionsGetLogical(const char[],const char [],PetscTruth *,int*);
extern int  OptionsGetDouble(const char[],const char[],double *,int*);
extern int  OptionsGetScalar(const char[],const char[],Scalar *,int*);
extern int  OptionsGetIntArray(const char[],const char[],int[],int *,int*);
extern int  OptionsGetDoubleArray(const char[],const char[],double[],int *,int*);
extern int  OptionsGetString(const char[],const char[],char[],int,int*);
extern int  OptionsGetStringArray(const char[],const char[],char**,int*,int*);

extern int  OptionsSetAlias(const char[],const char[]);
extern int  OptionsSetValue(const char[],const char[]);
extern int  OptionsClearValue(const char[]);

extern int  OptionsAllUsed(int *);
extern int  OptionsLeft(void);
extern int  OptionsPrint(FILE *);

extern int  OptionsCreate(void);
extern int  OptionsInsert(int *,char ***,const char[]);
extern int  OptionsInsertFile(const char[]);
extern int  OptionsDestroy(void);

extern int  OptionsReject(const char[],const char[]);
extern int  OptionsGetAll(char*[]);

extern int  OptionsGetenv(MPI_Comm,const char *,char[],int,PetscTruth *);
extern int  OptionsAtoi(const char[],int*);

#endif
