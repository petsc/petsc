/* $Id: options.h,v 1.27 1998/05/04 03:59:04 bsmith Exp bsmith $ */
/*
   Routines to determine options set in the options database.
*/
#if !defined(__OPTIONS_H)
#define __OPTIONS_H
#include "petsc.h"

extern int  OptionsHasName(char*,char *,int*);
extern int  OptionsGetInt(char*,char *,int *,int*);
extern int  OptionsGetDouble(char *,char *,double *,int*);
extern int  OptionsGetScalar(char *,char *,Scalar *,int*);
extern int  OptionsGetIntArray(char*,char *,int *,int *,int*);
extern int  OptionsGetDoubleArray(char *,char *,double *,int *,int*);
extern int  OptionsGetString(char*,char *,char *,int,int*);
extern int  OptionsGetStringArray(char*,char *,char **,int*,int*);

extern int  OptionsSetAlias(char *,char *);
extern int  OptionsSetValue(char*,char*);
extern int  OptionsClearValue(char*);

extern int  OptionsAllUsed(void);
extern int  OptionsLeft(void);
extern int  OptionsPrint(FILE *);

extern int  OptionsCreate(void);
extern int  OptionsInsert(int *,char ***,char*);
extern int  OptionsInsertFile(char *);
extern int  OptionsDestroy(void);

extern int  OptionsReject(char *,char*);

#endif
