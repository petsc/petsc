/* $Id: options.h,v 1.15 1996/01/12 22:10:35 bsmith Exp bsmith $ */
/*
   Application callable routines to determine options set in the options database.
*/
#if !defined(__OPTIONS_PACKAGE)
#define __OPTIONS_PACKAGE
#include "petsc.h"

int OptionsHasName(char*,char *,int*);
int OptionsGetInt(char*,char *,int *,int*);
int OptionsGetDouble(char *,char *,double *,int*);
int OptionsGetScalar(char *,char *,Scalar *,int*);
int OptionsGetIntArray(char*,char *,int *,int *,int*);
int OptionsGetDoubleArray(char *,char *,double *,int *,int*);
int OptionsGetString(char*,char *,char *,int,int*);
int OptionsAllUsed();

int OptionsSetValue(char*,char*);

int OptionsPrint(FILE *);

#endif
