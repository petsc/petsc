/* $Id: options.h,v 1.14 1996/01/01 01:05:46 bsmith Exp bsmith $ */
/*
   Application callable routines to determine options set in the options database.
*/
#if !defined(__OPTIONS_PACKAGE)
#define __OPTIONS_PACKAGE
#include "petsc.h"

/* returns -1 on error, 0 on not found and 1 on found */
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
