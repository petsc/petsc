
#if !defined(__OPTIONS_PACKAGE)
#define __OPTIONS_PACKAGE
#include "petsc.h"

#include <stdio.h>
int OptionsPrint(FILE *);

/* returns -1 on error, 0 on not found and 1 on found */
int OptionsHasName(char*,char *);
int OptionsGetInt(char*,char *,int *);
int OptionsGetDouble(char *,char *,double *);
int OptionsGetIntArray(char*,char *,int *,int *nmax);
int OptionsGetDoubleArray(char *,char *,double *,int *nmax);
int OptionsGetString(char*,char *,char *,int);
int OptionsGetScalar(char*,char *,Scalar *);
int OptionsAllUsed();

int OptionsSetValue(char*,char*);

#endif
