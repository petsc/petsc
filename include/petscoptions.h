
#if !defined(__OPTIONS_PACKAGE)
#define __OPTIONS_PACKAGE
#include "petsc.h"

int OptionsCreate(int* ,char ***,char *,char*);
int OptionsDestroy();
#include <stdio.h>
int OptionsPrint(FILE *);

/* returns -1 on error, 0 on not found and 1 on found */
int OptionsHasName(int, char*,char *);
int OptionsGetInt(int, char*,char *,int *);
int OptionsGetDouble(int, char *,char *,double *);
int OptionsGetInt(int, char*,char *,int *);
int OptionsGetDouble(int, char *,char *,double *);
int OptionsGetString(int, char*,char *,char *,int);
int OptionsGetScalar(int,char*,char *,Scalar *);
int OptionsAllUsed();

int OptionsSetValue(char*,char*);

#endif
