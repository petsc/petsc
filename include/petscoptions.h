
#if !defined(_OPTIONS_H)
#define _OPTIONS_H
#include "petsc.h"

int OptionsCreate(int *,char ***,char *,char*);
int OptionsDestroy();

/* returns -1 on error, 0 on not found and 1 on found */
int OptionsHasName(int, char *);
int OptionsGetInt(int, char *,int *);
int OptionsGetDouble(int, char *,double *);
int OptionsGetString(int, char *,char *,int);
int OptionsGetScalar(int,char *,Scalar *);

int OptionsSetValue(char*,char*);

#endif
