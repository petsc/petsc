/* $Id: options.h,v 1.16 1996/03/19 21:30:28 bsmith Exp bsmith $ */
/*
   Application callable routines to determine options set in the options database.
*/
#if !defined(__OPTIONS_PACKAGE)
#define __OPTIONS_PACKAGE
#include "petsc.h"

extern int OptionsHasName(char*,char *,int*);
extern int OptionsGetInt(char*,char *,int *,int*);
extern int OptionsGetDouble(char *,char *,double *,int*);
extern int OptionsGetScalar(char *,char *,Scalar *,int*);
extern int OptionsGetIntArray(char*,char *,int *,int *,int*);
extern int OptionsGetDoubleArray(char *,char *,double *,int *,int*);
extern int OptionsGetString(char*,char *,char *,int,int*);
extern int OptionsAllUsed();

extern int OptionsSetValue(char*,char*);

extern int OptionsPrint(FILE *);
extern char * OptionsGetProgramName();

#endif
