/* $Id: options.h,v 1.20 1997/08/13 22:27:41 bsmith Exp bsmith $ */
/*
   Routines to determine options set in the options database.
*/
#if !defined(__OPTIONS_PACKAGE)
#define __OPTIONS_PACKAGE
#include "petsc.h"

extern int  OptionsHasName(char*,char *,int*);
extern int  OptionsGetInt(char*,char *,int *,int*);
extern int  OptionsGetDouble(char *,char *,double *,int*);
extern int  OptionsGetScalar(char *,char *,Scalar *,int*);
extern int  OptionsGetIntArray(char*,char *,int *,int *,int*);
extern int  OptionsGetDoubleArray(char *,char *,double *,int *,int*);
extern int  OptionsGetString(char*,char *,char *,int,int*);
extern int  OptionsAllUsed();

extern int  OptionsReject(char *,char*);

extern int  OptionsSetValue(char*,char*);

extern int  OptionsPrint(FILE *);
extern int  OptionsGetProgramName(char**);

#endif
