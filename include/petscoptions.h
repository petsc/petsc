/* $Id: options.h,v 1.25 1998/03/24 20:39:10 balay Exp bsmith $ */
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
extern int  OptionsGetStringArray(char*,char *,char **,int*,int*);
extern int  OptionsAllUsed(void);
extern int  OptionsLeft(void);
extern int  OptionsDestroy(void);
extern int  OptionsCreate(void);

extern int  OptionsReject(char *,char*);

extern int  OptionsSetValue(char*,char*);
extern int  OptionsClearValue(char*);

extern int  OptionsPrint(FILE *);

#endif
