
#ifndef lint
static char vcid[] = "$Id: str.c,v 1.1 1995/09/30 15:24:54 bsmith Exp bsmith $";
#endif
/*
    We define the string operations here. The reason we just don't use 
  the standard string routines in teh PETSc code is that on some machines 
  they are broken.

*/
#include "petsc.h"        /*I  "petsc.h"   I*/
#include <stdio.h>
#include <math.h>
#if defined(HAVE_STRING_H)
#include <string.h>
#endif
#include "pinclude/petscfix.h"

int PetscStrlen(char *s)
{
  return strlen(s);
}

void PetscStrcpy(char *s,char *t)
{
  strcpy(s,t,n);
}

void PetscStrncpy(char *s,char *t,int n)
{
  strncpy(s,t,n);
}

void PetscStrcat(char *s,char *t)
{
  strcat(s,t,n);
}
