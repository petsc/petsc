
#ifndef lint
static char vcid[] = "$Id: str.c,v 1.4 1995/12/21 18:30:34 bsmith Exp bsmith $";
#endif
/*
    We define the string operations here. The reason we just don't use 
  the standard string routines in the PETSc code is that on some machines 
  they are broken.

*/
#include "petsc.h"        /*I  "petsc.h"   I*/
#if defined(HAVE_STRING_H)
#include <string.h>
#endif
#include "pinclude/petscfix.h"

int PetscStrlen(char *s)
{
  if (!s) return 0;
  return strlen(s);
}

void PetscStrcpy(char *s,char *t)
{
  strcpy(s,t);
}

void PetscStrncpy(char *s,char *t,int n)
{
  strncpy(s,t,n);
}

void PetscStrcat(char *s,char *t)
{
  strcat(s,t);
}

void PetscStrncat(char *s,char *t,int n)
{
  strncat(s,t,n);
}

int PetscStrcmp(char *a,char *b)
{
  return strcmp(a,b);
}

int PetscStrncmp(char *a,char *b,int n)
{
  return strncmp(a,b,n);
}

char *PetscStrchr(char *a,char b)
{
  return strchr(a,b);
}

/*
      This is slightly different then the system version. 
   It returns the position after the position of b and 
   if it does not find it then it returns the entire string.
*/
char *PetscStrrchr(char *a,char b)
{
  char *tmp = strrchr(a,b);
  if (!tmp) tmp = *a; else tmp += 1;
  return tmp;
}

char *PetscStrtok(char *a,char *b)
{
  return strtok(a,b);
}

char *PetscStrstr(char*a,char *b)
{
  return strstr(a,b);
}



