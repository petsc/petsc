
#ifndef lint
static char vcid[] = "$Id: str.c,v 1.6 1996/03/10 17:27:31 bsmith Exp balay $";
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

#undef __FUNCTION__  
#define __FUNCTION__ "PetscStrlen"
int PetscStrlen(char *s)
{
  if (!s) return 0;
  return strlen(s);
}

#undef __FUNCTION__  
#define __FUNCTION__ "PetscStrcpy"
void PetscStrcpy(char *s,char *t)
{
  strcpy(s,t);
}

#undef __FUNCTION__  
#define __FUNCTION__ "PetscStrncpy"
void PetscStrncpy(char *s,char *t,int n)
{
  strncpy(s,t,n);
}

#undef __FUNCTION__  
#define __FUNCTION__ "PetscStrcat"
void PetscStrcat(char *s,char *t)
{
  strcat(s,t);
}

#undef __FUNCTION__  
#define __FUNCTION__ "PetscStrncat"
void PetscStrncat(char *s,char *t,int n)
{
  strncat(s,t,n);
}

#undef __FUNCTION__  
#define __FUNCTION__ "PetscStrcmp"
int PetscStrcmp(char *a,char *b)
{
  return strcmp(a,b);
}

#undef __FUNCTION__  
#define __FUNCTION__ "PetscStrncmp"
int PetscStrncmp(char *a,char *b,int n)
{
  return strncmp(a,b,n);
}

#undef __FUNCTION__  
#define __FUNCTION__ "PetscStrchr"
char *PetscStrchr(char *a,char b)
{
  return strchr(a,b);
}

/*
      This is slightly different then the system version. 
   It returns the position after the position of b and 
   if it does not find it then it returns the entire string.
*/
#undef __FUNCTION__  
#define __FUNCTION__ "PetscStrrchr"
char *PetscStrrchr(char *a,char b)
{
  char *tmp = strrchr(a,b);
  if (!tmp) tmp = a; else tmp = tmp + 1;
  return tmp;
}

#undef __FUNCTION__  
#define __FUNCTION__ "PetscStrtok"
char *PetscStrtok(char *a,char *b)
{
  return strtok(a,b);
}

#undef __FUNCTION__  
#define __FUNCTION__ "PetscStrstr"
char *PetscStrstr(char*a,char *b)
{
  return strstr(a,b);
}



