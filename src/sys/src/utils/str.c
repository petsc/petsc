
#ifndef lint
static char vcid[] = "$Id: str.c,v 1.9 1997/01/27 18:15:50 bsmith Exp bsmith $";
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

#undef __FUNC__  
#define __FUNC__ "PetscStrlen" /* ADIC Ignore */
int PetscStrlen(char *s)
{
  if (!s) return 0;
  return strlen(s);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrcpy" /* ADIC Ignore */
void PetscStrcpy(char *s,char *t)
{
  strcpy(s,t);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrncpy" /* ADIC Ignore */
void PetscStrncpy(char *s,char *t,int n)
{
  strncpy(s,t,n);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrcat" /* ADIC Ignore */
void PetscStrcat(char *s,char *t)
{
  strcat(s,t);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrncat" /* ADIC Ignore */
void PetscStrncat(char *s,char *t,int n)
{
  strncat(s,t,n);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrcmp" /* ADIC Ignore */
int PetscStrcmp(char *a,char *b)
{
  if (!a && !b) return 0;
  if (!a || !b) return 1;
  return strcmp(a,b);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrncmp" /* ADIC Ignore */
int PetscStrncmp(char *a,char *b,int n)
{
  return strncmp(a,b,n);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrchr" /* ADIC Ignore */
char *PetscStrchr(char *a,char b)
{
  return strchr(a,b);
}

/*
      This is slightly different then the system version. 
   It returns the position after the position of b and 
   if it does not find it then it returns the entire string.
*/
#undef __FUNC__  
#define __FUNC__ "PetscStrrchr" /* ADIC Ignore */
char *PetscStrrchr(char *a,char b)
{
  char *tmp = strrchr(a,b);
  if (!tmp) tmp = a; else tmp = tmp + 1;
  return tmp;
}

#undef __FUNC__  
#define __FUNC__ "PetscStrtok" /* ADIC Ignore */
char *PetscStrtok(char *a,char *b)
{
  return strtok(a,b);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrstr" /* ADIC Ignore */
char *PetscStrstr(char*a,char *b)
{
  return strstr(a,b);
}



