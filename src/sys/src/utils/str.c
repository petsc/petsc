#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: str.c,v 1.13 1997/07/13 19:02:00 balay Exp bsmith $";
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
#if defined(HAVE_STRINGS_H)
#include <strings.h>
#endif
#include "pinclude/petscfix.h"

#undef __FUNC__  
#define __FUNC__ "PetscStrlen"
int PetscStrlen(char *s)
{
  if (!s) return 0;
  return strlen(s);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrcpy"
void PetscStrcpy(char *s,char *t)
{
  strcpy(s,t);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrncpy"
void PetscStrncpy(char *s,char *t,int n)
{
  strncpy(s,t,n);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrcat"
void PetscStrcat(char *s,char *t)
{
  strcat(s,t);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrncat"
void PetscStrncat(char *s,char *t,int n)
{
  strncat(s,t,n);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrcmp"
int PetscStrcmp(char *a,char *b)
{
  if (!a && !b) return 0;
  if (!a || !b) return 1;
  return strcmp(a,b);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrcasecmp"
int PetscStrcasecmp(char *a,char *b)
{
  if (!a && !b) return 0;
  if (!a || !b) return 1;
#if defined (PARCH_nt)
  return stricmp(a,b);
#else
  return strcasecmp(a,b);
#endif
}

#undef __FUNC__  
#define __FUNC__ "PetscStrncmp"
int PetscStrncmp(char *a,char *b,int n)
{
  return strncmp(a,b,n);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrchr"
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
#define __FUNC__ "PetscStrrchr"
char *PetscStrrchr(char *a,char b)
{
  char *tmp = strrchr(a,b);
  if (!tmp) tmp = a; else tmp = tmp + 1;
  return tmp;
}

#undef __FUNC__  
#define __FUNC__ "PetscStrtok"
char *PetscStrtok(char *a,char *b)
{
  return strtok(a,b);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrstr"
char *PetscStrstr(char*a,char *b)
{
  return strstr(a,b);
}



