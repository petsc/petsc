#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: str.c,v 1.15 1997/09/11 20:33:11 bsmith Exp bsmith $";
#endif
/*
    We define the string operations here. The reason we just don't use 
  the standard string routines in the PETSc code is that on some machines 
  they are broken or have the wrong prototypes.

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
int PetscStrcpy(char *s,char *t)
{
  strcpy(s,t);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscStrncpy"
int PetscStrncpy(char *s,char *t,int n)
{
  strncpy(s,t,n);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscStrcat"
int PetscStrcat(char *s,char *t)
{
  strcat(s,t);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscStrncat"
int PetscStrncat(char *s,char *t,int n)
{
  strncat(s,t,n);
  return 0;
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



