#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: str.c,v 1.24 1998/12/17 22:49:03 balay Exp bsmith $";
#endif
/*
    We define the string operations here. The reason we just don't use 
  the standard string routines in the PETSc code is that on some machines 
  they are broken or have the wrong prototypes.

*/
#include "petsc.h"                   /*I  "petsc.h"   I*/
#if defined(HAVE_STRING_H)
#include <string.h>
#endif
#if defined(HAVE_STRINGS_H)
#include <strings.h>
#endif
#include "pinclude/petscfix.h"

/*MC
   PetscTypeCompare - Compares two PETSc types, returns 1 if they are
      the same

   Input Parameter:
+    type1 - first type
-    type2 - second type

   Level: intermediate

   Synopsis:
   int PetscTypeCompare(type1,type2)

   Usage:
.vb
     VecType type;
     VecGetType(v,&type);
     if (PetscTypeCompare(type1,VEC_MPI)) {
       ....
     }
.ve

   Notes:
     Equivalent to PetscStrcmp((char*)type1,(char*)type2) 
 
     Only works for new-style types that are char*

.seealso: VecGetType(), KSPGetType(), PCGetType(), SNESGetType()

.keywords: comparing types
M*/


#undef __FUNC__  
#define __FUNC__ "PetscStrlen"
int PetscStrlen(const char s[])
{
  int len;

  PetscFunctionBegin;
  if (!s) PetscFunctionReturn(0);
  len = strlen(s);
  PetscFunctionReturn(len);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrcpy"
/*
    Handles copying null string correctly
*/
int PetscStrcpy(char s[],const char t[])
{
  PetscFunctionBegin;
  if (t && !s) {
    SETERRQ(1,1,"Trying to copy string into null pointer");
  }
  if (t) {strcpy(s,t);}
  else {s[0] = 0;}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrncpy"
int PetscStrncpy(char s[],const char t[],int n)
{
  PetscFunctionBegin;
  strncpy(s,t,n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrcat"
int PetscStrcat(char s[],const char t[])
{
  PetscFunctionBegin;
  strcat(s,t);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrncat"
int PetscStrncat(char s[],const char t[],int n)
{
  PetscFunctionBegin;
  strncat(s,t,n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrcmp"
int PetscStrcmp(const char a[],const char b[])
{
  int c;

  PetscFunctionBegin;
  if (!a && !b) PetscFunctionReturn(0);
  if (!a || !b) PetscFunctionReturn(1);
  c = strcmp(a,b);
  PetscFunctionReturn(c);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrcasecmp"
int PetscStrcasecmp(const char a[],const char b[])
{
  int c;

  PetscFunctionBegin;
  if (!a && !b) c = 0;
  else if (!a || !b) c = 1;
#if defined (PARCH_win32)
  else c = stricmp(a,b);
#else
  else c = strcasecmp(a,b);
#endif
  PetscFunctionReturn(c);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrncmp"
int PetscStrncmp(const char a[],const char b[],int n)
{
  int c;

  PetscFunctionBegin;
  c = strncmp(a,b,n);
  PetscFunctionReturn(c);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrchr"
char *PetscStrchr(const char a[],char b)
{
  char *c;

  PetscFunctionBegin;
  c = strchr(a,b);
  PetscFunctionReturn(c);
}

/*
      This is slightly different then the system version. 
   It returns the position after the position of b and 
   if it does not find it then it returns the entire string.
*/
#undef __FUNC__  
#define __FUNC__ "PetscStrrchr"
char *PetscStrrchr(const char a[],char b)
{
  char *tmp;

  PetscFunctionBegin;
  tmp = strrchr(a,b);
  if (!tmp) tmp = (char*)a; else tmp = tmp + 1;
  PetscFunctionReturn(tmp);
}

/*
     This version is different from the system version in that
  it allows you to pass a read-only string into the function.
  A copy is made that is then passed into the system strtok() 
  routine.

    Limitation: 
  String must be less than or equal 1024 bytes in length, otherwise
  it will bleed memory.

*/
#undef __FUNC__  
#define __FUNC__ "PetscStrtok"
char *PetscStrtok(const char a[],const char b[])
{
  static char init[1024];
         char *tmp=0,*ptr=0;
         int  len;

  PetscFunctionBegin;
  if (a) {
    len = strlen(a);
    if (len > 1023) {
      ptr = (char *) PetscMalloc((len+1)*sizeof(char));
      if (!ptr) {
         PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,1,1,"Malloc failed");
         PetscFunctionReturn(0);
      }
    } else {
      ptr = init;
    }
    PetscStrncpy(ptr,a,1024);
  }
  tmp = strtok(ptr,b);
  PetscFunctionReturn(tmp);
}

#undef __FUNC__  
#define __FUNC__ "PetscStrstr"
char *PetscStrstr(const char a[],const char b[])
{
  char *tmp;

  PetscFunctionBegin;
  tmp = strstr(a,b);
  PetscFunctionReturn(tmp);
}



