#define PETSC_DLL
/*
    We define the string operations here. The reason we just do not use 
  the standard string routines in the PETSc code is that on some machines 
  they are broken or have the wrong prototypes.

*/
#include "petsc.h"                   /*I  "petsc.h"   I*/
#include "petscsys.h"
#if defined(PETSC_HAVE_STRING_H)
#include <string.h>
#endif
#if defined(PETSC_HAVE_STRINGS_H)
#include <strings.h>
#endif
#include "petscfix.h"

#undef __FUNCT__  
#define __FUNCT__ "PetscStrlen"
/*@C
   PetscStrlen - Gets length of a string

   Not Collective

   Input Parameters:
.  s - pointer to string

   Output Parameter:
.  len - length in bytes

   Level: intermediate

   Note:
   This routine is analogous to strlen().

   Null string returns a length of zero

  Concepts: string length
  
@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrlen(const char s[],size_t *len)
{
  PetscFunctionBegin;
  if (!s) {
    *len = 0;
  } else {
    *len = strlen(s);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrallocpy"
/*@C
   PetscStrallocpy - Allocates space to hold a copy of a string then copies the string

   Not Collective

   Input Parameters:
.  s - pointer to string

   Output Parameter:
.  t - the copied string

   Level: intermediate

   Note:
      Null string returns a new null string

  Concepts: string copy
  
@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrallocpy(const char s[],char *t[])
{
  PetscErrorCode ierr;
  size_t         len;

  PetscFunctionBegin;
  if (s) {
    ierr = PetscStrlen(s,&len);CHKERRQ(ierr);
    ierr = PetscMalloc((1+len)*sizeof(char),t);CHKERRQ(ierr);
    ierr = PetscStrcpy(*t,s);CHKERRQ(ierr);
  } else {
    *t = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrcpy"
/*@C
   PetscStrcpy - Copies a string

   Not Collective

   Input Parameters:
.  t - pointer to string

   Output Parameter:
.  s - the copied string

   Level: intermediate

   Note:
     Null string returns a string starting with zero

  Concepts: string copy
  
.seealso: PetscStrncpy(), PetscStrcat(), PetscStrncat()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrcpy(char s[],const char t[])
{
  PetscFunctionBegin;
  if (t && !s) {
    SETERRQ(PETSC_ERR_ARG_NULL,"Trying to copy string into null pointer");
  }
  if (t) {strcpy(s,t);}
  else {s[0] = 0;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrncpy"
/*@C
   PetscStrncpy - Copies a string up to a certain length

   Not Collective

   Input Parameters:
+  t - pointer to string
-  n - the length to copy

   Output Parameter:
.  s - the copied string

   Level: intermediate

   Note:
     Null string returns a string starting with zero

  Concepts: string copy

.seealso: PetscStrcpy(), PetscStrcat(), PetscStrncat()
  
@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrncpy(char s[],const char t[],size_t n)
{
  PetscFunctionBegin;
  strncpy(s,t,n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrcat"
/*@C
   PetscStrcat - Concatenates a string onto a given string

   Not Collective

   Input Parameters:
+  s - pointer to string to be added to end
-  t - string to be added to

   Level: intermediate

  Concepts: string copy

.seealso: PetscStrcpy(), PetscStrncpy(), PetscStrncat()
  
@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrcat(char s[],const char t[])
{
  PetscFunctionBegin;
  strcat(s,t);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrncat"
/*@C
   PetscStrncat - Concatenates a string onto a given string, up to a given length

   Not Collective

   Input Parameters:
+  s - pointer to string to be added to end
.  t - string to be added to
.  n - maximum length to copy 

   Level: intermediate

  Concepts: string copy

.seealso: PetscStrcpy(), PetscStrncpy(), PetscStrcat()
  
@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrncat(char s[],const char t[],size_t n)
{
  PetscFunctionBegin;
  strncat(s,t,n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrcmp"
/*@C
   PetscStrcmp - Compares two strings,

   Not Collective

   Input Parameters:
+  a - pointer to string first string
-  b - pointer to second string

   Output Parameter:
.  flg - if the two strings are equal

   Level: intermediate

.seealso: PetscStrgrt(), PetscStrncmp(), PetscStrcasecmp()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrcmp(const char a[],const char b[],PetscTruth *flg)
{
  int c;

  PetscFunctionBegin;
  if (!a && !b) {
    *flg = PETSC_TRUE;
  } else if (!a || !b) {
    *flg = PETSC_FALSE;
  } else {
    c = strcmp(a,b);
    if (c) *flg = PETSC_FALSE;
    else   *flg = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrgrt"
/*@C
   PetscStrgrt - If first string is greater than the second

   Not Collective

   Input Parameters:
+  a - pointer to first string
-  b - pointer to second string

   Output Parameter:
.  flg - if the first string is greater

   Notes:
    Null arguments are ok, a null string is considered smaller than 
    all others

   Level: intermediate

.seealso: PetscStrcmp(), PetscStrncmp(), PetscStrcasecmp()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrgrt(const char a[],const char b[],PetscTruth *t)
{
  int c;

  PetscFunctionBegin;
  if (!a && !b) {
    *t = PETSC_FALSE;
  } else if (a && !b) {
    *t = PETSC_TRUE; 
  } else if (!a && b) {
    *t = PETSC_FALSE; 
  } else {
    c = strcmp(a,b);
    if (c > 0) *t = PETSC_TRUE;
    else       *t = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrcasecmp"
/*@C
   PetscStrcasecmp - Returns true if the two strings are the same
     except possibly for case.

   Not Collective

   Input Parameters:
+  a - pointer to first string
-  b - pointer to second string

   Output Parameter:
.  flg - if the two strings are the same

   Notes:
    Null arguments are ok

   Level: intermediate

.seealso: PetscStrcmp(), PetscStrncmp(), PetscStrgrt()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrcasecmp(const char a[],const char b[],PetscTruth *t)
{
  int c;

  PetscFunctionBegin;
  if (!a && !b) c = 0;
  else if (!a || !b) c = 1;
#if defined(PETSC_HAVE_STRICMP)
  else c = stricmp(a,b);
#else
  else c = strcasecmp(a,b);
#endif
  if (!c) *t = PETSC_TRUE;
  else    *t = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrncmp"
/*@C
   PetscStrncmp - Compares two strings, up to a certain length

   Not Collective

   Input Parameters:
+  a - pointer to first string
.  b - pointer to second string
-  n - length to compare up to

   Output Parameter:
.  t - if the two strings are equal

   Level: intermediate

.seealso: PetscStrgrt(), PetscStrcmp(), PetscStrcasecmp()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrncmp(const char a[],const char b[],size_t n,PetscTruth *t)
{
  int c;

  PetscFunctionBegin;
  c = strncmp(a,b,n);
  if (!c) *t = PETSC_TRUE;
  else    *t = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrchr"
/*@C
   PetscStrchr - Locates first occurance of a character in a string

   Not Collective

   Input Parameters:
+  a - pointer to string
-  b - character

   Output Parameter:
.  c - location of occurance, PETSC_NULL if not found

   Level: intermediate

@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrchr(const char a[],char b,char *c[])
{
  PetscFunctionBegin;
  *c = (char *)strchr(a,b);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrrchr"
/*@C
   PetscStrrchr - Locates one location past the last occurance of a character in a string,
      if the character is not found then returns entire string

   Not Collective

   Input Parameters:
+  a - pointer to string
-  b - character

   Output Parameter:
.  tmp - location of occurance, a if not found

   Level: intermediate

@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrrchr(const char a[],char b,char *tmp[])
{
  PetscFunctionBegin;
  *tmp = (char *)strrchr(a,b);
  if (!*tmp) *tmp = (char*)a; else *tmp = *tmp + 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrtolower"
/*@C
   PetscStrtolower - Converts string to lower case

   Not Collective

   Input Parameters:
.  a - pointer to string

   Level: intermediate

@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrtolower(char a[])
{
  PetscFunctionBegin;
  while (*a) {
    if (*a >= 'A' && *a <= 'Z') *a += 'a' - 'A';
    a++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTokenFind"
/*@C
   PetscTokenFind - Locates next "token" in a string

   Not Collective

   Input Parameters:
.  a - pointer to token

   Output Parameter:
.  result - location of occurance, PETSC_NULL if not found

   Notes:

     This version is different from the system version in that
  it allows you to pass a read-only string into the function.

   Level: intermediate

.seealso: PetscTokenCreate(), PetscTokenDestroy()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscTokenFind(PetscToken *a,char *result[])
{
  char *ptr = a->current;

  PetscFunctionBegin;
  *result = a->current;
  if (ptr && !*ptr) *result = 0;
  while (ptr) {
    if (*ptr == a->token) {
      *ptr++ = 0; 
      while (*ptr == a->token) ptr++;
      a->current = ptr;
      break;
    }
    if (!*ptr) {
      a->current = 0;
      break;
    }
    ptr++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTokenCreate"
/*@C
   PetscTokenCreate - Creates a PetscToken used to find tokens in a string

   Not Collective

   Input Parameters:
+  string - the string to look in
-  token - the character to look for

   Output Parameter:
.  a - pointer to token

   Notes:

     This version is different from the system version in that
  it allows you to pass a read-only string into the function.

   Level: intermediate

.seealso: PetscTokenFind(), PetscTokenDestroy()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscTokenCreate(const char a[],const char b,PetscToken **t)
{
  PetscErrorCode ierr;
  size_t         len;

  PetscFunctionBegin;
  ierr = PetscNew(PetscToken,t);CHKERRQ(ierr);
  ierr = PetscStrlen(a,&len);CHKERRQ(ierr);
  ierr = PetscStrallocpy(a,&(*t)->array);CHKERRQ(ierr);
  (*t)->current = (*t)->array;   
  (*t)->token   = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTokenDestroy"
/*@C
   PetscTokenDestroy - Destroys a PetscToken

   Not Collective

   Input Parameters:
.  a - pointer to token

   Level: intermediate

.seealso: PetscTokenCreate(), PetscTokenFind()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscTokenDestroy(PetscToken *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(a->array);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrstr"
/*@C
   PetscStrstr - Locates first occurance of string in another string

   Not Collective

   Input Parameters:
+  a - pointer to string
-  b - string to find

   Output Parameter:
.  tmp - location of occurance

   Level: intermediate

@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrstr(const char a[],const char b[],char *tmp[])
{
  PetscFunctionBegin;
  *tmp = (char *)strstr(a,b);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscGetPetscDir"
/*@C
   PetscGetPetscDir - Gets the directory PETSc is installed in

   Not Collective

   Output Parameter:
.  dir - the directory

   Level: developer

@*/
PetscErrorCode PETSC_DLLEXPORT PetscGetPetscDir(const char *dir[])
{
  PetscFunctionBegin;
  *dir = PETSC_DIR;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStrreplace"
/*@C
   PetscStrreplace - Replaces substrings in string with other substrings

   Not Collective

   Input Parameters:
+   comm - MPI_Comm of processors that are processing the string
.   a - the string to look in
.   b - the resulting copy of a with replaced strings
-   len - the length of b

   Notes:
      Replaces   ${PETSC_ARCH},${PETSC_DIR},${PETSC_LIB_DIR},${DISPLAY},
      ${HOMEDIRECTORY},${WORKINGDIRECTORY},${USERNAME} with appropriate values
      as well as any environmental variables.
   
   Level: intermediate

@*/
PetscErrorCode PETSC_DLLEXPORT PetscStrreplace(MPI_Comm comm,const char a[],char b[],size_t len)
{
  PetscErrorCode ierr;
  int            i = 0;
  size_t         l,l1,l2,l3;
  char           *work,*par,*epar,env[1024],*tfree;
  const char     *s[] = {"${PETSC_ARCH}","${PETSC_DIR}","${PETSC_LIB_DIR}","${DISPLAY}","${HOMEDIRECTORY}","${WORKINGDIRECTORY}","${USERNAME}",0};
  const char     *r[] = {PETSC_ARCH,PETSC_DIR,PETSC_LIB_DIR,0,0,0,0,0};
  PetscTruth     flag;

  PetscFunctionBegin;
  if (!a || !b) SETERRQ(PETSC_ERR_ARG_NULL,"a and b strings must be nonnull");
  ierr = PetscMalloc(len*sizeof(char*),&work);CHKERRQ(ierr);

  /* get values for replaced variables */
  ierr = PetscMalloc(256*sizeof(char),&r[4]);CHKERRQ(ierr);
  ierr = PetscMalloc(PETSC_MAX_PATH_LEN*sizeof(char),&r[5]);CHKERRQ(ierr);
  ierr = PetscMalloc(PETSC_MAX_PATH_LEN*sizeof(char),&r[6]);CHKERRQ(ierr);
  ierr = PetscMalloc(256*sizeof(char),&r[7]);CHKERRQ(ierr);
  ierr = PetscGetDisplay((char*)r[4],256);CHKERRQ(ierr);
  ierr = PetscGetHomeDirectory((char*)r[5],PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscGetWorkingDirectory((char*)r[6],PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscGetUserName((char*)r[7],256);CHKERRQ(ierr);

  /* replace the requested strings */
  ierr = PetscStrncpy(b,a,len);CHKERRQ(ierr);  
  while (s[i]) {
    ierr = PetscStrlen(s[i],&l);CHKERRQ(ierr);
    ierr = PetscStrstr(b,s[i],&par);CHKERRQ(ierr);
    while (par) {
      *par  =  0;
      par  += l;

      ierr = PetscStrlen(b,&l1);CHKERRQ(ierr);
      ierr = PetscStrlen(r[i],&l2);CHKERRQ(ierr);
      ierr = PetscStrlen(par,&l3);CHKERRQ(ierr);
      if (l1 + l2 + l3 >= len) {
        SETERRQ(PETSC_ERR_ARG_SIZ,"b len is not long enough to hold new values");
      }
      ierr  = PetscStrcpy(work,b);CHKERRQ(ierr);
      ierr  = PetscStrcat(work,r[i]);CHKERRQ(ierr);
      ierr  = PetscStrcat(work,par);CHKERRQ(ierr);
      ierr  = PetscStrncpy(b,work,len);CHKERRQ(ierr);
      ierr  = PetscStrstr(b,s[i],&par);CHKERRQ(ierr);
    }
    i++;
  }
  for ( i=4; i<8; i++){
    tfree = (char*)r[i];
    ierr = PetscFree(tfree);CHKERRQ(ierr);
  }

  /* look for any other ${xxx} strings to replace from environmental variables */
  ierr = PetscStrstr(b,"${",&par);CHKERRQ(ierr);
  while (par) {
    *par = 0;
    par += 2;
    ierr  = PetscStrcpy(work,b);CHKERRQ(ierr);
    ierr = PetscStrstr(par,"}",&epar);CHKERRQ(ierr);
    *epar = 0;
    epar += 1;
    ierr = PetscOptionsGetenv(comm,par,env,256,&flag);CHKERRQ(ierr);
    if (!flag) {
      SETERRQ1(PETSC_ERR_ARG_WRONG,"Substitution string ${%s} not found as environmental variable",par);
    }
    ierr = PetscStrcat(work,env);CHKERRQ(ierr);
    ierr = PetscStrcat(work,epar);CHKERRQ(ierr);
    ierr = PetscStrcpy(b,work);CHKERRQ(ierr);
    ierr = PetscStrstr(b,"${",&par);CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   PetscStrfree - Frees a string (if it is not null)

   Not Collective

   Synopsis:
   PetscErrorCode PetscStrfree(char *s)

   Input Parameter:
.  s - pointer to string

   Level: intermediate

  Concepts: string free
  
.seealso: PetscStrncpy(), PetscStrcat(), PetscStrncat(), PetscStrallocpy()

M*/
