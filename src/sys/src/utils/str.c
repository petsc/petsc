/*$Id: str.c,v 1.41 2000/04/09 03:09:18 bsmith Exp bsmith $*/
/*
    We define the string operations here. The reason we just do not use 
  the standard string routines in the PETSc code is that on some machines 
  they are broken or have the wrong prototypes.

*/
#include "petsc.h"                   /*I  "petsc.h"   I*/
#include "sys.h"
#if defined(PETSC_HAVE_STRING_H)
#include <string.h>
#endif
#if defined(PETSC_HAVE_STRINGS_H)
#include <strings.h>
#endif
#include "petscfix.h"

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrlen"
int PetscStrlen(const char s[],int *len)
{
  PetscFunctionBegin;
  if (!s) {
    *len = 0;
  } else {
    *len = strlen(s);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrallocpy"
int PetscStrallocpy(const char s[],char **t)
{
  int ierr,len;

  PetscFunctionBegin;
  if (s) {
    ierr  = PetscStrlen(s,&len);CHKERRQ(ierr);
    *t    = (char*)PetscMalloc((1+len)*sizeof(char));CHKPTRQ(*t);
    ierr  = PetscStrcpy(*t,s);CHKERRQ(ierr);
  } else {
    *t = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrcpy"
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
#define  __FUNC__ /*<a name=""></a>*/"PetscStrncpy"
int PetscStrncpy(char s[],const char t[],int n)
{
  PetscFunctionBegin;
  strncpy(s,t,n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrcat"
int PetscStrcat(char s[],const char t[])
{
  PetscFunctionBegin;
  strcat(s,t);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrncat"
int PetscStrncat(char s[],const char t[],int n)
{
  PetscFunctionBegin;
  strncat(s,t,n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrcmp"
int PetscStrcmp(const char a[],const char b[],PetscTruth *flg)
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

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrgrt"
int PetscStrgrt(const char a[],const char b[],PetscTruth *t)
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

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrcasecmp"
/*
    Note: This is different from system strncmp() this returns PETSC_TRUE
    if the strings are the same!
*/
int PetscStrcasecmp(const char a[],const char b[],PetscTruth *t)
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
  if (!c) *t = PETSC_TRUE;
  else    *t = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrncmp"
/*
    Note: This is different from system strncmp() this returns PETSC_TRUE
    if the strings are the same!
*/
int PetscStrncmp(const char a[],const char b[],int n,PetscTruth *t)
{
  int c;

  PetscFunctionBegin;
  c = strncmp(a,b,n);
  if (!c) *t = PETSC_TRUE;
  else    *t = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrchr"
int PetscStrchr(const char a[],char b,char **c)
{
  PetscFunctionBegin;
  *c = (char *)strchr(a,b);
  PetscFunctionReturn(0);
}

/*
      This is slightly different then the system version. 
   It returns the position after the position of b and 
   if it does not find it then it returns the entire string.
*/
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrrchr"
int PetscStrrchr(const char a[],char b,char **tmp)
{
  PetscFunctionBegin;
  *tmp = (char *)strrchr(a,b);
  if (!*tmp) *tmp = (char*)a; else *tmp = *tmp + 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrtolower"
int PetscStrtolower(char a[])
{
  PetscFunctionBegin;
  while (*a) {
    if (*a >= 'A' && *a <= 'Z') *a += 'a' - 'A';
    a++;
  }
  PetscFunctionReturn(0);
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
#define  __FUNC__ /*<a name=""></a>*/"PetscStrtok"
int PetscStrtok(const char a[],const char b[],char **result)
{
  static char init[1024];
  char        *ptr=0;
  int         ierr,len;
         

  PetscFunctionBegin;
  if (a) {
    ierr = PetscStrlen(a,&len);CHKERRQ(ierr);
    if (len > 1023) {
      ptr = (char*)PetscMalloc((len+1)*sizeof(char));
      if (!ptr) SETERRQ(1,1,"Malloc failed");
    } else {
      ptr = init;
    }
    ierr = PetscStrncpy(ptr,a,len+1);CHKERRQ(ierr);
  }
  *result = strtok(ptr,b);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrstr"
int PetscStrstr(const char a[],const char b[],char **tmp)
{
  PetscFunctionBegin;
  *tmp = (char *)strstr(a,b);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscStrreplace"
/*

      No proper error checking yet
*/
int PetscStrreplace(MPI_Comm comm,const char a[],char *b,int len)
{
  int        ierr,i = 0,l,l1,l2,l3;
  char       *work,*par,*epar,env[256];
  char       *s[] = {"${PETSC_ARCH}","${BOPT}","${PETSC_DIR}","${PETSC_LDIR}","${DISPLAY}","${HOMEDIRECTORY}","${WORKINGDIRECTORY}",0};
  char       *r[] = {PETSC_ARCH_NAME,PETSC_BOPT,PETSC_DIR,PETSC_LDIR,0,0,0,0};
  PetscTruth flag;

  PetscFunctionBegin;
  if (len <= 0) SETERRQ(1,1,"Length of b must be greater than 0");
  if (!a || !b) SETERRQ(1,1,"a and b strings must be nonnull");
  work = (char*)PetscMalloc(len*sizeof(char*));CHKPTRQ(work);

  /* get values for replaced variables */
  r[4] = (char*)PetscMalloc(256*sizeof(char));CHKPTRQ(r[4]);
  r[5] = (char*)PetscMalloc(256*sizeof(char));CHKPTRQ(r[5]);
  r[6] = (char*)PetscMalloc(256*sizeof(char));CHKPTRQ(r[6]);
  ierr = PetscGetDisplay(r[4],256);CHKERRQ(ierr);
  ierr = PetscGetHomeDirectory(r[5],256);CHKERRQ(ierr);
  ierr = PetscGetWorkingDirectory(r[6],256);CHKERRQ(ierr);

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
        SETERRQ(1,1,"b len is not long enough to hold new values");
      }
      ierr  = PetscStrcpy(work,b);CHKERRQ(ierr);
      ierr  = PetscStrcat(work,r[i]);CHKERRQ(ierr);
      ierr  = PetscStrcat(work,par);CHKERRQ(ierr);
      ierr  = PetscStrncpy(b,work,len);CHKERRQ(ierr);
      ierr  = PetscStrstr(b,s[i],&par);CHKERRQ(ierr);
    }
    i++;
  }
  ierr = PetscFree(r[4]);CHKERRQ(ierr);
  ierr = PetscFree(r[5]);CHKERRQ(ierr);
  ierr = PetscFree(r[6]);CHKERRQ(ierr);

  /* look for any other ${xxx} strings to replace from environmental variables */
  ierr = PetscStrstr(b,"${",&par);CHKERRQ(ierr);
  while (par) {
    *par = 0;
    par += 2;
    ierr  = PetscStrcpy(work,b);CHKERRQ(ierr);
    ierr = PetscStrstr(par,"}",&epar);CHKERRQ(ierr);
    *epar = 0;
    epar += 1;
    ierr = OptionsGetenv(comm,par,env,256,&flag);CHKERRQ(ierr);
    if (!flag) {
      SETERRQ1(1,1,"Substitution string ${%s} not found as environmental variable",par);
    }
    ierr = PetscStrcat(work,env);CHKERRQ(ierr);
    ierr = PetscStrcat(work,epar);CHKERRQ(ierr);
    ierr = PetscStrcpy(b,work);CHKERRQ(ierr);
    ierr = PetscStrstr(b,"${",&par);CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
