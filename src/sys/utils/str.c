/*
    We define the string operations here. The reason we just do not use
  the standard string routines in the PETSc code is that on some machines
  they are broken or have the wrong prototypes.

*/
#include <petscsys.h>                   /*I  "petscsys.h"   I*/
#if defined(PETSC_HAVE_STRINGS_H)
#  include <strings.h>          /* strcasecmp */
#endif

/*@C
   PetscStrToArray - Separates a string by a character (for example ' ' or '\n') and creates an array of strings

   Not Collective

   Input Parameters:
+  s - pointer to string
-  sp - separator character

   Output Parameters:
+   argc - the number of entries in the array
-   args - an array of the entries with a null at the end

   Level: intermediate

   Notes:
    this may be called before PetscInitialize() or after PetscFinalize()

   Not for use in Fortran

   Developer Notes:
    Using raw malloc() and does not call error handlers since this may be used before PETSc is initialized. Used
     to generate argc, args arguments passed to MPI_Init()

.seealso: PetscStrToArrayDestroy(), PetscToken, PetscTokenCreate()

@*/
PetscErrorCode  PetscStrToArray(const char s[],char sp,int *argc,char ***args)
{
  int       i,j,n,*lens,cnt = 0;
  PetscBool flg = PETSC_FALSE;

  if (!s) n = 0;
  else    n = strlen(s);
  *argc = 0;
  *args = NULL;
  for (; n>0; n--) {   /* remove separator chars at the end - and will empty the string if all chars are separator chars */
    if (s[n-1] != sp) break;
  }
  if (!n) {
    return(0);
  }
  for (i=0; i<n; i++) {
    if (s[i] != sp) break;
  }
  for (;i<n+1; i++) {
    if ((s[i] == sp || s[i] == 0) && !flg) {flg = PETSC_TRUE; (*argc)++;}
    else if (s[i] != sp) {flg = PETSC_FALSE;}
  }
  (*args) = (char**) malloc(((*argc)+1)*sizeof(char*)); if (!*args) return PETSC_ERR_MEM;
  lens    = (int*) malloc((*argc)*sizeof(int)); if (!lens) return PETSC_ERR_MEM;
  for (i=0; i<*argc; i++) lens[i] = 0;

  *argc = 0;
  for (i=0; i<n; i++) {
    if (s[i] != sp) break;
  }
  for (;i<n+1; i++) {
    if ((s[i] == sp || s[i] == 0) && !flg) {flg = PETSC_TRUE; (*argc)++;}
    else if (s[i] != sp) {lens[*argc]++;flg = PETSC_FALSE;}
  }

  for (i=0; i<*argc; i++) {
    (*args)[i] = (char*) malloc((lens[i]+1)*sizeof(char));
    if (!(*args)[i]) {
      free(lens);
      for (j=0; j<i; j++) free((*args)[j]);
      free(*args);
      return PETSC_ERR_MEM;
    }
  }
  free(lens);
  (*args)[*argc] = NULL;

  *argc = 0;
  for (i=0; i<n; i++) {
    if (s[i] != sp) break;
  }
  for (;i<n+1; i++) {
    if ((s[i] == sp || s[i] == 0) && !flg) {flg = PETSC_TRUE; (*args)[*argc][cnt++] = 0; (*argc)++; cnt = 0;}
    else if (s[i] != sp && s[i] != 0) {(*args)[*argc][cnt++] = s[i]; flg = PETSC_FALSE;}
  }
  return 0;
}

/*@C
   PetscStrToArrayDestroy - Frees array created with PetscStrToArray().

   Not Collective

   Output Parameters:
+  argc - the number of arguments
-  args - the array of arguments

   Level: intermediate

   Notes:
    This may be called before PetscInitialize() or after PetscFinalize()

   Not for use in Fortran

.seealso: PetscStrToArray()

@*/
PetscErrorCode  PetscStrToArrayDestroy(int argc,char **args)
{
  PetscInt i;

  for (i=0; i<argc; i++) free(args[i]);
  if (args) free(args);
  return 0;
}

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

   Not for use in Fortran

@*/
PetscErrorCode  PetscStrlen(const char s[],size_t *len)
{
  PetscFunctionBegin;
  if (!s) *len = 0;
  else    *len = strlen(s);
  PetscFunctionReturn(0);
}

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

      Not for use in Fortran

      Warning: If t has previously been allocated then that memory is lost, you may need to PetscFree()
      the array before calling this routine.

.seealso: PetscStrArrayallocpy(), PetscStrcpy(), PetscStrNArrayallocpy()

@*/
PetscErrorCode  PetscStrallocpy(const char s[],char *t[])
{
  PetscErrorCode ierr;
  size_t         len;
  char           *tmp = NULL;

  PetscFunctionBegin;
  if (s) {
    ierr = PetscStrlen(s,&len);CHKERRQ(ierr);
    ierr = PetscMalloc1(1+len,&tmp);CHKERRQ(ierr);
    ierr = PetscStrcpy(tmp,s);CHKERRQ(ierr);
  }
  *t = tmp;
  PetscFunctionReturn(0);
}

/*@C
   PetscStrArrayallocpy - Allocates space to hold a copy of an array of strings then copies the strings

   Not Collective

   Input Parameters:
.  s - pointer to array of strings (final string is a null)

   Output Parameter:
.  t - the copied array string

   Level: intermediate

   Note:
      Not for use in Fortran

      Warning: If t has previously been allocated then that memory is lost, you may need to PetscStrArrayDestroy()
      the array before calling this routine.

.seealso: PetscStrallocpy(), PetscStrArrayDestroy(), PetscStrNArrayallocpy()

@*/
PetscErrorCode  PetscStrArrayallocpy(const char *const *list,char ***t)
{
  PetscErrorCode ierr;
  PetscInt       i,n = 0;

  PetscFunctionBegin;
  while (list[n++]) ;
  ierr = PetscMalloc1(n+1,t);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscStrallocpy(list[i],(*t)+i);CHKERRQ(ierr);
  }
  (*t)[n] = NULL;
  PetscFunctionReturn(0);
}

/*@C
   PetscStrArrayDestroy - Frees array of strings created with PetscStrArrayallocpy().

   Not Collective

   Output Parameters:
.   list - array of strings

   Level: intermediate

   Notes:
    Not for use in Fortran

.seealso: PetscStrArrayallocpy()

@*/
PetscErrorCode PetscStrArrayDestroy(char ***list)
{
  PetscInt       n = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*list) PetscFunctionReturn(0);
  while ((*list)[n]) {
    ierr = PetscFree((*list)[n]);CHKERRQ(ierr);
    n++;
  }
  ierr = PetscFree(*list);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscStrNArrayallocpy - Allocates space to hold a copy of an array of strings then copies the strings

   Not Collective

   Input Parameters:
+  n - the number of string entries
-  s - pointer to array of strings

   Output Parameter:
.  t - the copied array string

   Level: intermediate

   Note:
      Not for use in Fortran

.seealso: PetscStrallocpy(), PetscStrArrayallocpy(), PetscStrNArrayDestroy()

@*/
PetscErrorCode  PetscStrNArrayallocpy(PetscInt n,const char *const *list,char ***t)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n,t);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscStrallocpy(list[i],(*t)+i);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscStrNArrayDestroy - Frees array of strings created with PetscStrArrayallocpy().

   Not Collective

   Output Parameters:
+   n - number of string entries
-   list - array of strings

   Level: intermediate

   Notes:
    Not for use in Fortran

.seealso: PetscStrArrayallocpy()

@*/
PetscErrorCode PetscStrNArrayDestroy(PetscInt n,char ***list)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!*list) PetscFunctionReturn(0);
  for (i=0; i<n; i++) {
    ierr = PetscFree((*list)[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(*list);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscStrcpy - Copies a string

   Not Collective

   Input Parameters:
.  t - pointer to string

   Output Parameter:
.  s - the copied string

   Level: intermediate

   Notes:
     Null string returns a string starting with zero

     Not for use in Fortran

     It is recommended you use PetscStrncpy() instead of this routine

.seealso: PetscStrncpy(), PetscStrcat(), PetscStrlcat()

@*/

PetscErrorCode  PetscStrcpy(char s[],const char t[])
{
  PetscFunctionBegin;
  PetscAssertFalse(t && !s,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to copy string into null pointer");
  if (t) strcpy(s,t);
  else if (s) s[0] = 0;
  PetscFunctionReturn(0);
}

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

     If the string that is being copied is of length n or larger then the entire string is not
     copied and the final location of s is set to NULL. This is different then the behavior of
     strncpy() which leaves s non-terminated if there is not room for the entire string.

  Developers Note: Should this be PetscStrlcpy() to reflect its behavior which is like strlcpy() not strncpy()

.seealso: PetscStrcpy(), PetscStrcat(), PetscStrlcat()

@*/
PetscErrorCode  PetscStrncpy(char s[],const char t[],size_t n)
{
  PetscFunctionBegin;
  PetscAssertFalse(t && !s,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to copy string into null pointer");
  PetscAssertFalse(s && !n,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Requires an output string of length at least 1 to hold the termination character");
  if (t) {
    if (n > 1) {
      strncpy(s,t,n-1);
      s[n-1] = '\0';
    } else {
      s[0] = '\0';
    }
  } else if (s) s[0] = 0;
  PetscFunctionReturn(0);
}

/*@C
   PetscStrcat - Concatenates a string onto a given string

   Not Collective

   Input Parameters:
+  s - string to be added to
-  t - pointer to string to be added to end

   Level: intermediate

   Notes:
    Not for use in Fortran

    It is recommended you use PetscStrlcat() instead of this routine

.seealso: PetscStrcpy(), PetscStrncpy(), PetscStrlcat()

@*/
PetscErrorCode  PetscStrcat(char s[],const char t[])
{
  PetscFunctionBegin;
  if (!t) PetscFunctionReturn(0);
  strcat(s,t);
  PetscFunctionReturn(0);
}

/*@C
   PetscStrlcat - Concatenates a string onto a given string, up to a given length

   Not Collective

   Input Parameters:
+  s - pointer to string to be added to at end
.  t - string to be added
-  n - length of the original allocated string

   Level: intermediate

  Notes:
  Not for use in Fortran

  Unlike the system call strncat(), the length passed in is the length of the
  original allocated space, not the length of the left-over space. This is
  similar to the BSD system call strlcat().

.seealso: PetscStrcpy(), PetscStrncpy(), PetscStrcat()

@*/
PetscErrorCode  PetscStrlcat(char s[],const char t[],size_t n)
{
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscAssertFalse(t && !n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"String buffer length must be positive");
  if (!t) PetscFunctionReturn(0);
  ierr = PetscStrlen(t,&len);CHKERRQ(ierr);
  strncat(s,t,n - len);
  s[n-1] = 0;
  PetscFunctionReturn(0);
}

void  PetscStrcmpNoError(const char a[],const char b[],PetscBool  *flg)
{
  int c;

  if (!a && !b)      *flg = PETSC_TRUE;
  else if (!a || !b) *flg = PETSC_FALSE;
  else {
    c = strcmp(a,b);
    if (c) *flg = PETSC_FALSE;
    else   *flg = PETSC_TRUE;
  }
}

/*@C
   PetscStrcmp - Compares two strings,

   Not Collective

   Input Parameters:
+  a - pointer to string first string
-  b - pointer to second string

   Output Parameter:
.  flg - PETSC_TRUE if the two strings are equal

   Level: intermediate

   Notes:
    Not for use in Fortran

.seealso: PetscStrgrt(), PetscStrncmp(), PetscStrcasecmp()

@*/
PetscErrorCode  PetscStrcmp(const char a[],const char b[],PetscBool  *flg)
{
  int c;

  PetscFunctionBegin;
  if (!a && !b)      *flg = PETSC_TRUE;
  else if (!a || !b) *flg = PETSC_FALSE;
  else {
    c = strcmp(a,b);
    if (c) *flg = PETSC_FALSE;
    else   *flg = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

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

   Not for use in Fortran

   Level: intermediate

.seealso: PetscStrcmp(), PetscStrncmp(), PetscStrcasecmp()

@*/
PetscErrorCode  PetscStrgrt(const char a[],const char b[],PetscBool  *t)
{
  int c;

  PetscFunctionBegin;
  if (!a && !b) *t = PETSC_FALSE;
  else if (a && !b) *t = PETSC_TRUE;
  else if (!a && b) *t = PETSC_FALSE;
  else {
    c = strcmp(a,b);
    if (c > 0) *t = PETSC_TRUE;
    else       *t = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

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

   Not for use in Fortran

   Level: intermediate

.seealso: PetscStrcmp(), PetscStrncmp(), PetscStrgrt()

@*/
PetscErrorCode  PetscStrcasecmp(const char a[],const char b[],PetscBool  *t)
{
  int c;

  PetscFunctionBegin;
  if (!a && !b) c = 0;
  else if (!a || !b) c = 1;
#if defined(PETSC_HAVE_STRCASECMP)
  else c = strcasecmp(a,b);
#elif defined(PETSC_HAVE_STRICMP)
  else c = stricmp(a,b);
#else
  else {
    char           *aa,*bb;
    PetscErrorCode ierr;
    ierr = PetscStrallocpy(a,&aa);CHKERRQ(ierr);
    ierr = PetscStrallocpy(b,&bb);CHKERRQ(ierr);
    ierr = PetscStrtolower(aa);CHKERRQ(ierr);
    ierr = PetscStrtolower(bb);CHKERRQ(ierr);
    ierr = PetscStrcmp(aa,bb,t);CHKERRQ(ierr);
    ierr = PetscFree(aa);CHKERRQ(ierr);
    ierr = PetscFree(bb);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif
  if (!c) *t = PETSC_TRUE;
  else    *t = PETSC_FALSE;
  PetscFunctionReturn(0);
}

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

   Notes:
    Not for use in Fortran

.seealso: PetscStrgrt(), PetscStrcmp(), PetscStrcasecmp()

@*/
PetscErrorCode  PetscStrncmp(const char a[],const char b[],size_t n,PetscBool  *t)
{
  int c;

  PetscFunctionBegin;
  c = strncmp(a,b,n);
  if (!c) *t = PETSC_TRUE;
  else    *t = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   PetscStrchr - Locates first occurrence of a character in a string

   Not Collective

   Input Parameters:
+  a - pointer to string
-  b - character

   Output Parameter:
.  c - location of occurrence, NULL if not found

   Level: intermediate

   Notes:
    Not for use in Fortran

@*/
PetscErrorCode  PetscStrchr(const char a[],char b,char *c[])
{
  PetscFunctionBegin;
  *c = (char*)strchr(a,b);
  PetscFunctionReturn(0);
}

/*@C
   PetscStrrchr - Locates one location past the last occurrence of a character in a string,
      if the character is not found then returns entire string

   Not Collective

   Input Parameters:
+  a - pointer to string
-  b - character

   Output Parameter:
.  tmp - location of occurrence, a if not found

   Level: intermediate

   Notes:
    Not for use in Fortran

@*/
PetscErrorCode  PetscStrrchr(const char a[],char b,char *tmp[])
{
  PetscFunctionBegin;
  *tmp = (char*)strrchr(a,b);
  if (!*tmp) *tmp = (char*)a;
  else *tmp = *tmp + 1;
  PetscFunctionReturn(0);
}

/*@C
   PetscStrtolower - Converts string to lower case

   Not Collective

   Input Parameters:
.  a - pointer to string

   Level: intermediate

   Notes:
    Not for use in Fortran

@*/
PetscErrorCode  PetscStrtolower(char a[])
{
  PetscFunctionBegin;
  while (*a) {
    if (*a >= 'A' && *a <= 'Z') *a += 'a' - 'A';
    a++;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscStrtoupper - Converts string to upper case

   Not Collective

   Input Parameters:
.  a - pointer to string

   Level: intermediate

   Notes:
    Not for use in Fortran

@*/
PetscErrorCode  PetscStrtoupper(char a[])
{
  PetscFunctionBegin;
  while (*a) {
    if (*a >= 'a' && *a <= 'z') *a += 'A' - 'a';
    a++;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscStrendswith - Determines if a string ends with a certain string

   Not Collective

   Input Parameters:
+  a - pointer to string
-  b - string to endwith

   Output Parameter:
.  flg - PETSC_TRUE or PETSC_FALSE

   Notes:
    Not for use in Fortran

   Level: intermediate

@*/
PetscErrorCode  PetscStrendswith(const char a[],const char b[],PetscBool *flg)
{
  char           *test;
  PetscErrorCode ierr;
  size_t         na,nb;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  ierr = PetscStrrstr(a,b,&test);CHKERRQ(ierr);
  if (test) {
    ierr = PetscStrlen(a,&na);CHKERRQ(ierr);
    ierr = PetscStrlen(b,&nb);CHKERRQ(ierr);
    if (a+na-nb == test) *flg = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscStrbeginswith - Determines if a string begins with a certain string

   Not Collective

   Input Parameters:
+  a - pointer to string
-  b - string to begin with

   Output Parameter:
.  flg - PETSC_TRUE or PETSC_FALSE

   Notes:
    Not for use in Fortran

   Level: intermediate

.seealso: PetscStrendswithwhich(), PetscStrendswith(), PetscStrtoupper, PetscStrtolower(), PetscStrrchr(), PetscStrchr(),
          PetscStrncmp(), PetscStrlen(), PetscStrncmp(), PetscStrcmp()

@*/
PetscErrorCode  PetscStrbeginswith(const char a[],const char b[],PetscBool *flg)
{
  char           *test;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  ierr = PetscStrrstr(a,b,&test);CHKERRQ(ierr);
  if (test && (test == a)) *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
   PetscStrendswithwhich - Determines if a string ends with one of several possible strings

   Not Collective

   Input Parameters:
+  a - pointer to string
-  bs - strings to end with (last entry must be NULL)

   Output Parameter:
.  cnt - the index of the string it ends with or the index of NULL

   Notes:
    Not for use in Fortran

   Level: intermediate

@*/
PetscErrorCode  PetscStrendswithwhich(const char a[],const char *const *bs,PetscInt *cnt)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *cnt = 0;
  while (bs[*cnt]) {
    ierr = PetscStrendswith(a,bs[*cnt],&flg);CHKERRQ(ierr);
    if (flg) PetscFunctionReturn(0);
    *cnt += 1;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscStrrstr - Locates last occurrence of string in another string

   Not Collective

   Input Parameters:
+  a - pointer to string
-  b - string to find

   Output Parameter:
.  tmp - location of occurrence

   Notes:
    Not for use in Fortran

   Level: intermediate

@*/
PetscErrorCode  PetscStrrstr(const char a[],const char b[],char *tmp[])
{
  const char *stmp = a, *ltmp = NULL;

  PetscFunctionBegin;
  while (stmp) {
    stmp = (char*)strstr(stmp,b);
    if (stmp) {ltmp = stmp;stmp++;}
  }
  *tmp = (char*)ltmp;
  PetscFunctionReturn(0);
}

/*@C
   PetscStrstr - Locates first occurrence of string in another string

   Not Collective

   Input Parameters:
+  haystack - string to search
-  needle - string to find

   Output Parameter:
.  tmp - location of occurrence, is a NULL if the string is not found

   Notes:
    Not for use in Fortran

   Level: intermediate

@*/
PetscErrorCode  PetscStrstr(const char haystack[],const char needle[],char *tmp[])
{
  PetscFunctionBegin;
  *tmp = (char*)strstr(haystack,needle);
  PetscFunctionReturn(0);
}

struct _p_PetscToken {char token;char *array;char *current;};

/*@C
   PetscTokenFind - Locates next "token" in a string

   Not Collective

   Input Parameters:
.  a - pointer to token

   Output Parameter:
.  result - location of occurrence, NULL if not found

   Notes:

     This version is different from the system version in that
  it allows you to pass a read-only string into the function.

     This version also treats all characters etc. inside a double quote "
   as a single token.

     For example if the separator character is + and the string is xxxx+y then the first fine will return a pointer to a null terminated xxxx and the
   second will return a null terminated y

     If the separator character is + and the string is xxxx then the first and only token found will be a pointer to a null terminated xxxx

    Not for use in Fortran

   Level: intermediate

.seealso: PetscTokenCreate(), PetscTokenDestroy()
@*/
PetscErrorCode  PetscTokenFind(PetscToken a,char *result[])
{
  char *ptr = a->current,token;

  PetscFunctionBegin;
  *result = a->current;
  if (ptr && !*ptr) {*result = NULL; PetscFunctionReturn(0);}
  token = a->token;
  if (ptr && (*ptr == '"')) {token = '"';(*result)++;ptr++;}
  while (ptr) {
    if (*ptr == token) {
      *ptr++ = 0;
      while (*ptr == a->token) ptr++;
      a->current = ptr;
      break;
    }
    if (!*ptr) {
      a->current = NULL;
      break;
    }
    ptr++;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscTokenCreate - Creates a PetscToken used to find tokens in a string

   Not Collective

   Input Parameters:
+  string - the string to look in
-  b - the separator character

   Output Parameter:
.  t- the token object

   Notes:

     This version is different from the system version in that
  it allows you to pass a read-only string into the function.

    Not for use in Fortran

   Level: intermediate

.seealso: PetscTokenFind(), PetscTokenDestroy()
@*/
PetscErrorCode  PetscTokenCreate(const char a[],const char b,PetscToken *t)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(t);CHKERRQ(ierr);
  ierr = PetscStrallocpy(a,&(*t)->array);CHKERRQ(ierr);

  (*t)->current = (*t)->array;
  (*t)->token   = b;
  PetscFunctionReturn(0);
}

/*@C
   PetscTokenDestroy - Destroys a PetscToken

   Not Collective

   Input Parameters:
.  a - pointer to token

   Level: intermediate

   Notes:
    Not for use in Fortran

.seealso: PetscTokenCreate(), PetscTokenFind()
@*/
PetscErrorCode  PetscTokenDestroy(PetscToken *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*a) PetscFunctionReturn(0);
  ierr = PetscFree((*a)->array);CHKERRQ(ierr);
  ierr = PetscFree(*a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscStrInList - search string in character-delimited list

   Not Collective

   Input Parameters:
+  str - the string to look for
.  list - the list to search in
-  sep - the separator character

   Output Parameter:
.  found - whether str is in list

   Level: intermediate

   Notes:
    Not for use in Fortran

.seealso: PetscTokenCreate(), PetscTokenFind(), PetscStrcmp()
@*/
PetscErrorCode PetscStrInList(const char str[],const char list[],char sep,PetscBool *found)
{
  PetscToken     token;
  char           *item;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *found = PETSC_FALSE;
  ierr = PetscTokenCreate(list,sep,&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&item);CHKERRQ(ierr);
  while (item) {
    ierr = PetscStrcmp(str,item,found);CHKERRQ(ierr);
    if (*found) break;
    ierr = PetscTokenFind(token,&item);CHKERRQ(ierr);
  }
  ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscGetPetscDir - Gets the directory PETSc is installed in

   Not Collective

   Output Parameter:
.  dir - the directory

   Level: developer

   Notes:
    Not for use in Fortran

@*/
PetscErrorCode  PetscGetPetscDir(const char *dir[])
{
  PetscFunctionBegin;
  *dir = PETSC_DIR;
  PetscFunctionReturn(0);
}

/*@C
   PetscStrreplace - Replaces substrings in string with other substrings

   Not Collective

   Input Parameters:
+   comm - MPI_Comm of processors that are processing the string
.   aa - the string to look in
.   b - the resulting copy of a with replaced strings (b can be the same as a)
-   len - the length of b

   Notes:
      Replaces   ${PETSC_ARCH},${PETSC_DIR},${PETSC_LIB_DIR},${DISPLAY},
      ${HOMEDIRECTORY},${WORKINGDIRECTORY},${USERNAME}, ${HOSTNAME} with appropriate values
      as well as any environmental variables.

      PETSC_LIB_DIR uses the environmental variable if it exists. PETSC_ARCH and PETSC_DIR use what
      PETSc was built with and do not use environmental variables.

      Not for use in Fortran

   Level: intermediate

@*/
PetscErrorCode  PetscStrreplace(MPI_Comm comm,const char aa[],char b[],size_t len)
{
  PetscErrorCode ierr;
  int            i = 0;
  size_t         l,l1,l2,l3;
  char           *work,*par,*epar,env[1024],*tfree,*a = (char*)aa;
  const char     *s[] = {"${PETSC_ARCH}","${PETSC_DIR}","${PETSC_LIB_DIR}","${DISPLAY}","${HOMEDIRECTORY}","${WORKINGDIRECTORY}","${USERNAME}","${HOSTNAME}",NULL};
  char           *r[] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
  PetscBool      flag;
  static size_t  DISPLAY_LENGTH = 265,USER_LENGTH = 256, HOST_LENGTH = 256;

  PetscFunctionBegin;
  PetscAssertFalse(!a || !b,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"a and b strings must be nonnull");
  if (aa == b) {
    ierr = PetscStrallocpy(aa,(char**)&a);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(len,&work);CHKERRQ(ierr);

  /* get values for replaced variables */
  ierr = PetscStrallocpy(PETSC_ARCH,&r[0]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(PETSC_DIR,&r[1]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(PETSC_LIB_DIR,&r[2]);CHKERRQ(ierr);
  ierr = PetscMalloc1(DISPLAY_LENGTH,&r[3]);CHKERRQ(ierr);
  ierr = PetscMalloc1(PETSC_MAX_PATH_LEN,&r[4]);CHKERRQ(ierr);
  ierr = PetscMalloc1(PETSC_MAX_PATH_LEN,&r[5]);CHKERRQ(ierr);
  ierr = PetscMalloc1(USER_LENGTH,&r[6]);CHKERRQ(ierr);
  ierr = PetscMalloc1(HOST_LENGTH,&r[7]);CHKERRQ(ierr);
  ierr = PetscGetDisplay(r[3],DISPLAY_LENGTH);CHKERRQ(ierr);
  ierr = PetscGetHomeDirectory(r[4],PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscGetWorkingDirectory(r[5],PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscGetUserName(r[6],USER_LENGTH);CHKERRQ(ierr);
  ierr = PetscGetHostName(r[7],HOST_LENGTH);CHKERRQ(ierr);

  /* replace that are in environment */
  ierr = PetscOptionsGetenv(comm,"PETSC_LIB_DIR",env,sizeof(env),&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PetscFree(r[2]);CHKERRQ(ierr);
    ierr = PetscStrallocpy(env,&r[2]);CHKERRQ(ierr);
  }

  /* replace the requested strings */
  ierr = PetscStrncpy(b,a,len);CHKERRQ(ierr);
  while (s[i]) {
    ierr = PetscStrlen(s[i],&l);CHKERRQ(ierr);
    ierr = PetscStrstr(b,s[i],&par);CHKERRQ(ierr);
    while (par) {
      *par =  0;
      par += l;

      ierr = PetscStrlen(b,&l1);CHKERRQ(ierr);
      ierr = PetscStrlen(r[i],&l2);CHKERRQ(ierr);
      ierr = PetscStrlen(par,&l3);CHKERRQ(ierr);
      PetscAssertFalse(l1 + l2 + l3 >= len,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"b len is not long enough to hold new values");
      ierr = PetscStrncpy(work,b,len);CHKERRQ(ierr);
      ierr = PetscStrlcat(work,r[i],len);CHKERRQ(ierr);
      ierr = PetscStrlcat(work,par,len);CHKERRQ(ierr);
      ierr = PetscStrncpy(b,work,len);CHKERRQ(ierr);
      ierr = PetscStrstr(b,s[i],&par);CHKERRQ(ierr);
    }
    i++;
  }
  i = 0;
  while (r[i]) {
    tfree = (char*)r[i];
    ierr  = PetscFree(tfree);CHKERRQ(ierr);
    i++;
  }

  /* look for any other ${xxx} strings to replace from environmental variables */
  ierr = PetscStrstr(b,"${",&par);CHKERRQ(ierr);
  while (par) {
    *par  = 0;
    par  += 2;
    ierr  = PetscStrncpy(work,b,len);CHKERRQ(ierr);
    ierr  = PetscStrstr(par,"}",&epar);CHKERRQ(ierr);
    *epar = 0;
    epar += 1;
    ierr  = PetscOptionsGetenv(comm,par,env,sizeof(env),&flag);CHKERRQ(ierr);
    PetscAssertFalse(!flag,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Substitution string ${%s} not found as environmental variable",par);
    ierr = PetscStrlcat(work,env,len);CHKERRQ(ierr);
    ierr = PetscStrlcat(work,epar,len);CHKERRQ(ierr);
    ierr = PetscStrncpy(b,work,len);CHKERRQ(ierr);
    ierr = PetscStrstr(b,"${",&par);CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  if (aa == b) {
    ierr = PetscFree(a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscEListFind - searches list of strings for given string, using case insensitive matching

   Not Collective

   Input Parameters:
+  n - number of strings in
.  list - list of strings to search
-  str - string to look for, empty string "" accepts default (first entry in list)

   Output Parameters:
+  value - index of matching string (if found)
-  found - boolean indicating whether string was found (can be NULL)

   Notes:
   Not for use in Fortran

   Level: advanced
@*/
PetscErrorCode PetscEListFind(PetscInt n,const char *const *list,const char *str,PetscInt *value,PetscBool *found)
{
  PetscErrorCode ierr;
  PetscBool matched;
  PetscInt i;

  PetscFunctionBegin;
  if (found) *found = PETSC_FALSE;
  for (i=0; i<n; i++) {
    ierr = PetscStrcasecmp(str,list[i],&matched);CHKERRQ(ierr);
    if (matched || !str[0]) {
      if (found) *found = PETSC_TRUE;
      *value = i;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscEnumFind - searches enum list of strings for given string, using case insensitive matching

   Not Collective

   Input Parameters:
+  enumlist - list of strings to search, followed by enum name, then enum prefix, then NUL
-  str - string to look for

   Output Parameters:
+  value - index of matching string (if found)
-  found - boolean indicating whether string was found (can be NULL)

   Notes:
   Not for use in Fortran

   Level: advanced
@*/
PetscErrorCode PetscEnumFind(const char *const *enumlist,const char *str,PetscEnum *value,PetscBool *found)
{
  PetscErrorCode ierr;
  PetscInt n = 0,evalue;
  PetscBool efound;

  PetscFunctionBegin;
  while (enumlist[n++]) PetscAssertFalse(n > 50,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument appears to be wrong or have more than 50 entries");
  PetscAssertFalse(n < 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument must have at least two entries: typename and type prefix");
  n -= 3; /* drop enum name, prefix, and null termination */
  ierr = PetscEListFind(n,enumlist,str,&evalue,&efound);CHKERRQ(ierr);
  if (efound) *value = (PetscEnum)evalue;
  if (found) *found = efound;
  PetscFunctionReturn(0);
}
