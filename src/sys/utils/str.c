/*
    We define the string operations here. The reason we just do not use
  the standard string routines in the PETSc code is that on some machines
  they are broken or have the wrong prototypes.

*/
#include <petsc/private/petscimpl.h> /*I  "petscsys.h"   I*/
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
PetscErrorCode PetscStrToArray(const char s[], char sp, int *argc, char ***args)
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
  if (!n) return 0;
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
PetscErrorCode PetscStrToArrayDestroy(int argc, char **args)
{
  for (int i = 0; i < argc; ++i) free(args[i]);
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
PetscErrorCode PetscStrlen(const char s[], size_t *len)
{
  PetscFunctionBegin;
  *len = s ? strlen(s) : 0;
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
PetscErrorCode PetscStrallocpy(const char s[], char *t[])
{
  char *tmp = NULL;

  PetscFunctionBegin;
  if (s) {
    size_t len;

    CHKERRQ(PetscStrlen(s,&len));
    CHKERRQ(PetscMalloc1(1+len,&tmp));
    CHKERRQ(PetscStrcpy(tmp,s));
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
PetscErrorCode PetscStrArrayallocpy(const char *const *list, char ***t)
{
  PetscInt n = 0;

  PetscFunctionBegin;
  while (list[n++]) ;
  CHKERRQ(PetscMalloc1(n+1,t));
  for (PetscInt i=0; i<n; i++) CHKERRQ(PetscStrallocpy(list[i],(*t)+i));
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
  PetscInt n = 0;

  PetscFunctionBegin;
  if (!*list) PetscFunctionReturn(0);
  while ((*list)[n]) {
    CHKERRQ(PetscFree((*list)[n]));
    ++n;
  }
  CHKERRQ(PetscFree(*list));
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
PetscErrorCode PetscStrNArrayallocpy(PetscInt n, const char *const *list, char ***t)
{
  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(n,t));
  for (PetscInt i=0; i<n; i++) CHKERRQ(PetscStrallocpy(list[i],(*t)+i));
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
PetscErrorCode PetscStrNArrayDestroy(PetscInt n, char ***list)
{
  PetscFunctionBegin;
  if (!*list) PetscFunctionReturn(0);
  for (PetscInt i=0; i<n; i++) CHKERRQ(PetscFree((*list)[i]));
  CHKERRQ(PetscFree(*list));
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

PetscErrorCode PetscStrcpy(char s[], const char t[])
{
  PetscFunctionBegin;
  if (t) {
    PetscValidCharPointer(s,1);
    PetscValidCharPointer(t,2);
    strcpy(s,t);
  } else if (s) s[0] = 0;
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
PetscErrorCode PetscStrncpy(char s[], const char t[], size_t n)
{
  PetscFunctionBegin;
  if (s) PetscCheck(n,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Requires an output string of length at least 1 to hold the termination character");
  if (t) {
    PetscValidCharPointer(s,1);
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
PetscErrorCode PetscStrcat(char s[], const char t[])
{
  PetscFunctionBegin;
  if (!t) PetscFunctionReturn(0);
  PetscValidCharPointer(s,1);
  PetscValidCharPointer(t,2);
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
PetscErrorCode PetscStrlcat(char s[], const char t[], size_t n)
{
  size_t len;

  PetscFunctionBegin;
  if (!t) PetscFunctionReturn(0);
  PetscValidCharPointer(s,1);
  PetscValidCharPointer(t,2);
  PetscCheck(n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"String buffer length must be positive");
  CHKERRQ(PetscStrlen(t,&len));
  strncat(s,t,n - len);
  s[n-1] = 0;
  PetscFunctionReturn(0);
}

void PetscStrcmpNoError(const char a[], const char b[], PetscBool *flg)
{
  if (!a && !b)      *flg = PETSC_TRUE;
  else if (!a || !b) *flg = PETSC_FALSE;
  else *flg = strcmp(a,b) ? PETSC_FALSE : PETSC_TRUE;
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
PetscErrorCode  PetscStrcmp(const char a[],const char b[],PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(flg,3);
  if (!a && !b)      *flg = PETSC_TRUE;
  else if (!a || !b) *flg = PETSC_FALSE;
  else               *flg = (PetscBool)!strcmp(a,b);
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
PetscErrorCode PetscStrgrt(const char a[], const char b[], PetscBool *t)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(t,3);
  if (!a && !b)     *t = PETSC_FALSE;
  else if (a && !b) *t = PETSC_TRUE;
  else if (!a && b) *t = PETSC_FALSE;
  else {
    PetscValidCharPointer(a,1);
    PetscValidCharPointer(b,2);
    *t = strcmp(a,b) > 0 ? PETSC_TRUE : PETSC_FALSE;
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
PetscErrorCode PetscStrcasecmp(const char a[], const char b[], PetscBool *t)
{
  int c;

  PetscFunctionBegin;
  PetscValidBoolPointer(t,3);
  if (!a && !b)      c = 0;
  else if (!a || !b) c = 1;
#if defined(PETSC_HAVE_STRCASECMP)
  else c = strcasecmp(a,b);
#elif defined(PETSC_HAVE_STRICMP)
  else c = stricmp(a,b);
#else
  else {
    char           *aa,*bb;
    PetscErrorCode ierr;
    CHKERRQ(PetscStrallocpy(a,&aa));
    CHKERRQ(PetscStrallocpy(b,&bb));
    CHKERRQ(PetscStrtolower(aa));
    CHKERRQ(PetscStrtolower(bb));
    CHKERRQ(PetscStrcmp(aa,bb,t));
    CHKERRQ(PetscFree(aa));
    CHKERRQ(PetscFree(bb));
    PetscFunctionReturn(0);
  }
#endif
  *t = c ? PETSC_FALSE : PETSC_TRUE;
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
PetscErrorCode PetscStrncmp(const char a[], const char b[], size_t n, PetscBool *t)
{
  PetscFunctionBegin;
  if (n) {
    PetscValidCharPointer(a,1);
    PetscValidCharPointer(b,2);
  }
  PetscValidBoolPointer(t,4);
  *t = strncmp(a,b,n) ? PETSC_FALSE : PETSC_TRUE;
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
PetscErrorCode PetscStrchr(const char a[], char b, char *c[])
{
  PetscFunctionBegin;
  PetscValidCharPointer(a,1);
  PetscValidPointer(c,3);
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
PetscErrorCode PetscStrrchr(const char a[], char b, char *tmp[])
{
  PetscFunctionBegin;
  PetscValidCharPointer(a,1);
  PetscValidPointer(tmp,3);
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
PetscErrorCode PetscStrtolower(char a[])
{
  PetscFunctionBegin;
  PetscValidCharPointer(a,1);
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
PetscErrorCode PetscStrtoupper(char a[])
{
  PetscFunctionBegin;
  PetscValidCharPointer(a,1);
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
PetscErrorCode PetscStrendswith(const char a[], const char b[], PetscBool *flg)
{
  char *test;

  PetscFunctionBegin;
  PetscValidBoolPointer(flg,3);
  *flg = PETSC_FALSE;
  CHKERRQ(PetscStrrstr(a,b,&test));
  if (test) {
    size_t na,nb;

    CHKERRQ(PetscStrlen(a,&na));
    CHKERRQ(PetscStrlen(b,&nb));
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
PetscErrorCode PetscStrbeginswith(const char a[], const char b[], PetscBool *flg)
{
  char *test;

  PetscFunctionBegin;
  PetscValidCharPointer(a,1);
  PetscValidCharPointer(b,2);
  PetscValidBoolPointer(flg,3);
  *flg = PETSC_FALSE;
  CHKERRQ(PetscStrrstr(a,b,&test));
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
PetscErrorCode PetscStrendswithwhich(const char a[], const char *const *bs, PetscInt *cnt)
{
  PetscFunctionBegin;
  PetscValidPointer(bs,2);
  PetscValidIntPointer(cnt,3);
  *cnt = 0;
  while (bs[*cnt]) {
    PetscBool flg;

    CHKERRQ(PetscStrendswith(a,bs[*cnt],&flg));
    if (flg) PetscFunctionReturn(0);
    ++(*cnt);
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
PetscErrorCode PetscStrrstr(const char a[], const char b[], char *tmp[])
{
  const char *ltmp = NULL;

  PetscFunctionBegin;
  PetscValidCharPointer(a,1);
  PetscValidCharPointer(b,2);
  PetscValidPointer(tmp,3);
  while (a) {
    a = (char*)strstr(a,b);
    if (a) ltmp = a++;
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
PetscErrorCode PetscStrstr(const char haystack[],const char needle[],char *tmp[])
{
  PetscFunctionBegin;
  PetscValidCharPointer(haystack,1);
  PetscValidCharPointer(needle,2);
  PetscValidPointer(tmp,3);
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
PetscErrorCode PetscTokenFind(PetscToken a, char *result[])
{
  char *ptr,token;

  PetscFunctionBegin;
  PetscValidPointer(a,1);
  PetscValidPointer(result,2);
  *result = ptr = a->current;
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
PetscErrorCode PetscTokenCreate(const char a[], const char b, PetscToken *t)
{
  PetscFunctionBegin;
  PetscValidCharPointer(a,1);
  PetscValidPointer(t,3);
  CHKERRQ(PetscNew(t));
  CHKERRQ(PetscStrallocpy(a,&(*t)->array));

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
PetscErrorCode PetscTokenDestroy(PetscToken *a)
{
  PetscFunctionBegin;
  if (!*a) PetscFunctionReturn(0);
  CHKERRQ(PetscFree((*a)->array));
  CHKERRQ(PetscFree(*a));
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
PetscErrorCode PetscStrInList(const char str[], const char list[], char sep, PetscBool *found)
{
  PetscToken  token;
  char       *item;

  PetscFunctionBegin;
  PetscValidBoolPointer(found,4);
  *found = PETSC_FALSE;
  CHKERRQ(PetscTokenCreate(list,sep,&token));
  CHKERRQ(PetscTokenFind(token,&item));
  while (item) {
    CHKERRQ(PetscStrcmp(str,item,found));
    if (*found) break;
    CHKERRQ(PetscTokenFind(token,&item));
  }
  CHKERRQ(PetscTokenDestroy(&token));
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
PetscErrorCode PetscGetPetscDir(const char *dir[])
{
  PetscFunctionBegin;
  PetscValidPointer(dir,1);
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
PetscErrorCode PetscStrreplace(MPI_Comm comm, const char aa[], char b[], size_t len)
{
  int            i = 0;
  size_t         l,l1,l2,l3;
  char           *work,*par,*epar,env[1024],*tfree,*a = (char*)aa;
  const char     *s[] = {"${PETSC_ARCH}","${PETSC_DIR}","${PETSC_LIB_DIR}","${DISPLAY}","${HOMEDIRECTORY}","${WORKINGDIRECTORY}","${USERNAME}","${HOSTNAME}",NULL};
  char           *r[] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
  PetscBool      flag;
  static size_t  DISPLAY_LENGTH = 265,USER_LENGTH = 256, HOST_LENGTH = 256;

  PetscFunctionBegin;
  PetscValidCharPointer(aa,2);
  PetscValidCharPointer(b,3);
  if (aa == b) CHKERRQ(PetscStrallocpy(aa,(char**)&a));
  CHKERRQ(PetscMalloc1(len,&work));

  /* get values for replaced variables */
  CHKERRQ(PetscStrallocpy(PETSC_ARCH,&r[0]));
  CHKERRQ(PetscStrallocpy(PETSC_DIR,&r[1]));
  CHKERRQ(PetscStrallocpy(PETSC_LIB_DIR,&r[2]));
  CHKERRQ(PetscMalloc1(DISPLAY_LENGTH,&r[3]));
  CHKERRQ(PetscMalloc1(PETSC_MAX_PATH_LEN,&r[4]));
  CHKERRQ(PetscMalloc1(PETSC_MAX_PATH_LEN,&r[5]));
  CHKERRQ(PetscMalloc1(USER_LENGTH,&r[6]));
  CHKERRQ(PetscMalloc1(HOST_LENGTH,&r[7]));
  CHKERRQ(PetscGetDisplay(r[3],DISPLAY_LENGTH));
  CHKERRQ(PetscGetHomeDirectory(r[4],PETSC_MAX_PATH_LEN));
  CHKERRQ(PetscGetWorkingDirectory(r[5],PETSC_MAX_PATH_LEN));
  CHKERRQ(PetscGetUserName(r[6],USER_LENGTH));
  CHKERRQ(PetscGetHostName(r[7],HOST_LENGTH));

  /* replace that are in environment */
  CHKERRQ(PetscOptionsGetenv(comm,"PETSC_LIB_DIR",env,sizeof(env),&flag));
  if (flag) {
    CHKERRQ(PetscFree(r[2]));
    CHKERRQ(PetscStrallocpy(env,&r[2]));
  }

  /* replace the requested strings */
  CHKERRQ(PetscStrncpy(b,a,len));
  while (s[i]) {
    CHKERRQ(PetscStrlen(s[i],&l));
    CHKERRQ(PetscStrstr(b,s[i],&par));
    while (par) {
      *par = 0;
      par += l;

      CHKERRQ(PetscStrlen(b,&l1));
      CHKERRQ(PetscStrlen(r[i],&l2));
      CHKERRQ(PetscStrlen(par,&l3));
      PetscCheckFalse(l1 + l2 + l3 >= len,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"b len is not long enough to hold new values");
      CHKERRQ(PetscStrncpy(work,b,len));
      CHKERRQ(PetscStrlcat(work,r[i],len));
      CHKERRQ(PetscStrlcat(work,par,len));
      CHKERRQ(PetscStrncpy(b,work,len));
      CHKERRQ(PetscStrstr(b,s[i],&par));
    }
    i++;
  }
  i = 0;
  while (r[i]) {
    tfree = (char*)r[i];
    CHKERRQ(PetscFree(tfree));
    i++;
  }

  /* look for any other ${xxx} strings to replace from environmental variables */
  CHKERRQ(PetscStrstr(b,"${",&par));
  while (par) {
    *par  = 0;
    par  += 2;
    CHKERRQ(PetscStrncpy(work,b,len));
    CHKERRQ(PetscStrstr(par,"}",&epar));
    *epar = 0;
    epar += 1;
    CHKERRQ(PetscOptionsGetenv(comm,par,env,sizeof(env),&flag));
    PetscCheckFalse(!flag,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Substitution string ${%s} not found as environmental variable",par);
    CHKERRQ(PetscStrlcat(work,env,len));
    CHKERRQ(PetscStrlcat(work,epar,len));
    CHKERRQ(PetscStrncpy(b,work,len));
    CHKERRQ(PetscStrstr(b,"${",&par));
  }
  CHKERRQ(PetscFree(work));
  if (aa == b) CHKERRQ(PetscFree(a));
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
PetscErrorCode PetscEListFind(PetscInt n, const char *const *list, const char *str, PetscInt *value, PetscBool *found)
{
  PetscFunctionBegin;
  if (found) {
    PetscValidBoolPointer(found,5);
    *found = PETSC_FALSE;
  }
  for (PetscInt i = 0; i < n; ++i) {
    PetscBool matched;

    CHKERRQ(PetscStrcasecmp(str,list[i],&matched));
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
PetscErrorCode PetscEnumFind(const char *const *enumlist, const char *str, PetscEnum *value, PetscBool *found)
{
  PetscInt  n = 0,evalue;
  PetscBool efound;

  PetscFunctionBegin;
  PetscValidPointer(enumlist,1);
  while (enumlist[n++]) PetscCheck(n <= 50,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument appears to be wrong or have more than 50 entries");
  PetscCheck(n >= 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument must have at least two entries: typename and type prefix");
  n -= 3; /* drop enum name, prefix, and null termination */
  CHKERRQ(PetscEListFind(n,enumlist,str,&evalue,&efound));
  if (efound) {
    PetscValidPointer(value,3);
    *value = (PetscEnum)evalue;
  }
  if (found) {
    PetscValidBoolPointer(found,4);
    *found = efound;
  }
  PetscFunctionReturn(0);
}
