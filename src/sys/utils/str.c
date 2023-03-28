/*
    We define the string operations here. The reason we just do not use
  the standard string routines in the PETSc code is that on some machines
  they are broken or have the wrong prototypes.
*/
#include <petsc/private/petscimpl.h> /*I  "petscsys.h"   I*/
#if defined(PETSC_HAVE_STRINGS_H)
  #include <strings.h> /* strcasecmp */
#endif

/*@C
   PetscStrToArray - Separates a string by a character (for example ' ' or '\n') and creates an array of strings

   Not Collective; No Fortran Support

   Input Parameters:
+  s - pointer to string
-  sp - separator character

   Output Parameters:
+   argc - the number of entries in the array
-   args - an array of the entries with a `NULL` at the end

   Level: intermediate

   Note:
    This may be called before `PetscInitialize()` or after `PetscFinalize()`

   Developer Notes:
   Uses raw `malloc()` and does not call error handlers since this may be used before PETSc is initialized.

   Used to generate argc, args arguments passed to `MPI_Init()`

.seealso: `PetscStrToArrayDestroy()`, `PetscToken`, `PetscTokenCreate()`
@*/
PetscErrorCode PetscStrToArray(const char s[], char sp, int *argc, char ***args)
{
  int       i, j, n, *lens, cnt = 0;
  PetscBool flg = PETSC_FALSE;

  if (!s) n = 0;
  else n = strlen(s);
  *argc = 0;
  *args = NULL;
  for (; n > 0; n--) { /* remove separator chars at the end - and will empty the string if all chars are separator chars */
    if (s[n - 1] != sp) break;
  }
  if (!n) return PETSC_SUCCESS;
  for (i = 0; i < n; i++) {
    if (s[i] != sp) break;
  }
  for (; i < n + 1; i++) {
    if ((s[i] == sp || s[i] == 0) && !flg) {
      flg = PETSC_TRUE;
      (*argc)++;
    } else if (s[i] != sp) {
      flg = PETSC_FALSE;
    }
  }
  (*args) = (char **)malloc(((*argc) + 1) * sizeof(char *));
  if (!*args) return PETSC_ERR_MEM;
  lens = (int *)malloc((*argc) * sizeof(int));
  if (!lens) return PETSC_ERR_MEM;
  for (i = 0; i < *argc; i++) lens[i] = 0;

  *argc = 0;
  for (i = 0; i < n; i++) {
    if (s[i] != sp) break;
  }
  for (; i < n + 1; i++) {
    if ((s[i] == sp || s[i] == 0) && !flg) {
      flg = PETSC_TRUE;
      (*argc)++;
    } else if (s[i] != sp) {
      lens[*argc]++;
      flg = PETSC_FALSE;
    }
  }

  for (i = 0; i < *argc; i++) {
    (*args)[i] = (char *)malloc((lens[i] + 1) * sizeof(char));
    if (!(*args)[i]) {
      free(lens);
      for (j = 0; j < i; j++) free((*args)[j]);
      free(*args);
      return PETSC_ERR_MEM;
    }
  }
  free(lens);
  (*args)[*argc] = NULL;

  *argc = 0;
  for (i = 0; i < n; i++) {
    if (s[i] != sp) break;
  }
  for (; i < n + 1; i++) {
    if ((s[i] == sp || s[i] == 0) && !flg) {
      flg                   = PETSC_TRUE;
      (*args)[*argc][cnt++] = 0;
      (*argc)++;
      cnt = 0;
    } else if (s[i] != sp && s[i] != 0) {
      (*args)[*argc][cnt++] = s[i];
      flg                   = PETSC_FALSE;
    }
  }
  return PETSC_SUCCESS;
}

/*@C
   PetscStrToArrayDestroy - Frees array created with `PetscStrToArray()`.

   Not Collective; No Fortran Support

   Output Parameters:
+  argc - the number of arguments
-  args - the array of arguments

   Level: intermediate

   Note:
    This may be called before `PetscInitialize()` or after `PetscFinalize()`

.seealso: `PetscStrToArray()`
@*/
PetscErrorCode PetscStrToArrayDestroy(int argc, char **args)
{
  for (int i = 0; i < argc; ++i) free(args[i]);
  if (args) free(args);
  return PETSC_SUCCESS;
}

/*@C
   PetscStrArrayallocpy - Allocates space to hold a copy of an array of strings then copies the strings

   Not Collective; No Fortran Support

   Input Parameter:
.  s - pointer to array of strings (final string is a `NULL`)

   Output Parameter:
.  t - the copied array string

   Level: intermediate

   Note:
   If `t` has previously been allocated then that memory is lost, you may need to `PetscStrArrayDestroy()`
   the array before calling this routine.

.seealso: `PetscStrallocpy()`, `PetscStrArrayDestroy()`, `PetscStrNArrayallocpy()`
@*/
PetscErrorCode PetscStrArrayallocpy(const char *const *list, char ***t)
{
  PetscInt n = 0;

  PetscFunctionBegin;
  while (list[n++])
    ;
  PetscCall(PetscMalloc1(n + 1, t));
  for (PetscInt i = 0; i < n; i++) PetscCall(PetscStrallocpy(list[i], (*t) + i));
  (*t)[n] = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscStrArrayDestroy - Frees array of strings created with `PetscStrArrayallocpy()`.

   Not Collective; No Fortran Support

   Output Parameter:
.   list - array of strings

   Level: intermediate

.seealso: `PetscStrArrayallocpy()`
@*/
PetscErrorCode PetscStrArrayDestroy(char ***list)
{
  PetscInt n = 0;

  PetscFunctionBegin;
  if (!*list) PetscFunctionReturn(PETSC_SUCCESS);
  while ((*list)[n]) {
    PetscCall(PetscFree((*list)[n]));
    ++n;
  }
  PetscCall(PetscFree(*list));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscStrNArrayallocpy - Allocates space to hold a copy of an array of strings then copies the strings

   Not Collective; No Fortran Support

   Input Parameters:
+  n - the number of string entries
-  s - pointer to array of strings

   Output Parameter:
.  t - the copied array string

   Level: intermediate

.seealso: `PetscStrallocpy()`, `PetscStrArrayallocpy()`, `PetscStrNArrayDestroy()`
@*/
PetscErrorCode PetscStrNArrayallocpy(PetscInt n, const char *const *list, char ***t)
{
  PetscFunctionBegin;
  PetscCall(PetscMalloc1(n, t));
  for (PetscInt i = 0; i < n; i++) PetscCall(PetscStrallocpy(list[i], (*t) + i));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscStrNArrayDestroy - Frees array of strings created with `PetscStrNArrayallocpy()`.

   Not Collective; No Fortran Support

   Output Parameters:
+   n - number of string entries
-   list - array of strings

   Level: intermediate

.seealso: `PetscStrNArrayallocpy()`, `PetscStrArrayallocpy()`
@*/
PetscErrorCode PetscStrNArrayDestroy(PetscInt n, char ***list)
{
  PetscFunctionBegin;
  if (!*list) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt i = 0; i < n; i++) PetscCall(PetscFree((*list)[i]));
  PetscCall(PetscFree(*list));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscBasename - returns a pointer to the last entry of a / or \ separated directory path

   Not Collective; No Fortran Support

   Input Parameter:
.  a - pointer to string

   Level: intermediate

.seealso: `PetscStrgrt()`, `PetscStrncmp()`, `PetscStrcasecmp()`, `PetscStrrchr()`, `PetscStrcmp()`, `PetscStrstr()`,
          `PetscTokenCreate()`, `PetscStrToArray()`, `PetscStrInList()`
@*/
const char *PetscBasename(const char a[])
{
  const char *ptr = NULL;

  (void)PetscStrrchr(a, '/', (char **)&ptr);
  if (ptr == a) {
    if (PetscStrrchr(a, '\\', (char **)&ptr)) ptr = NULL;
  }
  return ptr;
}

/*@C
   PetscStrcasecmp - Returns true if the two strings are the same
     except possibly for case.

   Not Collective; No Fortran Support

   Input Parameters:
+  a - pointer to first string
-  b - pointer to second string

   Output Parameter:
.  flg - if the two strings are the same

   Level: intermediate

   Note:
   `NULL` arguments are ok

.seealso: `PetscStrcmp()`, `PetscStrncmp()`, `PetscStrgrt()`
@*/
PetscErrorCode PetscStrcasecmp(const char a[], const char b[], PetscBool *t)
{
  int c;

  PetscFunctionBegin;
  PetscValidBoolPointer(t, 3);
  if (!a && !b) c = 0;
  else if (!a || !b) c = 1;
#if defined(PETSC_HAVE_STRCASECMP)
  else c = strcasecmp(a, b);
#elif defined(PETSC_HAVE_STRICMP)
  else c = stricmp(a, b);
#else
  else {
    char *aa, *bb;

    PetscCall(PetscStrallocpy(a, &aa));
    PetscCall(PetscStrallocpy(b, &bb));
    PetscCall(PetscStrtolower(aa));
    PetscCall(PetscStrtolower(bb));
    PetscCall(PetscStrcmp(aa, bb, t));
    PetscCall(PetscFree(aa));
    PetscCall(PetscFree(bb));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif
  *t = c ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscStrendswithwhich - Determines if a string ends with one of several possible strings

   Not Collective; No Fortran Support

   Input Parameters:
+  a - pointer to string
-  bs - strings to end with (last entry must be `NULL`)

   Output Parameter:
.  cnt - the index of the string it ends with or the index of `NULL`

   Level: intermediate

.seealso: `PetscStrbeginswithwhich()`, `PetscStrendswith()`, `PetscStrtoupper`, `PetscStrtolower()`, `PetscStrrchr()`, `PetscStrchr()`,
          `PetscStrncmp()`, `PetscStrlen()`, `PetscStrncmp()`, `PetscStrcmp()`
@*/
PetscErrorCode PetscStrendswithwhich(const char a[], const char *const *bs, PetscInt *cnt)
{
  PetscFunctionBegin;
  PetscValidPointer(bs, 2);
  PetscValidIntPointer(cnt, 3);
  *cnt = 0;
  while (bs[*cnt]) {
    PetscBool flg;

    PetscCall(PetscStrendswith(a, bs[*cnt], &flg));
    if (flg) PetscFunctionReturn(PETSC_SUCCESS);
    ++(*cnt);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct _p_PetscToken {
  char  token;
  char *array;
  char *current;
};

/*@C
   PetscTokenFind - Locates next "token" in a `PetscToken`

   Not Collective; No Fortran Support

   Input Parameter:
.  a - pointer to token

   Output Parameter:
.  result - location of occurrence, `NULL` if not found

   Level: intermediate

   Notes:
   Treats all characters etc. inside a double quote "
   as a single token.

     For example if the separator character is + and the string is xxxx+y then the first fine will return a pointer to a `NULL` terminated xxxx and the
   second will return a `NULL` terminated y

     If the separator character is + and the string is xxxx then the first and only token found will be a pointer to a `NULL` terminated xxxx

.seealso: `PetscToken`, `PetscTokenCreate()`, `PetscTokenDestroy()`
@*/
PetscErrorCode PetscTokenFind(PetscToken a, char *result[])
{
  char *ptr, token;

  PetscFunctionBegin;
  PetscValidPointer(a, 1);
  PetscValidPointer(result, 2);
  *result = ptr = a->current;
  if (ptr && !*ptr) {
    *result = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  token = a->token;
  if (ptr && (*ptr == '"')) {
    token = '"';
    (*result)++;
    ptr++;
  }
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscTokenCreate - Creates a `PetscToken` used to find tokens in a string

   Not Collective; No Fortran Support

   Input Parameters:
+  string - the string to look in
-  b - the separator character

   Output Parameter:
.  t - the token object

   Level: intermediate

   Note:
     This version is different from the system version in that
  it allows you to pass a read-only string into the function.

.seealso: `PetscToken`, `PetscTokenFind()`, `PetscTokenDestroy()`
@*/
PetscErrorCode PetscTokenCreate(const char a[], char b, PetscToken *t)
{
  PetscFunctionBegin;
  PetscValidCharPointer(a, 1);
  PetscValidPointer(t, 3);
  PetscCall(PetscNew(t));
  PetscCall(PetscStrallocpy(a, &(*t)->array));

  (*t)->current = (*t)->array;
  (*t)->token   = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscTokenDestroy - Destroys a `PetscToken`

   Not Collective; No Fortran Support

   Input Parameter:
.  a - pointer to token

   Level: intermediate

.seealso: `PetscToken`, `PetscTokenCreate()`, `PetscTokenFind()`
@*/
PetscErrorCode PetscTokenDestroy(PetscToken *a)
{
  PetscFunctionBegin;
  if (!*a) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFree((*a)->array));
  PetscCall(PetscFree(*a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscStrInList - search for a string in character-delimited list

   Not Collective; No Fortran Support

   Input Parameters:
+  str - the string to look for
.  list - the list to search in
-  sep - the separator character

   Output Parameter:
.  found - whether `str` is in `list`

   Level: intermediate

.seealso: `PetscTokenCreate()`, `PetscTokenFind()`, `PetscStrcmp()`
@*/
PetscErrorCode PetscStrInList(const char str[], const char list[], char sep, PetscBool *found)
{
  PetscToken token;
  char      *item;

  PetscFunctionBegin;
  PetscValidBoolPointer(found, 4);
  *found = PETSC_FALSE;
  PetscCall(PetscTokenCreate(list, sep, &token));
  PetscCall(PetscTokenFind(token, &item));
  while (item) {
    PetscCall(PetscStrcmp(str, item, found));
    if (*found) break;
    PetscCall(PetscTokenFind(token, &item));
  }
  PetscCall(PetscTokenDestroy(&token));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscGetPetscDir - Gets the directory PETSc is installed in

   Not Collective; No Fortran Support

   Output Parameter:
.  dir - the directory

   Level: developer

@*/
PetscErrorCode PetscGetPetscDir(const char *dir[])
{
  PetscFunctionBegin;
  PetscValidPointer(dir, 1);
  *dir = PETSC_DIR;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscStrreplace - Replaces substrings in string with other substrings

   Not Collective; No Fortran Support

   Input Parameters:
+   comm - `MPI_Comm` of processors that are processing the string
.   aa - the string to look in
.   b - the resulting copy of a with replaced strings (`b` can be the same as `a`)
-   len - the length of `b`

   Level: developer

   Notes:
      Replaces ${PETSC_ARCH},${PETSC_DIR},${PETSC_LIB_DIR},${DISPLAY},
      ${HOMEDIRECTORY},${WORKINGDIRECTORY},${USERNAME}, ${HOSTNAME}, ${PETSC_MAKE} with appropriate values
      as well as any environmental variables.

      `PETSC_LIB_DIR` uses the environmental variable if it exists. `PETSC_ARCH` and `PETSC_DIR` use what
      PETSc was built with and do not use environmental variables.

@*/
PetscErrorCode PetscStrreplace(MPI_Comm comm, const char aa[], char b[], size_t len)
{
  int           i = 0;
  size_t        l, l1, l2, l3;
  char         *work, *par, *epar = NULL, env[1024], *tfree, *a = (char *)aa;
  const char   *s[] = {"${PETSC_ARCH}", "${PETSC_DIR}", "${PETSC_LIB_DIR}", "${DISPLAY}", "${HOMEDIRECTORY}", "${WORKINGDIRECTORY}", "${USERNAME}", "${HOSTNAME}", "${PETSC_MAKE}", NULL};
  char         *r[] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  PetscBool     flag;
  static size_t DISPLAY_LENGTH = 265, USER_LENGTH = 256, HOST_LENGTH = 256;

  PetscFunctionBegin;
  PetscValidCharPointer(aa, 2);
  PetscValidCharPointer(b, 3);
  if (aa == b) PetscCall(PetscStrallocpy(aa, (char **)&a));
  PetscCall(PetscMalloc1(len, &work));

  /* get values for replaced variables */
  PetscCall(PetscStrallocpy(PETSC_ARCH, &r[0]));
  PetscCall(PetscStrallocpy(PETSC_DIR, &r[1]));
  PetscCall(PetscStrallocpy(PETSC_LIB_DIR, &r[2]));
  PetscCall(PetscMalloc1(DISPLAY_LENGTH, &r[3]));
  PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN, &r[4]));
  PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN, &r[5]));
  PetscCall(PetscMalloc1(USER_LENGTH, &r[6]));
  PetscCall(PetscMalloc1(HOST_LENGTH, &r[7]));
  PetscCall(PetscGetDisplay(r[3], DISPLAY_LENGTH));
  PetscCall(PetscGetHomeDirectory(r[4], PETSC_MAX_PATH_LEN));
  PetscCall(PetscGetWorkingDirectory(r[5], PETSC_MAX_PATH_LEN));
  PetscCall(PetscGetUserName(r[6], USER_LENGTH));
  PetscCall(PetscGetHostName(r[7], HOST_LENGTH));
  PetscCall(PetscStrallocpy(PETSC_OMAKE, &r[8]));

  /* replace that are in environment */
  PetscCall(PetscOptionsGetenv(comm, "PETSC_LIB_DIR", env, sizeof(env), &flag));
  if (flag) {
    PetscCall(PetscFree(r[2]));
    PetscCall(PetscStrallocpy(env, &r[2]));
  }

  /* replace the requested strings */
  PetscCall(PetscStrncpy(b, a, len));
  while (s[i]) {
    PetscCall(PetscStrlen(s[i], &l));
    PetscCall(PetscStrstr(b, s[i], &par));
    while (par) {
      *par = 0;
      par += l;

      PetscCall(PetscStrlen(b, &l1));
      PetscCall(PetscStrlen(r[i], &l2));
      PetscCall(PetscStrlen(par, &l3));
      PetscCheck(l1 + l2 + l3 < len, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "b len is not long enough to hold new values");
      PetscCall(PetscStrncpy(work, b, len));
      PetscCall(PetscStrlcat(work, r[i], len));
      PetscCall(PetscStrlcat(work, par, len));
      PetscCall(PetscStrncpy(b, work, len));
      PetscCall(PetscStrstr(b, s[i], &par));
    }
    i++;
  }
  i = 0;
  while (r[i]) {
    tfree = (char *)r[i];
    PetscCall(PetscFree(tfree));
    i++;
  }

  /* look for any other ${xxx} strings to replace from environmental variables */
  PetscCall(PetscStrstr(b, "${", &par));
  while (par) {
    *par = 0;
    par += 2;
    PetscCall(PetscStrncpy(work, b, len));
    PetscCall(PetscStrstr(par, "}", &epar));
    *epar = 0;
    epar += 1;
    PetscCall(PetscOptionsGetenv(comm, par, env, sizeof(env), &flag));
    PetscCheck(flag, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Substitution string ${%s} not found as environmental variable", par);
    PetscCall(PetscStrlcat(work, env, len));
    PetscCall(PetscStrlcat(work, epar, len));
    PetscCall(PetscStrncpy(b, work, len));
    PetscCall(PetscStrstr(b, "${", &par));
  }
  PetscCall(PetscFree(work));
  if (aa == b) PetscCall(PetscFree(a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscEListFind - searches list of strings for given string, using case insensitive matching

   Not Collective; No Fortran Support

   Input Parameters:
+  n - number of strings in
.  list - list of strings to search
-  str - string to look for, empty string "" accepts default (first entry in list)

   Output Parameters:
+  value - index of matching string (if found)
-  found - boolean indicating whether string was found (can be `NULL`)

   Level: developer

.seealso: `PetscEnumFind()`
@*/
PetscErrorCode PetscEListFind(PetscInt n, const char *const *list, const char *str, PetscInt *value, PetscBool *found)
{
  PetscFunctionBegin;
  if (found) {
    PetscValidBoolPointer(found, 5);
    *found = PETSC_FALSE;
  }
  for (PetscInt i = 0; i < n; ++i) {
    PetscBool matched;

    PetscCall(PetscStrcasecmp(str, list[i], &matched));
    if (matched || !str[0]) {
      if (found) *found = PETSC_TRUE;
      *value = i;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscEnumFind - searches enum list of strings for given string, using case insensitive matching

   Not Collective; No Fortran Support

   Input Parameters:
+  enumlist - list of strings to search, followed by enum name, then enum prefix, then `NULL`
-  str - string to look for

   Output Parameters:
+  value - index of matching string (if found)
-  found - boolean indicating whether string was found (can be `NULL`)

   Level: advanced

.seealso: `PetscEListFind()`
@*/
PetscErrorCode PetscEnumFind(const char *const *enumlist, const char *str, PetscEnum *value, PetscBool *found)
{
  PetscInt  n = 0, evalue;
  PetscBool efound;

  PetscFunctionBegin;
  PetscValidPointer(enumlist, 1);
  while (enumlist[n++]) PetscCheck(n <= 50, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "List argument appears to be wrong or have more than 50 entries");
  PetscCheck(n >= 3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "List argument must have at least two entries: typename and type prefix");
  n -= 3; /* drop enum name, prefix, and null termination */
  PetscCall(PetscEListFind(n, enumlist, str, &evalue, &efound));
  if (efound) {
    PetscValidPointer(value, 3);
    *value = (PetscEnum)evalue;
  }
  if (found) {
    PetscValidBoolPointer(found, 4);
    *found = efound;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscCIFilename - returns the basename of a file name when the PETSc CI portable error output mode is enabled.

  Not Collective; No Fortran Support

  Input Parameter:
. file - the file name

  Level: developer

  Note:
  PETSc CI mode is a mode of running PETSc where output (both error and non-error) is made portable across all systems
  so that comparisons of output between runs are easy to make.

  This mode is used for all tests in the test harness, it applies to both debug and optimized builds.

  Use the option `-petsc_ci` to turn on PETSc CI mode. It changes certain output in non-error situations to be portable for
  all systems, mainly the output of options. It is passed to all PETSc programs automatically by the test harness.

  Always uses the Unix / as the file separate even on Microsoft Windows systems

  The option `-petsc_ci_portable_error_output` attempts to output the same error messages on all systems for the test harness.
  In particular the output of filenames and line numbers in PETSc stacks. This is to allow (limited) checking of PETSc
  error handling by the test harness. This options also causes PETSc to attempt to return an error code of 0 so that the test
  harness can process the output for differences in the usual manner as for successful runs. It should be provided to the test
  harness in the args: argument for specific examples. It will not necessarily produce portable output if different errors
  (or no errors) occur on a subset of the MPI ranks.

.seealso: `PetscCILinenumber()`
@*/
const char *PetscCIFilename(const char *file)
{
  if (!PetscCIEnabledPortableErrorOutput) return file;
  return PetscBasename(file);
}

/*@C
  PetscCILinenumber - returns a line number except if `PetscCIEnablePortableErrorOutput` is set when it returns 0

  Not Collective; No Fortran Support

  Input Parameter:
. linenumber - the initial line number

  Level: developer

  Note:
  See `PetscCIFilename()` for details on usage

.seealso: `PetscCIFilename()`
@*/
int PetscCILinenumber(int linenumber)
{
  if (!PetscCIEnabledPortableErrorOutput) return linenumber;
  return 0;
}

/*@C
  PetscStrcat - Concatenates a string onto a given string

  Not Collective, No Fortran Support

  Input Parameters:
+ s - string to be added to
- t - pointer to string to be added to end

  Level: deprecated (since 3.18.5)

  Notes:
  It is recommended you use `PetscStrlcat()` instead of this routine.

.seealso: `PetscStrlcat()`
@*/
PetscErrorCode PetscStrcat(char s[], const char t[])
{
  PetscFunctionBegin;
  if (!t) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidCharPointer(s, 1);
  strcat(s, t);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStrcpy - Copies a string

  Not Collective, No Fortran Support

  Input Parameter:
. t - pointer to string

  Output Parameter:
. s - the copied string

  Level: deprecated (since 3.18.5)

  Notes:
  It is recommended you use `PetscStrncpy()` (equivalently `PetscArraycpy()` or
  `PetscMemcpy()`) instead of this routine.

  `NULL` strings returns a string starting with zero.

.seealso: `PetscStrncpy()`
@*/
PetscErrorCode PetscStrcpy(char s[], const char t[])
{
  PetscFunctionBegin;
  if (t) {
    PetscValidCharPointer(s, 1);
    PetscValidCharPointer(t, 2);
    strcpy(s, t);
  } else if (s) {
    s[0] = '\0';
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
