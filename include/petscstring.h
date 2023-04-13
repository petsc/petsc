#ifndef PETSC_STRING_H
#define PETSC_STRING_H

#include <petscsystypes.h>
#include <petscerror.h>
#include <petscmacros.h>
#include <petscsys.h>

/* SUBMANSEC = Sys */

#include <stddef.h> /* size_t */
#include <string.h> /* for memcpy, memset */

PETSC_EXTERN PetscErrorCode PetscMemcmp(const void *, const void *, size_t, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscStrToArray(const char[], char, int *, char ***);
PETSC_EXTERN PetscErrorCode PetscStrToArrayDestroy(int, char **);
PETSC_EXTERN PetscErrorCode PetscStrcasecmp(const char[], const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscStrendswithwhich(const char[], const char *const *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscStrArrayallocpy(const char *const *, char ***);
PETSC_EXTERN PetscErrorCode PetscStrArrayDestroy(char ***);
PETSC_EXTERN PetscErrorCode PetscStrNArrayallocpy(PetscInt, const char *const *, char ***);
PETSC_EXTERN PetscErrorCode PetscStrNArrayDestroy(PetscInt, char ***);
PETSC_EXTERN PetscErrorCode PetscStrreplace(MPI_Comm, const char[], char[], size_t);

PETSC_EXTERN PetscErrorCode PetscTokenCreate(const char[], char, PetscToken *);
PETSC_EXTERN PetscErrorCode PetscTokenFind(PetscToken, char *[]);
PETSC_EXTERN PetscErrorCode PetscTokenDestroy(PetscToken *);

PETSC_EXTERN PetscErrorCode PetscStrInList(const char[], const char[], char, PetscBool *);
PETSC_EXTERN const char    *PetscBasename(const char[]);
PETSC_EXTERN PetscErrorCode PetscEListFind(PetscInt, const char *const *, const char *, PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscEnumFind(const char *const *, const char *, PetscEnum *, PetscBool *);

PETSC_EXTERN PetscErrorCode PetscStrcat(char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscStrcpy(char[], const char[]);

#define PetscAssertPointer_Private(ptr, arg) PetscAssert((ptr), PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Null Pointer: Parameter '" PetscStringize(ptr) "' # " PetscStringize(arg))

/*@C
  PetscStrtolower - Converts a string to lower case

  Not Collective, No Fortran Support

  Input Parameter:
. a - pointer to string

  Level: intermediate

.seealso: `PetscStrtoupper()`
@*/
static inline PetscErrorCode PetscStrtolower(char a[])
{
  PetscFunctionBegin;
  PetscAssertPointer_Private(a, 1);
  while (*a) {
    if (*a >= 'A' && *a <= 'Z') *a += 'a' - 'A';
    a++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStrtoupper - Converts a string to upper case

  Not Collective, No Fortran Support

  Input Parameter:
. a - pointer to string

  Level: intermediate

.seealso: `PetscStrtolower()`
@*/
static inline PetscErrorCode PetscStrtoupper(char a[])
{
  PetscFunctionBegin;
  PetscAssertPointer_Private(a, 1);
  while (*a) {
    if (*a >= 'a' && *a <= 'z') *a += 'A' - 'a';
    a++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStrlen - Gets the length of a string

  Not Collective, No Fortran Support

  Input Parameter:
. s - pointer to string

  Output Parameter:
. len - length in bytes

  Level: intermediate

  Notes:
  This routine is analogous to `strlen()`. `NULL` string returns a length of zero.

.seealso: `PetscStrallocpy()`
@*/
static inline PetscErrorCode PetscStrlen(const char s[], size_t *len)
{
  PetscFunctionBegin;
  PetscAssertPointer_Private(len, 2);
  if (s) {
#if PetscHasBuiltin(__builtin_strlen)
    *len = __builtin_strlen(s);
#else
    *len = strlen(s);
#endif
  } else {
    *len = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStrallocpy - Allocates space to hold a copy of a string then copies the string into the new space

  Not Collective, No Fortran Support

  Input Parameter:
. s - pointer to string

  Output Parameter:
. t - the copied string

  Level: intermediate

  Notes:
  `NULL` string returns a new `NULL` string.

  If `t` has previously been allocated then that memory is lost, you may need to `PetscFree()`
  the array before calling this routine.

.seealso: `PetscStrArrayallocpy()`, `PetscStrNArrayallocpy()`
@*/
static inline PetscErrorCode PetscStrallocpy(const char s[], char *t[])
{
  PetscFunctionBegin;
  PetscAssertPointer_Private(t, 2);
  *t = PETSC_NULLPTR;
  if (s) {
    size_t len;
    char  *tmp;

    PetscAssertPointer_Private(s, 1);
    PetscCall(PetscStrlen(s, &len));
    PetscCall(PetscMalloc1(len + 1, &tmp));
#if PetscHasBuiltin(__builtin_memcpy)
    __builtin_memcpy(tmp, s, len);
#else
    memcpy(tmp, s, len);
#endif
    tmp[len] = '\0';
    *t       = tmp;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline void PetscStrcmpNoError(const char a[], const char b[], PetscBool *flg)
{
  if (!a && !b) {
    *flg = PETSC_TRUE;
  } else if (!a || !b) {
    *flg = PETSC_FALSE;
  } else {
#if PetscHasBuiltin(__builtin_strcmp)
    *flg = __builtin_strcmp(a, b) ? PETSC_FALSE : PETSC_TRUE;
#else
    *flg = strcmp(a, b) ? PETSC_FALSE : PETSC_TRUE;
#endif
  }
}

/*@C
  PetscStrcmp - Compares two strings

  Not Collective, No Fortran Support

  Input Parameters:
+ a - pointer to string first string
- b - pointer to second string

  Output Parameter:
. flg - `PETSC_TRUE` if the two strings are equal

  Level: intermediate

.seealso: `PetscStrgrt()`, `PetscStrncmp()`, `PetscStrcasecmp()`
@*/
static inline PetscErrorCode PetscStrcmp(const char a[], const char b[], PetscBool *flg)
{
  PetscFunctionBegin;
  PetscAssertPointer_Private(flg, 3);
  PetscStrcmpNoError(a, b, flg);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(__GNUC__) && !defined(__clang__)
  #if __GNUC__ >= 8
    #define PETSC_SILENCE_WSTRINGOP_TRUNCATION_BEGIN \
      do { \
        _Pragma("GCC diagnostic push"); \
        _Pragma("GCC diagnostic ignored \"-Wstringop-truncation\""); \
      } while (0)
    #define PETSC_SILENCE_WSTRINGOP_TRUNCATION_END _Pragma("GCC diagnostic pop")
  #endif
#endif

#ifndef PETSC_SILENCE_WSTRINGOP_TRUNCATION_BEGIN
  #define PETSC_SILENCE_WSTRINGOP_TRUNCATION_BEGIN (void)0
  #define PETSC_SILENCE_WSTRINGOP_TRUNCATION_END   (void)0
#endif

/*@C
  PetscStrncpy - Copies a string up to a certain length

  Not Collective

  Input Parameters:
+ t - pointer to string
- n - the length to copy

  Output Parameter:
. s - the copied string

  Level: intermediate

  Notes:
  `NULL` string returns a string starting with zero.

  If the string that is being copied is of length `n` or larger, then the entire string is not
  copied and the final location of `s` is set to `NULL`. This is different then the behavior of
  `strncpy()` which leaves `s` non-terminated if there is not room for the entire string.

  Developers Notes:
  Should this be `PetscStrlcpy()` to reflect its behavior which is like `strlcpy()` not
  `strncpy()`?

.seealso: `PetscStrlcat()`, `PetscStrallocpy()`
@*/
static inline PetscErrorCode PetscStrncpy(char s[], const char t[], size_t n)
{
  PetscFunctionBegin;
  if (s) PetscAssert(n, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Requires an output string of length at least 1 to hold the termination character");
  if (t) {
    PetscAssertPointer_Private(s, 1);
    PETSC_SILENCE_WSTRINGOP_TRUNCATION_BEGIN;
#if PetscHasBuiltin(__builtin_strncpy)
    __builtin_strncpy(s, t, n);
#else
    strncpy(s, t, n);
#endif
    PETSC_SILENCE_WSTRINGOP_TRUNCATION_END;
    s[n - 1] = '\0';
  } else if (s) {
    s[0] = '\0';
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStrlcat - Concatenates a string onto a given string, up to a given length

  Not Collective, No Fortran Support

  Input Parameters:
+ s - pointer to string to be added to at end
. t - string to be added
- n - length of the original allocated string

  Level: intermediate

  Notes:
  Unlike the system call `strncat()`, the length passed in is the length of the
  original allocated space, not the length of the left-over space. This is
  similar to the BSD system call `strlcat()`.

.seealso: `PetscStrncpy()`
@*/
static inline PetscErrorCode PetscStrlcat(char s[], const char t[], size_t n)
{
  size_t len;

  PetscFunctionBegin;
  if (!t) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssert(n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "String buffer length must be positive");
  PetscCall(PetscStrlen(s, &len));
  PETSC_SILENCE_WSTRINGOP_TRUNCATION_BEGIN;
#if PetscHasBuiltin(__builtin_strncat)
  __builtin_strncat(s, t, n - len);
#else
  strncat(s, t, n - len);
#endif
  PETSC_SILENCE_WSTRINGOP_TRUNCATION_END;
  s[n - 1] = '\0';
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef PETSC_SILENCE_WSTRINGOP_TRUNCATION_BEGIN
#undef PETSC_SILENCE_WSTRINGOP_TRUNCATION_END

/*@C
  PetscStrncmp - Compares two strings, up to a certain length

  Not Collective, No Fortran Support

  Input Parameters:
+ a - pointer to first string
. b - pointer to second string
- n - length to compare up to

  Output Parameter:
. t - `PETSC_TRUE` if the two strings are equal, `PETSC_FALSE` otherwise

  Level: intermediate

  Note:
  If `n` is `0`, `t` is set to `PETSC_FALSE`. `a` and/or `b` may be `NULL` in this case.

.seealso: `PetscStrgrt()`, `PetscStrcmp()`, `PetscStrcasecmp()`
@*/
static inline PetscErrorCode PetscStrncmp(const char a[], const char b[], size_t n, PetscBool *t)
{
  PetscFunctionBegin;
  PetscAssertPointer_Private(t, 4);
  *t = PETSC_FALSE;
  if (n) {
    PetscAssertPointer_Private(a, 1);
    PetscAssertPointer_Private(b, 2);
  }
#if PetscHasBuiltin(__builtin_strncmp)
  *t = __builtin_strncmp(a, b, n) ? PETSC_FALSE : PETSC_TRUE;
#else
  *t = strncmp(a, b, n) ? PETSC_FALSE : PETSC_TRUE;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStrrstr - Locates last occurrence of string in another string

  Not Collective, No Fortran Support

  Input Parameters:
+ a - pointer to string
- b - string to find

  Output Parameter:
. tmp - location of occurrence

  Level: intermediate

.seealso: `PetscStrbeginswithwhich()`, `PetscStrendswith()`, `PetscStrtoupper`,
          `PetscStrtolower()`, `PetscStrrchr()`, `PetscStrchr()`, `PetscStrncmp()`, `PetscStrlen()`,
          `PetscStrcmp()`
@*/
static inline PetscErrorCode PetscStrrstr(const char a[], const char b[], char *tmp[])
{
  const char *ltmp = PETSC_NULLPTR;

  PetscFunctionBegin;
  PetscAssertPointer_Private(a, 1);
  PetscAssertPointer_Private(b, 2);
  PetscAssertPointer_Private(tmp, 3);
  while (a) {
#if PetscHasBuiltin(__builtin_strstr)
    a = (char *)__builtin_strstr(a, b);
#else
    a = (char *)strstr(a, b);
#endif
    if (a) ltmp = a++;
  }
  *tmp = (char *)ltmp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStrstr - Locates first occurrence of string in another string

  Not Collective, No Fortran Support

  Input Parameters:
+ haystack - string to search
- needle   - string to find

  Output Parameter:
. tmp - location of `needle` within `haystack`, `NULL` if `needle` is not found

  Level: intermediate

.seealso: `PetscStrbeginswithwhich()`, `PetscStrendswith()`, `PetscStrtoupper`,
          `PetscStrtolower()`, `PetscStrrchr()`, `PetscStrchr()`, `PetscStrncmp()`, `PetscStrlen()`,
          `PetscStrcmp()`
@*/
static inline PetscErrorCode PetscStrstr(const char haystack[], const char needle[], char *tmp[])
{
  PetscFunctionBegin;
  PetscAssertPointer_Private(haystack, 1);
  PetscAssertPointer_Private(needle, 2);
  PetscAssertPointer_Private(tmp, 3);
#if PetscHasBuiltin(__builtin_strstr)
  *tmp = (char *)__builtin_strstr(haystack, needle);
#else
  *tmp = (char *)strstr(haystack, needle);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStrgrt - If first string is greater than the second

  Not Collective, No Fortran Support

  Input Parameters:
+ a - pointer to first string
- b - pointer to second string

  Output Parameter:
. flg - `PETSC_TRUE` if `a` is strictly greater than `b`, `PETSC_FALSE` otherwise

  Level: intermediate

  Notes:
  `NULL` arguments are OK, a `NULL` string is considered smaller than all others. If both `a`
  and `b` are `NULL` then `t` is set to `PETSC_FALSE`.

.seealso: `PetscStrcmp()`, `PetscStrncmp()`, `PetscStrcasecmp()`
@*/
static inline PetscErrorCode PetscStrgrt(const char a[], const char b[], PetscBool *t)
{
  PetscFunctionBegin;
  PetscAssertPointer_Private(t, 3);
  if (!a && !b) {
    *t = PETSC_FALSE;
  } else if (a && !b) {
    *t = PETSC_TRUE;
  } else if (!a && b) {
    *t = PETSC_FALSE;
  } else {
#if PetscHasBuiltin(__builtin_strcmp)
    *t = __builtin_strcmp(a, b) > 0 ? PETSC_TRUE : PETSC_FALSE;
#else
    *t = strcmp(a, b) > 0 ? PETSC_TRUE : PETSC_FALSE;
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStrchr - Locates first occurrence of a character in a string

  Not Collective, No Fortran Support

  Input Parameters:
+ a - pointer to string
- b - character

  Output Parameter:
. c - location of occurrence, `NULL` if not found

  Level: intermediate

.seealso: `PetscStrrchr()`, `PetscTokenCreate()`, `PetscStrendswith()`, `PetscStrbeginsswith()`
@*/
static inline PetscErrorCode PetscStrchr(const char a[], char b, char *c[])
{
  PetscFunctionBegin;
  PetscAssertPointer_Private(a, 1);
  PetscAssertPointer_Private(c, 3);
#if PetscHasBuiltin(__builtin_strchr)
  *c = (char *)__builtin_strchr(a, b);
#else
  *c = (char *)strchr(a, b);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStrrchr - Locates one location past the last occurrence of a character in a string, if
  the character is not found then returns entire string

  Not Collective, No Fortran Support

  Input Parameters:
+ a - pointer to string
- b - character

  Output Parameter:
. c - one past location of `b` in `a`, or `a` if `b` was not found

  Level: intermediate

.seealso: `PetscStrchr()`, `PetscTokenCreate()`, `PetscStrendswith()`, `PetscStrbeginsswith()`
@*/
static inline PetscErrorCode PetscStrrchr(const char a[], char b, char *c[])
{
  PetscFunctionBegin;
  PetscAssertPointer_Private(a, 1);
  PetscAssertPointer_Private(c, 3);
#if PetscHasBuiltin(__builtin_strrchr)
  *c = (char *)__builtin_strrchr(a, b);
#else
  *c = (char *)strrchr(a, b);
#endif
  if (!*c) *c = (char *)a;
  else *c = *c + 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStrendswith - Determines if a string ends with a certain string

  Not Collective, No Fortran Support

  Input Parameters:
+ a - string to search
- b - string to end with

  Output Parameter:
. flg - `PETSC_TRUE` if `a` ends with `b`, `PETSC_FALSE` otherwise

  Level: intermediate

  Notes:
  Both `a` and `b` may be `NULL` (in which case `flg` is set to `PETSC_FALSE`) bot not either.

.seealso: `PetscStrendswithwhich()`, `PetscStrbeginswith()`, `PetscStrtoupper`,
          `PetscStrtolower()`, `PetscStrrchr()`, `PetscStrchr()`, `PetscStrncmp()`, `PetscStrlen()`,
          `PetscStrcmp()`
@*/
static inline PetscErrorCode PetscStrendswith(const char a[], const char b[], PetscBool *flg)
{
  size_t na = 0, nb = 0;

  PetscFunctionBegin;
  PetscAssertPointer_Private(flg, 3);
  // do this here to silence stupid "may be used uninitialized"" warnings
  *flg = PETSC_FALSE;
  PetscCall(PetscStrlen(a, &na));
  PetscCall(PetscStrlen(b, &nb));
  if (na >= nb) {
#if PetscHasBuiltin(__builtin_memcmp)
    *flg = __builtin_memcmp(b, a + (na - nb), nb) == 0 ? PETSC_TRUE : PETSC_FALSE;
#else
    *flg = memcmp(b, a + (na - nb), nb) == 0 ? PETSC_TRUE : PETSC_FALSE;
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStrbeginswith - Determines if a string begins with a certain string

  Not Collective, No Fortran Support

  Input Parameters:
+ a - string to search
- b - string to begin with

  Output Parameter:
. flg - `PETSC_TRUE` if `a` begins with `b`, `PETSC_FALSE` otherwise

  Level: intermediate

  Notes:
  Both `a` and `b` may be `NULL` (in which case `flg` is set to `PETSC_FALSE`) but not
  either.

  `a` and `b` may point to the same string.

.seealso: `PetscStrendswithwhich()`, `PetscStrendswith()`, `PetscStrtoupper`,
          `PetscStrtolower()`, `PetscStrrchr()`, `PetscStrchr()`, `PetscStrncmp()`, `PetscStrlen()`,
          `PetscStrcmp()`
@*/
static inline PetscErrorCode PetscStrbeginswith(const char a[], const char b[], PetscBool *flg)
{
  size_t len = 0;

  PetscFunctionBegin;
  PetscAssertPointer_Private(flg, 3);
  // do this here to silence stupid "may be used uninitialized"" warnings
  *flg = PETSC_FALSE;
  PetscCall(PetscStrlen(b, &len));
  PetscCall(PetscStrncmp(a, b, len, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef PetscAssertPointer_Private

/*@C
   PetscMemmove - Copies `n` bytes, beginning at location `b`, to the space
   beginning at location `a`. Copying  between regions that overlap will
   take place correctly. Use `PetscMemcpy()` if the locations do not overlap

   Not Collective

   Input Parameters:
+  b - pointer to initial memory space
.  a - pointer to copy space
-  n - length (in bytes) of space to copy

   Level: intermediate

   Notes:
   `PetscArraymove()` is preferred

   This routine is analogous to `memmove()`.

.seealso: `PetscMemcpy()`, `PetscMemcmp()`, `PetscArrayzero()`, `PetscMemzero()`, `PetscArraycmp()`, `PetscArraycpy()`, `PetscStrallocpy()`,
          `PetscArraymove()`
@*/
static inline PetscErrorCode PetscMemmove(void *a, const void *b, size_t n)
{
  PetscFunctionBegin;
  if (PetscUnlikely((n == 0) || (a == b))) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssert(a, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Trying to copy %zu bytes to null pointer (Argument #1)", n);
  PetscAssert(b, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Trying to copy %zu bytes from a null pointer (Argument #2)", n);
#if PetscDefined(HAVE_MEMMOVE)
  memmove((char *)a, (const char *)b, n);
#else
  if (a < b) {
    if ((char *)a <= (char *)b - n) {
      memcpy(a, b, n);
    } else {
      const size_t ptr_diff = (size_t)((char *)b - (char *)a);

      memcpy(a, b, ptr_diff);
      PetscCall(PetscMemmove((void *)b, (char *)b + ptr_diff, n - ptr_diff));
    }
  } else {
    if ((char *)b <= (char *)a - n) {
      memcpy(a, b, n);
    } else {
      const size_t ptr_diff = (size_t)((char *)a - (char *)b);

      memcpy((void *)((char *)b + n), (char *)b + (n - ptr_diff), ptr_diff);
      PetscCall(PetscMemmove(a, b, n - ptr_diff));
    }
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscMemcpy - Copies `n` bytes, beginning at location `b`, to the space
   beginning at location `a`. The two memory regions CANNOT overlap, use
   `PetscMemmove()` in that case.

   Not Collective

   Input Parameters:
+  b - pointer to initial memory space
-  n - length (in bytes) of space to copy

   Output Parameter:
.  a - pointer to copy space

   Level: intermediate

   Compile Option:
    `PETSC_PREFER_DCOPY_FOR_MEMCPY` will cause the BLAS `dcopy()` routine to be used
                                  for memory copies on double precision values.
    `PETSC_PREFER_COPY_FOR_MEMCPY` will cause C code to be used
                                  for memory copies on double precision values.
    `PETSC_PREFER_FORTRAN_FORMEMCPY` will cause Fortran code to be used
                                  for memory copies on double precision values.

   Notes:
   Prefer `PetscArraycpy()`

   This routine is analogous to `memcpy()`.

.seealso: `PetscMemzero()`, `PetscMemcmp()`, `PetscArrayzero()`, `PetscArraycmp()`, `PetscArraycpy()`, `PetscMemmove()`, `PetscStrallocpy()`
@*/
static inline PetscErrorCode PetscMemcpy(void *a, const void *b, size_t n)
{
  const PETSC_UINTPTR_T al = (PETSC_UINTPTR_T)a;
  const PETSC_UINTPTR_T bl = (PETSC_UINTPTR_T)b;

  PetscFunctionBegin;
  if (PetscUnlikely((n == 0) || (a == b))) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssert(a, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Trying to copy %zu bytes to a null pointer (Argument #1)", n);
  PetscAssert(b, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Trying to copy %zu bytes from a null pointer (Argument #2)", n);
  PetscAssert(!(((al > bl) && (al - bl) < n) || (bl - al) < n), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Memory regions overlap: either use PetscMemmove()\nor make sure your copy regions and lengths are correct.\nLength (bytes) %zu first address %" PRIxPTR " second address %" PRIxPTR, n, al, bl);
  if (PetscDefined(PREFER_DCOPY_FOR_MEMCPY) || PetscDefined(PREFER_COPY_FOR_MEMCPY) || PetscDefined(PREFER_FORTRAN_FORMEMCPY)) {
    if (!(al % sizeof(PetscScalar)) && !(n % sizeof(PetscScalar))) {
      const size_t       scalar_len = n / sizeof(PetscScalar);
      const PetscScalar *x          = (PetscScalar *)b;
      PetscScalar       *y          = (PetscScalar *)a;

#if PetscDefined(PREFER_DCOPY_FOR_MEMCPY)
      {
        const PetscBLASInt one = 1;
        PetscBLASInt       blen;

        PetscCall(PetscBLASIntCast(scalar_len, &blen));
        PetscCallBLAS("BLAScopy", BLAScopy_(&blen, x, &one, y, &one));
      }
#elif PetscDefined(PREFER_FORTRAN_FORMEMCPY)
      fortrancopy_(&scalar_len, x, y);
#else
      for (size_t i = 0; i < scalar_len; i++) y[i] = x[i];
#endif
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  memcpy(a, b, n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscMemzero - Zeros the specified memory.

   Not Collective

   Input Parameters:
+  a - pointer to beginning memory location
-  n - length (in bytes) of memory to initialize

   Level: intermediate

   Compile Option:
   `PETSC_PREFER_BZERO` - on certain machines (the IBM RS6000) the bzero() routine happens
  to be faster than the memset() routine. This flag causes the bzero() routine to be used.

   Notes:
   Prefer `PetscArrayzero()`

.seealso: `PetscMemcpy()`, `PetscMemcmp()`, `PetscArrayzero()`, `PetscArraycmp()`, `PetscArraycpy()`, `PetscMemmove()`, `PetscStrallocpy()`
@*/
static inline PetscErrorCode PetscMemzero(void *a, size_t n)
{
  PetscFunctionBegin;
  if (PetscUnlikely(n == 0)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssert(a, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Trying to zero %zu bytes at a null pointer", n);
  if (PetscDefined(PREFER_ZERO_FOR_MEMZERO) || PetscDefined(PREFER_FORTRAN_FOR_MEMZERO)) {
    if (!(((PETSC_UINTPTR_T)a) % sizeof(PetscScalar)) && !(n % sizeof(PetscScalar))) {
      const size_t scalar_len = n / sizeof(PetscScalar);
      PetscScalar *x          = (PetscScalar *)a;

      if (PetscDefined(PREFER_ZERO_FOR_MEMZERO)) {
        for (size_t i = 0; i < scalar_len; ++i) x[i] = 0;
      } else {
#if PetscDefined(PREFER_FORTRAN_FOR_MEMZERO)
        fortranzero_(&scalar_len, x);
#else
        (void)scalar_len;
        (void)x;
#endif
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
#if PetscDefined(PREFER_BZERO)
  bzero(a, n);
#else
  memset(a, 0, n);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PetscArraycmp - Compares two arrays in memory.

   Synopsis:
    #include <petscstring.h>
    PetscErrorCode PetscArraycmp(const anytype *str1,const anytype *str2,size_t cnt,PetscBool *e)

   Not Collective

   Input Parameters:
+  str1 - First array
.  str2 - Second array
-  cnt  - Count of the array, not in bytes, but number of entries in the arrays

   Output Parameter:
.   e - `PETSC_TRUE` if equal else `PETSC_FALSE`.

   Level: intermediate

   Notes:
   This routine is a preferred replacement to `PetscMemcmp()`

   The arrays must be of the same type

.seealso: `PetscMemcpy()`, `PetscMemcmp()`, `PetscArrayzero()`, `PetscMemzero()`, `PetscArraycpy()`, `PetscMemmove()`, `PetscStrallocpy()`,
          `PetscArraymove()`
M*/
#define PetscArraycmp(str1, str2, cnt, e) ((sizeof(*(str1)) == sizeof(*(str2))) ? PetscMemcmp((str1), (str2), (size_t)(cnt) * sizeof(*(str1)), (e)) : PETSC_ERR_ARG_SIZ)

/*MC
   PetscArraymove - Copies from one array in memory to another, the arrays may overlap. Use `PetscArraycpy()` when the arrays
                    do not overlap

   Synopsis:
    #include <petscstring.h>
    PetscErrorCode PetscArraymove(anytype *str1,const anytype *str2,size_t cnt)

   Not Collective

   Input Parameters:
+  str1 - First array
.  str2 - Second array
-  cnt  - Count of the array, not in bytes, but number of entries in the arrays

   Level: intermediate

   Notes:
   This routine is a preferred replacement to `PetscMemmove()`

   The arrays must be of the same type

.seealso: `PetscMemcpy()`, `PetscMemcmp()`, `PetscArrayzero()`, `PetscMemzero()`, `PetscArraycpy()`, `PetscMemmove()`, `PetscArraycmp()`, `PetscStrallocpy()`
M*/
#define PetscArraymove(str1, str2, cnt) ((sizeof(*(str1)) == sizeof(*(str2))) ? PetscMemmove((str1), (str2), (size_t)(cnt) * sizeof(*(str1))) : PETSC_ERR_ARG_SIZ)

/*MC
   PetscArraycpy - Copies from one array in memory to another

   Synopsis:
    #include <petscstring.h>
    PetscErrorCode PetscArraycpy(anytype *str1,const anytype *str2,size_t cnt)

   Not Collective

   Input Parameters:
+  str1 - First array (destination)
.  str2 - Second array (source)
-  cnt  - Count of the array, not in bytes, but number of entries in the arrays

   Level: intermediate

   Notes:
   This routine is a preferred replacement to `PetscMemcpy()`

   The arrays must be of the same type

.seealso: `PetscMemcpy()`, `PetscMemcmp()`, `PetscArrayzero()`, `PetscMemzero()`, `PetscArraymove()`, `PetscMemmove()`, `PetscArraycmp()`, `PetscStrallocpy()`
M*/
#define PetscArraycpy(str1, str2, cnt) ((sizeof(*(str1)) == sizeof(*(str2))) ? PetscMemcpy((str1), (str2), (size_t)(cnt) * sizeof(*(str1))) : PETSC_ERR_ARG_SIZ)

/*MC
   PetscArrayzero - Zeros an array in memory.

   Synopsis:
    #include <petscstring.h>
    PetscErrorCode PetscArrayzero(anytype *str1,size_t cnt)

   Not Collective

   Input Parameters:
+  str1 - array
-  cnt  - Count of the array, not in bytes, but number of entries in the array

   Level: intermediate

   Notes:
   This routine is a preferred replacement to `PetscMemzero()`

.seealso: `PetscMemcpy()`, `PetscMemcmp()`, `PetscMemzero()`, `PetscArraycmp()`, `PetscArraycpy()`, `PetscMemmove()`, `PetscStrallocpy()`, `PetscArraymove()`
M*/
#define PetscArrayzero(str1, cnt) PetscMemzero((str1), (size_t)(cnt) * sizeof(*(str1)))

#endif // PETSC_STRING_H
