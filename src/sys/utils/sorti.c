
/*
   This file contains routines for sorting integers. Values are sorted in place.
   One can use src/sys/tests/ex52.c for benchmarking.
 */
#include <petsc/private/petscimpl.h> /*I  "petscsys.h"  I*/
#include <petsc/private/hashseti.h>

#define MEDIAN3(v, a, b, c) (v[a] < v[b] ? (v[b] < v[c] ? (b) : (v[a] < v[c] ? (c) : (a))) : (v[c] < v[b] ? (b) : (v[a] < v[c] ? (a) : (c))))

#define MEDIAN(v, right) MEDIAN3(v, right / 4, right / 2, right / 4 * 3)

/* Swap one, two or three pairs. Each pair can have its own type */
#define SWAP1(a, b, t1) \
  do { \
    t1 = a; \
    a  = b; \
    b  = t1; \
  } while (0)
#define SWAP2(a, b, c, d, t1, t2) \
  do { \
    t1 = a; \
    a  = b; \
    b  = t1; \
    t2 = c; \
    c  = d; \
    d  = t2; \
  } while (0)
#define SWAP3(a, b, c, d, e, f, t1, t2, t3) \
  do { \
    t1 = a; \
    a  = b; \
    b  = t1; \
    t2 = c; \
    c  = d; \
    d  = t2; \
    t3 = e; \
    e  = f; \
    f  = t3; \
  } while (0)

/* Swap a & b, *c & *d. c, d, t2 are pointers to a type of size <siz> */
#define SWAP2Data(a, b, c, d, t1, t2, siz) \
  do { \
    t1 = a; \
    a  = b; \
    b  = t1; \
    PetscCall(PetscMemcpy(t2, c, siz)); \
    PetscCall(PetscMemcpy(c, d, siz)); \
    PetscCall(PetscMemcpy(d, t2, siz)); \
  } while (0)

/*
   Partition X[lo,hi] into two parts: X[lo,l) <= pivot; X[r,hi] > pivot

   Input Parameters:
    + X         - array to partition
    . pivot     - a pivot of X[]
    . t1        - temp variable for X
    - lo,hi     - lower and upper bound of the array

   Output Parameters:
    + l,r       - of type PetscInt

   Note:
    The TwoWayPartition2/3 variants also partition other arrays along with X.
    These arrays can have different types, so they provide their own temp t2,t3
 */
#define TwoWayPartition1(X, pivot, t1, lo, hi, l, r) \
  do { \
    l = lo; \
    r = hi; \
    while (1) { \
      while (X[l] < pivot) l++; \
      while (X[r] > pivot) r--; \
      if (l >= r) { \
        r++; \
        break; \
      } \
      SWAP1(X[l], X[r], t1); \
      l++; \
      r--; \
    } \
  } while (0)

/*
   Partition X[lo,hi] into two parts: X[lo,l) >= pivot; X[r,hi] < pivot

   Input Parameters:
    + X         - array to partition
    . pivot     - a pivot of X[]
    . t1        - temp variable for X
    - lo,hi     - lower and upper bound of the array

   Output Parameters:
    + l,r       - of type PetscInt

   Note:
    The TwoWayPartition2/3 variants also partition other arrays along with X.
    These arrays can have different types, so they provide their own temp t2,t3
 */
#define TwoWayPartitionReverse1(X, pivot, t1, lo, hi, l, r) \
  do { \
    l = lo; \
    r = hi; \
    while (1) { \
      while (X[l] > pivot) l++; \
      while (X[r] < pivot) r--; \
      if (l >= r) { \
        r++; \
        break; \
      } \
      SWAP1(X[l], X[r], t1); \
      l++; \
      r--; \
    } \
  } while (0)

#define TwoWayPartition2(X, Y, pivot, t1, t2, lo, hi, l, r) \
  do { \
    l = lo; \
    r = hi; \
    while (1) { \
      while (X[l] < pivot) l++; \
      while (X[r] > pivot) r--; \
      if (l >= r) { \
        r++; \
        break; \
      } \
      SWAP2(X[l], X[r], Y[l], Y[r], t1, t2); \
      l++; \
      r--; \
    } \
  } while (0)

#define TwoWayPartition3(X, Y, Z, pivot, t1, t2, t3, lo, hi, l, r) \
  do { \
    l = lo; \
    r = hi; \
    while (1) { \
      while (X[l] < pivot) l++; \
      while (X[r] > pivot) r--; \
      if (l >= r) { \
        r++; \
        break; \
      } \
      SWAP3(X[l], X[r], Y[l], Y[r], Z[l], Z[r], t1, t2, t3); \
      l++; \
      r--; \
    } \
  } while (0)

/* Templates for similar functions used below */
#define QuickSort1(FuncName, X, n, pivot, t1) \
  do { \
    PetscCount i, j, p, l, r, hi = n - 1; \
    if (n < 8) { \
      for (i = 0; i < n; i++) { \
        pivot = X[i]; \
        for (j = i + 1; j < n; j++) { \
          if (pivot > X[j]) { \
            SWAP1(X[i], X[j], t1); \
            pivot = X[i]; \
          } \
        } \
      } \
    } else { \
      p     = MEDIAN(X, hi); \
      pivot = X[p]; \
      TwoWayPartition1(X, pivot, t1, 0, hi, l, r); \
      PetscCall(FuncName(l, X)); \
      PetscCall(FuncName(hi - r + 1, X + r)); \
    } \
  } while (0)

/* Templates for similar functions used below */
#define QuickSortReverse1(FuncName, X, n, pivot, t1) \
  do { \
    PetscCount i, j, p, l, r, hi = n - 1; \
    if (n < 8) { \
      for (i = 0; i < n; i++) { \
        pivot = X[i]; \
        for (j = i + 1; j < n; j++) { \
          if (pivot < X[j]) { \
            SWAP1(X[i], X[j], t1); \
            pivot = X[i]; \
          } \
        } \
      } \
    } else { \
      p     = MEDIAN(X, hi); \
      pivot = X[p]; \
      TwoWayPartitionReverse1(X, pivot, t1, 0, hi, l, r); \
      PetscCall(FuncName(l, X)); \
      PetscCall(FuncName(hi - r + 1, X + r)); \
    } \
  } while (0)

#define QuickSort2(FuncName, X, Y, n, pivot, t1, t2) \
  do { \
    PetscCount i, j, p, l, r, hi = n - 1; \
    if (n < 8) { \
      for (i = 0; i < n; i++) { \
        pivot = X[i]; \
        for (j = i + 1; j < n; j++) { \
          if (pivot > X[j]) { \
            SWAP2(X[i], X[j], Y[i], Y[j], t1, t2); \
            pivot = X[i]; \
          } \
        } \
      } \
    } else { \
      p     = MEDIAN(X, hi); \
      pivot = X[p]; \
      TwoWayPartition2(X, Y, pivot, t1, t2, 0, hi, l, r); \
      PetscCall(FuncName(l, X, Y)); \
      PetscCall(FuncName(hi - r + 1, X + r, Y + r)); \
    } \
  } while (0)

#define QuickSort3(FuncName, X, Y, Z, n, pivot, t1, t2, t3) \
  do { \
    PetscCount i, j, p, l, r, hi = n - 1; \
    if (n < 8) { \
      for (i = 0; i < n; i++) { \
        pivot = X[i]; \
        for (j = i + 1; j < n; j++) { \
          if (pivot > X[j]) { \
            SWAP3(X[i], X[j], Y[i], Y[j], Z[i], Z[j], t1, t2, t3); \
            pivot = X[i]; \
          } \
        } \
      } \
    } else { \
      p     = MEDIAN(X, hi); \
      pivot = X[p]; \
      TwoWayPartition3(X, Y, Z, pivot, t1, t2, t3, 0, hi, l, r); \
      PetscCall(FuncName(l, X, Y, Z)); \
      PetscCall(FuncName(hi - r + 1, X + r, Y + r, Z + r)); \
    } \
  } while (0)

/*@
   PetscSortedInt - Determines whether the `PetscInt` array is sorted.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Output Parameter:
.  sorted - flag whether the array is sorted

   Level: intermediate

.seealso: `PetscSortInt()`, `PetscSortedMPIInt()`, `PetscSortedReal()`
@*/
PetscErrorCode PetscSortedInt(PetscInt n, const PetscInt X[], PetscBool *sorted)
{
  PetscFunctionBegin;
  if (n) PetscValidIntPointer(X, 2);
  PetscValidBoolPointer(sorted, 3);
  PetscSorted(n, X, *sorted);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortedInt64 - Determines whether the `PetscInt64` array is sorted.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Output Parameter:
.  sorted - flag whether the array is sorted

   Level: intermediate

.seealso: `PetscSortInt64()`, `PetscSortInt()`, `PetscSortedMPIInt()`, `PetscSortedReal()`
@*/
PetscErrorCode PetscSortedInt64(PetscInt n, const PetscInt64 X[], PetscBool *sorted)
{
  PetscFunctionBegin;
  if (n) PetscValidInt64Pointer(X, 2);
  PetscValidBoolPointer(sorted, 3);
  PetscSorted(n, X, *sorted);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortInt - Sorts an array of `PetscInt` in place in increasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Note:
   This function serves as an alternative to `PetscIntSortSemiOrdered()`, and may perform faster especially if the array
   is completely random. There are exceptions to this and so it is __highly__ recommended that the user benchmark their
   code to see which routine is fastest.

   Level: intermediate

.seealso: `PetscIntSortSemiOrdered()`, `PetscSortReal()`, `PetscSortIntWithPermutation()`
@*/
PetscErrorCode PetscSortInt(PetscInt n, PetscInt X[])
{
  PetscInt pivot, t1;

  PetscFunctionBegin;
  if (n) PetscValidIntPointer(X, 2);
  QuickSort1(PetscSortInt, X, n, pivot, t1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortInt64 - Sorts an array of `PetscInt64` in place in increasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Notes:
   This function sorts `PetscCount`s assumed to be in completely random order

   Level: intermediate

.seealso: `PetscSortInt()`
@*/
PetscErrorCode PetscSortInt64(PetscInt n, PetscInt64 X[])
{
  PetscCount pivot, t1;

  PetscFunctionBegin;
  if (n) PetscValidInt64Pointer(X, 2);
  QuickSort1(PetscSortInt64, X, n, pivot, t1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortCount - Sorts an array of integers in place in increasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Notes:
   This function sorts `PetscCount`s assumed to be in completely random order

   Level: intermediate

.seealso: `PetscSortInt()`
@*/
PetscErrorCode PetscSortCount(PetscInt n, PetscCount X[])
{
  PetscCount pivot, t1;

  PetscFunctionBegin;
  if (n) PetscValidCountPointer(X, 2);
  QuickSort1(PetscSortCount, X, n, pivot, t1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortReverseInt - Sorts an array of integers in place in decreasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Level: intermediate

.seealso: `PetscIntSortSemiOrdered()`, `PetscSortInt()`, `PetscSortIntWithPermutation()`
@*/
PetscErrorCode PetscSortReverseInt(PetscInt n, PetscInt X[])
{
  PetscInt pivot, t1;

  PetscFunctionBegin;
  if (n) PetscValidIntPointer(X, 2);
  QuickSortReverse1(PetscSortReverseInt, X, n, pivot, t1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortedRemoveDupsInt - Removes all duplicate entries of a sorted `PetscInt` array

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - sorted array of integers

   Output Parameter:
.  n - number of non-redundant values

   Level: intermediate

.seealso: `PetscSortInt()`
@*/
PetscErrorCode PetscSortedRemoveDupsInt(PetscInt *n, PetscInt X[])
{
  PetscInt i, s = 0, N = *n, b = 0;

  PetscFunctionBegin;
  PetscValidIntPointer(n, 1);
  PetscCheckSorted(*n, X);
  for (i = 0; i < N - 1; i++) {
    if (X[b + s + 1] != X[b]) {
      X[b + 1] = X[b + s + 1];
      b++;
    } else s++;
  }
  *n = N - s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortedCheckDupsInt - Checks if a sorted `PetscInt` array has duplicates

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - sorted array of integers

   Output Parameter:
.  dups - True if the array has dups, otherwise false

   Level: intermediate

.seealso: `PetscSortInt()`, `PetscCheckDupsInt()`, `PetscSortRemoveDupsInt()`, `PetscSortedRemoveDupsInt()`
@*/
PetscErrorCode PetscSortedCheckDupsInt(PetscInt n, const PetscInt X[], PetscBool *flg)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscCheckSorted(n, X);
  *flg = PETSC_FALSE;
  for (i = 0; i < n - 1; i++) {
    if (X[i + 1] == X[i]) {
      *flg = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortRemoveDupsInt - Sorts an array of `PetscInt` in place in increasing order removes all duplicate entries

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Output Parameter:
.  n - number of non-redundant values

   Level: intermediate

.seealso: `PetscIntSortSemiOrdered()`, `PetscSortReal()`, `PetscSortIntWithPermutation()`, `PetscSortInt()`, `PetscSortedRemoveDupsInt()`
@*/
PetscErrorCode PetscSortRemoveDupsInt(PetscInt *n, PetscInt X[])
{
  PetscFunctionBegin;
  PetscValidIntPointer(n, 1);
  PetscCall(PetscSortInt(*n, X));
  PetscCall(PetscSortedRemoveDupsInt(n, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
 PetscFindInt - Finds `PetscInt` in a sorted array of `PetscInt`

   Not Collective

   Input Parameters:
+  key - the integer to locate
.  n   - number of values in the array
-  X  - array of integers

   Output Parameter:
.  loc - the location if found, otherwise -(slot+1) where slot is the place the value would go

   Level: intermediate

.seealso: `PetscIntSortSemiOrdered()`, `PetscSortInt()`, `PetscSortIntWithArray()`, `PetscSortRemoveDupsInt()`
@*/
PetscErrorCode PetscFindInt(PetscInt key, PetscInt n, const PetscInt X[], PetscInt *loc)
{
  PetscInt lo = 0, hi = n;

  PetscFunctionBegin;
  PetscValidIntPointer(loc, 4);
  if (!n) {
    *loc = -1;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscValidIntPointer(X, 3);
  PetscCheckSorted(n, X);
  while (hi - lo > 1) {
    PetscInt mid = lo + (hi - lo) / 2;
    if (key < X[mid]) hi = mid;
    else lo = mid;
  }
  *loc = key == X[lo] ? lo : -(lo + (key > X[lo]) + 1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscCheckDupsInt - Checks if an `PetscInt` array has duplicates

   Not Collective

   Input Parameters:
+  n  - number of values in the array
-  X  - array of integers

   Output Parameter:
.  dups - True if the array has dups, otherwise false

   Level: intermediate

.seealso: `PetscSortRemoveDupsInt()`, `PetscSortedCheckDupsInt()`
@*/
PetscErrorCode PetscCheckDupsInt(PetscInt n, const PetscInt X[], PetscBool *dups)
{
  PetscInt   i;
  PetscHSetI ht;
  PetscBool  missing;

  PetscFunctionBegin;
  if (n) PetscValidIntPointer(X, 2);
  PetscValidBoolPointer(dups, 3);
  *dups = PETSC_FALSE;
  if (n > 1) {
    PetscCall(PetscHSetICreate(&ht));
    PetscCall(PetscHSetIResize(ht, n));
    for (i = 0; i < n; i++) {
      PetscCall(PetscHSetIQueryAdd(ht, X[i], &missing));
      if (!missing) {
        *dups = PETSC_TRUE;
        break;
      }
    }
    PetscCall(PetscHSetIDestroy(&ht));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFindMPIInt - Finds `PetscMPIInt` in a sorted array of `PetscMPIInt`

   Not Collective

   Input Parameters:
+  key - the integer to locate
.  n   - number of values in the array
-  X   - array of integers

   Output Parameter:
.  loc - the location if found, otherwise -(slot+1) where slot is the place the value would go

   Level: intermediate

.seealso: `PetscMPIIntSortSemiOrdered()`, `PetscSortInt()`, `PetscSortIntWithArray()`, `PetscSortRemoveDupsInt()`
@*/
PetscErrorCode PetscFindMPIInt(PetscMPIInt key, PetscInt n, const PetscMPIInt X[], PetscInt *loc)
{
  PetscInt lo = 0, hi = n;

  PetscFunctionBegin;
  PetscValidIntPointer(loc, 4);
  if (!n) {
    *loc = -1;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscValidIntPointer(X, 3);
  PetscCheckSorted(n, X);
  while (hi - lo > 1) {
    PetscInt mid = lo + (hi - lo) / 2;
    if (key < X[mid]) hi = mid;
    else lo = mid;
  }
  *loc = key == X[lo] ? lo : -(lo + (key > X[lo]) + 1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortIntWithArray - Sorts an array of `PetscInt` in place in increasing order;
       changes a second array of `PetscInt` to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of integers
-  Y  - second array of integers

   Level: intermediate

.seealso: `PetscIntSortSemiOrderedWithArray()`, `PetscSortReal()`, `PetscSortIntWithPermutation()`, `PetscSortInt()`, `PetscSortIntWithCountArray()`
@*/
PetscErrorCode PetscSortIntWithArray(PetscInt n, PetscInt X[], PetscInt Y[])
{
  PetscInt pivot, t1, t2;

  PetscFunctionBegin;
  QuickSort2(PetscSortIntWithArray, X, Y, n, pivot, t1, t2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortIntWithArrayPair - Sorts an array of `PetscInt` in place in increasing order;
       changes a pair of `PetscInt` arrays to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of integers
.  Y  - second array of integers (first array of the pair)
-  Z  - third array of integers  (second array of the pair)

   Level: intermediate

.seealso: `PetscSortReal()`, `PetscSortIntWithPermutation()`, `PetscSortIntWithArray()`, `PetscIntSortSemiOrdered()`, `PetscSortIntWithIntCountArrayPair()`
@*/
PetscErrorCode PetscSortIntWithArrayPair(PetscInt n, PetscInt X[], PetscInt Y[], PetscInt Z[])
{
  PetscInt pivot, t1, t2, t3;

  PetscFunctionBegin;
  QuickSort3(PetscSortIntWithArrayPair, X, Y, Z, n, pivot, t1, t2, t3);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortIntWithCountArray - Sorts an array of `PetscInt` in place in increasing order;
       changes a second array of `PetscCount` to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of integers
-  Y  - second array of PetscCounts (signed integers)

   Level: intermediate

.seealso: `PetscIntSortSemiOrderedWithArray()`, `PetscSortReal()`, `PetscSortIntPermutation()`, `PetscSortInt()`, `PetscSortIntWithArray()`
@*/
PetscErrorCode PetscSortIntWithCountArray(PetscCount n, PetscInt X[], PetscCount Y[])
{
  PetscInt   pivot, t1;
  PetscCount t2;

  PetscFunctionBegin;
  QuickSort2(PetscSortIntWithCountArray, X, Y, n, pivot, t1, t2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortIntWithIntCountArrayPair - Sorts an array of `PetscInt` in place in increasing order;
       changes a `PetscInt`  array and a `PetscCount` array to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of integers
.  Y  - second array of integers (first array of the pair)
-  Z  - third array of PetscCounts  (second array of the pair)

   Level: intermediate

   Note:
    Usually X, Y are matrix row/column indices, and Z is a permutation array and therefore Z's type is PetscCount to allow 2B+ nonzeros even with 32-bit PetscInt.

.seealso: `PetscSortReal()`, `PetscSortIntPermutation()`, `PetscSortIntWithArray()`, `PetscIntSortSemiOrdered()`, `PetscSortIntWithArrayPair()`
@*/
PetscErrorCode PetscSortIntWithIntCountArrayPair(PetscCount n, PetscInt X[], PetscInt Y[], PetscCount Z[])
{
  PetscInt   pivot, t1, t2; /* pivot is take from X[], so its type is still PetscInt */
  PetscCount t3;            /* temp for Z[] */

  PetscFunctionBegin;
  QuickSort3(PetscSortIntWithIntCountArrayPair, X, Y, Z, n, pivot, t1, t2, t3);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSortedMPIInt - Determines whether the `PetscMPIInt` array is sorted.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Output Parameter:
.  sorted - flag whether the array is sorted

   Level: intermediate

.seealso: `PetscMPIIntSortSemiOrdered()`, `PetscSortMPIInt()`, `PetscSortedInt()`, `PetscSortedReal()`
@*/
PetscErrorCode PetscSortedMPIInt(PetscInt n, const PetscMPIInt X[], PetscBool *sorted)
{
  PetscFunctionBegin;
  PetscSorted(n, X, *sorted);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortMPIInt - Sorts an array of `PetscMPIInt` in place in increasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Level: intermediate

   Note:
   This function serves as an alternative to PetscMPIIntSortSemiOrdered(), and may perform faster especially if the array
   is completely random. There are exceptions to this and so it is __highly__ recommended that the user benchmark their
   code to see which routine is fastest.

.seealso: `PetscMPIIntSortSemiOrdered()`, `PetscSortReal()`, `PetscSortIntWithPermutation()`
@*/
PetscErrorCode PetscSortMPIInt(PetscInt n, PetscMPIInt X[])
{
  PetscMPIInt pivot, t1;

  PetscFunctionBegin;
  QuickSort1(PetscSortMPIInt, X, n, pivot, t1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortRemoveDupsMPIInt - Sorts an array of `PetscMPIInt` in place in increasing order removes all duplicate entries

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Output Parameter:
.  n - number of non-redundant values

   Level: intermediate

.seealso: `PetscSortReal()`, `PetscSortIntWithPermutation()`, `PetscSortInt()`
@*/
PetscErrorCode PetscSortRemoveDupsMPIInt(PetscInt *n, PetscMPIInt X[])
{
  PetscInt s = 0, N = *n, b = 0;

  PetscFunctionBegin;
  PetscCall(PetscSortMPIInt(N, X));
  for (PetscInt i = 0; i < N - 1; i++) {
    if (X[b + s + 1] != X[b]) {
      X[b + 1] = X[b + s + 1];
      b++;
    } else s++;
  }
  *n = N - s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortMPIIntWithArray - Sorts an array of `PetscMPIInt` in place in increasing order;
       changes a second `PetscMPIInt` array to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of integers
-  Y  - second array of integers

   Level: intermediate

.seealso: `PetscMPIIntSortSemiOrderedWithArray()`, `PetscSortReal()`, `PetscSortIntWithPermutation()`, `PetscSortInt()`
@*/
PetscErrorCode PetscSortMPIIntWithArray(PetscMPIInt n, PetscMPIInt X[], PetscMPIInt Y[])
{
  PetscMPIInt pivot, t1, t2;

  PetscFunctionBegin;
  QuickSort2(PetscSortMPIIntWithArray, X, Y, n, pivot, t1, t2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortMPIIntWithIntArray - Sorts an array of `PetscMPIInt` in place in increasing order;
       changes a second array of `PetscInt` to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of MPI integers
-  Y  - second array of Petsc integers

   Level: intermediate

   Note:
   This routine is useful when one needs to sort MPI ranks with other integer arrays.

.seealso: `PetscSortMPIIntWithArray()`, `PetscIntSortSemiOrderedWithArray()`, `PetscTimSortWithArray()`
@*/
PetscErrorCode PetscSortMPIIntWithIntArray(PetscMPIInt n, PetscMPIInt X[], PetscInt Y[])
{
  PetscMPIInt pivot, t1;
  PetscInt    t2;

  PetscFunctionBegin;
  QuickSort2(PetscSortMPIIntWithIntArray, X, Y, n, pivot, t1, t2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscSortIntWithScalarArray - Sorts an array of `PetscInt` in place in increasing order;
       changes a second `PetscScalar` array to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of integers
-  Y  - second array of scalars

   Level: intermediate

.seealso: `PetscTimSortWithArray()`, `PetscSortReal()`, `PetscSortIntWithPermutation()`, `PetscSortInt()`, `PetscSortIntWithArray()`
@*/
PetscErrorCode PetscSortIntWithScalarArray(PetscInt n, PetscInt X[], PetscScalar Y[])
{
  PetscInt    pivot, t1;
  PetscScalar t2;

  PetscFunctionBegin;
  QuickSort2(PetscSortIntWithScalarArray, X, Y, n, pivot, t1, t2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscSortIntWithDataArray - Sorts an array of `PetscInt` in place in increasing order;
       changes a second array to match the sorted first INTEGER array.  Unlike other sort routines, the user must
       provide workspace (the size of an element in the data array) to use when sorting.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of integers
.  Y  - second array of data
.  size - sizeof elements in the data array in bytes
-  t2   - workspace of "size" bytes used when sorting

   Level: intermediate

.seealso: `PetscTimSortWithArray()`, `PetscSortReal()`, `PetscSortIntWithPermutation()`, `PetscSortInt()`, `PetscSortIntWithArray()`
@*/
PetscErrorCode PetscSortIntWithDataArray(PetscInt n, PetscInt X[], void *Y, size_t size, void *t2)
{
  char    *YY = (char *)Y;
  PetscInt t1, pivot, hi = n - 1;

  PetscFunctionBegin;
  if (n < 8) {
    for (PetscInt i = 0; i < n; i++) {
      pivot = X[i];
      for (PetscInt j = i + 1; j < n; j++) {
        if (pivot > X[j]) {
          SWAP2Data(X[i], X[j], YY + size * i, YY + size * j, t1, t2, size);
          pivot = X[i];
        }
      }
    }
  } else {
    /* Two way partition */
    PetscInt l = 0, r = hi;

    pivot = X[MEDIAN(X, hi)];
    while (1) {
      while (X[l] < pivot) l++;
      while (X[r] > pivot) r--;
      if (l >= r) {
        r++;
        break;
      }
      SWAP2Data(X[l], X[r], YY + size * l, YY + size * r, t1, t2, size);
      l++;
      r--;
    }
    PetscCall(PetscSortIntWithDataArray(l, X, Y, size, t2));
    PetscCall(PetscSortIntWithDataArray(hi - r + 1, X + r, YY + size * r, size, t2));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscMergeIntArray -     Merges two SORTED `PetscInt` arrays, removes duplicate elements.

   Not Collective

   Input Parameters:
+  an  - number of values in the first array
.  aI  - first sorted array of integers
.  bn  - number of values in the second array
-  bI  - second array of integers

   Output Parameters:
+  n   - number of values in the merged array
-  L   - merged sorted array, this is allocated if an array is not provided

   Level: intermediate

.seealso: `PetscSortReal()`, `PetscSortIntWithPermutation()`, `PetscSortInt()`, `PetscSortIntWithArray()`
@*/
PetscErrorCode PetscMergeIntArray(PetscInt an, const PetscInt aI[], PetscInt bn, const PetscInt bI[], PetscInt *n, PetscInt **L)
{
  PetscInt *L_ = *L, ak, bk, k;

  PetscFunctionBegin;
  if (!L_) {
    PetscCall(PetscMalloc1(an + bn, L));
    L_ = *L;
  }
  k = ak = bk = 0;
  while (ak < an && bk < bn) {
    if (aI[ak] == bI[bk]) {
      L_[k] = aI[ak];
      ++ak;
      ++bk;
      ++k;
    } else if (aI[ak] < bI[bk]) {
      L_[k] = aI[ak];
      ++ak;
      ++k;
    } else {
      L_[k] = bI[bk];
      ++bk;
      ++k;
    }
  }
  if (ak < an) {
    PetscCall(PetscArraycpy(L_ + k, aI + ak, an - ak));
    k += (an - ak);
  }
  if (bk < bn) {
    PetscCall(PetscArraycpy(L_ + k, bI + bk, bn - bk));
    k += (bn - bk);
  }
  *n = k;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscMergeIntArrayPair -     Merges two SORTED `PetscInt` arrays that share NO common values along with an additional array of `PetscInt`.
                                The additional arrays are the same length as sorted arrays and are merged
                                in the order determined by the merging of the sorted pair.

   Not Collective

   Input Parameters:
+  an  - number of values in the first array
.  aI  - first sorted array of integers
.  aJ  - first additional array of integers
.  bn  - number of values in the second array
.  bI  - second array of integers
-  bJ  - second additional of integers

   Output Parameters:
+  n   - number of values in the merged array (== an + bn)
.  L   - merged sorted array
-  J   - merged additional array

   Note:
    if L or J point to non-null arrays then this routine will assume they are of the appropriate size and use them, otherwise this routine will allocate space for them

   Level: intermediate

.seealso: `PetscIntSortSemiOrdered()`, `PetscSortReal()`, `PetscSortIntWithPermutation()`, `PetscSortInt()`, `PetscSortIntWithArray()`
@*/
PetscErrorCode PetscMergeIntArrayPair(PetscInt an, const PetscInt aI[], const PetscInt aJ[], PetscInt bn, const PetscInt bI[], const PetscInt bJ[], PetscInt *n, PetscInt **L, PetscInt **J)
{
  PetscInt n_, *L_, *J_, ak, bk, k;

  PetscFunctionBegin;
  PetscValidPointer(L, 8);
  PetscValidPointer(J, 9);
  n_ = an + bn;
  *n = n_;
  if (!*L) PetscCall(PetscMalloc1(n_, L));
  L_ = *L;
  if (!*J) PetscCall(PetscMalloc1(n_, J));
  J_ = *J;
  k = ak = bk = 0;
  while (ak < an && bk < bn) {
    if (aI[ak] <= bI[bk]) {
      L_[k] = aI[ak];
      J_[k] = aJ[ak];
      ++ak;
      ++k;
    } else {
      L_[k] = bI[bk];
      J_[k] = bJ[bk];
      ++bk;
      ++k;
    }
  }
  if (ak < an) {
    PetscCall(PetscArraycpy(L_ + k, aI + ak, an - ak));
    PetscCall(PetscArraycpy(J_ + k, aJ + ak, an - ak));
    k += (an - ak);
  }
  if (bk < bn) {
    PetscCall(PetscArraycpy(L_ + k, bI + bk, bn - bk));
    PetscCall(PetscArraycpy(J_ + k, bJ + bk, bn - bk));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscMergeMPIIntArray -     Merges two SORTED `PetscMPIInt` arrays.

   Not Collective

   Input Parameters:
+  an  - number of values in the first array
.  aI  - first sorted array of integers
.  bn  - number of values in the second array
-  bI  - second array of integers

   Output Parameters:
+  n   - number of values in the merged array (<= an + bn)
-  L   - merged sorted array, allocated if address of NULL pointer is passed

   Level: intermediate

.seealso: `PetscIntSortSemiOrdered()`, `PetscSortReal()`, `PetscSortIntWithPermutation()`, `PetscSortInt()`, `PetscSortIntWithArray()`
@*/
PetscErrorCode PetscMergeMPIIntArray(PetscInt an, const PetscMPIInt aI[], PetscInt bn, const PetscMPIInt bI[], PetscInt *n, PetscMPIInt **L)
{
  PetscInt ai, bi, k;

  PetscFunctionBegin;
  if (!*L) PetscCall(PetscMalloc1((an + bn), L));
  for (ai = 0, bi = 0, k = 0; ai < an || bi < bn;) {
    PetscInt t = -1;
    for (; ai < an && (!bn || aI[ai] <= bI[bi]); ai++) (*L)[k++] = t = aI[ai];
    for (; bi < bn && bI[bi] == t; bi++)
      ;
    for (; bi < bn && (!an || bI[bi] <= aI[ai]); bi++) (*L)[k++] = t = bI[bi];
    for (; ai < an && aI[ai] == t; ai++)
      ;
  }
  *n = k;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscProcessTree - Prepares tree data to be displayed graphically

   Not Collective

   Input Parameters:
+  n  - number of values
.  mask - indicates those entries in the tree, location 0 is always masked
-  parentid - indicates the parent of each entry

   Output Parameters:
+  Nlevels - the number of levels
.  Level - for each node tells its level
.  Levelcnts - the number of nodes on each level
.  Idbylevel - a list of ids on each of the levels, first level followed by second etc
-  Column - for each id tells its column index

   Level: developer

   Note:
    This code is not currently used

.seealso: `PetscSortReal()`, `PetscSortIntWithPermutation()`
@*/
PetscErrorCode PetscProcessTree(PetscInt n, const PetscBool mask[], const PetscInt parentid[], PetscInt *Nlevels, PetscInt **Level, PetscInt **Levelcnt, PetscInt **Idbylevel, PetscInt **Column)
{
  PetscInt  i, j, cnt, nmask = 0, nlevels = 0, *level, *levelcnt, levelmax = 0, *workid, *workparentid, tcnt = 0, *idbylevel, *column;
  PetscBool done = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCheck(mask[0], PETSC_COMM_SELF, PETSC_ERR_SUP, "Mask of 0th location must be set");
  for (i = 0; i < n; i++) {
    if (mask[i]) continue;
    PetscCheck(parentid[i] != i, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Node labeled as own parent");
    PetscCheck(!parentid[i] || !mask[parentid[i]], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Parent is masked");
  }

  for (i = 0; i < n; i++) {
    if (!mask[i]) nmask++;
  }

  /* determine the level in the tree of each node */
  PetscCall(PetscCalloc1(n, &level));

  level[0] = 1;
  while (!done) {
    done = PETSC_TRUE;
    for (i = 0; i < n; i++) {
      if (mask[i]) continue;
      if (!level[i] && level[parentid[i]]) level[i] = level[parentid[i]] + 1;
      else if (!level[i]) done = PETSC_FALSE;
    }
  }
  for (i = 0; i < n; i++) {
    level[i]--;
    nlevels = PetscMax(nlevels, level[i]);
  }

  /* count the number of nodes on each level and its max */
  PetscCall(PetscCalloc1(nlevels, &levelcnt));
  for (i = 0; i < n; i++) {
    if (mask[i]) continue;
    levelcnt[level[i] - 1]++;
  }
  for (i = 0; i < nlevels; i++) levelmax = PetscMax(levelmax, levelcnt[i]);

  /* for each level sort the ids by the parent id */
  PetscCall(PetscMalloc2(levelmax, &workid, levelmax, &workparentid));
  PetscCall(PetscMalloc1(nmask, &idbylevel));
  for (j = 1; j <= nlevels; j++) {
    cnt = 0;
    for (i = 0; i < n; i++) {
      if (mask[i]) continue;
      if (level[i] != j) continue;
      workid[cnt]         = i;
      workparentid[cnt++] = parentid[i];
    }
    /*  PetscIntView(cnt,workparentid,0);
    PetscIntView(cnt,workid,0);
    PetscCall(PetscSortIntWithArray(cnt,workparentid,workid));
    PetscIntView(cnt,workparentid,0);
    PetscIntView(cnt,workid,0);*/
    PetscCall(PetscArraycpy(idbylevel + tcnt, workid, cnt));
    tcnt += cnt;
  }
  PetscCheck(tcnt == nmask, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent count of unmasked nodes");
  PetscCall(PetscFree2(workid, workparentid));

  /* for each node list its column */
  PetscCall(PetscMalloc1(n, &column));
  cnt = 0;
  for (j = 0; j < nlevels; j++) {
    for (i = 0; i < levelcnt[j]; i++) column[idbylevel[cnt++]] = i;
  }

  *Nlevels   = nlevels;
  *Level     = level;
  *Levelcnt  = levelcnt;
  *Idbylevel = idbylevel;
  *Column    = column;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscParallelSortedInt - Check whether a `PetscInt` array, distributed over a communicator, is globally sorted.

  Collective

  Input Parameters:
+ comm - the MPI communicator
. n - the local number of integers
- keys - the local array of integers

  Output Parameters:
. is_sorted - whether the array is globally sorted

  Level: developer

.seealso: `PetscParallelSortInt()`
@*/
PetscErrorCode PetscParallelSortedInt(MPI_Comm comm, PetscInt n, const PetscInt keys[], PetscBool *is_sorted)
{
  PetscBool   sorted;
  PetscInt    i, min, max, prevmax;
  PetscMPIInt rank;

  PetscFunctionBegin;
  sorted = PETSC_TRUE;
  min    = PETSC_MAX_INT;
  max    = PETSC_MIN_INT;
  if (n) {
    min = keys[0];
    max = keys[0];
  }
  for (i = 1; i < n; i++) {
    if (keys[i] < keys[i - 1]) break;
    min = PetscMin(min, keys[i]);
    max = PetscMax(max, keys[i]);
  }
  if (i < n) sorted = PETSC_FALSE;
  prevmax = PETSC_MIN_INT;
  PetscCallMPI(MPI_Exscan(&max, &prevmax, 1, MPIU_INT, MPI_MAX, comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) prevmax = PETSC_MIN_INT;
  if (prevmax > min) sorted = PETSC_FALSE;
  PetscCall(MPIU_Allreduce(&sorted, is_sorted, 1, MPIU_BOOL, MPI_LAND, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
