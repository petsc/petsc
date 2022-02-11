
/*
   This file contains routines for sorting integers. Values are sorted in place.
   One can use src/sys/tests/ex52.c for benchmarking.
 */
#include <petsc/private/petscimpl.h>                /*I  "petscsys.h"  I*/
#include <petsc/private/hashseti.h>

#define MEDIAN3(v,a,b,c)                                                        \
  (v[a]<v[b]                                                                    \
   ? (v[b]<v[c]                                                                 \
      ? (b)                                                                     \
      : (v[a]<v[c] ? (c) : (a)))                                                \
   : (v[c]<v[b]                                                                 \
      ? (b)                                                                     \
      : (v[a]<v[c] ? (a) : (c))))

#define MEDIAN(v,right) MEDIAN3(v,right/4,right/2,right/4*3)

/* Swap one, two or three pairs. Each pair can have its own type */
#define SWAP1(a,b,t1)               do {t1=a;a=b;b=t1;} while (0)
#define SWAP2(a,b,c,d,t1,t2)        do {t1=a;a=b;b=t1; t2=c;c=d;d=t2;} while (0)
#define SWAP3(a,b,c,d,e,f,t1,t2,t3) do {t1=a;a=b;b=t1; t2=c;c=d;d=t2; t3=e;e=f;f=t3;} while (0)

/* Swap a & b, *c & *d. c, d, t2 are pointers to a type of size <siz> */
#define SWAP2Data(a,b,c,d,t1,t2,siz)                                             \
  do {                                                                           \
    PetscErrorCode ierr;                                                         \
    t1=a; a=b; b=t1;                                                             \
    ierr = PetscMemcpy(t2,c,siz);CHKERRQ(ierr);                                  \
    ierr = PetscMemcpy(c,d,siz);CHKERRQ(ierr);                                   \
    ierr = PetscMemcpy(d,t2,siz);CHKERRQ(ierr);                                  \
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

   Notes:
    The TwoWayPartition2/3 variants also partition other arrays along with X.
    These arrays can have different types, so they provide their own temp t2,t3
 */
#define TwoWayPartition1(X,pivot,t1,lo,hi,l,r)                                   \
  do {                                                                           \
    l = lo;                                                                      \
    r = hi;                                                                      \
    while (1) {                                                                   \
      while (X[l] < pivot) l++;                                                  \
      while (X[r] > pivot) r--;                                                  \
      if (l >= r) {r++; break;}                                                  \
      SWAP1(X[l],X[r],t1);                                                       \
      l++;                                                                       \
      r--;                                                                       \
    }                                                                            \
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

   Notes:
    The TwoWayPartition2/3 variants also partition other arrays along with X.
    These arrays can have different types, so they provide their own temp t2,t3
 */
#define TwoWayPartitionReverse1(X,pivot,t1,lo,hi,l,r)                                   \
  do {                                                                           \
    l = lo;                                                                      \
    r = hi;                                                                      \
    while (1) {                                                                   \
      while (X[l] > pivot) l++;                                                  \
      while (X[r] < pivot) r--;                                                  \
      if (l >= r) {r++; break;}                                                  \
      SWAP1(X[l],X[r],t1);                                                       \
      l++;                                                                       \
      r--;                                                                       \
    }                                                                            \
  } while (0)

#define TwoWayPartition2(X,Y,pivot,t1,t2,lo,hi,l,r)                              \
  do {                                                                           \
    l = lo;                                                                      \
    r = hi;                                                                      \
    while (1) {                                                                   \
      while (X[l] < pivot) l++;                                                  \
      while (X[r] > pivot) r--;                                                  \
      if (l >= r) {r++; break;}                                                  \
      SWAP2(X[l],X[r],Y[l],Y[r],t1,t2);                                          \
      l++;                                                                       \
      r--;                                                                       \
    }                                                                            \
  } while (0)

#define TwoWayPartition3(X,Y,Z,pivot,t1,t2,t3,lo,hi,l,r)                         \
  do {                                                                           \
    l = lo;                                                                      \
    r = hi;                                                                      \
    while (1) {                                                                   \
      while (X[l] < pivot) l++;                                                  \
      while (X[r] > pivot) r--;                                                  \
      if (l >= r) {r++; break;}                                                  \
      SWAP3(X[l],X[r],Y[l],Y[r],Z[l],Z[r],t1,t2,t3);                             \
      l++;                                                                       \
      r--;                                                                       \
    }                                                                            \
  } while (0)

/* Templates for similar functions used below */
#define QuickSort1(FuncName,X,n,pivot,t1,ierr)                                   \
  do {                                                                           \
    PetscInt i,j,p,l,r,hi=n-1;                                                   \
    if (n < 8) {                                                                 \
      for (i=0; i<n; i++) {                                                      \
        pivot = X[i];                                                            \
        for (j=i+1; j<n; j++) {                                                  \
          if (pivot > X[j]) {                                                    \
            SWAP1(X[i],X[j],t1);                                                 \
            pivot = X[i];                                                        \
          }                                                                      \
        }                                                                        \
      }                                                                          \
    } else {                                                                     \
      p     = MEDIAN(X,hi);                                                      \
      pivot = X[p];                                                              \
      TwoWayPartition1(X,pivot,t1,0,hi,l,r);                                     \
      ierr  = FuncName(l,X);CHKERRQ(ierr);                                       \
      ierr  = FuncName(hi-r+1,X+r);CHKERRQ(ierr);                                \
    }                                                                            \
  } while (0)

/* Templates for similar functions used below */
#define QuickSortReverse1(FuncName,X,n,pivot,t1,ierr)                            \
  do {                                                                           \
    PetscInt i,j,p,l,r,hi=n-1;                                                   \
    if (n < 8) {                                                                 \
      for (i=0; i<n; i++) {                                                      \
        pivot = X[i];                                                            \
        for (j=i+1; j<n; j++) {                                                  \
          if (pivot < X[j]) {                                                    \
            SWAP1(X[i],X[j],t1);                                                 \
            pivot = X[i];                                                        \
          }                                                                      \
        }                                                                        \
      }                                                                          \
    } else {                                                                     \
      p     = MEDIAN(X,hi);                                                      \
      pivot = X[p];                                                              \
      TwoWayPartitionReverse1(X,pivot,t1,0,hi,l,r);                              \
      ierr  = FuncName(l,X);CHKERRQ(ierr);                                       \
      ierr  = FuncName(hi-r+1,X+r);CHKERRQ(ierr);                                \
    }                                                                            \
  } while (0)

#define QuickSort2(FuncName,X,Y,n,pivot,t1,t2,ierr)                              \
  do {                                                                           \
    PetscInt i,j,p,l,r,hi=n-1;                                                   \
    if (n < 8) {                                                                 \
      for (i=0; i<n; i++) {                                                      \
        pivot = X[i];                                                            \
        for (j=i+1; j<n; j++) {                                                  \
          if (pivot > X[j]) {                                                    \
            SWAP2(X[i],X[j],Y[i],Y[j],t1,t2);                                    \
            pivot = X[i];                                                        \
          }                                                                      \
        }                                                                        \
      }                                                                          \
    } else {                                                                     \
      p     = MEDIAN(X,hi);                                                      \
      pivot = X[p];                                                              \
      TwoWayPartition2(X,Y,pivot,t1,t2,0,hi,l,r);                                \
      ierr  = FuncName(l,X,Y);CHKERRQ(ierr);                                     \
      ierr  = FuncName(hi-r+1,X+r,Y+r);CHKERRQ(ierr);                            \
    }                                                                            \
  } while (0)

#define QuickSort3(FuncName,X,Y,Z,n,pivot,t1,t2,t3,ierr)                         \
  do {                                                                           \
    PetscInt i,j,p,l,r,hi=n-1;                                                   \
    if (n < 8) {                                                                 \
      for (i=0; i<n; i++) {                                                      \
        pivot = X[i];                                                            \
        for (j=i+1; j<n; j++) {                                                  \
          if (pivot > X[j]) {                                                    \
            SWAP3(X[i],X[j],Y[i],Y[j],Z[i],Z[j],t1,t2,t3);                       \
            pivot = X[i];                                                        \
          }                                                                      \
        }                                                                        \
      }                                                                          \
    } else {                                                                     \
      p     = MEDIAN(X,hi);                                                      \
      pivot = X[p];                                                              \
      TwoWayPartition3(X,Y,Z,pivot,t1,t2,t3,0,hi,l,r);                           \
      ierr  = FuncName(l,X,Y,Z);CHKERRQ(ierr);                                   \
      ierr  = FuncName(hi-r+1,X+r,Y+r,Z+r);CHKERRQ(ierr);                        \
    }                                                                            \
  } while (0)

/*@
   PetscSortedInt - Determines whether the array is sorted.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Output Parameters:
.  sorted - flag whether the array is sorted

   Level: intermediate

.seealso: PetscSortInt(), PetscSortedMPIInt(), PetscSortedReal()
@*/
PetscErrorCode  PetscSortedInt(PetscInt n,const PetscInt X[],PetscBool *sorted)
{
  PetscFunctionBegin;
  PetscSorted(n,X,*sorted);
  PetscFunctionReturn(0);
}

/*@
   PetscSortInt - Sorts an array of integers in place in increasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Notes:
   This function serves as an alternative to PetscIntSortSemiOrdered(), and may perform faster especially if the array
   is completely random. There are exceptions to this and so it is __highly__ recommended that the user benchmark their
   code to see which routine is fastest.

   Level: intermediate

.seealso: PetscIntSortSemiOrdered(), PetscSortReal(), PetscSortIntWithPermutation()
@*/
PetscErrorCode  PetscSortInt(PetscInt n,PetscInt X[])
{
  PetscErrorCode ierr;
  PetscInt       pivot,t1;

  PetscFunctionBegin;
  QuickSort1(PetscSortInt,X,n,pivot,t1,ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSortReverseInt - Sorts an array of integers in place in decreasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Level: intermediate

.seealso: PetscIntSortSemiOrdered(), PetscSortInt(), PetscSortIntWithPermutation()
@*/
PetscErrorCode  PetscSortReverseInt(PetscInt n,PetscInt X[])
{
  PetscErrorCode ierr;
  PetscInt       pivot,t1;

  PetscFunctionBegin;
  QuickSortReverse1(PetscSortReverseInt,X,n,pivot,t1,ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSortedRemoveDupsInt - Removes all duplicate entries of a sorted input array

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - sorted array of integers

   Output Parameter:
.  n - number of non-redundant values

   Level: intermediate

.seealso: PetscSortInt()
@*/
PetscErrorCode  PetscSortedRemoveDupsInt(PetscInt *n,PetscInt X[])
{
  PetscInt i,s = 0,N = *n, b = 0;

  PetscFunctionBegin;
  PetscCheckSorted(*n,X);
  for (i=0; i<N-1; i++) {
    if (X[b+s+1] != X[b]) {
      X[b+1] = X[b+s+1]; b++;
    } else s++;
  }
  *n = N - s;
  PetscFunctionReturn(0);
}

/*@
   PetscSortRemoveDupsInt - Sorts an array of integers in place in increasing order removes all duplicate entries

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Output Parameter:
.  n - number of non-redundant values

   Level: intermediate

.seealso: PetscIntSortSemiOrdered(), PetscSortReal(), PetscSortIntWithPermutation(), PetscSortInt(), PetscSortedRemoveDupsInt()
@*/
PetscErrorCode  PetscSortRemoveDupsInt(PetscInt *n,PetscInt X[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSortInt(*n,X);CHKERRQ(ierr);
  ierr = PetscSortedRemoveDupsInt(n,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscFindInt - Finds integer in a sorted array of integers

   Not Collective

   Input Parameters:
+  key - the integer to locate
.  n   - number of values in the array
-  X  - array of integers

   Output Parameter:
.  loc - the location if found, otherwise -(slot+1) where slot is the place the value would go

   Level: intermediate

.seealso: PetscIntSortSemiOrdered(), PetscSortInt(), PetscSortIntWithArray(), PetscSortRemoveDupsInt()
@*/
PetscErrorCode PetscFindInt(PetscInt key, PetscInt n, const PetscInt X[], PetscInt *loc)
{
  PetscInt lo = 0,hi = n;

  PetscFunctionBegin;
  PetscValidPointer(loc,4);
  if (!n) {*loc = -1; PetscFunctionReturn(0);}
  PetscValidPointer(X,3);
  PetscCheckSorted(n,X);
  while (hi - lo > 1) {
    PetscInt mid = lo + (hi - lo)/2;
    if (key < X[mid]) hi = mid;
    else               lo = mid;
  }
  *loc = key == X[lo] ? lo : -(lo + (key > X[lo]) + 1);
  PetscFunctionReturn(0);
}

/*@
  PetscCheckDupsInt - Checks if an integer array has duplicates

   Not Collective

   Input Parameters:
+  n  - number of values in the array
-  X  - array of integers

   Output Parameter:
.  dups - True if the array has dups, otherwise false

   Level: intermediate

.seealso: PetscSortRemoveDupsInt()
@*/
PetscErrorCode PetscCheckDupsInt(PetscInt n,const PetscInt X[],PetscBool *dups)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscHSetI     ht;
  PetscBool      missing;

  PetscFunctionBegin;
  PetscValidPointer(dups,3);
  *dups = PETSC_FALSE;
  if (n > 1) {
    ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);
    ierr = PetscHSetIResize(ht,n);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = PetscHSetIQueryAdd(ht,X[i],&missing);CHKERRQ(ierr);
      if (!missing) {*dups = PETSC_TRUE; break;}
    }
    ierr = PetscHSetIDestroy(&ht);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscFindMPIInt - Finds MPI integer in a sorted array of integers

   Not Collective

   Input Parameters:
+  key - the integer to locate
.  n   - number of values in the array
-  X   - array of integers

   Output Parameter:
.  loc - the location if found, otherwise -(slot+1) where slot is the place the value would go

   Level: intermediate

.seealso: PetscMPIIntSortSemiOrdered(), PetscSortInt(), PetscSortIntWithArray(), PetscSortRemoveDupsInt()
@*/
PetscErrorCode PetscFindMPIInt(PetscMPIInt key, PetscInt n, const PetscMPIInt X[], PetscInt *loc)
{
  PetscInt lo = 0,hi = n;

  PetscFunctionBegin;
  PetscValidPointer(loc,4);
  if (!n) {*loc = -1; PetscFunctionReturn(0);}
  PetscValidPointer(X,3);
  PetscCheckSorted(n,X);
  while (hi - lo > 1) {
    PetscInt mid = lo + (hi - lo)/2;
    if (key < X[mid]) hi = mid;
    else               lo = mid;
  }
  *loc = key == X[lo] ? lo : -(lo + (key > X[lo]) + 1);
  PetscFunctionReturn(0);
}

/*@
   PetscSortIntWithArray - Sorts an array of integers in place in increasing order;
       changes a second array to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of integers
-  Y  - second array of integers

   Level: intermediate

.seealso: PetscIntSortSemiOrderedWithArray(), PetscSortReal(), PetscSortIntWithPermutation(), PetscSortInt()
@*/
PetscErrorCode  PetscSortIntWithArray(PetscInt n,PetscInt X[],PetscInt Y[])
{
  PetscErrorCode ierr;
  PetscInt       pivot,t1,t2;

  PetscFunctionBegin;
  QuickSort2(PetscSortIntWithArray,X,Y,n,pivot,t1,t2,ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSortIntWithArrayPair - Sorts an array of integers in place in increasing order;
       changes a pair of integer arrays to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of integers
.  Y  - second array of integers (first array of the pair)
-  Z  - third array of integers  (second array of the pair)

   Level: intermediate

.seealso: PetscSortReal(), PetscSortIntWithPermutation(), PetscSortIntWithArray(), PetscIntSortSemiOrdered()
@*/
PetscErrorCode  PetscSortIntWithArrayPair(PetscInt n,PetscInt X[],PetscInt Y[],PetscInt Z[])
{
  PetscErrorCode ierr;
  PetscInt       pivot,t1,t2,t3;

  PetscFunctionBegin;
  QuickSort3(PetscSortIntWithArrayPair,X,Y,Z,n,pivot,t1,t2,t3,ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSortedMPIInt - Determines whether the array is sorted.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Output Parameters:
.  sorted - flag whether the array is sorted

   Level: intermediate

.seealso: PetscMPIIntSortSemiOrdered(), PetscSortMPIInt(), PetscSortedInt(), PetscSortedReal()
@*/
PetscErrorCode  PetscSortedMPIInt(PetscInt n,const PetscMPIInt X[],PetscBool *sorted)
{
  PetscFunctionBegin;
  PetscSorted(n,X,*sorted);
  PetscFunctionReturn(0);
}

/*@
   PetscSortMPIInt - Sorts an array of MPI integers in place in increasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Level: intermediate

   Notes:
   This function serves as an alternative to PetscMPIIntSortSemiOrdered(), and may perform faster especially if the array
   is completely random. There are exceptions to this and so it is __highly__ recommended that the user benchmark their
   code to see which routine is fastest.

.seealso: PetscMPIIntSortSemiOrdered(), PetscSortReal(), PetscSortIntWithPermutation()
@*/
PetscErrorCode  PetscSortMPIInt(PetscInt n,PetscMPIInt X[])
{
  PetscErrorCode ierr;
  PetscMPIInt    pivot,t1;

  PetscFunctionBegin;
  QuickSort1(PetscSortMPIInt,X,n,pivot,t1,ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSortRemoveDupsMPIInt - Sorts an array of MPI integers in place in increasing order removes all duplicate entries

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Output Parameter:
.  n - number of non-redundant values

   Level: intermediate

.seealso: PetscSortReal(), PetscSortIntWithPermutation(), PetscSortInt()
@*/
PetscErrorCode  PetscSortRemoveDupsMPIInt(PetscInt *n,PetscMPIInt X[])
{
  PetscErrorCode ierr;
  PetscInt       i,s = 0,N = *n, b = 0;

  PetscFunctionBegin;
  ierr = PetscSortMPIInt(N,X);CHKERRQ(ierr);
  for (i=0; i<N-1; i++) {
    if (X[b+s+1] != X[b]) {
      X[b+1] = X[b+s+1]; b++;
    } else s++;
  }
  *n = N - s;
  PetscFunctionReturn(0);
}

/*@
   PetscSortMPIIntWithArray - Sorts an array of integers in place in increasing order;
       changes a second array to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of integers
-  Y  - second array of integers

   Level: intermediate

.seealso: PetscMPIIntSortSemiOrderedWithArray(), PetscSortReal(), PetscSortIntWithPermutation(), PetscSortInt()
@*/
PetscErrorCode  PetscSortMPIIntWithArray(PetscMPIInt n,PetscMPIInt X[],PetscMPIInt Y[])
{
  PetscErrorCode ierr;
  PetscMPIInt    pivot,t1,t2;

  PetscFunctionBegin;
  QuickSort2(PetscSortMPIIntWithArray,X,Y,n,pivot,t1,t2,ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSortMPIIntWithIntArray - Sorts an array of MPI integers in place in increasing order;
       changes a second array of Petsc intergers to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of MPI integers
-  Y  - second array of Petsc integers

   Level: intermediate

   Notes: this routine is useful when one needs to sort MPI ranks with other integer arrays.

.seealso: PetscSortMPIIntWithArray(), PetscIntSortSemiOrderedWithArray(), PetscTimSortWithArray()
@*/
PetscErrorCode PetscSortMPIIntWithIntArray(PetscMPIInt n,PetscMPIInt X[],PetscInt Y[])
{
  PetscErrorCode ierr;
  PetscMPIInt    pivot,t1;
  PetscInt       t2;

  PetscFunctionBegin;
  QuickSort2(PetscSortMPIIntWithIntArray,X,Y,n,pivot,t1,t2,ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSortIntWithScalarArray - Sorts an array of integers in place in increasing order;
       changes a second SCALAR array to match the sorted first INTEGER array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  X  - array of integers
-  Y  - second array of scalars

   Level: intermediate

.seealso: PetscTimSortWithArray(), PetscSortReal(), PetscSortIntWithPermutation(), PetscSortInt(), PetscSortIntWithArray()
@*/
PetscErrorCode  PetscSortIntWithScalarArray(PetscInt n,PetscInt X[],PetscScalar Y[])
{
  PetscErrorCode ierr;
  PetscInt       pivot,t1;
  PetscScalar    t2;

  PetscFunctionBegin;
  QuickSort2(PetscSortIntWithScalarArray,X,Y,n,pivot,t1,t2,ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSortIntWithDataArray - Sorts an array of integers in place in increasing order;
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

.seealso: PetscTimSortWithArray(), PetscSortReal(), PetscSortIntWithPermutation(), PetscSortInt(), PetscSortIntWithArray()
@*/
PetscErrorCode  PetscSortIntWithDataArray(PetscInt n,PetscInt X[],void *Y,size_t size,void *t2)
{
  PetscErrorCode ierr;
  char           *YY = (char*)Y;
  PetscInt       i,j,p,t1,pivot,hi=n-1,l,r;

  PetscFunctionBegin;
  if (n<8) {
    for (i=0; i<n; i++) {
      pivot = X[i];
      for (j=i+1; j<n; j++) {
        if (pivot > X[j]) {
          SWAP2Data(X[i],X[j],YY+size*i,YY+size*j,t1,t2,size);
          pivot = X[i];
        }
      }
    }
  } else {
    /* Two way partition */
    p     = MEDIAN(X,hi);
    pivot = X[p];
    l     = 0;
    r     = hi;
    while (1) {
      while (X[l] < pivot) l++;
      while (X[r] > pivot) r--;
      if (l >= r) {r++; break;}
      SWAP2Data(X[l],X[r],YY+size*l,YY+size*r,t1,t2,size);
      l++;
      r--;
    }
    ierr = PetscSortIntWithDataArray(l,X,Y,size,t2);CHKERRQ(ierr);
    ierr = PetscSortIntWithDataArray(hi-r+1,X+r,YY+size*r,size,t2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscMergeIntArray -     Merges two SORTED integer arrays, removes duplicate elements.

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

.seealso: PetscSortReal(), PetscSortIntWithPermutation(), PetscSortInt(), PetscSortIntWithArray()
@*/
PetscErrorCode  PetscMergeIntArray(PetscInt an,const PetscInt aI[], PetscInt bn, const PetscInt bI[], PetscInt *n, PetscInt **L)
{
  PetscErrorCode ierr;
  PetscInt       *L_ = *L, ak, bk, k;

  PetscFunctionBegin;
  if (!L_) {
    ierr = PetscMalloc1(an+bn, L);CHKERRQ(ierr);
    L_   = *L;
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
    ierr = PetscArraycpy(L_+k,aI+ak,an-ak);CHKERRQ(ierr);
    k   += (an-ak);
  }
  if (bk < bn) {
    ierr = PetscArraycpy(L_+k,bI+bk,bn-bk);CHKERRQ(ierr);
    k   += (bn-bk);
  }
  *n = k;
  PetscFunctionReturn(0);
}

/*@
   PetscMergeIntArrayPair -     Merges two SORTED integer arrays that share NO common values along with an additional array of integers.
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

   Notes:
    if L or J point to non-null arrays then this routine will assume they are of the approproate size and use them, otherwise this routine will allocate space for them
   Level: intermediate

.seealso: PetscIntSortSemiOrdered(), PetscSortReal(), PetscSortIntWithPermutation(), PetscSortInt(), PetscSortIntWithArray()
@*/
PetscErrorCode  PetscMergeIntArrayPair(PetscInt an,const PetscInt aI[], const PetscInt aJ[], PetscInt bn, const PetscInt bI[], const PetscInt bJ[], PetscInt *n, PetscInt **L, PetscInt **J)
{
  PetscErrorCode ierr;
  PetscInt       n_, *L_, *J_, ak, bk, k;

  PetscFunctionBegin;
  PetscValidIntPointer(L,8);
  PetscValidIntPointer(J,9);
  n_ = an + bn;
  *n = n_;
  if (!*L) {
    ierr = PetscMalloc1(n_, L);CHKERRQ(ierr);
  }
  L_ = *L;
  if (!*J) {
    ierr = PetscMalloc1(n_, J);CHKERRQ(ierr);
  }
  J_   = *J;
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
    ierr = PetscArraycpy(L_+k,aI+ak,an-ak);CHKERRQ(ierr);
    ierr = PetscArraycpy(J_+k,aJ+ak,an-ak);CHKERRQ(ierr);
    k   += (an-ak);
  }
  if (bk < bn) {
    ierr = PetscArraycpy(L_+k,bI+bk,bn-bk);CHKERRQ(ierr);
    ierr = PetscArraycpy(J_+k,bJ+bk,bn-bk);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscMergeMPIIntArray -     Merges two SORTED integer arrays.

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

.seealso: PetscIntSortSemiOrdered(), PetscSortReal(), PetscSortIntWithPermutation(), PetscSortInt(), PetscSortIntWithArray()
@*/
PetscErrorCode PetscMergeMPIIntArray(PetscInt an,const PetscMPIInt aI[],PetscInt bn,const PetscMPIInt bI[],PetscInt *n,PetscMPIInt **L)
{
  PetscErrorCode ierr;
  PetscInt       ai,bi,k;

  PetscFunctionBegin;
  if (!*L) {ierr = PetscMalloc1((an+bn),L);CHKERRQ(ierr);}
  for (ai=0,bi=0,k=0; ai<an || bi<bn;) {
    PetscInt t = -1;
    for (; ai<an && (!bn || aI[ai] <= bI[bi]); ai++) (*L)[k++] = t = aI[ai];
    for (; bi<bn && bI[bi] == t; bi++);
    for (; bi<bn && (!an || bI[bi] <= aI[ai]); bi++) (*L)[k++] = t = bI[bi];
    for (; ai<an && aI[ai] == t; ai++);
  }
  *n = k;
  PetscFunctionReturn(0);
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

   Notes:
    This code is not currently used

.seealso: PetscSortReal(), PetscSortIntWithPermutation()
@*/
PetscErrorCode  PetscProcessTree(PetscInt n,const PetscBool mask[],const PetscInt parentid[],PetscInt *Nlevels,PetscInt **Level,PetscInt **Levelcnt,PetscInt **Idbylevel,PetscInt **Column)
{
  PetscInt       i,j,cnt,nmask = 0,nlevels = 0,*level,*levelcnt,levelmax = 0,*workid,*workparentid,tcnt = 0,*idbylevel,*column;
  PetscErrorCode ierr;
  PetscBool      done = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCheckFalse(!mask[0],PETSC_COMM_SELF,PETSC_ERR_SUP,"Mask of 0th location must be set");
  for (i=0; i<n; i++) {
    if (mask[i]) continue;
    PetscCheckFalse(parentid[i]  == i,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Node labeled as own parent");
    PetscCheckFalse(parentid[i] && mask[parentid[i]],PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Parent is masked");
  }

  for (i=0; i<n; i++) {
    if (!mask[i]) nmask++;
  }

  /* determine the level in the tree of each node */
  ierr = PetscCalloc1(n,&level);CHKERRQ(ierr);

  level[0] = 1;
  while (!done) {
    done = PETSC_TRUE;
    for (i=0; i<n; i++) {
      if (mask[i]) continue;
      if (!level[i] && level[parentid[i]]) level[i] = level[parentid[i]] + 1;
      else if (!level[i]) done = PETSC_FALSE;
    }
  }
  for (i=0; i<n; i++) {
    level[i]--;
    nlevels = PetscMax(nlevels,level[i]);
  }

  /* count the number of nodes on each level and its max */
  ierr = PetscCalloc1(nlevels,&levelcnt);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (mask[i]) continue;
    levelcnt[level[i]-1]++;
  }
  for (i=0; i<nlevels;i++) levelmax = PetscMax(levelmax,levelcnt[i]);

  /* for each level sort the ids by the parent id */
  ierr = PetscMalloc2(levelmax,&workid,levelmax,&workparentid);CHKERRQ(ierr);
  ierr = PetscMalloc1(nmask,&idbylevel);CHKERRQ(ierr);
  for (j=1; j<=nlevels;j++) {
    cnt = 0;
    for (i=0; i<n; i++) {
      if (mask[i]) continue;
      if (level[i] != j) continue;
      workid[cnt]         = i;
      workparentid[cnt++] = parentid[i];
    }
    /*  PetscIntView(cnt,workparentid,0);
    PetscIntView(cnt,workid,0);
    ierr = PetscSortIntWithArray(cnt,workparentid,workid);CHKERRQ(ierr);
    PetscIntView(cnt,workparentid,0);
    PetscIntView(cnt,workid,0);*/
    ierr  = PetscArraycpy(idbylevel+tcnt,workid,cnt);CHKERRQ(ierr);
    tcnt += cnt;
  }
  PetscCheckFalse(tcnt != nmask,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent count of unmasked nodes");
  ierr = PetscFree2(workid,workparentid);CHKERRQ(ierr);

  /* for each node list its column */
  ierr = PetscMalloc1(n,&column);CHKERRQ(ierr);
  cnt = 0;
  for (j=0; j<nlevels; j++) {
    for (i=0; i<levelcnt[j]; i++) {
      column[idbylevel[cnt++]] = i;
    }
  }

  *Nlevels   = nlevels;
  *Level     = level;
  *Levelcnt  = levelcnt;
  *Idbylevel = idbylevel;
  *Column    = column;
  PetscFunctionReturn(0);
}

/*@
  PetscParallelSortedInt - Check whether an integer array, distributed over a communicator, is globally sorted.

  Collective

  Input Parameters:
+ comm - the MPI communicator
. n - the local number of integers
- keys - the local array of integers

  Output Parameters:
. is_sorted - whether the array is globally sorted

  Level: developer

.seealso: PetscParallelSortInt()
@*/
PetscErrorCode PetscParallelSortedInt(MPI_Comm comm, PetscInt n, const PetscInt keys[], PetscBool *is_sorted)
{
  PetscBool      sorted;
  PetscInt       i, min, max, prevmax;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

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
    min = PetscMin(min,keys[i]);
    max = PetscMax(max,keys[i]);
  }
  if (i < n) sorted = PETSC_FALSE;
  prevmax = PETSC_MIN_INT;
  ierr = MPI_Exscan(&max, &prevmax, 1, MPIU_INT, MPI_MAX, comm);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  if (rank == 0) prevmax = PETSC_MIN_INT;
  if (prevmax > min) sorted = PETSC_FALSE;
  ierr = MPI_Allreduce(&sorted, is_sorted, 1, MPIU_BOOL, MPI_LAND, comm);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}
