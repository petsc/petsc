
/*
   This file contains routines for sorting doubles.  Values are sorted in place.
   These are provided because the general sort routines incur a great deal
   of overhead in calling the comparison routines.

 */
#include <petscsys.h>                /*I  "petscsys.h"  I*/
#include <petsc/private/petscimpl.h>

#define SWAP(a,b,t) {t=a;a=b;b=t;}

/*@
   PetscSortedReal - Determines whether the array is sorted.

   Not Collective

   Input Parameters:
+  n  - number of values
-  X  - array of integers

   Output Parameters:
.  sorted - flag whether the array is sorted

   Level: intermediate

.seealso: PetscSortReal(), PetscSortedInt(), PetscSortedMPIInt()
@*/
PetscErrorCode  PetscSortedReal(PetscInt n,const PetscReal X[],PetscBool *sorted)
{
  PetscFunctionBegin;
  PetscSorted(n,X,*sorted);
  PetscFunctionReturn(0);
}

/* A simple version of quicksort; taken from Kernighan and Ritchie, page 87 */
static PetscErrorCode PetscSortReal_Private(PetscReal *v,PetscInt right)
{
  PetscInt  i,last;
  PetscReal vl,tmp;

  PetscFunctionBegin;
  if (right <= 1) {
    if (right == 1) {
      if (v[0] > v[1]) SWAP(v[0],v[1],tmp);
    }
    PetscFunctionReturn(0);
  }
  SWAP(v[0],v[right/2],tmp);
  vl   = v[0];
  last = 0;
  for (i=1; i<=right; i++) {
    if (v[i] < vl) {last++; SWAP(v[last],v[i],tmp);}
  }
  SWAP(v[0],v[last],tmp);
  PetscSortReal_Private(v,last-1);
  PetscSortReal_Private(v+last+1,right-(last+1));
  PetscFunctionReturn(0);
}

/*@
   PetscSortReal - Sorts an array of doubles in place in increasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  v  - array of doubles

   Notes:
   This function serves as an alternative to PetscRealSortSemiOrdered(), and may perform faster especially if the array
   is completely random. There are exceptions to this and so it is __highly__ recommended that the user benchmark their
   code to see which routine is fastest.

   Level: intermediate

.seealso: PetscRealSortSemiOrdered(), PetscSortInt(), PetscSortRealWithPermutation(), PetscSortRealWithArrayInt()
@*/
PetscErrorCode  PetscSortReal(PetscInt n,PetscReal v[])
{
  PetscInt  j,k;
  PetscReal tmp,vk;

  PetscFunctionBegin;
  PetscValidRealPointer(v,2);
  if (n<8) {
    for (k=0; k<n; k++) {
      vk = v[k];
      for (j=k+1; j<n; j++) {
        if (vk > v[j]) {
          SWAP(v[k],v[j],tmp);
          vk = v[k];
        }
      }
    }
  } else PetscSortReal_Private(v,n-1);
  PetscFunctionReturn(0);
}

#define SWAP2ri(a,b,c,d,rt,it) {rt=a;a=b;b=rt;it=c;c=d;d=it;}

/* modified from PetscSortIntWithArray_Private */
static PetscErrorCode PetscSortRealWithArrayInt_Private(PetscReal *v,PetscInt *V,PetscInt right)
{
  PetscInt       i,last,itmp;
  PetscReal      rvl,rtmp;

  PetscFunctionBegin;
  if (right <= 1) {
    if (right == 1) {
      if (v[0] > v[1]) SWAP2ri(v[0],v[1],V[0],V[1],rtmp,itmp);
    }
    PetscFunctionReturn(0);
  }
  SWAP2ri(v[0],v[right/2],V[0],V[right/2],rtmp,itmp);
  rvl  = v[0];
  last = 0;
  for (i=1; i<=right; i++) {
    if (v[i] < rvl) {last++; SWAP2ri(v[last],v[i],V[last],V[i],rtmp,itmp);}
  }
  SWAP2ri(v[0],v[last],V[0],V[last],rtmp,itmp);
  CHKERRQ(PetscSortRealWithArrayInt_Private(v,V,last-1));
  CHKERRQ(PetscSortRealWithArrayInt_Private(v+last+1,V+last+1,right-(last+1)));
  PetscFunctionReturn(0);
}
/*@
   PetscSortRealWithArrayInt - Sorts an array of PetscReal in place in increasing order;
       changes a second PetscInt array to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  i  - array of integers
-  I - second array of integers

   Level: intermediate

.seealso: PetscSortReal()
@*/
PetscErrorCode  PetscSortRealWithArrayInt(PetscInt n,PetscReal r[],PetscInt Ii[])
{
  PetscInt       j,k,itmp;
  PetscReal      rk,rtmp;

  PetscFunctionBegin;
  PetscValidRealPointer(r,2);
  PetscValidIntPointer(Ii,3);
  if (n<8) {
    for (k=0; k<n; k++) {
      rk = r[k];
      for (j=k+1; j<n; j++) {
        if (rk > r[j]) {
          SWAP2ri(r[k],r[j],Ii[k],Ii[j],rtmp,itmp);
          rk = r[k];
        }
      }
    }
  } else {
    CHKERRQ(PetscSortRealWithArrayInt_Private(r,Ii,n-1));
  }
  PetscFunctionReturn(0);
}

/*@
  PetscFindReal - Finds a PetscReal in a sorted array of PetscReals

   Not Collective

   Input Parameters:
+  key - the value to locate
.  n   - number of values in the array
.  ii  - array of values
-  eps - tolerance used to compare

   Output Parameter:
.  loc - the location if found, otherwise -(slot+1) where slot is the place the value would go

   Level: intermediate

.seealso: PetscSortReal(), PetscSortRealWithArrayInt()
@*/
PetscErrorCode PetscFindReal(PetscReal key, PetscInt n, const PetscReal t[], PetscReal eps, PetscInt *loc)
{
  PetscInt lo = 0,hi = n;

  PetscFunctionBegin;
  PetscValidIntPointer(loc,5);
  if (!n) {*loc = -1; PetscFunctionReturn(0);}
  PetscValidRealPointer(t,3);
  PetscCheckSorted(n,t);
  while (hi - lo > 1) {
    PetscInt mid = lo + (hi - lo)/2;
    if (key < t[mid]) hi = mid;
    else              lo = mid;
  }
  *loc = (PetscAbsReal(key - t[lo]) < eps) ? lo : -(lo + (key > t[lo]) + 1);
  PetscFunctionReturn(0);
}

/*@
   PetscSortRemoveDupsReal - Sorts an array of doubles in place in increasing order removes all duplicate entries

   Not Collective

   Input Parameters:
+  n  - number of values
-  v  - array of doubles

   Output Parameter:
.  n - number of non-redundant values

   Level: intermediate

.seealso: PetscSortReal(), PetscSortRemoveDupsInt()
@*/
PetscErrorCode  PetscSortRemoveDupsReal(PetscInt *n,PetscReal v[])
{
  PetscInt       i,s = 0,N = *n, b = 0;

  PetscFunctionBegin;
  CHKERRQ(PetscSortReal(N,v));
  for (i=0; i<N-1; i++) {
    if (v[b+s+1] != v[b]) {
      v[b+1] = v[b+s+1]; b++;
    } else s++;
  }
  *n = N - s;
  PetscFunctionReturn(0);
}

/*@
   PetscSortSplit - Quick-sort split of an array of PetscScalars in place.

   Not Collective

   Input Parameters:
+  ncut  - splitig index
-  n     - number of values to sort

   Input/Output Parameters:
+  a     - array of values, on output the values are permuted such that its elements satisfy:
           abs(a[i]) >= abs(a[ncut-1]) for i < ncut and
           abs(a[i]) <= abs(a[ncut-1]) for i >= ncut
-  idx   - index for array a, on output permuted accordingly

   Level: intermediate

.seealso: PetscSortInt(), PetscSortRealWithPermutation()
@*/
PetscErrorCode  PetscSortSplit(PetscInt ncut,PetscInt n,PetscScalar a[],PetscInt idx[])
{
  PetscInt    i,mid,last,itmp,j,first;
  PetscScalar d,tmp;
  PetscReal   abskey;

  PetscFunctionBegin;
  first = 0;
  last  = n-1;
  if (ncut < first || ncut > last) PetscFunctionReturn(0);

  while (1) {
    mid    = first;
    d      = a[mid];
    abskey = PetscAbsScalar(d);
    i      = last;
    for (j = first + 1; j <= i; ++j) {
      d = a[j];
      if (PetscAbsScalar(d) >= abskey) {
        ++mid;
        /* interchange */
        tmp = a[mid];  itmp = idx[mid];
        a[mid] = a[j]; idx[mid] = idx[j];
        a[j] = tmp;    idx[j] = itmp;
      }
    }

    /* interchange */
    tmp = a[mid];      itmp = idx[mid];
    a[mid] = a[first]; idx[mid] = idx[first];
    a[first] = tmp;    idx[first] = itmp;

    /* test for while loop */
    if (mid == ncut) break;
    else if (mid > ncut) last = mid - 1;
    else first = mid + 1;
  }
  PetscFunctionReturn(0);
}

/*@
   PetscSortSplitReal - Quick-sort split of an array of PetscReals in place.

   Not Collective

   Input Parameters:
+  ncut  - splitig index
-  n     - number of values to sort

   Input/Output Parameters:
+  a     - array of values, on output the values are permuted such that its elements satisfy:
           abs(a[i]) >= abs(a[ncut-1]) for i < ncut and
           abs(a[i]) <= abs(a[ncut-1]) for i >= ncut
-  idx   - index for array a, on output permuted accordingly

   Level: intermediate

.seealso: PetscSortInt(), PetscSortRealWithPermutation()
@*/
PetscErrorCode  PetscSortSplitReal(PetscInt ncut,PetscInt n,PetscReal a[],PetscInt idx[])
{
  PetscInt  i,mid,last,itmp,j,first;
  PetscReal d,tmp;
  PetscReal abskey;

  PetscFunctionBegin;
  first = 0;
  last  = n-1;
  if (ncut < first || ncut > last) PetscFunctionReturn(0);

  while (1) {
    mid    = first;
    d      = a[mid];
    abskey = PetscAbsReal(d);
    i      = last;
    for (j = first + 1; j <= i; ++j) {
      d = a[j];
      if (PetscAbsReal(d) >= abskey) {
        ++mid;
        /* interchange */
        tmp = a[mid];  itmp = idx[mid];
        a[mid] = a[j]; idx[mid] = idx[j];
        a[j] = tmp;    idx[j] = itmp;
      }
    }

    /* interchange */
    tmp = a[mid];      itmp = idx[mid];
    a[mid] = a[first]; idx[mid] = idx[first];
    a[first] = tmp;    idx[first] = itmp;

    /* test for while loop */
    if (mid == ncut) break;
    else if (mid > ncut) last = mid - 1;
    else first = mid + 1;
  }
  PetscFunctionReturn(0);
}
