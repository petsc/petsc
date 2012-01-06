
/*
   This file contains routines for sorting doubles.  Values are sorted in place.
   These are provided because the general sort routines incur a great deal
   of overhead in calling the comparision routines.

 */
#include <petscsys.h>                /*I  "petscsys.h"  I*/

#define SWAP(a,b,t) {t=a;a=b;b=t;}
   
#undef __FUNCT__  
#define __FUNCT__ "PetscSortReal_Private"
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

#undef __FUNCT__  
#define __FUNCT__ "PetscSortReal"
/*@
   PetscSortReal - Sorts an array of doubles in place in increasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  v  - array of doubles

   Level: intermediate

   Concepts: sorting^doubles

.seealso: PetscSortInt(), PetscSortRealWithPermutation()
@*/
PetscErrorCode  PetscSortReal(PetscInt n,PetscReal v[])
{
  PetscInt  j,k;
  PetscReal tmp,vk;

  PetscFunctionBegin;
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
  } else {
    PetscSortReal_Private(v,n-1);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSortSplit"
/*@
   PetscSortSplit - Quick-sort split of an array of PetscScalars in place.

   Not Collective

   Input Parameters:
+  ncut  - splitig index
.  n     - number of values to sort
.  a     - array of values
-  idx   - index for array a

   Output Parameters:
+  a     - permuted array of values such that its elements satisfy:
           abs(a[i]) >= abs(a[ncut-1]) for i < ncut and 
           abs(a[i]) <= abs(a[ncut-1]) for i >= ncut 
-  idx   - permuted index of array a

   Level: intermediate

   Concepts: sorting^doubles

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

  while (1){
    mid = first;
    abskey = (d = a[mid],PetscAbsScalar(d));
    i = last;
    for (j = first + 1; j <= i; ++j) {
      if ((d = a[j],PetscAbsScalar(d)) >= abskey) {
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
    if (mid == ncut) {
      break;
    } else if (mid > ncut){
      last = mid - 1;
    } else {
      first = mid + 1;
    }
  } 
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "PetscSortSplitReal"
/*@
   PetscSortSplitReal - Quick-sort split of an array of PetscReals in place.

   Not Collective

   Input Parameters:
+  ncut  - splitig index
.  n     - number of values to sort
.  a     - array of values in PetscReal
-  idx   - index for array a

   Output Parameters:
+  a     - permuted array of real values such that its elements satisfy:
           abs(a[i]) >= abs(a[ncut-1]) for i < ncut and 
           abs(a[i]) <= abs(a[ncut-1]) for i >= ncut 
-  idx   - permuted index of array a

   Level: intermediate

   Concepts: sorting^doubles

.seealso: PetscSortInt(), PetscSortRealWithPermutation()
@*/
PetscErrorCode  PetscSortSplitReal(PetscInt ncut,PetscInt n,PetscReal a[],PetscInt idx[])
{
  PetscInt    i,mid,last,itmp,j,first;
  PetscReal   d,tmp;
  PetscReal   abskey;

  PetscFunctionBegin;
  first = 0;
  last  = n-1;
  if (ncut < first || ncut > last) PetscFunctionReturn(0);

  while (1){
    mid = first;
    abskey = (d = a[mid],PetscAbsReal(d));
    i = last;
    for (j = first + 1; j <= i; ++j) {
      if ((d = a[j],PetscAbsReal(d)) >= abskey) {
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
    if (mid == ncut) {
      break;
    } else if (mid > ncut){
      last = mid - 1;
    } else {
      first = mid + 1;
    }
  } 
  PetscFunctionReturn(0);
} 

