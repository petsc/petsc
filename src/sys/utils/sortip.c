
/*
   This file contains routines for sorting integers and doubles with a permutation array.

   The word "register"  in this code is used to identify data that is not
   aliased.  For some compilers, this can cause the compiler to fail to
   place inner-loop variables into registers.
 */
#include <petscsys.h>                /*I  "petscsys.h"  I*/

#define SWAP(a,b,t) {t=a;a=b;b=t;}

static PetscErrorCode PetscSortIntWithPermutation_Private(const PetscInt v[],PetscInt vdx[],PetscInt right)
{
  PetscInt       tmp,i,vl,last;

  PetscFunctionBegin;
  if (right <= 1) {
    if (right == 1) {
      if (v[vdx[0]] > v[vdx[1]]) SWAP(vdx[0],vdx[1],tmp);
    }
    PetscFunctionReturn(0);
  }
  SWAP(vdx[0],vdx[right/2],tmp);
  vl   = v[vdx[0]];
  last = 0;
  for (i=1; i<=right; i++) {
    if (v[vdx[i]] < vl) {last++; SWAP(vdx[last],vdx[i],tmp);}
  }
  SWAP(vdx[0],vdx[last],tmp);
  PetscCall(PetscSortIntWithPermutation_Private(v,vdx,last-1));
  PetscCall(PetscSortIntWithPermutation_Private(v,vdx+last+1,right-(last+1)));
  PetscFunctionReturn(0);
}

/*@
   PetscSortIntWithPermutation - Computes the permutation of values that gives
   a sorted sequence.

   Not Collective

   Input Parameters:
+  n  - number of values to sort
.  i  - values to sort
-  idx - permutation array.  Must be initialized to 0:n-1 on input.

   Level: intermediate

   Notes:
   On output i is unchanged and idx[i] is the position of the i-th smallest index in i.

.seealso: `PetscSortInt()`, `PetscSortRealWithPermutation()`, `PetscSortIntWithArray()`
 @*/
PetscErrorCode  PetscSortIntWithPermutation(PetscInt n,const PetscInt i[],PetscInt idx[])
{
  PetscInt       j,k,tmp,ik;

  PetscFunctionBegin;
  if (n<8) {
    for (k=0; k<n; k++) {
      ik = i[idx[k]];
      for (j=k+1; j<n; j++) {
        if (ik > i[idx[j]]) {
          SWAP(idx[k],idx[j],tmp);
          ik = i[idx[k]];
        }
      }
    }
  } else {
    PetscCall(PetscSortIntWithPermutation_Private(i,idx,n-1));
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------- */

static PetscErrorCode PetscSortRealWithPermutation_Private(const PetscReal v[],PetscInt vdx[],PetscInt right)
{
  PetscReal      vl;
  PetscInt       tmp,i,last;

  PetscFunctionBegin;
  if (right <= 1) {
    if (right == 1) {
      if (v[vdx[0]] > v[vdx[1]]) SWAP(vdx[0],vdx[1],tmp);
    }
    PetscFunctionReturn(0);
  }
  SWAP(vdx[0],vdx[right/2],tmp);
  vl   = v[vdx[0]];
  last = 0;
  for (i=1; i<=right; i++) {
    if (v[vdx[i]] < vl) {last++; SWAP(vdx[last],vdx[i],tmp);}
  }
  SWAP(vdx[0],vdx[last],tmp);
  PetscCall(PetscSortRealWithPermutation_Private(v,vdx,last-1));
  PetscCall(PetscSortRealWithPermutation_Private(v,vdx+last+1,right-(last+1)));
  PetscFunctionReturn(0);
}

/*@
   PetscSortRealWithPermutation - Computes the permutation of values that gives
   a sorted sequence.

   Not Collective

   Input Parameters:
+  n  - number of values to sort
.  i  - values to sort
-  idx - permutation array.  Must be initialized to 0:n-1 on input.

   Level: intermediate

   Notes:
   i is unchanged on output.

.seealso: `PetscSortReal()`, `PetscSortIntWithPermutation()`
 @*/
PetscErrorCode  PetscSortRealWithPermutation(PetscInt n,const PetscReal i[],PetscInt idx[])
{
  PetscInt       j,k,tmp;
  PetscReal      ik;

  PetscFunctionBegin;
  if (n<8) {
    for (k=0; k<n; k++) {
      ik = i[idx[k]];
      for (j=k+1; j<n; j++) {
        if (ik > i[idx[j]]) {
          SWAP(idx[k],idx[j],tmp);
          ik = i[idx[k]];
        }
      }
    }
  } else {
    PetscCall(PetscSortRealWithPermutation_Private(i,idx,n-1));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSortStrWithPermutation_Private(const char* v[],PetscInt vdx[],PetscInt right)
{
  PetscInt       tmp,i,last;
  PetscBool      gt;
  const char     *vl;

  PetscFunctionBegin;
  if (right <= 1) {
    if (right == 1) {
      PetscCall(PetscStrgrt(v[vdx[0]],v[vdx[1]],&gt));
      if (gt) SWAP(vdx[0],vdx[1],tmp);
    }
    PetscFunctionReturn(0);
  }
  SWAP(vdx[0],vdx[right/2],tmp);
  vl   = v[vdx[0]];
  last = 0;
  for (i=1; i<=right; i++) {
    PetscCall(PetscStrgrt(vl,v[vdx[i]],&gt));
    if (gt) {last++; SWAP(vdx[last],vdx[i],tmp);}
  }
  SWAP(vdx[0],vdx[last],tmp);
  PetscCall(PetscSortStrWithPermutation_Private(v,vdx,last-1));
  PetscCall(PetscSortStrWithPermutation_Private(v,vdx+last+1,right-(last+1)));
  PetscFunctionReturn(0);
}

/*@C
   PetscSortStrWithPermutation - Computes the permutation of values that gives
   a sorted sequence.

   Not Collective

   Input Parameters:
+  n  - number of values to sort
.  i  - values to sort
-  idx - permutation array.  Must be initialized to 0:n-1 on input.

   Level: intermediate

   Notes:
   i is unchanged on output.

.seealso: `PetscSortInt()`, `PetscSortRealWithPermutation()`
 @*/
PetscErrorCode  PetscSortStrWithPermutation(PetscInt n,const char* i[],PetscInt idx[])
{
  PetscInt       j,k,tmp;
  const char     *ik;
  PetscBool      gt;

  PetscFunctionBegin;
  if (n<8) {
    for (k=0; k<n; k++) {
      ik = i[idx[k]];
      for (j=k+1; j<n; j++) {
        PetscCall(PetscStrgrt(ik,i[idx[j]],&gt));
        if (gt) {
          SWAP(idx[k],idx[j],tmp);
          ik = i[idx[k]];
        }
      }
    }
  } else {
    PetscCall(PetscSortStrWithPermutation_Private(i,idx,n-1));
  }
  PetscFunctionReturn(0);
}
