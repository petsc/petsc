/*$Id: sortip.c,v 1.31 2000/04/23 19:01:00 bsmith Exp balay $*/
/*
   This file contains routines for sorting integers and doubles with a permutation array.

   The word "register"  in this code is used to identify data that is not
   aliased.  For some compilers, this can cause the compiler to fail to
   place inner-loop variables into registers.
 */
#include "petsc.h"                /*I  "petsc.h"  I*/
#include "petscsys.h"             /*I  "petscsys.h"    I*/

#define SWAP(a,b,t) {t=a;a=b;b=t;}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscSortIntWithPermutation_Private"
static int PetscSortIntWithPermutation_Private(const int v[],int vdx[],int right)
{
  int ierr,tmp,i,vl,last;

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
  ierr = PetscSortIntWithPermutation_Private(v,vdx,last-1);CHKERRQ(ierr);
  ierr = PetscSortIntWithPermutation_Private(v,vdx+last+1,right-(last+1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscSortIntWithPermutation"
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
   i is unchanged on output.

.keywords: sort, integer, permutation

.seealso: PetscSortInt(), PetscSortDoubleWithPermutation()
 @*/
int PetscSortIntWithPermutation(int n,const int i[],int idx[])
{
  int ierr,j,k,tmp,ik;

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
    ierr = PetscSortIntWithPermutation_Private(i,idx,n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------- */

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscSortDoubleWithPermutation_Private"
static int PetscSortDoubleWithPermutation_Private(const double v[],int vdx[],int right)
{
  double vl;
  int    ierr,tmp,i,last;

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
  ierr = PetscSortDoubleWithPermutation_Private(v,vdx,last-1);CHKERRQ(ierr);
  ierr = PetscSortDoubleWithPermutation_Private(v,vdx+last+1,right-(last+1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscSortDoubleWithPermutation"
/*@
   PetscSortDoubleWithPermutation - Computes the permutation of values that gives 
   a sorted sequence.

   Not Collective

   Input Parameters:
+  n  - number of values to sort
.  i  - values to sort
-  idx - permutation array.  Must be initialized to 0:n-1 on input.

   Level: intermediate

   Notes: 
   i is unchanged on output.

.keywords: sort, double, permutation

.seealso: PetscSortDouble(), PetscSortIntWithPermutation()
 @*/
int PetscSortDoubleWithPermutation(int n,const double i[],int idx[])
{
  int    j,k,tmp,ierr;
  double ik;

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
    ierr = PetscSortDoubleWithPermutation_Private(i,idx,n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
