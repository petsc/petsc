/*$Id: sortip.c,v 1.37 2001/08/07 21:29:06 bsmith Exp $*/
/*
   This file contains routines for sorting integers and doubles with a permutation array.

   The word "register"  in this code is used to identify data that is not
   aliased.  For some compilers, this can cause the compiler to fail to
   place inner-loop variables into registers.
 */
#include "petsc.h"                /*I  "petsc.h"  I*/
#include "petscsys.h"             /*I  "petscsys.h"    I*/

#define SWAP(a,b,t) {t=a;a=b;b=t;}

#undef __FUNCT__  
#define __FUNCT__ "PetscSortIntWithPermutation_Private"
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

#undef __FUNCT__  
#define __FUNCT__ "PetscSortIntWithPermutation"
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

   Concepts: sorting^ints with permutation

.seealso: PetscSortInt(), PetscSortRealWithPermutation()
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

#undef __FUNCT__  
#define __FUNCT__ "PetscSortRealWithPermutation_Private"
static int PetscSortRealWithPermutation_Private(const PetscReal v[],int vdx[],int right)
{
  PetscReal vl;
  int       ierr,tmp,i,last;

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
  ierr = PetscSortRealWithPermutation_Private(v,vdx,last-1);CHKERRQ(ierr);
  ierr = PetscSortRealWithPermutation_Private(v,vdx+last+1,right-(last+1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSortRealWithPermutation"
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

   Concepts: sorting^doubles with permutation

.seealso: PetscSortReal(), PetscSortIntWithPermutation()
 @*/
int PetscSortRealWithPermutation(int n,const PetscReal i[],int idx[])
{
  int       j,k,tmp,ierr;
  PetscReal ik;

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
    ierr = PetscSortRealWithPermutation_Private(i,idx,n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
