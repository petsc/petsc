#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: sortip.c,v 1.24 1998/08/26 22:01:52 balay Exp bsmith $";
#endif

/*
   This file contains routines for sorting integers and doubles with a permutation array.

   The word "register"  in this code is used to identify data that is not
   aliased.  For some compilers, this can cause the compiler to fail to
   place inner-loop variables into registers.
 */
#include "petsc.h"           /*I  "petsc.h"  I*/
#include "sys.h"             /*I  "sys.h"    I*/

#define SWAP(a,b,t) {t=a;a=b;b=t;}

#undef __FUNC__  
#define __FUNC__ "PetsciIqsortPerm"
static int PetsciIqsortPerm(const int v[],int vdx[],int right)
{
  int          tmp;
  register int i, vl, last;

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
  for ( i=1; i<=right; i++ ) {
    if (v[vdx[i]] < vl ) {last++; SWAP(vdx[last],vdx[i],tmp);}
  }
  SWAP(vdx[0],vdx[last],tmp);
  PetsciIqsortPerm(v,vdx,last-1);
  PetsciIqsortPerm(v,vdx+last+1,right-(last+1));
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscSortIntWithPermutation"
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
int PetscSortIntWithPermutation(int n, const int i[], int idx[] )
{
  register int j, k, tmp, ik;

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
    PetsciIqsortPerm(i,idx,n-1);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------- */

#undef __FUNC__  
#define __FUNC__ "PetsciDqsortPerm"
static int PetsciDqsortPerm(const double v[],int vdx[],int right)
{
  double       vl;
  register int tmp,i, last;

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
  for ( i=1; i<=right; i++ ) {
    if (v[vdx[i]] < vl ) {last++; SWAP(vdx[last],vdx[i],tmp);}
  }
  SWAP(vdx[0],vdx[last],tmp);
  PetsciDqsortPerm(v,vdx,last-1);
  PetsciDqsortPerm(v,vdx+last+1,right-(last+1));
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscSortDoubleWithPermutation"
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
int PetscSortDoubleWithPermutation(int n, const double i[], int idx[])
{
  register int j, k,tmp;
  double       ik;

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
    PetsciDqsortPerm(i,idx,n-1);
  }
  PetscFunctionReturn(0);
}
