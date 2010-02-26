#define PETSC_DLL
/*
   This file contains routines for sorting integers. Values are sorted in place.


   The word "register"  in this code is used to identify data that is not
   aliased.  For some compilers, marking variables as register can improve 
   the compiler optimizations.
 */
#include "petscsys.h"                /*I  "petscsys.h"  I*/

#define SWAP(a,b,t) {t=a;a=b;b=t;}

/* -----------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PetscSortInt_Private"
/* 
   A simple version of quicksort; taken from Kernighan and Ritchie, page 87.
   Assumes 0 origin for v, number of elements = right+1 (right is index of
   right-most member). 
*/
static PetscErrorCode PetscSortInt_Private(PetscInt *v,PetscInt right)
{
  PetscErrorCode ierr;
  PetscInt       i,vl,last,tmp;

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
  ierr = PetscSortInt_Private(v,last-1);CHKERRQ(ierr);
  ierr = PetscSortInt_Private(v+last+1,right-(last+1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSortInt" 
/*@
   PetscSortInt - Sorts an array of integers in place in increasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  i  - array of integers

   Level: intermediate

   Concepts: sorting^ints

.seealso: PetscSortReal(), PetscSortIntWithPermutation()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscSortInt(PetscInt n,PetscInt i[])
{
  PetscErrorCode ierr;
  PetscInt       j,k,tmp,ik;

  PetscFunctionBegin;
  if (n<8) {
    for (k=0; k<n; k++) {
      ik = i[k];
      for (j=k+1; j<n; j++) {
	if (ik > i[j]) {
	  SWAP(i[k],i[j],tmp);
	  ik = i[k];
	}
      }
    }
  } else {
    ierr = PetscSortInt_Private(i,n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------*/
#define SWAP2(a,b,c,d,t) {t=a;a=b;b=t;t=c;c=d;d=t;}

#undef __FUNCT__  
#define __FUNCT__ "PetscSortIntWithArray_Private"
/* 
   A simple version of quicksort; taken from Kernighan and Ritchie, page 87.
   Assumes 0 origin for v, number of elements = right+1 (right is index of
   right-most member). 
*/
static PetscErrorCode PetscSortIntWithArray_Private(PetscInt *v,PetscInt *V,PetscInt right)
{
  PetscErrorCode ierr;
  PetscInt       i,vl,last,tmp;

  PetscFunctionBegin;
  if (right <= 1) {
    if (right == 1) {
      if (v[0] > v[1]) SWAP2(v[0],v[1],V[0],V[1],tmp);
    }
    PetscFunctionReturn(0);
  }
  SWAP2(v[0],v[right/2],V[0],V[right/2],tmp);
  vl   = v[0];
  last = 0;
  for (i=1; i<=right; i++) {
    if (v[i] < vl) {last++; SWAP2(v[last],v[i],V[last],V[i],tmp);}
  }
  SWAP2(v[0],v[last],V[0],V[last],tmp);
  ierr = PetscSortIntWithArray_Private(v,V,last-1);CHKERRQ(ierr);
  ierr = PetscSortIntWithArray_Private(v+last+1,V+last+1,right-(last+1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSortIntWithArray" 
/*@
   PetscSortIntWithArray - Sorts an array of integers in place in increasing order;
       changes a second array to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  i  - array of integers
-  I - second array of integers

   Level: intermediate

   Concepts: sorting^ints with array

.seealso: PetscSortReal(), PetscSortIntPermutation(), PetscSortInt()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscSortIntWithArray(PetscInt n,PetscInt i[],PetscInt Ii[])
{
  PetscErrorCode ierr;
  PetscInt       j,k,tmp,ik;

  PetscFunctionBegin;
  if (n<8) {
    for (k=0; k<n; k++) {
      ik = i[k];
      for (j=k+1; j<n; j++) {
	if (ik > i[j]) {
	  SWAP2(i[k],i[j],Ii[k],Ii[j],tmp);
	  ik = i[k];
	}
      }
    }
  } else {
    ierr = PetscSortIntWithArray_Private(i,Ii,n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSortMPIIntWithArray_Private"
/* 
   A simple version of quicksort; taken from Kernighan and Ritchie, page 87.
   Assumes 0 origin for v, number of elements = right+1 (right is index of
   right-most member). 
*/
static PetscErrorCode PetscSortMPIIntWithArray_Private(PetscMPIInt *v,PetscMPIInt *V,PetscMPIInt right)
{
  PetscErrorCode ierr;
  PetscMPIInt    i,vl,last,tmp;

  PetscFunctionBegin;
  if (right <= 1) {
    if (right == 1) {
      if (v[0] > v[1]) SWAP2(v[0],v[1],V[0],V[1],tmp);
    }
    PetscFunctionReturn(0);
  }
  SWAP2(v[0],v[right/2],V[0],V[right/2],tmp);
  vl   = v[0];
  last = 0;
  for (i=1; i<=right; i++) {
    if (v[i] < vl) {last++; SWAP2(v[last],v[i],V[last],V[i],tmp);}
  }
  SWAP2(v[0],v[last],V[0],V[last],tmp);
  ierr = PetscSortMPIIntWithArray_Private(v,V,last-1);CHKERRQ(ierr);
  ierr = PetscSortMPIIntWithArray_Private(v+last+1,V+last+1,right-(last+1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSortMPIIntWithArray" 
/*@
   PetscSortMPIIntWithArray - Sorts an array of integers in place in increasing order;
       changes a second array to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  i  - array of integers
-  I - second array of integers

   Level: intermediate

   Concepts: sorting^ints with array

.seealso: PetscSortReal(), PetscSortIntPermutation(), PetscSortInt()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscSortMPIIntWithArray(PetscMPIInt n,PetscMPIInt i[],PetscMPIInt Ii[])
{
  PetscErrorCode ierr;
  PetscMPIInt    j,k,tmp,ik;

  PetscFunctionBegin;
  if (n<8) {
    for (k=0; k<n; k++) {
      ik = i[k];
      for (j=k+1; j<n; j++) {
	if (ik > i[j]) {
	  SWAP2(i[k],i[j],Ii[k],Ii[j],tmp);
	  ik = i[k];
	}
      }
    }
  } else {
    ierr = PetscSortMPIIntWithArray_Private(i,Ii,n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------*/
#define SWAP2IntScalar(a,b,c,d,t,ts) {t=a;a=b;b=t;ts=c;c=d;d=ts;}

#undef __FUNCT__  
#define __FUNCT__ "PetscSortIntWithScalarArray_Private"
/* 
   Modified from PetscSortIntWithArray_Private(). 
*/
static PetscErrorCode PetscSortIntWithScalarArray_Private(PetscInt *v,PetscScalar *V,PetscInt right)
{
  PetscErrorCode ierr;
  PetscInt       i,vl,last,tmp;
  PetscScalar    stmp;

  PetscFunctionBegin;
  if (right <= 1) {
    if (right == 1) {
      if (v[0] > v[1]) SWAP2IntScalar(v[0],v[1],V[0],V[1],tmp,stmp);
    }
    PetscFunctionReturn(0);
  }
  SWAP2IntScalar(v[0],v[right/2],V[0],V[right/2],tmp,stmp);
  vl   = v[0];
  last = 0;
  for (i=1; i<=right; i++) {
    if (v[i] < vl) {last++; SWAP2IntScalar(v[last],v[i],V[last],V[i],tmp,stmp);}
  }
  SWAP2IntScalar(v[0],v[last],V[0],V[last],tmp,stmp);
  ierr = PetscSortIntWithScalarArray_Private(v,V,last-1);CHKERRQ(ierr);
  ierr = PetscSortIntWithScalarArray_Private(v+last+1,V+last+1,right-(last+1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSortIntWithScalarArray" 
/*@
   PetscSortIntWithScalarArray - Sorts an array of integers in place in increasing order;
       changes a second SCALAR array to match the sorted first INTEGER array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  i  - array of integers
-  I - second array of scalars

   Level: intermediate

   Concepts: sorting^ints with array

.seealso: PetscSortReal(), PetscSortIntPermutation(), PetscSortInt(), PetscSortIntWithArray()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscSortIntWithScalarArray(PetscInt n,PetscInt i[],PetscScalar Ii[])
{
  PetscErrorCode ierr;
  PetscInt       j,k,tmp,ik;
  PetscScalar    stmp;

  PetscFunctionBegin;
  if (n<8) {
    for (k=0; k<n; k++) {
      ik = i[k];
      for (j=k+1; j<n; j++) {
	if (ik > i[j]) {
	  SWAP2IntScalar(i[k],i[j],Ii[k],Ii[j],tmp,stmp);
	  ik = i[k];
	}
      }
    }
  } else {
    ierr = PetscSortIntWithScalarArray_Private(i,Ii,n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


