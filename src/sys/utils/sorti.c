
/*
   This file contains routines for sorting integers. Values are sorted in place.
 */
#include <petscsys.h>                /*I  "petscsys.h"  I*/

#define SWAP(a,b,t) {t=a;a=b;b=t;}

#define MEDIAN3(v,a,b,c)                        \
  (v[a]<v[b]                                    \
   ? (v[b]<v[c]                                 \
      ? b                                       \
      : (v[a]<v[c] ? c : a))                    \
   : (v[c]<v[b]                                 \
      ? b                                       \
      : (v[a]<v[c] ? a : c)))

#define MEDIAN(v,right) MEDIAN3(v,right/4,right/2,right/4*3)

/* -----------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "PetscSortInt_Private"
/*
   A simple version of quicksort; taken from Kernighan and Ritchie, page 87.
   Assumes 0 origin for v, number of elements = right+1 (right is index of
   right-most member).
*/
static void PetscSortInt_Private(PetscInt *v,PetscInt right)
{
  PetscInt       i,j,pivot,tmp;

  if (right <= 1) {
    if (right == 1) {
      if (v[0] > v[1]) SWAP(v[0],v[1],tmp);
    }
    return;
  }
  i = MEDIAN(v,right);          /* Choose a pivot */
  SWAP(v[0],v[i],tmp);          /* Move it out of the way */
  pivot = v[0];
  for (i=0,j=right+1;;) {
    while (++i < j && v[i] <= pivot); /* Scan from the left */
    while (v[--j] > pivot);           /* Scan from the right */
    if (i >= j) break;
    SWAP(v[i],v[j],tmp);
  }
  SWAP(v[0],v[j],tmp);          /* Put pivot back in place. */
  PetscSortInt_Private(v,j-1);
  PetscSortInt_Private(v+j+1,right-(j+1));
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
PetscErrorCode  PetscSortInt(PetscInt n,PetscInt i[])
{
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
    PetscSortInt_Private(i,n-1);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSortRemoveDupsInt"
/*@
   PetscSortRemoveDupsInt - Sorts an array of integers in place in increasing order removes all duplicate entries

   Not Collective

   Input Parameters:
+  n  - number of values
-  ii  - array of integers

   Output Parameter:
.  n - number of non-redundant values

   Level: intermediate

   Concepts: sorting^ints

.seealso: PetscSortReal(), PetscSortIntWithPermutation(), PetscSortInt()
@*/
PetscErrorCode  PetscSortRemoveDupsInt(PetscInt *n,PetscInt ii[])
{
  PetscErrorCode ierr;
  PetscInt       i,s = 0,N = *n, b = 0;

  PetscFunctionBegin;
  ierr = PetscSortInt(N,ii);CHKERRQ(ierr);
  for (i=0; i<N-1; i++) {
    if (ii[b+s+1] != ii[b]) {ii[b+1] = ii[b+s+1]; b++;}
    else s++;
  }
  *n = N - s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFindInt"
/*@
  PetscFindInt - Finds integer in a sorted array of integers

   Not Collective

   Input Parameters:
+  key - the integer to locate
.  n   - number of values in the array
-  ii  - array of integers

   Output Parameter:
.  loc - the location if found, otherwise -(slot+1) where slot is the place the value would go

   Level: intermediate

   Concepts: sorting^ints

.seealso: PetscSortInt(), PetscSortIntWithArray(), PetscSortRemoveDupsInt()
@*/
PetscErrorCode PetscFindInt(PetscInt key, PetscInt n, const PetscInt ii[], PetscInt *loc)
{
  PetscInt lo = 0,hi = n;

  PetscFunctionBegin;
  PetscValidPointer(loc,4);
  if (!n) {*loc = -1; PetscFunctionReturn(0);}
  PetscValidPointer(ii,3);
  while (hi - lo > 1) {
    PetscInt mid = lo + (hi - lo)/2;
    if (key < ii[mid]) hi = mid;
    else               lo = mid;
  }
  *loc = key == ii[lo] ? lo : -(lo + (key > ii[lo]) + 1);
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
PetscErrorCode  PetscSortIntWithArray(PetscInt n,PetscInt i[],PetscInt Ii[])
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


#define SWAP3(a,b,c,d,e,f,t) {t=a;a=b;b=t;t=c;c=d;d=t;t=e;e=f;f=t;}

#undef __FUNCT__
#define __FUNCT__ "PetscSortIntWithArrayPair_Private"
/*
   A simple version of quicksort; taken from Kernighan and Ritchie, page 87.
   Assumes 0 origin for v, number of elements = right+1 (right is index of
   right-most member).
*/
static PetscErrorCode PetscSortIntWithArrayPair_Private(PetscInt *L,PetscInt *J, PetscInt *K, PetscInt right)
{
  PetscErrorCode ierr;
  PetscInt       i,vl,last,tmp;

  PetscFunctionBegin;
  if (right <= 1) {
    if (right == 1) {
      if (L[0] > L[1]) SWAP3(L[0],L[1],J[0],J[1],K[0],K[1],tmp);
    }
    PetscFunctionReturn(0);
  }
  SWAP3(L[0],L[right/2],J[0],J[right/2],K[0],K[right/2],tmp);
  vl   = L[0];
  last = 0;
  for (i=1; i<=right; i++) {
    if (L[i] < vl) {last++; SWAP3(L[last],L[i],J[last],J[i],K[last],K[i],tmp);}
  }
  SWAP3(L[0],L[last],J[0],J[last],K[0],K[last],tmp);
  ierr = PetscSortIntWithArrayPair_Private(L,J,K,last-1);CHKERRQ(ierr);
  ierr = PetscSortIntWithArrayPair_Private(L+last+1,J+last+1,K+last+1,right-(last+1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSortIntWithArrayPair"
/*@
   PetscSortIntWithArrayPair - Sorts an array of integers in place in increasing order;
       changes a pair of integer arrays to match the sorted first array.

   Not Collective

   Input Parameters:
+  n  - number of values
.  I  - array of integers
.  J  - second array of integers (first array of the pair)
-  K  - third array of integers  (second array of the pair)

   Level: intermediate

   Concepts: sorting^ints with array pair

.seealso: PetscSortReal(), PetscSortIntPermutation(), PetscSortIntWithArray()
@*/
PetscErrorCode  PetscSortIntWithArrayPair(PetscInt n,PetscInt *L,PetscInt *J, PetscInt *K)
{
  PetscErrorCode ierr;
  PetscInt       j,k,tmp,ik;

  PetscFunctionBegin;
  if (n<8) {
    for (k=0; k<n; k++) {
      ik = L[k];
      for (j=k+1; j<n; j++) {
	if (ik > L[j]) {
	  SWAP3(L[k],L[j],J[k],J[j],K[k],K[j],tmp);
	  ik = L[k];
	}
      }
    }
  } else {
    ierr = PetscSortIntWithArrayPair_Private(L,J,K,n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSortMPIInt_Private"
/*
   A simple version of quicksort; taken from Kernighan and Ritchie, page 87.
   Assumes 0 origin for v, number of elements = right+1 (right is index of
   right-most member).
*/
static void PetscSortMPIInt_Private(PetscMPIInt *v,PetscInt right)
{
  PetscInt          i,j;
  PetscMPIInt       pivot,tmp;

  if (right <= 1) {
    if (right == 1) {
      if (v[0] > v[1]) SWAP(v[0],v[1],tmp);
    }
    return;
  }
  i = MEDIAN(v,right);          /* Choose a pivot */
  SWAP(v[0],v[i],tmp);          /* Move it out of the way */
  pivot = v[0];
  for (i=0,j=right+1;;) {
    while (++i < j && v[i] <= pivot); /* Scan from the left */
    while (v[--j] > pivot);           /* Scan from the right */
    if (i >= j) break;
    SWAP(v[i],v[j],tmp);
  }
  SWAP(v[0],v[j],tmp);          /* Put pivot back in place. */
  PetscSortMPIInt_Private(v,j-1);
  PetscSortMPIInt_Private(v+j+1,right-(j+1));
}

#undef __FUNCT__
#define __FUNCT__ "PetscSortMPIInt"
/*@
   PetscSortMPIInt - Sorts an array of MPI integers in place in increasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  i  - array of integers

   Level: intermediate

   Concepts: sorting^ints

.seealso: PetscSortReal(), PetscSortIntWithPermutation()
@*/
PetscErrorCode  PetscSortMPIInt(PetscInt n,PetscMPIInt i[])
{
  PetscInt       j,k;
  PetscMPIInt    tmp,ik;

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
    PetscSortMPIInt_Private(i,n-1);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSortRemoveDupsMPIInt"
/*@
   PetscSortRemoveDupsMPIInt - Sorts an array of MPI integers in place in increasing order removes all duplicate entries

   Not Collective

   Input Parameters:
+  n  - number of values
-  ii  - array of integers

   Output Parameter:
.  n - number of non-redundant values

   Level: intermediate

   Concepts: sorting^ints

.seealso: PetscSortReal(), PetscSortIntWithPermutation(), PetscSortInt()
@*/
PetscErrorCode  PetscSortRemoveDupsMPIInt(PetscInt *n,PetscMPIInt ii[])
{
  PetscErrorCode ierr;
  PetscInt       i,s = 0,N = *n, b = 0;

  PetscFunctionBegin;
  ierr = PetscSortMPIInt(N,ii);CHKERRQ(ierr);
  for (i=0; i<N-1; i++) {
    if (ii[b+s+1] != ii[b]) {ii[b+1] = ii[b+s+1]; b++;}
    else s++;
  }
  *n = N - s;
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
PetscErrorCode  PetscSortMPIIntWithArray(PetscMPIInt n,PetscMPIInt i[],PetscMPIInt Ii[])
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
PetscErrorCode  PetscSortIntWithScalarArray(PetscInt n,PetscInt i[],PetscScalar Ii[])
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



#undef __FUNCT__
#define __FUNCT__ "PetscMergeIntArrayPair"
/*@
   PetscMergeIntArrayPair -     Merges two SORTED integer arrays along with an additional array of integers.
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
.  I   - merged sorted array
-  J   - merged additional array

   Level: intermediate

   Concepts: merging^arrays

.seealso: PetscSortReal(), PetscSortIntPermutation(), PetscSortInt(), PetscSortIntWithArray()
@*/
PetscErrorCode  PetscMergeIntArrayPair(PetscInt an,const PetscInt *aI, const PetscInt *aJ, PetscInt bn, const PetscInt *bI, const PetscInt *bJ, PetscInt *n, PetscInt **L, PetscInt **J)
{
  PetscErrorCode ierr;
  PetscInt n_, *L_ = *L, *J_= *J, ak, bk, k;

  n_ = an + bn;
  *n = n_;
  if (!L_) {
    ierr = PetscMalloc(n_*sizeof(PetscInt), L); CHKERRQ(ierr);
    L_ = *L;
  }
  if (!J_){
    ierr = PetscMalloc(n_*sizeof(PetscInt), &J_); CHKERRQ(ierr);
    J_ = *J;
  }
  k = ak = bk = 0;
  while(ak < an && bk < bn) {
    if (aI[ak] <= bI[bk]) {
      L_[k] = aI[ak];
      J_[k] = aJ[ak];
      ++ak;
      ++k;
    }
    else {
      L_[k] = bI[bk];
      J_[k] = bJ[bk];
      ++bk;
      ++k;
    }
  }
  if (ak < an) {
    ierr = PetscMemcpy(L_+k,aI+ak,(an-ak)*sizeof(PetscInt)); CHKERRQ(ierr);
    ierr = PetscMemcpy(J_+k,aJ+ak,(an-ak)*sizeof(PetscInt)); CHKERRQ(ierr);
    k += (an-ak);
  }
  if (bk < bn) {
    ierr = PetscMemcpy(L_+k,bI+bk,(bn-bk)*sizeof(PetscInt)); CHKERRQ(ierr);
    ierr = PetscMemcpy(J_+k,bJ+bk,(bn-bk)*sizeof(PetscInt)); CHKERRQ(ierr);
    k += (bn-bk);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscProcessTree"
/*@
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

   Level: intermediate


.seealso: PetscSortReal(), PetscSortIntWithPermutation()
@*/
PetscErrorCode  PetscProcessTree(PetscInt n,const PetscBool  mask[],const PetscInt parentid[],PetscInt *Nlevels,PetscInt **Level,PetscInt **Levelcnt,PetscInt **Idbylevel,PetscInt **Column)
{
  PetscInt       i,j,cnt,nmask = 0,nlevels = 0,*level,*levelcnt,levelmax = 0,*workid,*workparentid,tcnt = 0,*idbylevel,*column;
  PetscErrorCode ierr;
  PetscBool      done = PETSC_FALSE;

  PetscFunctionBegin;
  if (!mask[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Mask of 0th location must be set");
  for (i=0; i<n; i++) {
    if (mask[i]) continue;
    if (parentid[i]  == i) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Node labeled as own parent");
    if (parentid[i] && mask[parentid[i]]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Parent is masked");
  }


  for (i=0; i<n; i++) {
    if (!mask[i]) nmask++;
  }

  /* determine the level in the tree of each node */
  ierr = PetscMalloc(n*sizeof(PetscInt),&level);CHKERRQ(ierr);
  ierr = PetscMemzero(level,n*sizeof(PetscInt));CHKERRQ(ierr);
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
  ierr = PetscMalloc(nlevels*sizeof(PetscInt),&levelcnt);CHKERRQ(ierr);
  ierr = PetscMemzero(levelcnt,nlevels*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (mask[i]) continue;
    levelcnt[level[i]-1]++;
  }
  for (i=0; i<nlevels;i++) {
    levelmax = PetscMax(levelmax,levelcnt[i]);
  }

  /* for each level sort the ids by the parent id */
  ierr = PetscMalloc2(levelmax,PetscInt,&workid,levelmax,PetscInt,&workparentid);CHKERRQ(ierr);
  ierr = PetscMalloc(nmask*sizeof(PetscInt),&idbylevel);CHKERRQ(ierr);
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
    ierr = PetscMemcpy(idbylevel+tcnt,workid,cnt*sizeof(PetscInt));CHKERRQ(ierr);
    tcnt += cnt;
  }
  if (tcnt != nmask) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent count of unmasked nodes");
  ierr = PetscFree2(workid,workparentid);CHKERRQ(ierr);

  /* for each node list its column */
  ierr = PetscMalloc(n*sizeof(PetscInt),&column);CHKERRQ(ierr);
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
