
/*
   This file contains routines for sorting integers. Values are sorted in place.
 */
#include <petsc/private/petscimpl.h>                /*I  "petscsys.h"  I*/

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

/*
   A simple version of quicksort; taken from Kernighan and Ritchie, page 87.
   Assumes 0 origin for v, number of elements = right+1 (right is index of
   right-most member).
*/
static void PetscSortInt_Private(PetscInt *v,PetscInt right)
{
  PetscInt i,j,pivot,tmp;

  if (right <= 1) {
    if (right == 1) {
      if (v[0] > v[1]) SWAP(v[0],v[1],tmp);
    }
    return;
  }
  i = MEDIAN(v,right);          /* Choose a pivot */
  SWAP(v[0],v[i],tmp);          /* Move it out of the way */
  pivot = v[0];
  for (i=0,j=right+1;; ) {
    while (++i < j && v[i] <= pivot) ; /* Scan from the left */
    while (v[--j] > pivot) ;           /* Scan from the right */
    if (i >= j) break;
    SWAP(v[i],v[j],tmp);
  }
  SWAP(v[0],v[j],tmp);          /* Put pivot back in place. */
  PetscSortInt_Private(v,j-1);
  PetscSortInt_Private(v+j+1,right-(j+1));
}

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
  PetscInt j,k,tmp,ik;

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
  } else PetscSortInt_Private(i,n-1);
  PetscFunctionReturn(0);
}

/*@
   PetscSortedRemoveDupsInt - Removes all duplicate entries of a sorted input array

   Not Collective

   Input Parameters:
+  n  - number of values
-  ii  - sorted array of integers

   Output Parameter:
.  n - number of non-redundant values

   Level: intermediate

   Concepts: sorting^ints

.seealso: PetscSortInt()
@*/
PetscErrorCode  PetscSortedRemoveDupsInt(PetscInt *n,PetscInt ii[])
{
  PetscInt i,s = 0,N = *n, b = 0;

  PetscFunctionBegin;
  for (i=0; i<N-1; i++) {
    if (PetscUnlikely(ii[b+s+1] < ii[b])) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Input array is not sorted");
    if (ii[b+s+1] != ii[b]) {
      ii[b+1] = ii[b+s+1]; b++;
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
-  ii  - array of integers

   Output Parameter:
.  n - number of non-redundant values

   Level: intermediate

   Concepts: sorting^ints

.seealso: PetscSortReal(), PetscSortIntWithPermutation(), PetscSortInt(), PetscSortedRemoveDupsInt()
@*/
PetscErrorCode  PetscSortRemoveDupsInt(PetscInt *n,PetscInt ii[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSortInt(*n,ii);CHKERRQ(ierr);
  ierr = PetscSortedRemoveDupsInt(n,ii);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

/*@
  PetscFindMPIInt - Finds MPI integer in a sorted array of integers

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
PetscErrorCode PetscFindMPIInt(PetscMPIInt key, PetscInt n, const PetscMPIInt ii[], PetscInt *loc)
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
PetscErrorCode  PetscSortIntWithArrayPair(PetscInt n,PetscInt L[],PetscInt J[], PetscInt K[])
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
  for (i=0,j=right+1;; ) {
    while (++i < j && v[i] <= pivot) ; /* Scan from the left */
    while (v[--j] > pivot) ;           /* Scan from the right */
    if (i >= j) break;
    SWAP(v[i],v[j],tmp);
  }
  SWAP(v[0],v[j],tmp);          /* Put pivot back in place. */
  PetscSortMPIInt_Private(v,j-1);
  PetscSortMPIInt_Private(v+j+1,right-(j+1));
}

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
  PetscInt    j,k;
  PetscMPIInt tmp,ik;

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
  } else PetscSortMPIInt_Private(i,n-1);
  PetscFunctionReturn(0);
}

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
    if (ii[b+s+1] != ii[b]) {
      ii[b+1] = ii[b+s+1]; b++;
    } else s++;
  }
  *n = N - s;
  PetscFunctionReturn(0);
}

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

#define SWAP2IntData(a,b,c,d,t,td,siz)          \
  do {                                          \
  PetscErrorCode _ierr;                         \
  t=a;a=b;b=t;                                  \
  _ierr = PetscMemcpy(td,c,siz);CHKERRQ(_ierr); \
  _ierr = PetscMemcpy(c,d,siz);CHKERRQ(_ierr);  \
  _ierr = PetscMemcpy(d,td,siz);CHKERRQ(_ierr); \
} while(0)

/*
   Modified from PetscSortIntWithArray_Private().
*/
static PetscErrorCode PetscSortIntWithDataArray_Private(PetscInt *v,char *V,PetscInt right,size_t size,void *work)
{
  PetscErrorCode ierr;
  PetscInt       i,vl,last,tmp;

  PetscFunctionBegin;
  if (right <= 1) {
    if (right == 1) {
      if (v[0] > v[1]) SWAP2IntData(v[0],v[1],V,V+size,tmp,work,size);
    }
    PetscFunctionReturn(0);
  }
  SWAP2IntData(v[0],v[right/2],V,V+size*(right/2),tmp,work,size);
  vl   = v[0];
  last = 0;
  for (i=1; i<=right; i++) {
    if (v[i] < vl) {last++; SWAP2IntData(v[last],v[i],V+size*last,V+size*i,tmp,work,size);}
  }
  SWAP2IntData(v[0],v[last],V,V + size*last,tmp,work,size);
  ierr = PetscSortIntWithDataArray_Private(v,V,last-1,size,work);CHKERRQ(ierr);
  ierr = PetscSortIntWithDataArray_Private(v+last+1,V+size*(last+1),right-(last+1),size,work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSortIntWithDataArray - Sorts an array of integers in place in increasing order;
       changes a second array to match the sorted first INTEGER array.  Unlike other sort routines, the user must
       provide workspace (the size of an element in the data array) to use when sorting.

   Not Collective

   Input Parameters:
+  n  - number of values
.  i  - array of integers
.  Ii - second array of data
.  size - sizeof elements in the data array in bytes
-  work - workspace of "size" bytes used when sorting

   Level: intermediate

   Concepts: sorting^ints with array

.seealso: PetscSortReal(), PetscSortIntPermutation(), PetscSortInt(), PetscSortIntWithArray()
@*/
PetscErrorCode  PetscSortIntWithDataArray(PetscInt n,PetscInt i[],void *Ii,size_t size,void *work)
{
  char           *V = (char *) Ii;
  PetscErrorCode ierr;
  PetscInt       j,k,tmp,ik;

  PetscFunctionBegin;
  if (n<8) {
    for (k=0; k<n; k++) {
      ik = i[k];
      for (j=k+1; j<n; j++) {
        if (ik > i[j]) {
          SWAP2IntData(i[k],i[j],V+size*k,V+size*j,tmp,work,size);
          ik = i[k];
        }
      }
    }
  } else {
    ierr = PetscSortIntWithDataArray_Private(i,V,n-1,size,work);CHKERRQ(ierr);
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

   Concepts: merging^arrays

.seealso: PetscSortReal(), PetscSortIntPermutation(), PetscSortInt(), PetscSortIntWithArray()
@*/
PetscErrorCode  PetscMergeIntArray(PetscInt an,const PetscInt aI[], PetscInt bn, const PetscInt bI[],  PetscInt *n, PetscInt **L)
{
  PetscErrorCode ierr;
  PetscInt       *L_ = *L, ak, bk, k;

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
    ierr = PetscMemcpy(L_+k,aI+ak,(an-ak)*sizeof(PetscInt));CHKERRQ(ierr);
    k   += (an-ak);
  }
  if (bk < bn) {
    ierr = PetscMemcpy(L_+k,bI+bk,(bn-bk)*sizeof(PetscInt));CHKERRQ(ierr);
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

   Notes: if L or J point to non-null arrays then this routine will assume they are of the approproate size and use them, otherwise this routine will allocate space for them 
   Level: intermediate

   Concepts: merging^arrays

.seealso: PetscSortReal(), PetscSortIntPermutation(), PetscSortInt(), PetscSortIntWithArray()
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
    ierr = PetscMemcpy(L_+k,aI+ak,(an-ak)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(J_+k,aJ+ak,(an-ak)*sizeof(PetscInt));CHKERRQ(ierr);
    k   += (an-ak);
  }
  if (bk < bn) {
    ierr = PetscMemcpy(L_+k,bI+bk,(bn-bk)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(J_+k,bJ+bk,(bn-bk)*sizeof(PetscInt));CHKERRQ(ierr);
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

   Concepts: merging^arrays

.seealso: PetscSortReal(), PetscSortIntPermutation(), PetscSortInt(), PetscSortIntWithArray()
@*/
PetscErrorCode PetscMergeMPIIntArray(PetscInt an,const PetscMPIInt aI[],PetscInt bn,const PetscMPIInt bI[],PetscInt *n,PetscMPIInt **L)
{
  PetscErrorCode ierr;
  PetscInt       ai,bi,k;

  PetscFunctionBegin;
  if (!*L) {ierr = PetscMalloc1((an+bn),L);CHKERRQ(ierr);}
  for (ai=0,bi=0,k=0; ai<an || bi<bn; ) {
    PetscInt t = -1;
    for ( ; ai<an && (!bn || aI[ai] <= bI[bi]); ai++) (*L)[k++] = t = aI[ai];
    for ( ; bi<bn && bI[bi] == t; bi++);
    for ( ; bi<bn && (!an || bI[bi] <= aI[ai]); bi++) (*L)[k++] = t = bI[bi];
    for ( ; ai<an && aI[ai] == t; ai++);
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

   Notes: This code is not currently used

.seealso: PetscSortReal(), PetscSortIntWithPermutation()
@*/
PetscErrorCode  PetscProcessTree(PetscInt n,const PetscBool mask[],const PetscInt parentid[],PetscInt *Nlevels,PetscInt **Level,PetscInt **Levelcnt,PetscInt **Idbylevel,PetscInt **Column)
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
    ierr  = PetscMemcpy(idbylevel+tcnt,workid,cnt*sizeof(PetscInt));CHKERRQ(ierr);
    tcnt += cnt;
  }
  if (tcnt != nmask) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent count of unmasked nodes");
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
