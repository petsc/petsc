/*$Id: sorti.c,v 1.22 2000/01/11 20:59:39 bsmith Exp bsmith $*/
/*
   This file contains routines for sorting integers. Values are sorted in place.


   The word "register"  in this code is used to identify data that is not
   aliased.  For some compilers, marking variables as register can improve 
   the compiler optimizations.
 */
#include "petsc.h"           /*I  "petsc.h"  I*/
#include "sys.h"             /*I  "sys.h"    I*/

#define SWAP(a,b,t) {t=a;a=b;b=t;}

#if defined(PETSC_USE_KNUTH_QUICKSORT)

/* 
   This quick-sort is from George Karypis METIS, and he says it
     originally comes from Donald Earvin Knuth TeX. 

   Reportedly it runs faster on some machines than quicksort; with PETSc 
   we have not seen faster performance, hence it is not used by default.

   Contributed by: Matthew Knepley

*/
#define THRESH      1        /* Threshold for insertion */
#define MTHRESH     6        /* Threshold for median */

#undef __FUNC__
#define  __FUNC__ /*<a name=""></a>*/"TeXqsort_Private" 
static int TeXqsort_Private(int *base,int *max)
{
  int *i;
  int *j;
  int *jj;
  int *mid;
  int c;
  int *tmp;
  int lo;
  int hi;

  PetscFunctionBegin;
  lo = max - base;              /* number of elements as ints */
  do
  {
    mid = base + ((unsigned) lo>>1);
    if (lo >= MTHRESH)
    {
      j = (*base > *mid ? base : mid);
      tmp = max - 1;
      if (*j > *tmp)
      {
        j = (j == base ? mid : base); /* switch to first loser */
        if (*j < *tmp)
          j = tmp;
      }

      if (j != mid)
      {  /* SWAP */
        c = *mid;
        *mid = *j;
        *j = c;
      }
    }

    /* Semi-standard quicksort partitioning/swapping */
    for (i = base,j = max - 1; ;)
    {
      while(i < mid && *i <= *mid)
        i++;
      while (j > mid)
      {
        if (*mid <= *j)
        {
          j--;
          continue;
        }
        tmp = i + 1;    /* value of i after swap */
        if (i == mid)   /* j <-> mid, new mid is j */
          mid = jj = j;
        else            /* i <-> j */
          jj = j--;
        goto swap;
      }

      if (i == mid)
        break;
      else
      {                 /* i <-> mid, new mid is i */
        jj = mid;
        tmp = mid = i;  /* value of i after swap */
        j--;
      }
    swap:
      c   = *i;
      *i  = *jj;
      *jj = c;
      i   = tmp;
    }

    i = (j = mid) + 1;
    if ((lo = j - base) <= (hi = max - i))
    {
      if (lo >= THRESH)
        TeXqsort_Private(base,j);
      base = i;
      lo = hi;
    }
    else
    {
      if (hi >= THRESH)
        TeXqsort_Private(i,max);
      max = j;
    }
  } while (lo >= THRESH);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define  __FUNC__ /*<a name=""></a>*/"PetsciIqsort" 
static int PetsciIqsort(int *base,int right)
{
  int *i;
  int *j;
  int *lo;
  int *hi;
  int *min;
  int c;
  int *max;

  PetscFunctionBegin;
  max = base + right + 1;

  if (right >= THRESH + 1)
  {
    TeXqsort_Private(base,max);
    hi = base + THRESH;
  }
  else
    hi = max;

  for(j = lo = base; lo++ < hi;)
  {
    if (*j > *lo)
      j = lo;
  }
  if (j != base)
  { /* swap j into place */
    c     = *base;
    *base = *j;
    *j    = c;
  }

  for(min = base; (hi = min += 1) < max;)
  {
    while (*(--hi) > *min);
    if ((hi += 1) != min)
    {
      for(lo = min + 1; --lo >= min;)
      {
        c = *lo;
        for(i = j = lo; (j -= 1) >= hi; i = j)
          *i = *j;
        *i = c;
      }
    }
  }
  PetscFunctionReturn(0);
}

#else

/* -----------------------------------------------------------------------*/

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetsciIqsort"
/* 
   A simple version of quicksort; taken from Kernighan and Ritchie, page 87.
   Assumes 0 origin for v, number of elements = right+1 (right is index of
   right-most member). 

   Tests on the IBM 590 show this performing about 10% better then the 
   Knuth quicksort above. Thus this is the default. With the compile flag
   PETSC_USE_KNUTH_QUICKSORT the above sort will be used.

*/
static int PetsciIqsort(int *v,int right)
{
  int i,vl,last,tmp;

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
  PetsciIqsort(v,last-1);
  PetsciIqsort(v+last+1,right-(last+1));
  PetscFunctionReturn(0);
}

#endif

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscSortInt" 
/*@
   PetscSortInt - Sorts an array of integers in place in increasing order.

   Not Collective

   Input Parameters:
+  n  - number of values
-  i  - array of integers

   Level: intermediate

.keywords: sort, integer

.seealso: PetscSortDouble(), PetscSortIntWithPermutation()
@*/
int PetscSortInt(int n,int i[])
{
  int j,k,tmp,ik;

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
  } else PetsciIqsort(i,n-1);
  PetscFunctionReturn(0);
}


