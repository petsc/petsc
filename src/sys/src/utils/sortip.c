

/*
   This file contains routines for sorting "common" objects.
   So far, this is integers and reals.  Values are sorted in-place.
   These are provided because the general sort routines incure a great deal
   of overhead in calling the comparision routines.

   In addition, we'll want to provide a routine that generates the permutation
   vector for integer and double orders.

   The word "register"  in this code is used to identify data that is not
   aliased.  For some compilers, this can cause the compiler to fail to
   place inner-loop variables into registers.
 */
#include "petsc.h"
#include "sys.h"

#define SWAP(a,b,t) {t=a;a=b;b=t;}

int SYiIqsort(int *,int), SYiDqsort(double*,int), SYiIqsortPerm(int*,int*,int);

/*@
  SYIsort - sort an array of integer inplace in increasing order

  Input Parameters:
. n  - number of values
. i  - array of integers
@*/
int SYIsort( int n, int *i )
{
register int j, k, tmp, ik;

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
    }
else 
    SYiIqsort(i,n-1);
  return 0;
}

/* A simple version of quicksort; taken from Kernighan and Ritchie, page 87.
   Assumes 0 origin for v, number of elements = right+1 (right is index of
   right-most member). */
int SYiIqsort(int *v,int right)
{
  int          tmp;
  register int i, vl, last;
  if (right <= 1) {
      if (right == 1) {
	  if (v[0] > v[1]) SWAP(v[0],v[1],tmp);
	  }
      return 0;
      }
  SWAP(v[0],v[right/2],tmp);
  vl   = v[0];
  last = 0;
  for ( i=1; i<=right; i++ ) {
    if (v[i] < vl ) {last++; SWAP(v[last],v[i],tmp);}
  }
  SWAP(v[0],v[last],tmp);
  SYiIqsort(v,last-1);
  SYiIqsort(v+last+1,right-(last+1));
  return 0;
}

/*@
  SYDsort - sort an array of doubles inplace in increasing order

  Input Parameters:
. n  - number of values
. v  - array of doubles
@*/
int SYDsort(int n, double *v )
{
register int    j, k;
register double tmp, vk;

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
    }
else
    SYiDqsort( v, n-1 );
  return 0;
}
   
/* A simple version of quicksort; taken from Kernighan and Ritchie, page 87 */
int SYiDqsort(double *v,int right)
{
  register int    i,last;
  register double vl;
  double          tmp;
  
  if (right <= 1) {
      if (right == 1) {
	  if (v[0] > v[1]) SWAP(v[0],v[1],tmp);
	  }
      return 0;
      }
  SWAP(v[0],v[right/2],tmp);
  vl   = v[0];
  last = 0;
  for ( i=1; i<=right; i++ ) {
    if (v[i] < vl ) {last++; SWAP(v[last],v[i],tmp);}
  }
  SWAP(v[0],v[last],tmp);
  SYiDqsort(v,last-1);
  SYiDqsort(v+last+1,right-(last+1));
  return 0;
}

/*@
   Compute the permutation of values that gives a sorted sequence

   Input Parameters:
.  n  - number of values to sort
.  i  - values to sort
.  idx - permutation array.  Must be initialized to 0:n-1 on input.

   Notes: 
   i is unchanged on output.
 @*/
int SYIsortperm(int n, int *i, int *idx )
{
register int j, k, tmp, ik;

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
    }
else 
    SYiIqsortPerm(i,idx,n-1);
  return 0;
}

int SYiIqsortPerm(int *v,int *vdx,int right)
{
  int          tmp;
  register int i, vl, last;
  if (right <= 1) {
      if (right == 1) {
	  if (v[vdx[0]] > v[vdx[1]]) SWAP(vdx[0],vdx[1],tmp);
	  }
      return 0;
      }
  SWAP(vdx[0],vdx[right/2],tmp);
  vl   = v[vdx[0]];
  last = 0;
  for ( i=1; i<=right; i++ ) {
    if (v[vdx[i]] < vl ) {last++; SWAP(vdx[last],vdx[i],tmp);}
  }
  SWAP(vdx[0],vdx[last],tmp);
  SYiIqsortPerm(v,vdx,last-1);
  SYiIqsortPerm(v,vdx+last+1,right-(last+1));
  return 0;
}
