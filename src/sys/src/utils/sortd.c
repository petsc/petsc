#ifndef lint
static char vcid[] = "$Id: sortd.c,v 1.6 1996/04/20 04:19:15 bsmith Exp curfman $";
#endif

/*
   This file contains routines for sorting "common" objects.
   So far, this includes integers and reals.  Values are sorted in place.
   These are provided because the general sort routines incur a great deal
   of overhead in calling the comparision routines.

   The word "register"  in this code is used to identify data that is not
   aliased.  For some compilers, this can cause the compiler to fail to
   place inner-loop variables into registers.
 */
#include "petsc.h"           /*I  "petsc.h"  I*/
#include "sys.h"             /*I  "sys.h"    I*/

#define SWAP(a,b,t) {t=a;a=b;b=t;}
   
/* A simple version of quicksort; taken from Kernighan and Ritchie, page 87 */
static int PetsciDqsort(double *v,int right)
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
  PetsciDqsort(v,last-1);
  PetsciDqsort(v+last+1,right-(last+1));
  return 0;
}

/*@
   PetscSortDouble - Sorts an array of doubles in place in increasing order.

   Input Parameters:
.  n  - number of values
.  v  - array of doubles

.keywords: sort, double

.seealso: PetscSortInt(), PetscSortDoubleWithPermutation()
@*/
int PetscSortDouble(int n,double *v)
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
    PetsciDqsort( v, n-1 );
  return 0;
}

