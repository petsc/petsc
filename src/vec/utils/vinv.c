#ifndef lint
static char vcid[] = "$Id: vinv.c,v 1.6 1995/03/06 03:58:54 bsmith Exp curfman $";
#endif

#include "vec.h"   /*I "vec.h" I*/
#include "vecimpl.h"

/*@
   VecReciprocal - Replaces each component of a vector by its reciprocal.

   Input Parameter:
.  v - the vector 

   Output Parameter:
.  v - the vector reciprocal

.keywords: vector, reciprocal
@*/
int VecReciprocal(Vec v)
{
  int    ierr, i,n;
  Scalar *x;
  VALIDHEADER(v,VEC_COOKIE);
  if ((ierr = VecGetLocalSize(v,&n))) return ierr;
  if ((ierr = VecGetArray(v,&x))) return ierr;
  for ( i=0; i<n; i++ ) {
    if (x[i] != 0.0) x[i] = 1.0/x[i];
  }
  return 0;
}

/*@
   VecSum - Computes the sum of all the components of a vector.

   Input Parameter:
.  v - the vector 

   Output Parameter:
.  sum - the result

.keywords: vector, sum

.seealso: VecASum()
@*/
int VecSum(Vec v,Scalar *sum)
{
  int    ierr, i,n;
  Scalar *x,lsum = 0.0;
  VALIDHEADER(v,VEC_COOKIE);
  if ((ierr = VecGetLocalSize(v,&n))) return ierr;
  if ((ierr = VecGetArray(v,&x))) return ierr;
  for ( i=0; i<n; i++ ) {
    lsum += x[i];
  }
  *sum = lsum;
  return 0;
}

/*@
   VecShift - Shifts all of the components of a vector by computing
   x[i] <- x[i] + shift.

   Input Parameters:
.  v - the vector 
.  sum - the shift

   Output Parameter:
.  v - the shifted vector 

.keywords: vector, shift
@*/
int VecShift(Scalar *shift,Vec v)
{
  int    ierr, i,n;
  Scalar *x,lsum = *shift;
  VALIDHEADER(v,VEC_COOKIE);
  if ((ierr = VecGetLocalSize(v,&n))) return ierr;
  if ((ierr = VecGetArray(v,&x))) return ierr;
  for ( i=0; i<n; i++ ) {
    x[i] += lsum;
  }
  return 0;
}
