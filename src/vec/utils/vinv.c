
#include "vec.h"   /*I "vec.h" I*/
#include "vecimpl.h"

/*@
     VecReciprocal - Replaces each component in a vector by 
      its reciprocal.

  Input Parameters:
.   v - the vector 
@*/
int VecReciprocal(Vec v)
{
  int    ierr, i,n;
  Scalar *x;
  VALIDHEADER(v,VEC_COOKIE);
  if (ierr = VecGetLocalSize(v,&n)) return ierr;
  if (ierr = VecGetArray(v,&x)) return ierr;
  for ( i=0; i<n; i++ ) {
    if (x[i] != 0.0) x[i] = 1.0/x[i];
  }
  return 0;
}

/*@
     VecSum - Sum of all the components of a vector.

  Input Parameters:
.   v - the vector 

  Output Parameters:
.   sum - the result
@*/
int VecSum(Vec v,Scalar *sum)
{
  int    ierr, i,n;
  Scalar *x,lsum = 0.0;
  VALIDHEADER(v,VEC_COOKIE);
  if (ierr = VecGetLocalSize(v,&n)) return ierr;
  if (ierr = VecGetArray(v,&x)) return ierr;
  for ( i=0; i<n; i++ ) {
    lsum += x[i];
  }
  *sum = lsum;
  return 0;
}

/*@
     VecShift - Shift all of the components of a vector.

  Input Parameters:
.   v - the vector 
.   sum - the shift

@*/
int VecShift(Scalar *sum,Vec v)
{
  int    ierr, i,n;
  Scalar *x,lsum = *sum;
  VALIDHEADER(v,VEC_COOKIE);
  if (ierr = VecGetLocalSize(v,&n)) return ierr;
  if (ierr = VecGetArray(v,&x)) return ierr;
  for ( i=0; i<n; i++ ) {
    x[i] += lsum;
  }
  return 0;
}
