
#include "vec.h"   /*I "vec.h" I*/
#include "vecimpl.h"

/*@
     VecReciprocal - replaces each component in a vector by 
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
