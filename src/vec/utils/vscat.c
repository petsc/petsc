
#include "vecimpl.h"    /*I "vec.h" I*/
#include "impls/dvecimpl.h"
#include "impls/mpi/pvecimpl.h"

/*@
     VecScatterBegin  -  Scatters from one vector into another.
                         This is far more general than the usual
                         scatter. Depending on ix, iy it can be 
                         a gather or a scatter or a combination.
                         If x is a parallel vector and y sequential
                         it can serve to gather values to a single
                         processor. Similar if y is paralle and 
                         x sequential it can scatter from one processor
                         to many.

  Input Parameters:
.  x - vector to scatter from
.  ix - indices of elements in x to take
.  iy - indices of locations in y to insert 

  Output Parameters:
.  y - vector to scatter to 

  Notes:
.   y[iy[i]] = x[ix[i]], for i=0,...,ni-1
@*/
int VecScatterBegin(Vec x,IS ix,Vec y,IS iy,ISScatterCtx *ctx)
{
  VALIDHEADER(y,VEC_COOKIE);
  if (x->ops->scatterbegin) return (*x->ops->scatterbegin)( x, ix, y, iy,ctx);
}
/*@
     VecScatterEnd  -  End scatter from one vector into another.
            Call after call to VecScatterBegin().

  Input Parameters:
.  x - vector to scatter from
.  ix - indices of elements in x to take
.  iy - indices of locations in y to insert 

  Output Parameters:
.  y - vector to scatter to 

  Notes:
.   y[iy[i]] = x[ix[i]], for i=0,...,ni-1
@*/
int VecScatterEnd(Vec x,IS ix,Vec y,IS iy,ISScatterCtx *ctx)
{
  VALIDHEADER(y,VEC_COOKIE);
  if (x->ops->scatterend) return (*x->ops->scatterend)( x, ix, y, iy,ctx);
}

/*@
     VecScatterAddBegin  -  Scatters from one vector into another.
                            See VecScatterBegin().
  Input Parameters:
.  x - vector to scatter from
.  ix - indices of elements in x to take
.  iy - indices of locations in y to insert 

  Output Parameters:
.  y - vector to scatter to 

  Notes:
.   y[iy[i]] += x[ix[i]], for i=0,...,ni-1
@*/
int VecScatterAddBegin(Vec x,IS ix,Vec y,IS iy,ISScatterCtx *ctx)
{
  VALIDHEADER(y,VEC_COOKIE);
  return (*y->ops->scatteraddbegin)( x, ix, y, iy,ctx);
}

/*@
     VecScatterAddEnd  -  End scatter from one vector into another.
            Call after call to VecScatterAddBegin().

  Input Parameters:
.  x - vector to scatter from
.  ix - indices of elements in x to take
.  iy - indices of locations in y to insert 

  Output Parameters:
.  y - vector to scatter to 

  Notes:
.   y[iy[i]] += x[ix[i]], for i=0,...,ni-1
@*/
int VecScatterAddEnd(Vec x,IS ix,Vec y,IS iy,ISScatterCtx *ctx)
{
  VALIDHEADER(y,VEC_COOKIE);
  return (*y->ops->scatteraddend)( x, ix, y, iy,ctx);
}
