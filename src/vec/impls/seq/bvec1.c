
/*  cannot have vcid because include in other files */
/*
   Defines the BLAS based vector operations
*/

#include "sys/flog.h"
#include <math.h>
#include "vecimpl.h" 
#include "dvecimpl.h" 
#include "plapack.h"

static int VeiDVBdot(Vec xin, Vec yin,Scalar *z )
{
  DvVector *x = (DvVector *)xin->data,*y = (DvVector *)yin->data;
  int  one = 1;
  *z = BLdot_( &x->n, x->array, &one, y->array, &one );
  return 0;
}

static int VeiDVBnorm(Vec xin,double* z )
{
  DvVector * x = (DvVector *) xin->data;
  int  one = 1;
  *z = BLnrm2_( &x->n, x->array, &one );
  return 0;
}

static int VeiDVBasum( Vec xin, double *z )
{
  DvVector *x = (DvVector *) xin->data;
  int one = 1;
  *z = BLasum_( &x->n, x->array, &one );
  return 0;
}

static int VeiDVBscal( Scalar *alpha,Vec xin )
{
  DvVector *x = (DvVector *) xin->data;
  int one = 1;
  BLscal_( &x->n, alpha, x->array, &one );
  return 0;
}

static int VeiDVBcopy(Vec xin, Vec yin )
{
  DvVector *x = (DvVector *)xin->data, *y = (DvVector *)yin->data;
  int one = 1;
  BLcopy_( &x->n, x->array, &one, y->array, &one );
  return 0;
}

static int VeiDVBswap(  Vec xin,Vec yin )
{
  DvVector *x = (DvVector *)xin->data, *y = (DvVector *)yin->data;
  int  one = 1;
  BLswap_( &x->n, x->array, &one, y->array, &one );
  return 0;
}

static int VeiDVBaxpy(  Scalar *alpha, Vec xin, Vec yin )
{
  DvVector  *x = (DvVector *)xin->data, *y = (DvVector *)yin->data;
  int one = 1;
  BLaxpy_( &x->n, alpha, x->array, &one, y->array, &one );
  return 0;
}

