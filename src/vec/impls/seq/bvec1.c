
/* $Id: dvec2.c,v 1.11 1995/06/07 17:27:28 bsmith Exp $ */

/*
   Defines the BLAS based vector operations
*/

#include <math.h>
#include "vecimpl.h" 
#include "dvecimpl.h" 
#include "plapack.h"

static int VecDot_Blas(Vec xin, Vec yin,Scalar *z )
{
  Vec_Seq *x = (Vec_Seq *)xin->data,*y = (Vec_Seq *)yin->data;
  int  one = 1;
  *z = BLdot_( &x->n, x->array, &one, y->array, &one );
  PLogFlops(2*x->n-1);
  return 0;
}

static int VecAsum_Blas( Vec xin, double *z )
{
  Vec_Seq *x = (Vec_Seq *) xin->data;
  int one = 1;
  *z = BLasum_( &x->n, x->array, &one );
  PLogFlops(x->n-1);
  return 0;
}

static int VecScale_Blas( Scalar *alpha,Vec xin )
{
  Vec_Seq *x = (Vec_Seq *) xin->data;
  int one = 1;
  BLscal_( &x->n, alpha, x->array, &one );
  PLogFlops(x->n);
  return 0;
}

static int VecCopy_Blas(Vec xin, Vec yin )
{
  Vec_Seq *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int one = 1;
  BLcopy_( &x->n, x->array, &one, y->array, &one );
  return 0;
}

static int VecSwap_Blas(  Vec xin,Vec yin )
{
  Vec_Seq *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int  one = 1;
  BLswap_( &x->n, x->array, &one, y->array, &one );
  return 0;
}

static int VecAXPY_Blas(  Scalar *alpha, Vec xin, Vec yin )
{
  Vec_Seq  *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int one = 1;
  BLaxpy_( &x->n, alpha, x->array, &one, y->array, &one );
  PLogFlops(2*x->n);
  return 0;
}

