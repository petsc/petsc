
/* $Id: bvec1.c,v 1.13 1996/03/19 21:23:01 bsmith Exp curfman $ */

/*
   Defines the BLAS based vector operations. Code shared by parallel
  and sequential vectors.
*/

#include <math.h>
#include "vecimpl.h" 
#include "dvecimpl.h" 
#include "pinclude/plapack.h"

static int VecDot_Seq(Vec xin, Vec yin,Scalar *z )
{
  Vec_Seq *x = (Vec_Seq *)xin->data,*y = (Vec_Seq *)yin->data;
  int     one = 1;
#if defined(PETSC_COMPLEX)
  /* cannot use BLAS dot for complex because compiler/linker is 
     not happy about returning a double complex */
  int    i;
  Scalar sum = 0.0, *xa = x->array, *ya = y->array;
  for ( i=0; i<x->n; i++ ) {
    sum += xa[i]*conj(ya[i]);
  }
  *z = sum;
#else
  *z = BLdot_( &x->n, x->array, &one, y->array, &one );
#endif
  PLogFlops(2*x->n-1);
  return 0;
}

static int VecScale_Seq( Scalar *alpha,Vec xin )
{
  Vec_Seq *x = (Vec_Seq *) xin->data;
  int     one = 1;
  BLscal_( &x->n, alpha, x->array, &one );
  PLogFlops(x->n);
  return 0;
}

static int VecCopy_Seq(Vec xin, Vec yin )
{
  Vec_Seq *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int     one = 1;
  BLcopy_( &x->n, x->array, &one, y->array, &one );
  return 0;
}

static int VecSwap_Seq(  Vec xin,Vec yin )
{
  Vec_Seq *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int     one = 1;
  BLswap_( &x->n, x->array, &one, y->array, &one );
  return 0;
}

static int VecAXPY_Seq(  Scalar *alpha, Vec xin, Vec yin )
{
  Vec_Seq  *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int      one = 1;
  BLaxpy_( &x->n, alpha, x->array, &one, y->array, &one );
  PLogFlops(2*x->n);
  return 0;
}

static int VecAXPBY_Seq(Scalar *alpha, Scalar *beta,Vec xin, Vec yin)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int      n = x->n, i;
  Scalar   *xx = x->array, *yy = y->array, a = *alpha, b = *beta;

  for ( i=0; i<n; i++ ) {
    yy[i] = a*xx[i] + b*yy[i];
  }

  PLogFlops(3*x->n);
  return 0;
}

