
/* $Id: bvec1.c,v 1.22 1998/03/16 19:28:19 balay Exp bsmith $ */

/*
   Defines the BLAS based vector operations. Code shared by parallel
  and sequential vectors.
*/

#include <math.h>
#include "src/vec/vecimpl.h" 
#include "src/vec/impls/dvecimpl.h" 
#include "pinclude/blaslapack.h"

#undef __FUNC__  
#define __FUNC__ "VecDot_Seq"
int VecDot_Seq(Vec xin, Vec yin,Scalar *z )
{
  Vec_Seq *x = (Vec_Seq *)xin->data,*y = (Vec_Seq *)yin->data;
#if !defined(USE_PETSC_COMPLEX)
  int     one = 1;
#endif

  PetscFunctionBegin;
#if defined(USE_PETSC_COMPLEX)
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
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecTDot_Seq"
int VecTDot_Seq(Vec xin, Vec yin,Scalar *z )
{
  Vec_Seq *x = (Vec_Seq *)xin->data,*y = (Vec_Seq *)yin->data;
#if !defined(USE_PETSC_COMPLEX)
 int     one = 1;
#endif

  PetscFunctionBegin;
#if defined(USE_PETSC_COMPLEX)
  /* cannot use BLAS dot for complex because compiler/linker is 
     not happy about returning a double complex */
  int    i;
  Scalar sum = 0.0, *xa = x->array, *ya = y->array;
  for ( i=0; i<x->n; i++ ) {
    sum += xa[i]*ya[i];
  }
  *z = sum;
#else
  *z = BLdot_( &x->n, x->array, &one, y->array, &one );
#endif
  PLogFlops(2*x->n-1);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecScale_Seq"
int VecScale_Seq( Scalar *alpha,Vec xin )
{
  Vec_Seq *x = (Vec_Seq *) xin->data;
  int     one = 1;

  PetscFunctionBegin;
  BLscal_( &x->n, alpha, x->array, &one );
  PLogFlops(x->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecCopy_Seq"
int VecCopy_Seq(Vec xin, Vec yin )
{
  Vec_Seq *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int     one = 1;

  PetscFunctionBegin;
  BLcopy_( &x->n, x->array, &one, y->array, &one );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSwap_Seq"
int VecSwap_Seq(  Vec xin,Vec yin )
{
  Vec_Seq *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int     one = 1;

  PetscFunctionBegin;
  BLswap_( &x->n, x->array, &one, y->array, &one );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecAXPY_Seq"
int VecAXPY_Seq(  Scalar *alpha, Vec xin, Vec yin )
{
  Vec_Seq  *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int      one = 1;

  PetscFunctionBegin;
  BLaxpy_( &x->n, alpha, x->array, &one, y->array, &one );
  PLogFlops(2*x->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecAXPBY_Seq"
int VecAXPBY_Seq(Scalar *alpha, Scalar *beta,Vec xin, Vec yin)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int      n = x->n, i;
  Scalar   *xx = x->array, *yy = y->array, a = *alpha, b = *beta;

  PetscFunctionBegin;
  for ( i=0; i<n; i++ ) {
    yy[i] = a*xx[i] + b*yy[i];
  }

  PLogFlops(3*x->n);
  PetscFunctionReturn(0);
}

