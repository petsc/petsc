#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: bvec1.c,v 1.30 1999/10/04 18:50:28 bsmith Exp bsmith $"
#endif

/*
   Defines the BLAS based vector operations. Code shared by parallel
  and sequential vectors.
*/

#include "src/vec/vecimpl.h" 
#include "src/vec/impls/dvecimpl.h" 
#include "pinclude/blaslapack.h"

#undef __FUNC__  
#define __FUNC__ "VecDot_Seq"
int VecDot_Seq(Vec xin, Vec yin,Scalar *z )
{
  Vec_Seq *x = (Vec_Seq *)xin->data,*y = (Vec_Seq *)yin->data;
#if !defined(PETSC_USE_COMPLEX)
  int     one = 1;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  /* cannot use BLAS dot for complex because compiler/linker is 
     not happy about returning a double complex */
  {
    int    i;
    Scalar sum = 0.0, *xa = x->array, *ya = y->array;
    for ( i=0; i<xin->n; i++ ) {
      sum += xa[i]*PetscConj(ya[i]);
    }
    *z = sum;
  }
#else
  *z = BLdot_( &xin->n, x->array, &one, y->array, &one );
#endif
  PLogFlops(2*xin->n-1);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecTDot_Seq"
int VecTDot_Seq(Vec xin, Vec yin,Scalar *z )
{
  Vec_Seq *x = (Vec_Seq *)xin->data,*y = (Vec_Seq *)yin->data;
#if !defined(PETSC_USE_COMPLEX)
 int     one = 1;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  /* cannot use BLAS dot for complex because compiler/linker is 
     not happy about returning a double complex */
  int    i;
  Scalar sum = 0.0, *xa = x->array, *ya = y->array;
  for ( i=0; i<xin->n; i++ ) {
    sum += xa[i]*ya[i];
  }
  *z = sum;
#else
  *z = BLdot_( &xin->n, x->array, &one, y->array, &one );
#endif
  PLogFlops(2*xin->n-1);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecScale_Seq"
int VecScale_Seq(const Scalar *alpha,Vec xin )
{
  Vec_Seq *x = (Vec_Seq *) xin->data;
  int     one = 1;

  PetscFunctionBegin;
  BLscal_( &xin->n, (Scalar *)alpha, x->array, &one );
  PLogFlops(xin->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecCopy_Seq"
int VecCopy_Seq(Vec xin, Vec yin )
{
  Vec_Seq *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int     ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(y->array,x->array,xin->n*sizeof(Scalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSwap_Seq"
int VecSwap_Seq(  Vec xin,Vec yin )
{
  Vec_Seq *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int     one = 1;

  PetscFunctionBegin;
  BLswap_( &xin->n, x->array, &one, y->array, &one );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecAXPY_Seq"
int VecAXPY_Seq(const Scalar *alpha, Vec xin, Vec yin )
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  int      one = 1,ierr;
  Scalar   *yarray;

  PetscFunctionBegin;
  ierr = VecGetArray(yin,&yarray);CHKERRQ(ierr);
  BLaxpy_( &xin->n, (Scalar *)alpha, x->array, &one, yarray, &one );
  ierr = VecRestoreArray(yin,&yarray);CHKERRQ(ierr);
  PLogFlops(2*xin->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecAXPBY_Seq"
int VecAXPBY_Seq(const Scalar *alpha,const Scalar *beta,Vec xin, Vec yin)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  int      n = xin->n, i;
  Scalar   *xx = x->array, *yy = y->array, a = *alpha, b = *beta;

  PetscFunctionBegin;
  for ( i=0; i<n; i++ ) {
    yy[i] = a*xx[i] + b*yy[i];
  }

  PLogFlops(3*xin->n);
  PetscFunctionReturn(0);
}

