/*$Id: bvec1.c,v 1.39 2001/03/23 23:21:25 balay Exp bsmith $*/

/*
   Defines the BLAS based vector operations. Code shared by parallel
  and sequential vectors.
*/

#include "src/vec/vecimpl.h" 
#include "src/vec/impls/dvecimpl.h" 
#include "petscblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "VecDot_Seq"
int VecDot_Seq(Vec xin,Vec yin,PetscScalar *z)
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
    PetscScalar sum = 0.0,*xa = x->array,*ya = y->array;
    for (i=0; i<xin->n; i++) {
      sum += xa[i]*PetscConj(ya[i]);
    }
    *z = sum;
  }
#else
  *z = BLdot_(&xin->n,x->array,&one,y->array,&one);
#endif
  PetscLogFlops(2*xin->n-1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecTDot_Seq"
int VecTDot_Seq(Vec xin,Vec yin,PetscScalar *z)
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
  PetscScalar sum = 0.0,*xa = x->array,*ya = y->array;
  for (i=0; i<xin->n; i++) {
    sum += xa[i]*ya[i];
  }
  *z = sum;
#else
  *z = BLdot_(&xin->n,x->array,&one,y->array,&one);
#endif
  PetscLogFlops(2*xin->n-1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScale_Seq"
int VecScale_Seq(const PetscScalar *alpha,Vec xin)
{
  Vec_Seq *x = (Vec_Seq*)xin->data;
  int     one = 1;

  PetscFunctionBegin;
  BLscal_(&xin->n,(PetscScalar *)alpha,x->array,&one);
  PetscLogFlops(xin->n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCopy_Seq"
int VecCopy_Seq(Vec xin,Vec yin)
{
  Vec_Seq *x = (Vec_Seq *)xin->data,*y = (Vec_Seq *)yin->data;
  int     ierr;

  PetscFunctionBegin;
  if (x->array != y->array) {
    ierr = PetscMemcpy(y->array,x->array,xin->n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSwap_Seq"
int VecSwap_Seq(Vec xin,Vec yin)
{
  Vec_Seq *x = (Vec_Seq *)xin->data,*y = (Vec_Seq *)yin->data;
  int     one = 1;

  PetscFunctionBegin;
  BLswap_(&xin->n,x->array,&one,y->array,&one);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPY_Seq"
int VecAXPY_Seq(const PetscScalar *alpha,Vec xin,Vec yin)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  int      one = 1,ierr;
  PetscScalar   *yarray;

  PetscFunctionBegin;
  ierr = VecGetArray(yin,&yarray);CHKERRQ(ierr);
  BLaxpy_(&xin->n,(PetscScalar *)alpha,x->array,&one,yarray,&one);
  ierr = VecRestoreArray(yin,&yarray);CHKERRQ(ierr);
  PetscLogFlops(2*xin->n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPBY_Seq"
int VecAXPBY_Seq(const PetscScalar *alpha,const PetscScalar *beta,Vec xin,Vec yin)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data,*y = (Vec_Seq *)yin->data;
  int      n = xin->n,i;
  PetscScalar   *xx = x->array,*yy = y->array,a = *alpha,b = *beta;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    yy[i] = a*xx[i] + b*yy[i];
  }

  PetscLogFlops(3*xin->n);
  PetscFunctionReturn(0);
}

