#define PETSCVEC_DLL
/*
   Defines the BLAS based vector operations. Code shared by parallel
  and sequential vectors.
*/

#include "private/vecimpl.h" 
#include "../src/vec/vec/impls/dvecimpl.h" 
#include "petscblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "VecDot_Seq"
PetscErrorCode VecDot_Seq(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    *ya,*xa;
  PetscErrorCode ierr;
#if !defined(PETSC_USE_COMPLEX)
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(xin->map->n);
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);}
  else ya = xa;
#if defined(PETSC_USE_COMPLEX)
  /* cannot use BLAS dot for complex because compiler/linker is 
     not happy about returning a double complex */
  {
    PetscInt    i;
    PetscScalar sum = 0.0;
    for (i=0; i<xin->map->n; i++) {
      sum += xa[i]*PetscConj(ya[i]);
    }
    *z = sum;
  }
#else
  *z = BLASdot_(&bn,xa,&one,ya,&one);
#endif
  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);}
  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecTDot_Seq"
PetscErrorCode VecTDot_Seq(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    *ya,*xa;
  PetscErrorCode ierr;
#if !defined(PETSC_USE_COMPLEX)
  PetscBLASInt    one = 1, bn = PetscBLASIntCast(xin->map->n);
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);}
  else ya = xa;
#if defined(PETSC_USE_COMPLEX)
  /* cannot use BLAS dot for complex because compiler/linker is 
     not happy about returning a double complex */
 {
   PetscInt    i;
   PetscScalar sum = 0.0;
   for (i=0; i<xin->map->n; i++) {
     sum += xa[i]*ya[i];
   }
   *z = sum;
 }
#else
  *z = BLASdot_(&bn,xa,&one,ya,&one);
#endif
  ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);}
  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScale_Seq"
PetscErrorCode VecScale_Seq(Vec xin, PetscScalar alpha)
{
  Vec_Seq        *x = (Vec_Seq*)xin->data;
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(xin->map->n);

  PetscFunctionBegin;
  if (alpha == 0.0) {
    ierr = VecSet_Seq(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != 1.0) {
    PetscScalar a = alpha;
    BLASscal_(&bn,&a,x->array,&one);
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCopy_Seq"
PetscErrorCode VecCopy_Seq(Vec xin,Vec yin)
{
  Vec_Seq        *x = (Vec_Seq *)xin->data;
  PetscScalar    *ya;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    ierr = PetscMemcpy(ya,x->array,xin->map->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSwap_Seq"
PetscErrorCode VecSwap_Seq(Vec xin,Vec yin)
{
  Vec_Seq        *x = (Vec_Seq *)xin->data;
  PetscScalar    *ya;
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(xin->map->n);

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    BLASswap_(&bn,x->array,&one,ya,&one);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPY_Seq"
PetscErrorCode VecAXPY_Seq(Vec yin,PetscScalar alpha,Vec xin)
{
  Vec_Seq        *y = (Vec_Seq *)yin->data;
  PetscErrorCode ierr;
  PetscScalar    *xarray;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(yin->map->n);

  PetscFunctionBegin;
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
  if (alpha != 0.0) {
    ierr = VecGetArray(xin,&xarray);CHKERRQ(ierr);
    BLASaxpy_(&bn,&alpha,xarray,&one,y->array,&one);
    ierr = VecRestoreArray(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPBY_Seq"
PetscErrorCode VecAXPBY_Seq(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  Vec_Seq           *y = (Vec_Seq *)yin->data;
  PetscErrorCode    ierr;
  PetscInt          n = yin->map->n,i;
  PetscScalar       *yy = y->array,a = alpha,b = beta;
  const PetscScalar *xx;

  PetscFunctionBegin;
  if (a == 0.0) {
    ierr = VecScale_Seq(yin,beta);CHKERRQ(ierr);
  } else if (b == 1.0) {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == 1.0) {
    ierr = VecAYPX_Seq(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == 0.0) {
    ierr = VecGetArray(xin,(PetscScalar**)&xx);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      yy[i] = a*xx[i];
    }
    ierr = VecRestoreArray(xin,(PetscScalar**)&xx);CHKERRQ(ierr);
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(xin,(PetscScalar**)&xx);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      yy[i] = a*xx[i] + b*yy[i];
    }
    ierr = VecRestoreArray(xin,(PetscScalar**)&xx);CHKERRQ(ierr);
    ierr = PetscLogFlops(3.0*xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPBYPCZ_Seq"
PetscErrorCode VecAXPBYPCZ_Seq(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode     ierr;
  PetscInt           n = zin->map->n,i;
  const PetscScalar  *yy,*xx;
  PetscScalar        *zz;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,(PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecGetArray(yin,(PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecGetArray(zin,&zz);CHKERRQ(ierr);
  if (alpha == 1.0) {
   for (i=0; i<n; i++) {
      zz[i] = xx[i] + beta*yy[i] + gamma*zz[i];
    }
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  } else if (gamma == 1.0) {
    for (i=0; i<n; i++) {
      zz[i] = alpha*xx[i] + beta*yy[i] + zz[i];
    }
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  } else {
    for (i=0; i<n; i++) {
      zz[i] = alpha*xx[i] + beta*yy[i] + gamma*zz[i];
    }
    ierr = PetscLogFlops(5.0*n);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xin,(PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,(PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(zin,&zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
