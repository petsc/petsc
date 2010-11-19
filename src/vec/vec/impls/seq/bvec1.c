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
#if defined(PETSC_USE_COMPLEX)
  /* cannot use BLAS dot for complex because compiler/linker is 
     not happy about returning a double complex */
  {
    ierr = VecGetArrayPrivate2(xin,&xa,yin,&ya);CHKERRQ(ierr);

    PetscInt    i;
    PetscScalar sum = 0.0;
    for (i=0; i<xin->map->n; i++) {
      sum += xa[i]*PetscConj(ya[i]);
    }
    *z = sum;
    ierr = VecRestoreArrayPrivate2(xin,&xa,yin,&ya);CHKERRQ(ierr);
  }
#else
  ierr = VecGetArrayPrivate2(xin,&xa,yin,&ya);CHKERRQ(ierr);
  *z = BLASdot_(&bn,xa,&one,ya,&one);
  ierr = VecRestoreArrayPrivate2(xin,&xa,yin,&ya);CHKERRQ(ierr);
#endif
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
#if defined(PETSC_USE_COMPLEX)
  /* cannot use BLAS dot for complex because compiler/linker is 
     not happy about returning a double complex */
 ierr = VecGetArrayPrivate2(xin,&xa,yin,&ya);CHKERRQ(ierr);
 {
   PetscInt    i;
   PetscScalar sum = 0.0;
   for (i=0; i<xin->map->n; i++) {
     sum += xa[i]*ya[i];
   }
   *z = sum;
   ierr = VecRestoreArrayPrivate2(xin,&xa,yin,&ya);CHKERRQ(ierr);
 }
#else
  ierr = VecGetArrayPrivate2(xin,&xa,yin,&ya);CHKERRQ(ierr);
  *z = BLASdot_(&bn,xa,&one,ya,&one);
  ierr = VecRestoreArrayPrivate2(xin,&xa,yin,&ya);CHKERRQ(ierr);
#endif
  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "VecScale_Seq"
PetscErrorCode VecScale_Seq(Vec xin, PetscScalar alpha)
{
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(xin->map->n);
  PetscFunctionBegin;
  if (alpha == 0.0) {
    ierr = VecSet_Seq(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != 1.0) {
    PetscScalar a = alpha;
    BLASscal_(&bn,&a,*(PetscScalar**)xin->data,&one);
  }
  ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecAXPY_Seq"
PetscErrorCode VecAXPY_Seq(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode    ierr;
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1,bn = PetscBLASIntCast(yin->map->n);

  PetscFunctionBegin;
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
  if (alpha != 0.0) {
    ierr = VecGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&yarray);CHKERRQ(ierr);
    BLASaxpy_(&bn,&alpha,xarray,&one,yarray,&one);
    ierr = VecRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPBY_Seq"
PetscErrorCode VecAXPBY_Seq(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode    ierr;
  PetscInt          n = yin->map->n,i;
  const PetscScalar *xx;
  PetscScalar       *yy,a = alpha,b = beta;

  PetscFunctionBegin;
  if (a == 0.0) {
    ierr = VecScale_Seq(yin,beta);CHKERRQ(ierr);
  } else if (b == 1.0) {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == 1.0) {
    ierr = VecAYPX_Seq(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == 0.0) {
    ierr = VecGetArrayPrivate2(xin,(PetscScalar**)&xx,yin,&yy);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      yy[i] = a*xx[i];
    }
    ierr = VecRestoreArrayPrivate2(xin,(PetscScalar**)&xx,yin,&yy);CHKERRQ(ierr);
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  } else {
    ierr = VecGetArrayPrivate2(xin,(PetscScalar**)&xx,yin,&yy);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      yy[i] = a*xx[i] + b*yy[i];
    }
    ierr = VecRestoreArrayPrivate2(xin,(PetscScalar**)&xx,yin,&yy);CHKERRQ(ierr);
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
  if (alpha == 1.0) {
    ierr = VecGetArrayPrivate3(xin,(PetscScalar**)&xx,yin,(PetscScalar**)&yy,zin,&zz);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      zz[i] = xx[i] + beta*yy[i] + gamma*zz[i];
    }
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
    ierr = VecRestoreArrayPrivate3(xin,(PetscScalar**)&xx,yin,(PetscScalar**)&yy,zin,&zz);CHKERRQ(ierr);
  } else if (gamma == 1.0) {
    ierr = VecGetArrayPrivate3(xin,(PetscScalar**)&xx,yin,(PetscScalar**)&yy,zin,&zz);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      zz[i] = alpha*xx[i] + beta*yy[i] + zz[i];
    }
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
    ierr = VecRestoreArrayPrivate3(xin,(PetscScalar**)&xx,yin,(PetscScalar**)&yy,zin,&zz);CHKERRQ(ierr);
  } else {
    ierr = VecGetArrayPrivate3(xin,(PetscScalar**)&xx,yin,(PetscScalar**)&yy,zin,&zz);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      zz[i] = alpha*xx[i] + beta*yy[i] + gamma*zz[i];
    }
    ierr = VecRestoreArrayPrivate3(xin,(PetscScalar**)&xx,yin,(PetscScalar**)&yy,zin,&zz);CHKERRQ(ierr);
    ierr = PetscLogFlops(5.0*n);CHKERRQ(ierr);    
  }
  PetscFunctionReturn(0);
}
