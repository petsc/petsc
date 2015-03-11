
/*
   Defines the BLAS based vector operations. Code shared by parallel
  and sequential vectors.
*/

#include <../src/vec/vec/impls/dvecimpl.h>          /*I "petscvec.h" I*/
#include <petscblaslapack.h>
#include <petscthreadcomm.h>

#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecDot_kernel(PetscInt thread_id,Vec xin,Vec yin,PetscThreadCommReduction red)
{
  PetscErrorCode    ierr;
  PetscInt          *trstarts=xin->map->trstarts;
  PetscInt          start,end,n;
  PetscBLASInt      one = 1,bn;
  const PetscScalar *xa,*ya;
  PetscScalar       z_loc;

  ierr  = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr  = VecGetArrayRead(yin,&ya);CHKERRQ(ierr);
  start = trstarts[thread_id];
  end   = trstarts[thread_id+1];
  n     = end-start;
  ierr  = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  /* arguments ya, xa are reversed because BLAS complex conjugates the first argument, PETSc the second */
  PetscStackCallBLAS("BLASdot",z_loc = BLASdot_(&bn,ya+start,&one,xa+start,&one));

  ierr = PetscThreadReductionKernelPost(thread_id,red,(void*)&z_loc);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&ya);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "VecDot_Seq"
PetscErrorCode VecDot_Seq(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode           ierr;
  PetscThreadCommReduction red;

  PetscFunctionBegin;
  ierr = PetscThreadReductionBegin(PetscObjectComm((PetscObject)xin),THREADCOMM_SUM,PETSC_SCALAR,1,&red);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel3(PetscObjectComm((PetscObject)xin),(PetscThreadKernel)VecDot_kernel,xin,yin,red);CHKERRQ(ierr);
  ierr = PetscThreadReductionEnd(red,z);CHKERRQ(ierr);
  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "VecDot_Seq"
PetscErrorCode VecDot_Seq(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *ya,*xa;
  PetscBLASInt      one = 1,bn;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&ya);CHKERRQ(ierr);
  /* arguments ya, xa are reversed because BLAS complex conjugates the first argument, PETSc the second */
  PetscStackCallBLAS("BLASdot",*z   = BLASdot_(&bn,ya,&one,xa,&one));
  ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&ya);CHKERRQ(ierr);
  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecTDot_kernel(PetscInt thread_id,Vec xin,Vec yin,PetscThreadCommReduction red)
{
  PetscErrorCode    ierr;
  PetscInt          *trstarts=xin->map->trstarts;
  PetscInt          start,end,n;
  PetscBLASInt      one=1,bn;
  const PetscScalar *xa,*ya;
  PetscScalar       z_loc;

  ierr  = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr  = VecGetArrayRead(yin,&ya);CHKERRQ(ierr);
  start = trstarts[thread_id];
  end   = trstarts[thread_id+1];
  n     = end-start;
  ierr  = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASdot",z_loc = BLASdotu_(&bn,xa+start,&one,ya+start,&one));

  ierr = PetscThreadReductionKernelPost(thread_id,red,(void*)&z_loc);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&ya);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "VecTDot_Seq"
PetscErrorCode VecTDot_Seq(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode           ierr;
  PetscThreadCommReduction red;

  PetscFunctionBegin;
  ierr = PetscThreadReductionBegin(PetscObjectComm((PetscObject)xin),THREADCOMM_SUM,PETSC_SCALAR,1,&red);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel3(PetscObjectComm((PetscObject)xin),(PetscThreadKernel)VecTDot_kernel,xin,yin,red);CHKERRQ(ierr);
  ierr = PetscThreadReductionEnd(red,z);CHKERRQ(ierr);
  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "VecTDot_Seq"
PetscErrorCode VecTDot_Seq(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *ya,*xa;
  PetscBLASInt      one = 1,bn;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&ya);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASdot",*z   = BLASdotu_(&bn,xa,&one,ya,&one));
  ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&ya);CHKERRQ(ierr);
  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecScale_kernel(PetscInt thread_id,Vec xin,PetscScalar *alpha_p)
{
  PetscErrorCode ierr;
  PetscScalar    *xx;
  PetscInt       start,end,n;
  PetscBLASInt   one=1,bn;
  PetscInt       *trstarts=xin->map->trstarts;
  PetscScalar    alpha=*alpha_p;

  start = trstarts[thread_id];
  end   = trstarts[thread_id+1];
  n     = end-start;
  ierr  = VecGetArray(xin,&xx);CHKERRQ(ierr);
  ierr  = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASscal",BLASscal_(&bn,&alpha,xx+start,&one));
  ierr  = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "VecScale_Seq"
PetscErrorCode VecScale_Seq(Vec xin,PetscScalar alpha)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecSet_Seq(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != (PetscScalar)1.0) {
    PetscScalar *scalar;
    ierr    = PetscThreadCommGetScalars(PetscObjectComm((PetscObject)xin),&scalar,NULL,NULL);CHKERRQ(ierr);
    *scalar = alpha;
    ierr    = PetscThreadCommRunKernel2(PetscObjectComm((PetscObject)xin),(PetscThreadKernel)VecScale_kernel,xin,scalar);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "VecScale_Seq"
PetscErrorCode VecScale_Seq(Vec xin, PetscScalar alpha)
{
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    ierr = VecSet_Seq(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != (PetscScalar)1.0) {
    PetscScalar a = alpha,*xarray;
    ierr = VecGetArray(xin,&xarray);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASscal",BLASscal_(&bn,&a,xarray,&one));
    ierr = VecRestoreArray(xin,&xarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecAXPY_kernel(PetscInt thread_id,Vec yin,PetscScalar *alpha_p,Vec xin)
{
  PetscErrorCode    ierr;
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one=1,bn;
  PetscInt          *trstarts=yin->map->trstarts,start,end,n;
  PetscScalar       alpha = *alpha_p;

  start = trstarts[thread_id];
  end   = trstarts[thread_id+1];
  n     = end - start;
  ierr  = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr  = VecGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr  = VecGetArray(yin,&yarray);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bn,&alpha,xarray+start,&one,yarray+start,&one));
  ierr = VecRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&yarray);CHKERRQ(ierr);

  return 0;
}
#undef __FUNCT__
#define __FUNCT__ "VecAXPY_Seq"
PetscErrorCode VecAXPY_Seq(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
  if (alpha != (PetscScalar)0.0) {
    PetscScalar *scalar;
    ierr    = PetscThreadCommGetScalars(PetscObjectComm((PetscObject)yin),&scalar,NULL,NULL);CHKERRQ(ierr);
    *scalar = alpha;
    ierr    = PetscThreadCommRunKernel3(PetscObjectComm((PetscObject)yin),(PetscThreadKernel)VecAXPY_kernel,yin,scalar,xin);CHKERRQ(ierr);
    ierr    = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "VecAXPY_Seq"
PetscErrorCode VecAXPY_Seq(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode    ierr;
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1,bn;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
  if (alpha != (PetscScalar)0.0) {
    ierr = VecGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&yarray);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bn,&alpha,xarray,&one,yarray,&one));
    ierr = VecRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecAXPBY_kernel(PetscInt thread_id,Vec yin,PetscScalar *alpha_p,PetscScalar *beta_p,Vec xin)
{
  PetscErrorCode    ierr;
  const PetscScalar *xx;
  PetscScalar       *yy;
  PetscInt          *trstarts=yin->map->trstarts,i;
  PetscScalar       a=*alpha_p,b=*beta_p;

  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&yy);CHKERRQ(ierr);

  if (b == (PetscScalar)0.0) {
    for (i=trstarts[thread_id];i < trstarts[thread_id+1];i++) yy[i] = a*xx[i];
  } else {
    for (i=trstarts[thread_id];i < trstarts[thread_id+1];i++) yy[i] = a*xx[i] + b*yy[i];
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&yy);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "VecAXPBY_Seq"
PetscErrorCode VecAXPBY_Seq(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecScale_Seq(yin,beta);CHKERRQ(ierr);
  } else if (beta == (PetscScalar)1.0) {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  } else if (alpha == (PetscScalar)1.0) {
    ierr = VecAYPX_Seq(yin,beta,xin);CHKERRQ(ierr);
  } else {
    PetscScalar *scal1,*scal2;
    ierr   = PetscThreadCommGetScalars(PetscObjectComm((PetscObject)yin),&scal1,&scal2,NULL);CHKERRQ(ierr);
    *scal1 = alpha; *scal2 = beta;
    ierr   = PetscThreadCommRunKernel4(PetscObjectComm((PetscObject)yin),(PetscThreadKernel)VecAXPBY_kernel,yin,scal1,scal2,xin);CHKERRQ(ierr);
    if (beta == (PetscScalar)0.0) {
      ierr = PetscLogFlops(yin->map->n);CHKERRQ(ierr);
    } else {
      ierr = PetscLogFlops(3.0*yin->map->n);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "VecAXPBY_Seq"
PetscErrorCode VecAXPBY_Seq(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode    ierr;
  PetscInt          n = yin->map->n,i;
  const PetscScalar *xx;
  PetscScalar       *yy,a = alpha,b = beta;

  PetscFunctionBegin;
  if (a == (PetscScalar)0.0) {
    ierr = VecScale_Seq(yin,beta);CHKERRQ(ierr);
  } else if (b == (PetscScalar)1.0) {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == (PetscScalar)1.0) {
    ierr = VecAYPX_Seq(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == (PetscScalar)0.0) {
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(yin,(PetscScalar**)&yy);CHKERRQ(ierr);

    for (i=0; i<n; i++) yy[i] = a*xx[i];

    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,(PetscScalar**)&yy);CHKERRQ(ierr);
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  } else {
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(yin,(PetscScalar**)&yy);CHKERRQ(ierr);

    for (i=0; i<n; i++) yy[i] = a*xx[i] + b*yy[i];

    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,(PetscScalar**)&yy);CHKERRQ(ierr);
    ierr = PetscLogFlops(3.0*xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecAXPBYPCZ_kernel(PetscInt thread_id,Vec zin,PetscScalar *alpha_p,PetscScalar *beta_p,PetscScalar *gamma_p,Vec xin,Vec yin)
{
  PetscErrorCode    ierr;
  const PetscScalar *xx,*yy;
  PetscScalar       *zz;
  PetscInt          *trstarts=zin->map->trstarts,i;
  PetscScalar       alpha=*alpha_p,beta=*beta_p,gamma=*gamma_p;

  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecGetArray(zin,&zz);CHKERRQ(ierr);

  if (alpha == (PetscScalar)1.0) {
    for (i=trstarts[thread_id]; i < trstarts[thread_id+1]; i++) zz[i] = xx[i] + beta*yy[i] + gamma*zz[i];
  } else if (gamma == (PetscScalar)1.0) {
    for (i=trstarts[thread_id]; i < trstarts[thread_id+1]; i++) zz[i] = alpha*xx[i] + beta*yy[i] + zz[i];
  } else if (gamma == (PetscScalar)0.0) {
    for (i=trstarts[thread_id]; i < trstarts[thread_id+1]; i++) zz[i] = alpha*xx[i] + beta*yy[i];
  } else {
    for (i=trstarts[thread_id]; i < trstarts[thread_id+1]; i++) zz[i] = alpha*xx[i] + beta*yy[i] + gamma*zz[i];
  }
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecGetArray(zin,&zz);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "VecAXPBYPCZ_Seq"
PetscErrorCode VecAXPBYPCZ_Seq(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscScalar    *scal1,*scal2,*scal3;

  PetscFunctionBegin;
  ierr   = PetscThreadCommGetScalars(PetscObjectComm((PetscObject)zin),&scal1,&scal2,&scal3);CHKERRQ(ierr);
  *scal1 = alpha; *scal2 = beta; *scal3 = gamma;
  ierr   = PetscThreadCommRunKernel6(PetscObjectComm((PetscObject)zin),(PetscThreadKernel)VecAXPBYPCZ_kernel,zin,scal1,scal2,scal3,xin,yin);CHKERRQ(ierr);
  if (alpha == (PetscScalar)1.0) {
    ierr = PetscLogFlops(4.0*zin->map->n);CHKERRQ(ierr);
  } else if (gamma == (PetscScalar)1.0) {
    ierr = PetscLogFlops(4.0*zin->map->n);CHKERRQ(ierr);
  } else if (gamma == (PetscScalar)0.0) {
    ierr = PetscLogFlops(3.0*zin->map->n);CHKERRQ(ierr);
  } else {
    ierr = PetscLogFlops(5.0*zin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "VecAXPBYPCZ_Seq"
PetscErrorCode VecAXPBYPCZ_Seq(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode    ierr;
  PetscInt          n = zin->map->n,i;
  const PetscScalar *yy,*xx;
  PetscScalar       *zz;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecGetArray(zin,&zz);CHKERRQ(ierr);
  if (alpha == (PetscScalar)1.0) {
    for (i=0; i<n; i++) zz[i] = xx[i] + beta*yy[i] + gamma*zz[i];
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  } else if (gamma == (PetscScalar)1.0) {
    for (i=0; i<n; i++) zz[i] = alpha*xx[i] + beta*yy[i] + zz[i];
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  } else if (gamma == (PetscScalar)0.0) {
    for (i=0; i<n; i++) zz[i] = alpha*xx[i] + beta*yy[i];
    ierr = PetscLogFlops(3.0*n);CHKERRQ(ierr);
  } else {
    for (i=0; i<n; i++) zz[i] = alpha*xx[i] + beta*yy[i] + gamma*zz[i];
    ierr = PetscLogFlops(5.0*n);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(zin,&zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
