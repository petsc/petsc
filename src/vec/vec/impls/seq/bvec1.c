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
#if defined(PETSC_HAVE_CUDA)
    ierr = VecCUDACopyFromGPU(xin);
    ierr = VecCUDACopyFromGPU(yin);
#endif
    ierr = VecGetArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);

    PetscInt    i;
    PetscScalar sum = 0.0;
    for (i=0; i<xin->map->n; i++) {
      sum += xa[i]*PetscConj(ya[i]);
    }
    *z = sum;
    ierr = VecRestoreArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
  }
#else
#if defined(PETSC_HAVE_CUDA)
  Vec_Seq *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;

  ierr = VecCUDACopyToGPU(xin);
  ierr = VecCUDACopyToGPU(yin);
  *z = cublasSdot(bn,x->GPUarray,one,y->GPUarray,one);
  ierr = cublasGetError();CHKERRCUDA(ierr);
#else
  ierr = VecGetArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
  *z = BLASdot_(&bn,xa,&one,ya,&one);
  ierr = VecRestoreArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
#endif
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
#if defined(PETSC_HAVE_CUDA)
  ierr = VecCUDACopyFromGPU(xin);CHKERRQ(ierr);
  ierr = VecCUDACopyFromGPU(yin);CHKERRQ(ierr);
#endif
 ierr = VecGetArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
 {
   PetscInt    i;
   PetscScalar sum = 0.0;
   for (i=0; i<xin->map->n; i++) {
     sum += xa[i]*ya[i];
   }
   *z = sum;
   ierr = VecRestoreArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
 }
#else
#if defined(PETSC_HAVE_CUDA)
 ierr = VecCUDACopyToGPU(xin);
 ierr = VecCUDACopyToGPU(yin);
 *z = cublasSdot(bn,xa,one,ya,one);
 ierr = cublasGetError();CHKERRCUDA(ierr);
#else
  ierr = VecGetArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
  *z = BLASdot_(&bn,xa,&one,ya,&one);
  ierr = VecRestoreArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
#endif
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
  Vec_Seq        *x = (Vec_Seq*)xin->data;
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(xin->map->n);
  PetscScalar    *xx;
 
  PetscFunctionBegin;

#if defined(PETSC_HAVE_CUDA)
  if (alpha == 0.0) {
    ierr = VecSet_Seq(xin,alpha);CHKERRQ(ierr);
  }
  else if (alpha != 1.0) {
  PetscScalar a = alpha;
  ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
  cublasSscal(bn,a,x->GPUarray,one);
  ierr = cublasGetError();CHKERRCUDA(ierr);
  x->valid_GPU_array = GPU;
  //for now, we always copy back from GPU
  ierr = VecCUDACopyFromGPU(xin);CHKERRQ(ierr);
  }
#else
  if (alpha == 0.0) {
    ierr = VecSet_Seq(xin,alpha);CHKERRQ(ierr);
  }
  else if (alpha != 1.0) {
    ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
    PetscScalar a = alpha;
    BLASscal_(&bn,&a,xx,&one);
    ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  }
#endif
  ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCopy_Seq"
PetscErrorCode VecCopy_Seq(Vec xin,Vec yin)
{
  PetscScalar    *ya, *xa;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin != yin) {
#if defined(PETSC_HAVE_CUDA)
    Vec_Seq        *x = (Vec_Seq *)xin->data;
    Vec_Seq        *y = (Vec_Seq *)yin->data;

    if (x->valid_GPU_array != GPU){
      ierr = VecGetArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
      ierr = PetscMemcpy(ya,xa,xin->map->n*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = VecRestoreArray2(xin,&xa,yin,&ya);
      if (y->valid_GPU_array != UNALLOCATED){
        //if the vector we copy to had not allocated GPU space, it still hasn't.
        y->valid_GPU_array = CPU;
      }
    } else {
      PetscInt one = 1;

      ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);
      cublasScopy(xin->map->n,x->GPUarray,one,y->GPUarray,one);
      ierr = cublasGetError();CHKERRCUDA(ierr);
      y->valid_GPU_array = GPU;
      //for now, we always copy back from GPU
      ierr = VecCUDACopyFromGPU(yin);CHKERRQ(ierr);
    }
#else
    ierr = VecGetArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
    ierr = PetscMemcpy(ya,xa,xin->map->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSwap_Seq"
PetscErrorCode VecSwap_Seq(Vec xin,Vec yin)
{
  PetscScalar    *ya, *xa;
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(xin->map->n);

  PetscFunctionBegin;
  if (xin != yin) {
#if defined(PETSC_HAVE_CUDA)
    Vec_Seq      *x=(Vec_Seq *)xin->data,*y=(Vec_Seq *)yin->data;
    
    //We perform the swap on the GPU unless both vectors are already on the CPU
    if ((x->valid_GPU_array == CPU || x->valid_GPU_array == SAME) && (y->valid_GPU_array == CPU || y->valid_GPU_array == SAME)){
      ierr = VecGetArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
      BLASswap_(&bn,xa,&one,ya,&one);
      ierr = VecRestoreArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
    }
    else{
      ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
      ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);
      cublasSswap(bn,x->GPUarray,one,y->GPUarray,one);
      ierr = cublasGetError();CHKERRCUDA(ierr);
      x->valid_GPU_array = GPU;
      y->valid_GPU_array = GPU;
      //for now, we always copy back from GPU
      ierr = VecCUDACopyFromGPU(yin);CHKERRQ(ierr);
      ierr = VecCUDACopyFromGPU(xin);CHKERRQ(ierr);
    }
#else
    ierr = VecGetArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
    BLASswap_(&bn,xa,&one,ya,&one);
    ierr = VecRestoreArray2(xin,&xa,yin,&ya);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPY_Seq"
PetscErrorCode VecAXPY_Seq(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode ierr;
  PetscScalar    *xarray,*yarray;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(yin->map->n);

  PetscFunctionBegin;
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
  if (alpha != 0.0) {
#if defined(PETSC_HAVE_CUDA)
    Vec_Seq         *x = (Vec_Seq *)xin->data,*y = (Vec_Seq *)yin->data;

    ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
    ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);
    cublasSaxpy(bn,alpha,x->GPUarray,one,y->GPUarray,one);
    ierr = cublasGetError();CHKERRCUDA(ierr);
    y->valid_GPU_array = GPU;
    //For now we always copy back from GPU
    ierr = VecCUDACopyFromGPU(xin);CHKERRQ(ierr);
    ierr = VecCUDACopyFromGPU(yin);CHKERRQ(ierr);
#else
    ierr = VecGetArray2(yin,&yarray,xin,&xarray);CHKERRQ(ierr);
    BLASaxpy_(&bn,&alpha,xarray,&one,yarray,&one);
    ierr = VecRestoreArray2(xin,&xarray,yin,&yarray);CHKERRQ(ierr);
#endif
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
  PetscScalar       *xx,*yy,a = alpha,b = beta;

  PetscFunctionBegin;
  if (a == 0.0) {
    ierr = VecScale_Seq(yin,beta);CHKERRQ(ierr);
  } else if (b == 1.0) {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == 1.0) {
    ierr = VecAYPX_Seq(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == 0.0) {
#if defined(PETSC_HAVE_CUDA)
    ierr = VecCUDACopyFromGPU(xin);CHKERRQ(ierr);
#endif
    ierr = VecGetArray2(xin,(PetscScalar**)&xx,yin,(PetscScalar**)&yy);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      yy[i] = a*xx[i];
    }
    ierr = VecRestoreArray2(xin,(PetscScalar**)&xx,yin,(PetscScalar**)&yy);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    Vec_Seq *y = (Vec_Seq *)yin->data;

    y->valid_GPU_array = CPU;
#endif
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  } else {
#if defined(PETSC_HAVE_CUDA)
    ierr = VecCUDACopyFromGPU(xin);CHKERRQ(ierr);
#endif
    ierr = VecGetArray2(xin,(PetscScalar**)&xx,yin,(PetscScalar**)&yy);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      yy[i] = a*xx[i] + b*yy[i];
    }
    ierr = VecRestoreArray2(xin,(PetscScalar**)&xx,yin,(PetscScalar**)&yy);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    Vec_Seq *y = (Vec_Seq *)yin->data;

    y->valid_GPU_array = CPU;
#endif
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
  PetscScalar  *yy,*xx,*zz;

  PetscFunctionBegin;
#if defined(VEC_HAVE_CUDA)
  ierr = VecCUDACopyFromGPU(xin);CHKERRQ(ierr);
  ierr = VecCUDACopyFromGPU(yin);CHKERRQ(ierr);
  ierr = VecCUDACopyFromGPU(zin)CHKERRQ(ierr);
#endif
  ierr = VecGetArray3(xin,(PetscScalar**)&xx,yin,(PetscScalar**)&yy,zin,(PetscScalar**)&zz);CHKERRQ(ierr);
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
  ierr = VecRestoreArray3(xin,(PetscScalar**)&xx,yin,(PetscScalar**)&yy,zin,(PetscScalar**)&zz);CHKERRQ(ierr);
#if defined(VEC_HAVE_CUDA)
  Vec_Seq *z = (Veq_Seq *)zin->data;

  z->valid_GPU_array = CPU;
#endif
  PetscFunctionReturn(0);
}
