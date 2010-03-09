#define PETSCVEC_DLL
/*
     Code for some of the parallel vector primatives.
*/
#include "../src/vec/vec/impls/mpi/pvecimpl.h" 
#include "petscblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "VecMDot_MPI"
PetscErrorCode VecMDot_MPI(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc(nv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  }
  ierr = VecMDot_Seq(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMTDot_MPI"
PetscErrorCode VecMTDot_MPI(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc(nv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  }
  ierr = VecMTDot_Seq(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include "../src/vec/vec/impls/seq/ftn-kernels/fnorm.h"
#undef __FUNCT__  
#define __FUNCT__ "VecNorm_MPI"
PetscErrorCode VecNorm_MPI(Vec xin,NormType type,PetscReal *z)
{
  Vec_MPI        *x = (Vec_MPI*)xin->data;
  PetscReal      sum,work = 0.0;
  PetscScalar    *xx = x->array;
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {

#if defined(PETSC_HAVE_SLOW_BLAS_NORM2)
#if defined(PETSC_USE_FORTRAN_KERNEL_NORM)
    fortrannormsqr_(xx,&n,&work);
#elif defined(PETSC_USE_UNROLLED_NORM)
    switch (n & 0x3) {
      case 3: work += PetscRealPart(xx[0]*PetscConj(xx[0])); xx++;
      case 2: work += PetscRealPart(xx[0]*PetscConj(xx[0])); xx++;
      case 1: work += PetscRealPart(xx[0]*PetscConj(xx[0])); xx++; n -= 4;
    }
    while (n>0) {
      work += PetscRealPart(xx[0]*PetscConj(xx[0])+xx[1]*PetscConj(xx[1])+
                        xx[2]*PetscConj(xx[2])+xx[3]*PetscConj(xx[3]));
      xx += 4; n -= 4;
    } 
#else
    {PetscInt i; for (i=0; i<n; i++) work += PetscRealPart((xx[i])*(PetscConj(xx[i])));}
#endif
#else
    {PetscBLASInt one = 1,bn = PetscBLASIntCast(n);
      work  = BLASnrm2_(&bn,xx,&one);
      work *= work;
    }
#endif
    ierr = MPI_Allreduce(&work,&sum,1,MPIU_REAL,MPI_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
    *z = sqrt(sum);
    ierr = PetscLogFlops(2.0*xin->map->n);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_Seq(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPI_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_Seq(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPI_MAX,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_Seq(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_Seq(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPI_Allreduce(temp,z,2,MPIU_REAL,MPI_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
    z[1] = sqrt(z[1]);
  }
  PetscFunctionReturn(0);
}

/*
       These two functions are the MPI reduction operation used for max and min with index
   The call below to MPI_Op_create() converts the function Vec[Max,Min]_Local() to the 
   MPI operator Vec[Max,Min]_Local_Op.
*/
MPI_Op VecMax_Local_Op = 0;
MPI_Op VecMin_Local_Op = 0;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecMax_Local"
void PETSCVEC_DLLEXPORT MPIAPI VecMax_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscReal *xin = (PetscReal *)in,*xout = (PetscReal*)out;

  PetscFunctionBegin;
  if (*datatype != MPIU_REAL) {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL data types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  if (xin[0] > xout[0]) {
    xout[0] = xin[0];
    xout[1] = xin[1];
  } else if (xin[0] == xout[0]) {
    xout[1] = PetscMin(xin[1],xout[1]);
  }
  PetscFunctionReturnVoid(); /* cannot return a value */
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecMin_Local"
void PETSCVEC_DLLEXPORT MPIAPI VecMin_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscReal *xin = (PetscReal *)in,*xout = (PetscReal*)out;

  PetscFunctionBegin;
  if (*datatype != MPIU_REAL) {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL data types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  if (xin[0] < xout[0]) {
    xout[0] = xin[0];
    xout[1] = xin[1];
  } else if (xin[0] == xout[0]) {
    xout[1] = PetscMin(xin[1],xout[1]);
  }
  PetscFunctionReturnVoid();
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "VecMax_MPI"
PetscErrorCode VecMax_MPI(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local max */
  ierr = VecMax_Seq(xin,idx,&work);CHKERRQ(ierr);

  /* Find the global max */
  if (!idx) {
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPI_MAX,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else {
    PetscReal work2[2],z2[2];
    PetscInt  rstart;
    rstart = xin->map->rstart;
    work2[0] = work;
    work2[1] = *idx + rstart;
    ierr = MPI_Allreduce(work2,z2,2,MPIU_REAL,VecMax_Local_Op,((PetscObject)xin)->comm);CHKERRQ(ierr);
    *z   = z2[0];
    *idx = (PetscInt)z2[1];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMin_MPI"
PetscErrorCode VecMin_MPI(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal work;

  PetscFunctionBegin;
  /* Find the local Min */
  ierr = VecMin_Seq(xin,idx,&work);CHKERRQ(ierr);

  /* Find the global Min */
  if (!idx) {
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPI_MIN,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else {
    PetscReal work2[2],z2[2];
    PetscInt       rstart;

    ierr = VecGetOwnershipRange(xin,&rstart,PETSC_NULL);CHKERRQ(ierr);
    work2[0] = work;
    work2[1] = *idx + rstart;
    ierr = MPI_Allreduce(work2,z2,2,MPIU_REAL,VecMin_Local_Op,((PetscObject)xin)->comm);CHKERRQ(ierr);
    *z   = z2[0];
    *idx = (PetscInt)z2[1];
  }
  PetscFunctionReturn(0);
}








