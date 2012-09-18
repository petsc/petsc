
/*
     Code for some of the parallel vector primatives.
*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petscblaslapack.h>

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

#include <../src/vec/vec/impls/seq/ftn-kernels/fnorm.h>
#undef __FUNCT__
#define __FUNCT__ "VecNorm_MPI"
PetscErrorCode VecNorm_MPI(Vec xin,NormType type,PetscReal *z)
{
  PetscReal         sum,work = 0.0;
  const PetscScalar *xx;
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n;
  PetscBLASInt      one = 1,bn = PetscBLASIntCast(n);

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    work = PetscRealPart(BLASdot_(&bn,xx,&one,xx,&one));
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
    *z = PetscSqrtReal(sum);
    ierr = PetscLogFlops(2.0*xin->map->n);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_Seq(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_Seq(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_Seq(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_Seq(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPI_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
    z[1] = PetscSqrtReal(z[1]);
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
void  MPIAPI VecMax_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscReal *xin = (PetscReal *)in,*xout = (PetscReal*)out;

  PetscFunctionBegin;
  if (*datatype != MPIU_REAL) {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL data types");
    MPI_Abort(MPI_COMM_SELF,1);
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
void  MPIAPI VecMin_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscReal *xin = (PetscReal *)in,*xout = (PetscReal*)out;

  PetscFunctionBegin;
  if (*datatype != MPIU_REAL) {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL data types");
    MPI_Abort(MPI_COMM_SELF,1);
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
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,((PetscObject)xin)->comm);CHKERRQ(ierr);
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
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local Min */
  ierr = VecMin_Seq(xin,idx,&work);CHKERRQ(ierr);

  /* Find the global Min */
  if (!idx) {
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPIU_MIN,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else {
    PetscReal work2[2],z2[2];
    PetscInt  rstart;

    ierr = VecGetOwnershipRange(xin,&rstart,PETSC_NULL);CHKERRQ(ierr);
    work2[0] = work;
    work2[1] = *idx + rstart;
    ierr = MPI_Allreduce(work2,z2,2,MPIU_REAL,VecMin_Local_Op,((PetscObject)xin)->comm);CHKERRQ(ierr);
    *z   = z2[0];
    *idx = (PetscInt)z2[1];
  }
  PetscFunctionReturn(0);
}








