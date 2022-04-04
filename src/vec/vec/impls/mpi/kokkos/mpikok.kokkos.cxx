
/*
   This file contains routines for Parallel vector operations.
 */

#include <petscvec_kokkos.hpp>
#include <petsc/private/vecimpl.h> /* for struct Vec */
#include <../src/vec/vec/impls/mpi/pvecimpl.h> /* for VecCreate/Destroy_MPI */
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp>

PetscErrorCode VecDestroy_MPIKokkos(Vec v)
{
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  delete veckok;
  PetscCall(VecDestroy_MPI(v));
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_MPIKokkos(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    PetscCall(VecNorm_SeqKokkos(xin,NORM_2,&work));
    work *= work;
    PetscCall(MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
    *z    = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    PetscCall(VecNorm_SeqKokkos(xin,NORM_1,&work));
    /* Find the global max */
    PetscCall(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    PetscCall(VecNorm_SeqKokkos(xin,NORM_INFINITY,&work));
    /* Find the global max */
    PetscCall(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    PetscCall(VecNorm_SeqKokkos(xin,NORM_1,temp));
    PetscCall(VecNorm_SeqKokkos(xin,NORM_2,temp+1));
    temp[1] = temp[1]*temp[1];
    PetscCall(MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

/* z = y^H x */
PetscErrorCode VecDot_MPIKokkos(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;

  PetscFunctionBegin;
  PetscCall(VecDot_SeqKokkos(xin,yin,&work));
  PetscCall(MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPIKokkos(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;

  PetscFunctionBegin;
  if (nv > 128) PetscCall(PetscMalloc1(nv,&work));
  PetscCall(VecMDot_SeqKokkos(xin,nv,y,work));
  PetscCall(MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  if (nv > 128) PetscCall(PetscFree(work));
  PetscFunctionReturn(0);
}

/* z = y^T x */
PetscErrorCode VecTDot_MPIKokkos(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;

  PetscFunctionBegin;
  PetscCall(VecTDot_SeqKokkos(xin,yin,&work));
  PetscCall(MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMTDot_MPIKokkos(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;

  PetscFunctionBegin;
  if (nv > 128) PetscCall(PetscMalloc1(nv,&work));
  PetscCall(VecMTDot_SeqKokkos(xin,nv,y,work));
  PetscCall(MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  if (nv > 128) PetscCall(PetscFree(work));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMax_MPIKokkos(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local max */
  PetscCall(VecMax_SeqKokkos(xin,idx,&work));
#if defined(PETSC_HAVE_MPIUNI)
  *z = work;
#else
  /* Find the global max */
  if (!idx) { /* User does not need idx */
    PetscCall(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx + xin->map->rstart;
    PetscCall(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MAXLOC,PetscObjectComm((PetscObject)xin)));
    *z    = out.v;
    *idx  = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_MPIKokkos(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local Min */
  PetscCall(VecMin_SeqKokkos(xin,idx,&work));
#if defined(PETSC_HAVE_MPIUNI)
  *z = work;
#else
  /* Find the global Min */
  if (!idx) {
    PetscCall(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)xin)));
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx + xin->map->rstart;
    PetscCall(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MINLOC,PetscObjectComm((PetscObject)xin)));
    *z    = out.v;
    *idx  = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecDuplicate_MPIKokkos(Vec win,Vec *vv)
{
  Vec            v;
  Vec_MPI        *vecmpi;
  Vec_Kokkos     *veckok;

  PetscFunctionBegin;
  /* Reuse VecDuplicate_MPI, which contains a lot of stuff */
  PetscCall(VecDuplicate_MPI(win,&v)); /* after the call, v is a VECMPI, with data zero'ed */
  PetscCall(PetscObjectChangeTypeName((PetscObject)v,VECMPIKOKKOS));
  PetscCall(PetscMemcpy(v->ops,win->ops,sizeof(struct _VecOps)));

  /* Build the Vec_Kokkos struct */
  vecmpi = static_cast<Vec_MPI*>(v->data);
  veckok = new Vec_Kokkos(v->map->n,vecmpi->array);
  Kokkos::deep_copy(veckok->v_dual.view_device(),0.0);
  v->spptr       = veckok;
  v->offloadmask = PETSC_OFFLOAD_KOKKOS;
  *vv = v;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_MPIKokkos(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscScalar    work[2],sum[2];

  PetscFunctionBegin;
  PetscCall(VecDotNorm2_SeqKokkos(s,t,work,work+1));
  PetscCall(MPIU_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s)));
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

static PetscErrorCode VecGetSubVector_MPIKokkos(Vec x,IS is,Vec *y)
{
  PetscFunctionBegin;
  PetscCall(VecGetSubVector_Kokkos_Private(x,PETSC_TRUE,is,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetOps_MPIKokkos(Vec v)
{
  PetscFunctionBegin;
  v->ops->abs                    = VecAbs_SeqKokkos;
  v->ops->reciprocal             = VecReciprocal_SeqKokkos;
  v->ops->pointwisemult          = VecPointwiseMult_SeqKokkos;
  v->ops->setrandom              = VecSetRandom_SeqKokkos;
  v->ops->dotnorm2               = VecDotNorm2_MPIKokkos;
  v->ops->waxpy                  = VecWAXPY_SeqKokkos;
  v->ops->norm                   = VecNorm_MPIKokkos;
  v->ops->min                    = VecMin_MPIKokkos;
  v->ops->max                    = VecMax_MPIKokkos;
  v->ops->sum                    = VecSum_SeqKokkos;
  v->ops->shift                  = VecShift_SeqKokkos;
  v->ops->scale                  = VecScale_SeqKokkos;
  v->ops->copy                   = VecCopy_SeqKokkos;
  v->ops->set                    = VecSet_SeqKokkos;
  v->ops->swap                   = VecSwap_SeqKokkos;
  v->ops->axpy                   = VecAXPY_SeqKokkos;
  v->ops->axpby                  = VecAXPBY_SeqKokkos;
  v->ops->maxpy                  = VecMAXPY_SeqKokkos;
  v->ops->aypx                   = VecAYPX_SeqKokkos;
  v->ops->axpbypcz               = VecAXPBYPCZ_SeqKokkos;
  v->ops->pointwisedivide        = VecPointwiseDivide_SeqKokkos;
  v->ops->placearray             = VecPlaceArray_SeqKokkos;
  v->ops->replacearray           = VecReplaceArray_SeqKokkos;
  v->ops->resetarray             = VecResetArray_SeqKokkos;

  v->ops->dot                    = VecDot_MPIKokkos;
  v->ops->tdot                   = VecTDot_MPIKokkos;
  v->ops->mdot                   = VecMDot_MPIKokkos;
  v->ops->mtdot                  = VecMTDot_MPIKokkos;

  v->ops->dot_local              = VecDot_SeqKokkos;
  v->ops->tdot_local             = VecTDot_SeqKokkos;
  v->ops->mdot_local             = VecMDot_SeqKokkos;
  v->ops->mtdot_local            = VecMTDot_SeqKokkos;

  v->ops->norm_local             = VecNorm_SeqKokkos;
  v->ops->duplicate              = VecDuplicate_MPIKokkos;
  v->ops->destroy                = VecDestroy_MPIKokkos;
  v->ops->getlocalvector         = VecGetLocalVector_SeqKokkos;
  v->ops->restorelocalvector     = VecRestoreLocalVector_SeqKokkos;
  v->ops->getlocalvectorread     = VecGetLocalVector_SeqKokkos;
  v->ops->restorelocalvectorread = VecRestoreLocalVector_SeqKokkos;
  v->ops->getarraywrite          = VecGetArrayWrite_SeqKokkos;
  v->ops->getarray               = VecGetArray_SeqKokkos;
  v->ops->restorearray           = VecRestoreArray_SeqKokkos;
  v->ops->getarrayandmemtype     = VecGetArrayAndMemType_SeqKokkos;
  v->ops->restorearrayandmemtype = VecRestoreArrayAndMemType_SeqKokkos;
  v->ops->getarraywriteandmemtype= VecGetArrayWriteAndMemType_SeqKokkos;
  v->ops->getsubvector           = VecGetSubVector_MPIKokkos;
  v->ops->restoresubvector       = VecRestoreSubVector_SeqKokkos;
  PetscFunctionReturn(0);
}

/*MC
   VECMPIKOKKOS - VECMPIKOKKOS = "mpikokkos" - The basic parallel vector, modified to use Kokkos

   Options Database Keys:
. -vec_type mpikokkos - sets the vector type to VECMPIKOKKOS during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIKokkosWithArray(), VECMPI, VecType, VecCreateMPI()
M*/
PetscErrorCode VecCreate_MPIKokkos(Vec v)
{
  Vec_MPI    *vecmpi;
  Vec_Kokkos *veckok;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscLayoutSetUp(v->map));
  PetscCall(VecCreate_MPI(v));  /* Calloc host array */

  vecmpi = static_cast<Vec_MPI*>(v->data);
  PetscCall(PetscObjectChangeTypeName((PetscObject)v,VECMPIKOKKOS));
  PetscCall(VecSetOps_MPIKokkos(v));
  veckok = new Vec_Kokkos(v->map->n,vecmpi->array,NULL); /* Alloc device array but do not init it */
  v->spptr = static_cast<void*>(veckok);
  v->offloadmask = PETSC_OFFLOAD_KOKKOS;
  PetscFunctionReturn(0);
}

/*@C
   VecCreateMPIKokkosWithArray - Creates a parallel, array-style vector,
   where the user provides the GPU array space to store the vector values.

   Collective

   Input Parameters:
+  comm  - the MPI communicator to use
.  bs    - block size, same meaning as VecSetBlockSize()
.  n     - local vector length, cannot be PETSC_DECIDE
.  N     - global vector length (or PETSC_DECIDE to have calculated)
-  array - the user provided GPU array to store the vector values

   Output Parameter:
.  vv - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is NULL, then VecKokkosPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateSeqKokkosWithArray(), VecCreateMPIWithArray(), VecCreateSeqWithArray(),
          VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()

@*/
PetscErrorCode  VecCreateMPIKokkosWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar darray[],Vec *v)
{
  Vec            w;
  Vec_Kokkos     *veckok;
  Vec_MPI        *vecmpi;
  PetscScalar    *harray;

  PetscFunctionBegin;
  PetscCheck(n != PETSC_DECIDE,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscSplitOwnership(comm,&n,&N));
  PetscCall(VecCreate(comm,&w));
  PetscCall(VecSetSizes(w,n,N));
  PetscCall(VecSetBlockSize(w,bs));
  PetscCall(PetscLayoutSetUp(w->map));

  if (std::is_same<DefaultMemorySpace,Kokkos::HostSpace>::value) {harray = const_cast<PetscScalar*>(darray);}
  else PetscCall(PetscMalloc1(w->map->n,&harray)); /* If device is not the same as host, allocate the host array ourselves */

  PetscCall(VecCreate_MPI_Private(w,PETSC_FALSE/*alloc*/,0/*nghost*/,harray)); /* Build a sequential vector with provided data */
  vecmpi = static_cast<Vec_MPI*>(w->data);

  if (!std::is_same<DefaultMemorySpace,Kokkos::HostSpace>::value) vecmpi->array_allocated = harray; /* The host array was allocated by petsc */

  PetscCall(PetscObjectChangeTypeName((PetscObject)w,VECMPIKOKKOS));
  PetscCall(VecSetOps_MPIKokkos(w));
  veckok = new Vec_Kokkos(n,harray,const_cast<PetscScalar*>(darray));
  veckok->v_dual.modify_device(); /* Mark the device is modified */
  w->spptr = static_cast<void*>(veckok);
  w->offloadmask = PETSC_OFFLOAD_KOKKOS;
  *v = w;
  PetscFunctionReturn(0);
}

/*
   VecCreateMPIKokkosWithArrays_Private - Creates a Kokkos parallel, array-style vector
   with user-provided arrays on host and device.

   Collective

   Input Parameter:
+  comm - the communicator
.  bs - the block size
.  n - the local vector length
.  N - the global vector length
-  harray - host memory where the vector elements are to be stored.
-  darray - device memory where the vector elements are to be stored.

   Output Parameter:
.  v - the vector

   Notes:
   If there is no device, then harray and darray must be the same.
   If n is not zero, then harray and darray must be allocated.
   After the call, the created vector is supposed to be in a synchronized state, i.e.,
   we suppose harray and darray have the same data.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.
*/
PetscErrorCode  VecCreateMPIKokkosWithArrays_Private(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar harray[],const PetscScalar darray[],Vec *v)
{
  Vec w;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  if (n) {
    PetscValidScalarPointer(harray,5);
    PetscCheck(darray,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"darray cannot be NULL");
  }
  if (std::is_same<DefaultMemorySpace,Kokkos::HostSpace>::value) PetscCheck(harray == darray,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"harray and darray must be the same");
  PetscCall(VecCreateMPIWithArray(comm,bs,n,N,harray,&w));
  PetscCall(PetscObjectChangeTypeName((PetscObject)w,VECMPIKOKKOS)); /* Change it to Kokkos */
  PetscCall(VecSetOps_MPIKokkos(w));
  PetscCallCXX(w->spptr = new Vec_Kokkos(n,const_cast<PetscScalar*>(harray),const_cast<PetscScalar*>(darray)));
  w->offloadmask = PETSC_OFFLOAD_KOKKOS;
  *v = w;
  PetscFunctionReturn(0);
}

/*MC
   VECKOKKOS - VECKOKKOS = "kokkos" - The basic vector, modified to use Kokkos

   Options Database Keys:
. -vec_type kokkos - sets the vector type to VECKOKKOS during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIKokkosWithArray(), VECMPI, VecType, VecCreateMPI()
M*/
PetscErrorCode VecCreate_Kokkos(Vec v)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)v),&size));
  if (size == 1) PetscCall(VecSetType(v,VECSEQKOKKOS));
  else PetscCall(VecSetType(v,VECMPIKOKKOS));
  PetscFunctionReturn(0);
}
