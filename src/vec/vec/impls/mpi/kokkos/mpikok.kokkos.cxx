
/*
   This file contains routines for Parallel vector operations.
 */

#include <petscvec_kokkos.hpp>
#include <petsc/private/vecimpl.h> /* for struct Vec */
#include <../src/vec/vec/impls/mpi/pvecimpl.h> /* for VecCreate/Destroy_MPI */
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp>

PetscErrorCode VecDestroy_MPIKokkos(Vec v)
{
  PetscErrorCode ierr;
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  delete veckok;
  ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_MPIKokkos(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr  = VecNorm_SeqKokkos(xin,NORM_2,&work);CHKERRQ(ierr);
    work *= work;
    ierr  = MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z    = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_SeqKokkos(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_SeqKokkos(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_SeqKokkos(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_SeqKokkos(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

/* z = y^H x */
PetscErrorCode VecDot_MPIKokkos(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqKokkos(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPIKokkos(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);}
  ierr = VecMDot_SeqKokkos(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  if (nv > 128) {ierr = PetscFree(work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* z = y^T x */
PetscErrorCode VecTDot_MPIKokkos(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDot_SeqKokkos(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMTDot_MPIKokkos(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);}
  ierr = VecMTDot_SeqKokkos(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  if (nv > 128) {ierr = PetscFree(work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode VecMax_MPIKokkos(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local max */
  ierr = VecMax_SeqKokkos(xin,idx,&work);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIUNI)
  *z = work;
#else
  /* Find the global max */
  if (!idx) { /* User does not need idx */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx + xin->map->rstart;
    ierr  = MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MAXLOC,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z    = out.v;
    *idx  = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_MPIKokkos(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local Min */
  ierr = VecMin_SeqKokkos(xin,idx,&work);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIUNI)
  *z = work;
#else
  /* Find the global Min */
  if (!idx) {
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx + xin->map->rstart;
    ierr  = MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MINLOC,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z    = out.v;
    *idx  = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecDuplicate_MPIKokkos(Vec win,Vec *vv)
{
  PetscErrorCode ierr;
  Vec            v;
  Vec_MPI        *vecmpi;
  Vec_Kokkos     *veckok;

  PetscFunctionBegin;
  /* Reuse VecDuplicate_MPI, which contains a lot of stuff */
  ierr = VecDuplicate_MPI(win,&v);CHKERRQ(ierr); /* after the call, v is a VECMPI, with data zero'ed */
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECMPIKOKKOS);CHKERRQ(ierr);
  ierr = PetscMemcpy(v->ops,win->ops,sizeof(struct _VecOps));CHKERRQ(ierr);

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
  PetscErrorCode ierr;
  PetscScalar    work[2],sum[2];

  PetscFunctionBegin;
  ierr = VecDotNorm2_SeqKokkos(s,t,work,work+1);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s));CHKERRMPI(ierr);
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

static PetscErrorCode VecGetSubVector_MPIKokkos(Vec x,IS is,Vec *y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetSubVector_Kokkos_Private(x,PETSC_TRUE,is,y);CHKERRQ(ierr);
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

PetscErrorCode VecCreate_MPIKokkos(Vec v)
{
  PetscErrorCode ierr;
  Vec_MPI        *vecmpi;
  Vec_Kokkos     *veckok;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);
  ierr = VecCreate_MPI(v);CHKERRQ(ierr);  /* Build a sequential vector, allocate array */
  ierr = VecSet_Seq(v,0.0);CHKERRQ(ierr); /* Zero the host array */

  vecmpi = static_cast<Vec_MPI*>(v->data);
  ierr   = PetscObjectChangeTypeName((PetscObject)v,VECMPIKOKKOS);CHKERRQ(ierr);
  ierr   = VecSetOps_MPIKokkos(v);CHKERRQ(ierr);
  veckok = new Vec_Kokkos(v->map->n,vecmpi->array);
  Kokkos::deep_copy(veckok->v_dual.view_device(),0.0);
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
  PetscErrorCode ierr;
  Vec            w;
  Vec_Kokkos     *veckok;
  Vec_MPI        *vecmpi;
  PetscScalar    *harray;

  PetscFunctionBegin;
  if (n == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  ierr = VecCreate(comm,&w);CHKERRQ(ierr);
  ierr = VecSetSizes(w,n,N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(w,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(w->map);CHKERRQ(ierr);

  if (std::is_same<DefaultMemorySpace,Kokkos::HostSpace>::value) {harray = const_cast<PetscScalar*>(darray);}
  else {ierr = PetscMalloc1(w->map->n,&harray);CHKERRQ(ierr);} /* If device is not the same as host, allocate the host array ourselves */

  ierr   = VecCreate_MPI_Private(w,PETSC_FALSE/*alloc*/,0/*nghost*/,harray);CHKERRQ(ierr); /* Build a sequential vector with provided data */
  vecmpi = static_cast<Vec_MPI*>(w->data);

  if (!std::is_same<DefaultMemorySpace,Kokkos::HostSpace>::value) vecmpi->array_allocated = harray; /* The host array was allocated by petsc */

  ierr   = PetscObjectChangeTypeName((PetscObject)w,VECMPIKOKKOS);CHKERRQ(ierr);
  ierr   = VecSetOps_MPIKokkos(w);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  Vec            w;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  if (n && !harray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"harray cannot be NULL");
  if (n && !darray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"darray cannot be NULL");
  if (std::is_same<DefaultMemorySpace,Kokkos::HostSpace>::value && harray != darray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"harray and darray must be the same");
  ierr = VecCreateMPIWithArray(comm,bs,n,N,harray,&w);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)w,VECMPIKOKKOS);CHKERRQ(ierr); /* Change it to Kokkos */
  ierr = VecSetOps_MPIKokkos(w);CHKERRQ(ierr);
  CHKERRCXX(w->spptr = new Vec_Kokkos(n,const_cast<PetscScalar*>(harray),const_cast<PetscScalar*>(darray)));
  w->offloadmask = PETSC_OFFLOAD_KOKKOS;
  *v = w;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_Kokkos(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)v),&size);CHKERRMPI(ierr);
  if (size == 1) {ierr = VecSetType(v,VECSEQKOKKOS);CHKERRQ(ierr);}
  else {ierr = VecSetType(v,VECMPIKOKKOS);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
