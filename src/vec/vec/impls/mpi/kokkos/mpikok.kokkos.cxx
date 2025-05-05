/*
   This file contains routines for Parallel vector operations.
 */
#include <petsc_kokkos.hpp>
#include <petscvec_kokkos.hpp>
#include <petsc/private/deviceimpl.h>
#include <petsc/private/vecimpl.h>             /* for struct Vec */
#include <../src/vec/vec/impls/mpi/pvecimpl.h> /* for VecCreate/Destroy_MPI */
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp>
#include <petscsf.h>

static PetscErrorCode VecDestroy_MPIKokkos(Vec v)
{
  PetscFunctionBegin;
  delete static_cast<Vec_Kokkos *>(v->spptr);
  PetscCall(VecDestroy_MPI(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecNorm_MPIKokkos(Vec xin, NormType type, PetscReal *z)
{
  PetscFunctionBegin;
  PetscCall(VecNorm_MPI_Default(xin, type, z, VecNorm_SeqKokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecErrorWeightedNorms_MPIKokkos(Vec U, Vec Y, Vec E, NormType wnormtype, PetscReal atol, Vec vatol, PetscReal rtol, Vec vrtol, PetscReal ignore_max, PetscReal *norm, PetscInt *norm_loc, PetscReal *norma, PetscInt *norma_loc, PetscReal *normr, PetscInt *normr_loc)
{
  PetscFunctionBegin;
  PetscCall(VecErrorWeightedNorms_MPI_Default(U, Y, E, wnormtype, atol, vatol, rtol, vrtol, ignore_max, norm, norm_loc, norma, norma_loc, normr, normr_loc, VecErrorWeightedNorms_SeqKokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = y^H x */
static PetscErrorCode VecDot_MPIKokkos(Vec xin, Vec yin, PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecXDot_MPI_Default(xin, yin, z, VecDot_SeqKokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = y^T x */
static PetscErrorCode VecTDot_MPIKokkos(Vec xin, Vec yin, PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecXDot_MPI_Default(xin, yin, z, VecTDot_SeqKokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMDot_MPIKokkos(Vec xin, PetscInt nv, const Vec y[], PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI_Default(xin, nv, y, z, VecMDot_SeqKokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMTDot_MPIKokkos(Vec xin, PetscInt nv, const Vec y[], PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI_Default(xin, nv, y, z, VecMTDot_SeqKokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMDot_MPIKokkos_GEMV(Vec xin, PetscInt nv, const Vec y[], PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI_Default(xin, nv, y, z, VecMDot_SeqKokkos_GEMV));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMTDot_MPIKokkos_GEMV(Vec xin, PetscInt nv, const Vec y[], PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI_Default(xin, nv, y, z, VecMTDot_SeqKokkos_GEMV));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMax_MPIKokkos(Vec xin, PetscInt *idx, PetscReal *z)
{
  const MPI_Op ops[] = {MPIU_MAXLOC, MPIU_MAX};

  PetscFunctionBegin;
  PetscCall(VecMinMax_MPI_Default(xin, idx, z, VecMax_SeqKokkos, ops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMin_MPIKokkos(Vec xin, PetscInt *idx, PetscReal *z)
{
  const MPI_Op ops[] = {MPIU_MINLOC, MPIU_MIN};

  PetscFunctionBegin;
  PetscCall(VecMinMax_MPI_Default(xin, idx, z, VecMin_SeqKokkos, ops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecCreate_MPIKokkos_Common(Vec); // forward declaration

static PetscErrorCode VecDuplicate_MPIKokkos(Vec win, Vec *vv)
{
  Vec         v;
  Vec_Kokkos *veckok;
  Vec_MPI    *wdata = (Vec_MPI *)win->data;

  PetscScalarKokkosDualView w_dual;

  PetscFunctionBegin;
  PetscCallCXX(w_dual = PetscScalarKokkosDualView("w_dual", win->map->n + wdata->nghost)); // Kokkos init's v_dual to zero

  /* Reuse VecDuplicate_MPI, which contains a lot of stuff */
  PetscCall(VecDuplicateWithArray_MPI(win, w_dual.view_host().data(), &v)); /* after the call, v is a VECMPI */
  PetscCall(PetscObjectChangeTypeName((PetscObject)v, VECMPIKOKKOS));
  PetscCall(VecCreate_MPIKokkos_Common(v));
  v->ops[0] = win->ops[0]; // always follow ops[] in win

  /* Build the Vec_Kokkos struct */
  veckok         = new Vec_Kokkos(v->map->n, w_dual.view_host().data(), w_dual.view_device().data());
  veckok->w_dual = w_dual;
  v->spptr       = veckok;
  *vv            = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecDotNorm2_MPIKokkos(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscFunctionBegin;
  PetscCall(VecDotNorm2_MPI_Default(s, t, dp, nm, VecDotNorm2_SeqKokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecGetSubVector_MPIKokkos(Vec x, IS is, Vec *y)
{
  PetscFunctionBegin;
  PetscCall(VecGetSubVector_Kokkos_Private(x, PETSC_TRUE, is, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSetPreallocationCOO_MPIKokkos(Vec x, PetscCount ncoo, const PetscInt coo_i[])
{
  const auto vecmpi = static_cast<Vec_MPI *>(x->data);
  const auto veckok = static_cast<Vec_Kokkos *>(x->spptr);
  PetscInt   m;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(x, &m));
  PetscCall(VecSetPreallocationCOO_MPI(x, ncoo, coo_i));
  PetscCall(veckok->SetUpCOO(vecmpi, m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSetValuesCOO_MPIKokkos(Vec x, const PetscScalar v[], InsertMode imode)
{
  const auto                  vecmpi  = static_cast<Vec_MPI *>(x->data);
  const auto                  veckok  = static_cast<Vec_Kokkos *>(x->spptr);
  const PetscCountKokkosView &jmap1   = veckok->jmap1_d;
  const PetscCountKokkosView &perm1   = veckok->perm1_d;
  const PetscCountKokkosView &imap2   = veckok->imap2_d;
  const PetscCountKokkosView &jmap2   = veckok->jmap2_d;
  const PetscCountKokkosView &perm2   = veckok->perm2_d;
  const PetscCountKokkosView &Cperm   = veckok->Cperm_d;
  PetscScalarKokkosView      &sendbuf = veckok->sendbuf_d;
  PetscScalarKokkosView      &recvbuf = veckok->recvbuf_d;
  PetscScalarKokkosView       xv;
  ConstPetscScalarKokkosView  vv;
  PetscMemType                memtype;
  PetscInt                    m;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(x, &m));
  PetscCall(PetscGetMemType(v, &memtype));
  if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we might need to copy it to device if any */
    vv = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscScalarKokkosViewHost(const_cast<PetscScalar *>(v), vecmpi->coo_n));
  } else {
    vv = ConstPetscScalarKokkosView(v, vecmpi->coo_n); /* Directly use v[]'s memory */
  }

  /* Pack entries to be sent to remote */
  Kokkos::parallel_for(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, vecmpi->sendlen), KOKKOS_LAMBDA(const PetscCount i) { sendbuf(i) = vv(Cperm(i)); });
  PetscCall(PetscSFReduceWithMemTypeBegin(vecmpi->coo_sf, MPIU_SCALAR, PETSC_MEMTYPE_KOKKOS, sendbuf.data(), PETSC_MEMTYPE_KOKKOS, recvbuf.data(), MPI_REPLACE));

  if (imode == INSERT_VALUES) PetscCall(VecGetKokkosViewWrite(x, &xv)); /* write vector */
  else PetscCall(VecGetKokkosView(x, &xv));                             /* read & write vector */

  Kokkos::parallel_for(
    Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, m), KOKKOS_LAMBDA(const PetscCount i) {
      PetscScalar sum = 0.0;
      for (PetscCount k = jmap1(i); k < jmap1(i + 1); k++) sum += vv(perm1(k));
      xv(i) = (imode == INSERT_VALUES ? 0.0 : xv(i)) + sum;
    });

  PetscCall(PetscSFReduceEnd(vecmpi->coo_sf, MPIU_SCALAR, sendbuf.data(), recvbuf.data(), MPI_REPLACE));

  /* Add received remote entries */
  Kokkos::parallel_for(
    Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, vecmpi->nnz2), KOKKOS_LAMBDA(PetscCount i) {
      for (PetscCount k = jmap2(i); k < jmap2(i + 1); k++) xv(imap2(i)) += recvbuf(perm2(k));
    });

  if (imode == INSERT_VALUES) PetscCall(VecRestoreKokkosViewWrite(x, &xv));
  else PetscCall(VecRestoreKokkosView(x, &xv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Shared by all VecCreate/Duplicate routines for VecMPIKokkos
static PetscErrorCode VecCreate_MPIKokkos_Common(Vec v)
{
  PetscFunctionBegin;
  v->ops->abs             = VecAbs_SeqKokkos;
  v->ops->reciprocal      = VecReciprocal_SeqKokkos;
  v->ops->pointwisemult   = VecPointwiseMult_SeqKokkos;
  v->ops->setrandom       = VecSetRandom_SeqKokkos;
  v->ops->dotnorm2        = VecDotNorm2_MPIKokkos;
  v->ops->waxpy           = VecWAXPY_SeqKokkos;
  v->ops->norm            = VecNorm_MPIKokkos;
  v->ops->min             = VecMin_MPIKokkos;
  v->ops->max             = VecMax_MPIKokkos;
  v->ops->sum             = VecSum_SeqKokkos;
  v->ops->shift           = VecShift_SeqKokkos;
  v->ops->scale           = VecScale_SeqKokkos;
  v->ops->copy            = VecCopy_SeqKokkos;
  v->ops->set             = VecSet_SeqKokkos;
  v->ops->swap            = VecSwap_SeqKokkos;
  v->ops->axpy            = VecAXPY_SeqKokkos;
  v->ops->axpby           = VecAXPBY_SeqKokkos;
  v->ops->maxpy           = VecMAXPY_SeqKokkos;
  v->ops->aypx            = VecAYPX_SeqKokkos;
  v->ops->axpbypcz        = VecAXPBYPCZ_SeqKokkos;
  v->ops->pointwisedivide = VecPointwiseDivide_SeqKokkos;
  v->ops->placearray      = VecPlaceArray_SeqKokkos;
  v->ops->replacearray    = VecReplaceArray_SeqKokkos;
  v->ops->resetarray      = VecResetArray_SeqKokkos;

  v->ops->dot   = VecDot_MPIKokkos;
  v->ops->tdot  = VecTDot_MPIKokkos;
  v->ops->mdot  = VecMDot_MPIKokkos;
  v->ops->mtdot = VecMTDot_MPIKokkos;

  v->ops->dot_local   = VecDot_SeqKokkos;
  v->ops->tdot_local  = VecTDot_SeqKokkos;
  v->ops->mdot_local  = VecMDot_SeqKokkos;
  v->ops->mtdot_local = VecMTDot_SeqKokkos;

  v->ops->norm_local              = VecNorm_SeqKokkos;
  v->ops->duplicate               = VecDuplicate_MPIKokkos;
  v->ops->destroy                 = VecDestroy_MPIKokkos;
  v->ops->getlocalvector          = VecGetLocalVector_SeqKokkos;
  v->ops->restorelocalvector      = VecRestoreLocalVector_SeqKokkos;
  v->ops->getlocalvectorread      = VecGetLocalVector_SeqKokkos;
  v->ops->restorelocalvectorread  = VecRestoreLocalVector_SeqKokkos;
  v->ops->getarraywrite           = VecGetArrayWrite_SeqKokkos;
  v->ops->getarray                = VecGetArray_SeqKokkos;
  v->ops->restorearray            = VecRestoreArray_SeqKokkos;
  v->ops->getarrayandmemtype      = VecGetArrayAndMemType_SeqKokkos;
  v->ops->restorearrayandmemtype  = VecRestoreArrayAndMemType_SeqKokkos;
  v->ops->getarraywriteandmemtype = VecGetArrayWriteAndMemType_SeqKokkos;
  v->ops->getsubvector            = VecGetSubVector_MPIKokkos;
  v->ops->restoresubvector        = VecRestoreSubVector_SeqKokkos;

  v->ops->setpreallocationcoo = VecSetPreallocationCOO_MPIKokkos;
  v->ops->setvaluescoo        = VecSetValuesCOO_MPIKokkos;

  v->ops->errorwnorm = VecErrorWeightedNorms_MPIKokkos;

  v->offloadmask = PETSC_OFFLOAD_KOKKOS; // Mark this is a VECKOKKOS; We use this flag for cheap VECKOKKOS test.
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode VecConvert_MPI_MPIKokkos_inplace(Vec v)
{
  Vec_MPI *vecmpi;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscLayoutSetUp(v->map));
  PetscCall(PetscObjectChangeTypeName((PetscObject)v, VECMPIKOKKOS));
  PetscCall(VecCreate_MPIKokkos_Common(v));
  PetscCheck(!v->spptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "v->spptr not NULL");
  vecmpi = static_cast<Vec_MPI *>(v->data);
  PetscCallCXX(v->spptr = new Vec_Kokkos(v->map->n, vecmpi->array, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Duplicate a VECMPIKOKKOS
static PetscErrorCode VecDuplicateVecs_MPIKokkos_GEMV(Vec w, PetscInt m, Vec *V[])
{
  PetscInt64                lda; // use 64-bit as we will do "m * lda"
  PetscScalar              *array_h, *array_d;
  PetscLayout               map;
  Vec_MPI                  *wmpi = (Vec_MPI *)w->data;
  PetscScalarKokkosDualView w_dual;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck()); // as we'll call kokkos_malloc()
  if (wmpi->nghost) {                      // currently only do GEMV optimization for vectors without ghosts
    w->ops->duplicatevecs = VecDuplicateVecs_Default;
    PetscCall(VecDuplicateVecs(w, m, V));
  } else {
    PetscCall(PetscMalloc1(m, V));
    PetscCall(VecGetLayout(w, &map));
    VecGetLocalSizeAligned(w, 64, &lda); // get in lda the 64-bytes aligned local size

    // See comments in VecCreate_SeqKokkos() on why we use DualView to allocate the memory
    PetscCallCXX(w_dual = PetscScalarKokkosDualView("VecDuplicateVecs", m * lda)); // Kokkos init's w_dual to zero

    // create the m vectors with raw arrays
    array_h = w_dual.view_host().data();
    array_d = w_dual.view_device().data();
    for (PetscInt i = 0; i < m; i++) {
      Vec v;
      PetscCall(VecCreateMPIKokkosWithLayoutAndArrays_Private(map, &array_h[i * lda], &array_d[i * lda], &v));
      PetscCallCXX(static_cast<Vec_Kokkos *>(v->spptr)->v_dual.modify_host()); // as we only init'ed array_h
      PetscCall(PetscObjectListDuplicate(((PetscObject)w)->olist, &((PetscObject)v)->olist));
      PetscCall(PetscFunctionListDuplicate(((PetscObject)w)->qlist, &((PetscObject)v)->qlist));
      v->ops[0]             = w->ops[0];
      v->stash.donotstash   = w->stash.donotstash;
      v->stash.ignorenegidx = w->stash.ignorenegidx;
      v->stash.bs           = w->stash.bs;
      v->bstash.bs          = w->bstash.bs;
      (*V)[i]               = v;
    }

    // let the first vector own the raw arrays, so when it is destroyed it will free the arrays
    if (m) {
      Vec v = (*V)[0];

      static_cast<Vec_Kokkos *>(v->spptr)->w_dual = w_dual; // stash the memory
      // disable replacearray of the first vector, as freeing its memory also frees others in the group.
      // But replacearray of others is ok, as they don't own their array.
      if (m > 1) v->ops->replacearray = VecReplaceArray_Default_GEMV_Error;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   VECMPIKOKKOS - VECMPIKOKKOS = "mpikokkos" - The basic parallel vector, modified to use Kokkos

   Options Database Keys:
. -vec_type mpikokkos - sets the vector type to VECMPIKOKKOS during a call to VecSetFromOptions()

  Level: beginner

.seealso: `VecCreate()`, `VecSetType()`, `VecSetFromOptions()`, `VecCreateMPIKokkosWithArray()`, `VECMPI`, `VecType`, `VecCreateMPI()`
M*/
PetscErrorCode VecCreate_MPIKokkos(Vec v)
{
  PetscBool                 mdot_use_gemv  = PETSC_TRUE;
  PetscBool                 maxpy_use_gemv = PETSC_FALSE; // default is false as we saw bad performance with vendors' GEMV with tall skinny matrices.
  PetscScalarKokkosDualView v_dual;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscLayoutSetUp(v->map));

  PetscCallCXX(v_dual = PetscScalarKokkosDualView("v_dual", v->map->n)); // Kokkos init's v_dual to zero
  PetscCall(VecCreate_MPI_Private(v, PETSC_FALSE, 0, v_dual.view_host().data()));

  PetscCall(PetscObjectChangeTypeName((PetscObject)v, VECMPIKOKKOS));
  PetscCall(VecCreate_MPIKokkos_Common(v));
  PetscCheck(!v->spptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "v->spptr not NULL");
  PetscCallCXX(v->spptr = new Vec_Kokkos(v_dual));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-vec_mdot_use_gemv", &mdot_use_gemv, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-vec_maxpy_use_gemv", &maxpy_use_gemv, NULL));

  // allocate multiple vectors together
  if (mdot_use_gemv || maxpy_use_gemv) v->ops[0].duplicatevecs = VecDuplicateVecs_MPIKokkos_GEMV;

  if (mdot_use_gemv) {
    v->ops[0].mdot        = VecMDot_MPIKokkos_GEMV;
    v->ops[0].mtdot       = VecMTDot_MPIKokkos_GEMV;
    v->ops[0].mdot_local  = VecMDot_SeqKokkos_GEMV;
    v->ops[0].mtdot_local = VecMTDot_SeqKokkos_GEMV;
  }

  if (maxpy_use_gemv) v->ops[0].maxpy = VecMAXPY_SeqKokkos_GEMV;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create a VECMPIKOKKOS with layout and arrays
PetscErrorCode VecCreateMPIKokkosWithLayoutAndArrays_Private(PetscLayout map, const PetscScalar harray[], const PetscScalar darray[], Vec *v)
{
  Vec w;

  PetscFunctionBegin;
  if (map->n > 0) PetscCheck(darray, map->comm, PETSC_ERR_ARG_WRONG, "darray cannot be NULL");
#if defined(KOKKOS_ENABLE_UNIFIED_MEMORY)
  PetscCheck(harray == darray, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "harray and darray must be the same");
#endif
  PetscCall(VecCreateMPIWithLayoutAndArray_Private(map, harray, &w));
  PetscCall(PetscObjectChangeTypeName((PetscObject)w, VECMPIKOKKOS)); // Change it to VECKOKKOS
  PetscCall(VecCreate_MPIKokkos_Common(w));
  PetscCallCXX(w->spptr = new Vec_Kokkos(map->n, const_cast<PetscScalar *>(harray), const_cast<PetscScalar *>(darray)));
  *v = w;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  VecCreateMPIKokkosWithArray - Creates a parallel, array-style vector,
  where the user provides the GPU array space to store the vector values.

  Collective

  Input Parameters:
+ comm   - the MPI communicator to use
. bs     - block size, same meaning as VecSetBlockSize()
. n      - local vector length, cannot be PETSC_DECIDE
. N      - global vector length (or PETSC_DECIDE to have calculated)
- darray - the user provided GPU array to store the vector values

  Output Parameter:
. v - the vector

  Notes:
  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
  same type as an existing vector.

  If the user-provided array is NULL, then VecKokkosPlaceArray() can be used
  at a later stage to SET the array for storing the vector values.

  PETSc does NOT free the array when the vector is destroyed via VecDestroy().
  The user should not free the array until the vector is destroyed.

  Level: intermediate

.seealso: `VecCreateSeqKokkosWithArray()`, `VecCreateMPIWithArray()`, `VecCreateSeqWithArray()`,
          `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`, `VecCreateGhost()`,
          `VecCreateMPI()`, `VecCreateGhostWithArray()`, `VecPlaceArray()`

@*/
PetscErrorCode VecCreateMPIKokkosWithArray(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar darray[], Vec *v)
{
  Vec          w;
  Vec_Kokkos  *veckok;
  Vec_MPI     *vecmpi;
  PetscScalar *harray;

  PetscFunctionBegin;
  PetscCheck(n != PETSC_DECIDE, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must set local size of vector");
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscSplitOwnership(comm, &n, &N));
  PetscCall(VecCreate(comm, &w));
  PetscCall(VecSetSizes(w, n, N));
  PetscCall(VecSetBlockSize(w, bs));
  PetscCall(PetscLayoutSetUp(w->map));

  if (std::is_same<DefaultMemorySpace, HostMirrorMemorySpace>::value) {
    harray = const_cast<PetscScalar *>(darray);
  } else PetscCall(PetscMalloc1(w->map->n, &harray)); /* If device is not the same as host, allocate the host array ourselves */

  PetscCall(VecCreate_MPI_Private(w, PETSC_FALSE /*alloc*/, 0 /*nghost*/, harray)); /* Build a sequential vector with provided data */
  vecmpi = static_cast<Vec_MPI *>(w->data);

  if (!std::is_same<DefaultMemorySpace, HostMirrorMemorySpace>::value) vecmpi->array_allocated = harray; /* The host array was allocated by PETSc */

  PetscCall(PetscObjectChangeTypeName((PetscObject)w, VECMPIKOKKOS));
  PetscCall(VecCreate_MPIKokkos_Common(w));
  veckok = new Vec_Kokkos(n, harray, const_cast<PetscScalar *>(darray));
  veckok->v_dual.modify_device(); /* Mark the device is modified */
  w->spptr = static_cast<void *>(veckok);
  *v       = w;
  PetscFunctionReturn(PETSC_SUCCESS);
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
PetscErrorCode VecCreateMPIKokkosWithArrays_Private(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar harray[], const PetscScalar darray[], Vec *v)
{
  Vec w;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  if (n) {
    PetscAssertPointer(harray, 5);
    PetscCheck(darray, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "darray cannot be NULL");
  }
  if (std::is_same<DefaultMemorySpace, HostMirrorMemorySpace>::value) PetscCheck(harray == darray, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "harray and darray must be the same");
  PetscCall(VecCreateMPIWithArray(comm, bs, n, N, harray, &w));
  PetscCall(PetscObjectChangeTypeName((PetscObject)w, VECMPIKOKKOS)); /* Change it to Kokkos */
  PetscCall(VecCreate_MPIKokkos_Common(w));
  PetscCallCXX(w->spptr = new Vec_Kokkos(n, const_cast<PetscScalar *>(harray), const_cast<PetscScalar *>(darray)));
  *v = w;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   VECKOKKOS - VECKOKKOS = "kokkos" - The basic vector, modified to use Kokkos

   Options Database Keys:
. -vec_type kokkos - sets the vector type to VECKOKKOS during a call to VecSetFromOptions()

  Level: beginner

.seealso: `VecCreate()`, `VecSetType()`, `VecSetFromOptions()`, `VecCreateMPIKokkosWithArray()`, `VECMPI`, `VecType`, `VecCreateMPI()`
M*/
PetscErrorCode VecCreate_Kokkos(Vec v)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)v), &size));
  if (size == 1) PetscCall(VecSetType(v, VECSEQKOKKOS));
  else PetscCall(VecSetType(v, VECMPIKOKKOS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
