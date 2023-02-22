#include <petscvec_kokkos.hpp>
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp>
#include <petscdevice.h>
#include <../src/ksp/pc/impls/pbjacobi/pbjacobi.h>

struct PC_PBJacobi_Kokkos {
  PetscScalarKokkosDualView diag_dual;

  PC_PBJacobi_Kokkos(PetscInt len, PetscScalar *diag_ptr_h)
  {
    PetscScalarKokkosViewHost diag_h(diag_ptr_h, len);
    auto                      diag_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), diag_h);
    diag_dual                        = PetscScalarKokkosDualView(diag_d, diag_h);
  }

  PetscErrorCode Update(const PetscScalar *diag_ptr_h)
  {
    PetscFunctionBegin;
    PetscCheck(diag_dual.view_host().data() == diag_ptr_h, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Host pointer has changed since last call");
    PetscCallCXX(diag_dual.modify_host()); /* mark the host has newer data */
    PetscCallCXX(diag_dual.sync_device());
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

/* Make 'transpose' a template parameter instead of a function input parameter, so that
 it will be a const in template instantiation and gets optimized out.
*/
template <PetscBool transpose>
static PetscErrorCode PCApplyOrTranspose_PBJacobi_Kokkos(PC pc, Vec x, Vec y)
{
  PC_PBJacobi               *jac   = (PC_PBJacobi *)pc->data;
  PC_PBJacobi_Kokkos        *pckok = static_cast<PC_PBJacobi_Kokkos *>(jac->spptr);
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      yv;
  PetscScalarKokkosView      Av = pckok->diag_dual.view_device();
  const PetscInt             bs = jac->bs, mbs = jac->mbs, bs2 = bs * bs;
  const char                *label = transpose ? "PCApplyTranspose_PBJacobi_Kokkos" : "PCApply_PBJacobi_Kokkos";

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  VecErrorIfNotKokkos(x);
  VecErrorIfNotKokkos(y);
  PetscCall(VecGetKokkosView(x, &xv));
  PetscCall(VecGetKokkosViewWrite(y, &yv));
  PetscCallCXX(Kokkos::parallel_for(
    label, bs * mbs, KOKKOS_LAMBDA(PetscInt row) {
      const PetscScalar *Ap, *xp;
      PetscScalar       *yp;
      PetscInt           i, j, k;

      k  = row / bs;                                /* k-th block */
      i  = row % bs;                                /* this thread deals with i-th row of the block */
      Ap = &Av(bs2 * k + i * (transpose ? bs : 1)); /* Ap points to the first entry of i-th row */
      xp = &xv(bs * k);
      yp = &yv(bs * k);
      /* multiply i-th row (column) with x */
      yp[i] = 0.0;
      for (j = 0; j < bs; j++) {
        yp[i] += Ap[0] * xp[j];
        Ap += (transpose ? 1 : bs); /* block is in column major order */
      }
    }));
  PetscCall(VecRestoreKokkosView(x, &xv));
  PetscCall(VecRestoreKokkosViewWrite(y, &yv));
  PetscCall(PetscLogGpuFlops(bs * bs * mbs * 2)); /* FMA on entries in all blocks */
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_PBJacobi_Kokkos(PC pc)
{
  PC_PBJacobi *jac = (PC_PBJacobi *)pc->data;

  PetscFunctionBegin;
  PetscCallCXX(delete static_cast<PC_PBJacobi_Kokkos *>(jac->spptr));
  PetscCall(PCDestroy_PBJacobi(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PCSetUp_PBJacobi_Kokkos(PC pc)
{
  PC_PBJacobi *jac = (PC_PBJacobi *)pc->data;
  PetscInt     len;

  PetscFunctionBegin;
  PetscCall(PCSetUp_PBJacobi_Host(pc)); /* Compute the inverse on host now. Might worth doing it on device directly */
  len = jac->bs * jac->bs * jac->mbs;
  if (!jac->spptr) {
    PetscCallCXX(jac->spptr = new PC_PBJacobi_Kokkos(len, const_cast<PetscScalar *>(jac->diag)));
  } else {
    PC_PBJacobi_Kokkos *pckok = static_cast<PC_PBJacobi_Kokkos *>(jac->spptr);
    PetscCall(pckok->Update(jac->diag));
  }
  PetscCall(PetscLogCpuToGpu(sizeof(PetscScalar) * len));

  pc->ops->apply          = PCApplyOrTranspose_PBJacobi_Kokkos<PETSC_FALSE>;
  pc->ops->applytranspose = PCApplyOrTranspose_PBJacobi_Kokkos<PETSC_TRUE>;
  pc->ops->destroy        = PCDestroy_PBJacobi_Kokkos;
  PetscFunctionReturn(PETSC_SUCCESS);
}
