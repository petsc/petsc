#include <petsc_kokkos.hpp>
#include <petscvec_kokkos.hpp>
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp>
#include <petscdevice.h>
#include <../src/ksp/pc/impls/vpbjacobi/vpbjacobi.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp> // for MatInvertVariableBlockDiagonal_SeqAIJKokkos
#include <../src/mat/impls/aij/mpi/mpiaij.h>          // for Mat_MPIAIJ
#include <KokkosBlas2_gemv.hpp>

/* A class that manages helper arrays assisting parallel PCApply() with Kokkos */
struct PC_VPBJacobi_Kokkos {
  /* Cache the old sizes to check if we need realloc */
  PetscInt n;       /* number of rows of the local matrix */
  PetscInt nblocks; /* number of point blocks */
  PetscInt nsize;   /* sum of sizes (elements) of the point blocks */

  /* Helper arrays that are pre-computed on host and then copied to device.
    bs:     [nblocks+1], "csr" version of bsizes[]
    bs2:    [nblocks+1], "csr" version of squares of bsizes[]
    blkMap: [n], row i of the local matrix belongs to the blkMap[i]-th block
  */
  PetscIntKokkosDualView bs_dual, bs2_dual, blkMap_dual;
  PetscScalarKokkosView  diag; // buffer to store diagonal blocks
  PetscScalarKokkosView  work; // work buffer, with the same size as diag[]
  PetscLogDouble         setupFlops;

  // clang-format off
  // n:               size of the matrix
  // nblocks:         number of blocks
  // nsize:           sum bsizes[i]^2 for i=0..nblocks
  // bsizes[nblocks]: sizes of blocks
  PC_VPBJacobi_Kokkos(PetscInt n, PetscInt nblocks, PetscInt nsize, const PetscInt *bsizes) :
    n(n), nblocks(nblocks), nsize(nsize), bs_dual(NoInit("bs_dual"), nblocks + 1),
    bs2_dual(NoInit("bs2_dual"), nblocks + 1), blkMap_dual(NoInit("blkMap_dual"), n),
    diag(NoInit("diag"), nsize), work(NoInit("work"), nsize)
  {
    PetscCallVoid(BuildHelperArrays(bsizes));
  }
  // clang-format on

private:
  PetscErrorCode BuildHelperArrays(const PetscInt *bsizes)
  {
    PetscInt *bs_h     = bs_dual.view_host().data();
    PetscInt *bs2_h    = bs2_dual.view_host().data();
    PetscInt *blkMap_h = blkMap_dual.view_host().data();

    PetscFunctionBegin;
    setupFlops = 0.0;
    bs_h[0] = bs2_h[0] = 0;
    for (PetscInt i = 0; i < nblocks; i++) {
      PetscInt m   = bsizes[i];
      bs_h[i + 1]  = bs_h[i] + m;
      bs2_h[i + 1] = bs2_h[i] + m * m;
      for (PetscInt j = 0; j < m; j++) blkMap_h[bs_h[i] + j] = i;
      // m^3/3 FMA for A=LU factorization; m^3 FMA for solving (LU)X=I to get the inverse
      setupFlops += 8.0 * m * m * m / 3;
    }

    PetscCallCXX(bs_dual.modify_host());
    PetscCallCXX(bs2_dual.modify_host());
    PetscCallCXX(blkMap_dual.modify_host());
    PetscCall(KokkosDualViewSyncDevice(bs_dual, PetscGetKokkosExecutionSpace()));
    PetscCall(KokkosDualViewSyncDevice(bs2_dual, PetscGetKokkosExecutionSpace()));
    PetscCall(KokkosDualViewSyncDevice(blkMap_dual, PetscGetKokkosExecutionSpace()));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

template <PetscBool transpose>
static PetscErrorCode PCApplyOrTranspose_VPBJacobi_Kokkos(PC pc, Vec x, Vec y)
{
  PC_VPBJacobi              *jac   = (PC_VPBJacobi *)pc->data;
  PC_VPBJacobi_Kokkos       *pckok = static_cast<PC_VPBJacobi_Kokkos *>(jac->spptr);
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      yv;
  PetscScalarKokkosView      diag   = pckok->diag;
  PetscIntKokkosView         bs     = pckok->bs_dual.view_device();
  PetscIntKokkosView         bs2    = pckok->bs2_dual.view_device();
  PetscIntKokkosView         blkMap = pckok->blkMap_dual.view_device();
  const char                *label  = transpose ? "PCApplyTranspose_VPBJacobi" : "PCApply_VPBJacobi";

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  VecErrorIfNotKokkos(x);
  VecErrorIfNotKokkos(y);
  PetscCall(VecGetKokkosView(x, &xv));
  PetscCall(VecGetKokkosViewWrite(y, &yv));
#if 0 // TODO: Why the TeamGemv version is 2x worse than the naive one?
  PetscCallCXX(Kokkos::parallel_for(
    label, Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), jac->nblocks, Kokkos::AUTO()), KOKKOS_LAMBDA(const KokkosTeamMemberType &team) {
      PetscInt           bid  = team.league_rank();    // block id
      PetscInt           n    = bs(bid + 1) - bs(bid); // size of this block
      const PetscScalar *bbuf = &diag(bs2(bid));
      const PetscScalar *xbuf = &xv(bs(bid));
      PetscScalar       *ybuf = &yv(bs(bid));
      const auto        &B    = Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft>(bbuf, n, n); // wrap it in a 2D view in column-major order
      const auto        &x1   = ConstPetscScalarKokkosView(xbuf, n);
      const auto        &y1   = PetscScalarKokkosView(ybuf, n);
      if (transpose) {
        KokkosBlas::TeamGemv<KokkosTeamMemberType, KokkosBlas::Trans::Transpose>::invoke(team, 1., B, x1, 0., y1); // y1 = 0.0 * y1 + 1.0 * B^T * x1
      } else {
        KokkosBlas::TeamGemv<KokkosTeamMemberType, KokkosBlas::Trans::NoTranspose>::invoke(team, 1., B, x1, 0., y1); // y1 = 0.0 * y1 + 1.0 * B * x1
      }
    }));
#else
  PetscCallCXX(Kokkos::parallel_for(
    label, Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, pckok->n), KOKKOS_LAMBDA(PetscInt row) {
      const PetscScalar *Bp, *xp;
      PetscScalar       *yp;
      PetscInt           i, j, k, m;

      k  = blkMap(row);                             /* k-th block/matrix */
      m  = bs(k + 1) - bs(k);                       /* block size of the k-th block */
      i  = row - bs(k);                             /* i-th row of the block */
      Bp = &diag(bs2(k) + i * (transpose ? m : 1)); /* Bp points to the first entry of i-th row/column */
      xp = &xv(bs(k));
      yp = &yv(bs(k));

      yp[i] = 0.0;
      for (j = 0; j < m; j++) {
        yp[i] += Bp[0] * xp[j];
        Bp += transpose ? 1 : m;
      }
    }));
#endif
  PetscCall(VecRestoreKokkosView(x, &xv));
  PetscCall(VecRestoreKokkosViewWrite(y, &yv));
  PetscCall(PetscLogGpuFlops(pckok->nsize * 2)); /* FMA on entries in all blocks */
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_VPBJacobi_Kokkos(PC pc)
{
  PC_VPBJacobi *jac = (PC_VPBJacobi *)pc->data;

  PetscFunctionBegin;
  PetscCallCXX(delete static_cast<PC_VPBJacobi_Kokkos *>(jac->spptr));
  PetscCall(PCDestroy_VPBJacobi(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PCSetUp_VPBJacobi_Kokkos(PC pc, Mat diagVPB)
{
  PC_VPBJacobi        *jac   = (PC_VPBJacobi *)pc->data;
  PC_VPBJacobi_Kokkos *pckok = static_cast<PC_VPBJacobi_Kokkos *>(jac->spptr);
  PetscInt             i, nlocal, nblocks, nsize = 0;
  const PetscInt      *bsizes;
  PetscBool            ismpi;
  Mat                  A;

  PetscFunctionBegin;
  PetscCall(MatGetVariableBlockSizes(pc->pmat, &nblocks, &bsizes));
  PetscCall(MatGetLocalSize(pc->pmat, &nlocal, NULL));
  PetscCheck(!nlocal || nblocks, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call MatSetVariableBlockSizes() before using PCVPBJACOBI");

  if (!jac->diag) {
    PetscInt max_bs = -1, min_bs = PETSC_INT_MAX;
    for (i = 0; i < nblocks; i++) {
      min_bs = PetscMin(min_bs, bsizes[i]);
      max_bs = PetscMax(max_bs, bsizes[i]);
      nsize += bsizes[i] * bsizes[i];
    }
    jac->nblocks = nblocks;
    jac->min_bs  = min_bs;
    jac->max_bs  = max_bs;
  }

  // If one calls MatSetVariableBlockSizes() multiple times and sizes have been changed (is it allowed?), we delete the old and rebuild anyway
  if (pckok && (pckok->n != nlocal || pckok->nblocks != nblocks || pckok->nsize != nsize)) {
    PetscCallCXX(delete pckok);
    pckok = nullptr;
  }

  PetscCall(PetscLogGpuTimeBegin());
  if (!pckok) {
    PetscCallCXX(pckok = new PC_VPBJacobi_Kokkos(nlocal, nblocks, nsize, bsizes));
    jac->spptr = pckok;
  }

  // Extract diagonal blocks from the matrix and compute their inverse
  const auto &bs     = pckok->bs_dual.view_device();
  const auto &bs2    = pckok->bs2_dual.view_device();
  const auto &blkMap = pckok->blkMap_dual.view_device();
  if (diagVPB) { // If caller provided a matrix made of the diagonal blocks, use it
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)diagVPB, MATMPIAIJ, &ismpi));
    A = ismpi ? static_cast<Mat_MPIAIJ *>(diagVPB->data)->A : diagVPB;
  } else {
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)pc->pmat, MATMPIAIJ, &ismpi));
    A = ismpi ? static_cast<Mat_MPIAIJ *>(pc->pmat->data)->A : pc->pmat;
  }
  PetscCall(MatInvertVariableBlockDiagonal_SeqAIJKokkos(A, bs, bs2, blkMap, pckok->work, pckok->diag));
  pc->ops->apply          = PCApplyOrTranspose_VPBJacobi_Kokkos<PETSC_FALSE>;
  pc->ops->applytranspose = PCApplyOrTranspose_VPBJacobi_Kokkos<PETSC_TRUE>;
  pc->ops->destroy        = PCDestroy_VPBJacobi_Kokkos;
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(pckok->setupFlops));
  PetscFunctionReturn(PETSC_SUCCESS);
}
