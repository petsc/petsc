#include <petscvec_kokkos.hpp>
#include <petsc_kokkos.hpp>
#include <petsc/private/kokkosimpl.hpp>
#include <petsc/private/vecimpl.h>
#include <petsc/private/matimpl.h>

PETSC_INTERN PetscErrorCode MatADot_Diagonal_SeqKokkos(Mat A, Vec x, Vec y, PetscScalar *z)
{
  Mat_Diagonal              *ctx = (Mat_Diagonal *)A->data;
  ConstPetscScalarKokkosView xv, yv, wv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(x, &xv));
  PetscCall(VecGetKokkosView(y, &yv));
  PetscCall(VecGetKokkosView(ctx->diag, &wv));
  // Kokkos always overwrites z, so no need to init it
  PetscCallCXX(Kokkos::parallel_reduce("MatADot_Diagonal", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, x->map->n), KOKKOS_LAMBDA(const PetscInt &i, PetscScalar &update) { update += PetscConj(yv(i)) * wv(i) * xv(i); }, *z));
  PetscCall(VecRestoreKokkosView(x, &xv));
  PetscCall(VecRestoreKokkosView(y, &yv));
  PetscCall(VecRestoreKokkosView(ctx->diag, &wv));
  PetscCall(PetscLogGpuTimeEnd());
  if (x->map->n > 0) PetscCall(PetscLogGpuFlops(3.0 * x->map->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatANormSq_Diagonal_SeqKokkos(Mat A, Vec x, PetscReal *z)
{
  Mat_Diagonal              *ctx = (Mat_Diagonal *)A->data;
  ConstPetscScalarKokkosView xv, wv;
  PetscScalar                res = 0.;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(x, &xv));
  PetscCall(VecGetKokkosView(ctx->diag, &wv));
  PetscCallCXX(Kokkos::parallel_reduce("MatANorm_Diagonal", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, x->map->n), KOKKOS_LAMBDA(const PetscInt &i, PetscScalar &update) { update += PetscConj(xv(i)) * wv(i) * xv(i); }, res));
  PetscCall(VecRestoreKokkosView(x, &xv));
  PetscCall(VecRestoreKokkosView(ctx->diag, &wv));
  PetscCall(PetscLogGpuTimeEnd());
  *z = PetscRealPart(res);
  if (x->map->n > 0) PetscCall(PetscLogGpuFlops(3.0 * x->map->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}
