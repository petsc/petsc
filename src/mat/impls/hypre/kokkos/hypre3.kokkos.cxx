#include <petsc/private/petschypre.h>
#include <Kokkos_Core.hpp>
#include <../src/mat/impls/hypre/mhypre.h>

PetscErrorCode MatZeroRows_Kokkos(PetscInt n, const PetscInt rows[], const HYPRE_Int i[], const HYPRE_Int j[], HYPRE_Complex a[], HYPRE_Complex diag)
{
  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscKokkosInitializeCheck()); // As we might have not created any petsc/kokkos object yet
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(n, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &t) {
      PetscInt r = rows[t.league_rank()]; // row r
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, i[r + 1] - i[r]), [&](PetscInt c) {
        if (r == j[i[r] + c]) a[i[r] + c] = diag;
        else a[i[r] + c] = 0.0;
      });
    });
  PetscFunctionReturn(PETSC_SUCCESS);
}
