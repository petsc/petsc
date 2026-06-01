#include <petsc_kokkos.hpp>
#include <petsc/private/petschypre.h>
#include <Kokkos_Core.hpp>
#include <../src/mat/impls/hypre/mhypre.h>
#include <petsc/private/kokkosimpl.hpp>

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

PetscErrorCode PetscHypreIntCastArray_Kokkos(PetscInt n, const PetscInt *a, HYPRE_Int *b)
{
  PetscFunctionBegin;
  if (n) PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(const size_t i) { b[i] = static_cast<HYPRE_Int>(a[i]); }));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatHypreDeviceMalloc_Kokkos(size_t size, void **ptr)
{
  PetscFunctionBegin;
  if (size) PetscCallCXX(*ptr = Kokkos::kokkos_malloc<DefaultMemorySpace>(size));
  else *ptr = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatHypreDeviceFree_Kokkos(void *a)
{
  PetscFunctionBegin;
  PetscCallCXX(Kokkos::kokkos_free<DefaultMemorySpace>(a));
  PetscFunctionReturn(PETSC_SUCCESS);
}
