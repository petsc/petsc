#pragma once

#include <Kokkos_Core.hpp>

/* SUBMANSEC = Sys */

extern Kokkos::DefaultExecutionSpace *PetscKokkosExecutionSpacePtr;

/*MC
  PetscGetKokkosExecutionSpace - Return the Kokkos execution space that PETSc is using

  Level: beginner

M*/
inline Kokkos::DefaultExecutionSpace PetscGetKokkosExecutionSpace(void)
{
  return *PetscKokkosExecutionSpacePtr;
}
