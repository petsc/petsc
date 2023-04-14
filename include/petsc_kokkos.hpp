#ifndef PETSC_KOKKOS_HPP
#define PETSC_KOKKOS_HPP

#include <Kokkos_Core.hpp>

/* SUBMANSEC = Sys */

extern Kokkos::DefaultExecutionSpace *PetscKokkosExecutionSpacePtr;

/*MC
  PetscGetKokkosExecutionSpace - Return the Kokkos execution space that petsc is using

  Level: beginner

M*/
inline Kokkos::DefaultExecutionSpace &PetscGetKokkosExecutionSpace(void)
{
  return *PetscKokkosExecutionSpacePtr;
}

#endif
