#include <Kokkos_Core.hpp>
#include <petscvec_kokkos.hpp>
#include "ex18.h"

using DefaultMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

PetscErrorCode FillMatrixKokkosCOO(FEStruct *fe,Mat A)
{
  Kokkos::View<PetscScalar*,DefaultMemorySpace> v("v",3*3*fe->Ne);

  PetscFunctionBeginUser;
  // Simulation of GPU based finite assembly process with COO
  Kokkos::parallel_for("AssembleElementMatrices", fe->Ne, KOKKOS_LAMBDA (PetscInt i) {
      PetscScalar *s = &v(3*3*i);
      for (PetscInt vi=0; vi<3; vi++) {
        for (PetscInt vj=0; vj<3; vj++) {
          s[vi*3+vj] = vi+2*vj;
        }
      }
    });
  CHKERRQ(MatSetValuesCOO(A,v.data(),INSERT_VALUES));
  PetscFunctionReturn(0);
}
