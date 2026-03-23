#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/

#include "../matdiagonalcupm.hpp"
#include "../src/vec/vec/impls/seq/cupm/vecseqcupm.hpp"
#include "../src/vec/vec/impls/seq/cupm/vecseqcupm_impl.hpp"

using namespace Petsc::device::cupm;
using Petsc::device::cupm::DeviceType;

static constexpr impl::MatDiagonal_CUPM<DeviceType::CUDA, ::Petsc::vec::cupm::impl::VecSeq_CUPM<DeviceType::CUDA>> cupm_mat{};

PETSC_INTERN PetscErrorCode MatADot_Diagonal_SeqCUDA(Mat A, Vec x, Vec y, PetscScalar *val)
{
  PetscFunctionBegin;
  PetscCall(cupm_mat.ADot(A, x, y, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatANormSq_Diagonal_SeqCUDA(Mat A, Vec x, PetscReal *val)
{
  PetscFunctionBegin;
  PetscCall(cupm_mat.ANormSq(A, x, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}
