
/*
     Code for some of the parallel vector primitives.
*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petscblaslapack.h>

PetscErrorCode VecDot_MPI(Vec xin, Vec yin, PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecXDot_MPI_Default(xin, yin, z, VecDot_Seq));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecTDot_MPI(Vec xin, Vec yin, PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecXDot_MPI_Default(xin, yin, z, VecTDot_Seq));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMDot_MPI(Vec xin, PetscInt nv, const Vec y[], PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI_Default(xin, nv, y, z, VecMDot_Seq));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMTDot_MPI(Vec xin, PetscInt nv, const Vec y[], PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI_Default(xin, nv, y, z, VecMTDot_Seq));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecNorm_MPI(Vec xin, NormType type, PetscReal *z)
{
  PetscFunctionBegin;
  PetscCall(VecNorm_MPI_Default(xin, type, z, VecNorm_Seq));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMax_MPI(Vec xin, PetscInt *idx, PetscReal *z)
{
  const MPI_Op ops[] = {MPIU_MAXLOC, MPIU_MAX};

  PetscFunctionBegin;
  PetscCall(VecMinMax_MPI_Default(xin, idx, z, VecMax_Seq, ops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMin_MPI(Vec xin, PetscInt *idx, PetscReal *z)
{
  const MPI_Op ops[] = {MPIU_MINLOC, MPIU_MIN};

  PetscFunctionBegin;
  PetscCall(VecMinMax_MPI_Default(xin, idx, z, VecMin_Seq, ops));
  PetscFunctionReturn(PETSC_SUCCESS);
}
