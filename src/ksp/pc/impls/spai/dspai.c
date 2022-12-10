
#include <petscmat.h>
#include <petsc/private/petscimpl.h>

/*
     MatDumpSPAI - Dumps a PETSc matrix to a file in an ASCII format
  suitable for the SPAI code of Stephen Barnard to solve. This routine
  is simply here to allow testing of matrices directly with the SPAI
  code, rather then through the PETSc interface.

*/
PetscErrorCode MatDumpSPAI(Mat A, FILE *file)
{
  PetscMPIInt size;
  PetscInt    n;
  MPI_Comm    comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(file, 2);
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_SUP, "Only single processor dumps");
  PetscCall(MatGetSize(A, &n, &n));
  /* print the matrix */
  fprintf(file, "%" PetscInt_FMT "\n", n);
  for (PetscInt i = 0; i < n; i++) {
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscInt           nz;

    PetscCall(MatGetRow(A, i, &nz, &cols, &vals));
    for (PetscInt j = 0; j < nz; j++) fprintf(file, "%" PetscInt_FMT " %d" PetscInt_FMT " %16.14e\n", i + 1, cols[j] + 1, vals[j]);
    PetscCall(MatRestoreRow(A, i, &nz, &cols, &vals));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecDumpSPAI(Vec b, FILE *file)
{
  PetscInt           n;
  const PetscScalar *array;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(b, VEC_CLASSID, 1);
  PetscValidPointer(file, 2);
  PetscCall(VecGetSize(b, &n));
  PetscCall(VecGetArrayRead(b, &array));
  fprintf(file, "%" PetscInt_FMT "\n", n);
  for (PetscInt i = 0; i < n; i++) fprintf(file, "%" PetscInt_FMT " %16.14e\n", i + 1, array[i]);
  PetscCall(VecRestoreArrayRead(b, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}
