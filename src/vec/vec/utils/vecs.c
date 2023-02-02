
#include <petscvec.h>

PetscErrorCode VecsDestroy(Vecs x)
{
  PetscFunctionBegin;
  PetscCall(VecDestroy(&(x)->v));
  PetscCall(PetscFree(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecsCreateSeq(MPI_Comm comm, PetscInt p, PetscInt m, Vecs *x)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(x));
  PetscCall(VecCreateSeq(comm, p * m, &(*x)->v));
  (*x)->n = m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecsCreateSeqWithArray(MPI_Comm comm, PetscInt p, PetscInt m, PetscScalar *a, Vecs *x)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(x));
  PetscCall(VecCreateSeqWithArray(comm, 1, p * m, a, &(*x)->v));
  (*x)->n = m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecsDuplicate(Vecs x, Vecs *y)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(y));
  PetscCall(VecDuplicate(x->v, &(*y)->v));
  (*y)->n = x->n;
  PetscFunctionReturn(PETSC_SUCCESS);
}
