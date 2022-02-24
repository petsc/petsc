
#include <petscvec.h>

PetscErrorCode VecsDestroy(Vecs x)
{
  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&(x)->v));
  CHKERRQ(PetscFree(x));
  PetscFunctionReturn(0);
}

PetscErrorCode VecsCreateSeq(MPI_Comm comm,PetscInt p,PetscInt m,Vecs *x)
{
  PetscFunctionBegin;
  CHKERRQ(PetscNew(x));
  CHKERRQ(VecCreateSeq(comm,p*m,&(*x)->v));
  (*x)->n = m;
  PetscFunctionReturn(0);
}

PetscErrorCode VecsCreateSeqWithArray(MPI_Comm comm,PetscInt p,PetscInt m,PetscScalar *a,Vecs *x)
{
  PetscFunctionBegin;
  CHKERRQ(PetscNew(x));
  CHKERRQ(VecCreateSeqWithArray(comm,1,p*m,a,&(*x)->v));
  (*x)->n = m;
  PetscFunctionReturn(0);
}

PetscErrorCode VecsDuplicate(Vecs x,Vecs *y)
{
  PetscFunctionBegin;
  CHKERRQ(PetscNew(y));
  CHKERRQ(VecDuplicate(x->v,&(*y)->v));
  (*y)->n = x->n;
  PetscFunctionReturn(0);
}
