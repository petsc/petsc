
#include <petscvec.h>

PetscErrorCode VecsDestroy(Vecs x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDestroy(&(x)->v);CHKERRQ(ierr);
  ierr = PetscFree(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecsCreateSeq(MPI_Comm comm,PetscInt p,PetscInt m,Vecs *x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNew(x);CHKERRQ(ierr);
  ierr = VecCreateSeq(comm,p*m,&(*x)->v);CHKERRQ(ierr);
  (*x)->n = m;
  PetscFunctionReturn(0);
}

PetscErrorCode VecsCreateSeqWithArray(MPI_Comm comm,PetscInt p,PetscInt m,PetscScalar *a,Vecs *x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNew(x);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(comm,1,p*m,a,&(*x)->v);CHKERRQ(ierr);
  (*x)->n = m;
  PetscFunctionReturn(0);
}

PetscErrorCode VecsDuplicate(Vecs x,Vecs *y)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNew(y);CHKERRQ(ierr);
  ierr = VecDuplicate(x->v,&(*y)->v);CHKERRQ(ierr);
  (*y)->n = x->n;
  PetscFunctionReturn(0);
}

