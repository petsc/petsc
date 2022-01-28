#include <petsc/private/matimpl.h>
#include <petscsf.h>

/* this function maps rows to locally owned rows */
PETSC_INTERN PetscErrorCode MatZeroRowsMapLocal_Private(Mat A,PetscInt N,const PetscInt *rows,PetscInt *nr,PetscInt **olrows)
{
  PetscInt      *owners = A->rmap->range;
  PetscInt       n      = A->rmap->n;
  PetscSF        sf;
  PetscInt      *lrows;
  PetscSFNode   *rrows;
  PetscMPIInt    rank, p = 0;
  PetscInt       r, len = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create SF where leaves are input rows and roots are owned rows */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank);CHKERRMPI(ierr);
  ierr = PetscMalloc1(n, &lrows);CHKERRQ(ierr);
  for (r = 0; r < n; ++r) lrows[r] = -1;
  if (!A->nooffproczerorows) {ierr = PetscMalloc1(N, &rrows);CHKERRQ(ierr);}
  for (r = 0; r < N; ++r) {
    const PetscInt idx   = rows[r];
    PetscAssertFalse(idx < 0 || A->rmap->N <= idx,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %" PetscInt_FMT " out of range [0,%" PetscInt_FMT ")",idx,A->rmap->N);
    if (idx < owners[p] || owners[p+1] <= idx) { /* short-circuit the search if the last p owns this row too */
      ierr = PetscLayoutFindOwner(A->rmap,idx,&p);CHKERRQ(ierr);
    }
    if (A->nooffproczerorows) {
      PetscAssertFalse(p != rank,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"MAT_NO_OFF_PROC_ZERO_ROWS set, but row %" PetscInt_FMT " is not owned by rank %d",idx,rank);
      lrows[len++] = idx - owners[p];
    } else {
      rrows[r].rank = p;
      rrows[r].index = rows[r] - owners[p];
    }
  }
  if (!A->nooffproczerorows) {
    ierr = PetscSFCreate(PetscObjectComm((PetscObject) A), &sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sf, n, N, NULL, PETSC_OWN_POINTER, rrows, PETSC_OWN_POINTER);CHKERRQ(ierr);
    /* Collect flags for rows to be zeroed */
    ierr = PetscSFReduceBegin(sf, MPIU_INT, (PetscInt*)rows, lrows, MPI_LOR);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf, MPIU_INT, (PetscInt*)rows, lrows, MPI_LOR);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    /* Compress and put in row numbers */
    for (r = 0; r < n; ++r) if (lrows[r] >= 0) lrows[len++] = r;
  }
  if (nr) *nr = len;
  if (olrows) *olrows = lrows;
  PetscFunctionReturn(0);
}
