
#include <petscmat.h>
#include <petsc/private/petscimpl.h>

/*
     MatDumpSPAI - Dumps a PETSc matrix to a file in an ASCII format
  suitable for the SPAI code of Stephen Barnard to solve. This routine
  is simply here to allow testing of matrices directly with the SPAI
  code, rather then through the PETSc interface.

*/
PetscErrorCode  MatDumpSPAI(Mat A,FILE *file)
{
  const PetscScalar *vals;
  PetscErrorCode    ierr;
  int               i,j,n,size,nz;
  const int         *cols;
  MPI_Comm          comm;

  PetscObjectGetComm((PetscObject)A,&comm);

  MPI_Comm_size(comm,&size);
  PetscAssertFalse(size > 1,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only single processor dumps");

  ierr = MatGetSize(A,&n,&n);CHKERRQ(ierr);

  /* print the matrix */
  fprintf(file,"%d\n",n);
  for (i=0; i<n; i++) {
    ierr = MatGetRow(A,i,&nz,&cols,&vals);CHKERRQ(ierr);
    for (j=0; j<nz; j++) fprintf(file,"%d %d %16.14e\n",i+1,cols[j]+1,vals[j]);
    ierr = MatRestoreRow(A,i,&nz,&cols,&vals);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  VecDumpSPAI(Vec b,FILE *file)
{
  PetscErrorCode ierr;
  int            n,i;
  PetscScalar    *array;

  ierr = VecGetSize(b,&n);CHKERRQ(ierr);
  ierr = VecGetArray(b,&array);CHKERRQ(ierr);

  fprintf(file,"%d\n",n);
  for (i=0; i<n; i++) fprintf(file,"%d %16.14e\n",i+1,array[i]);
  PetscFunctionReturn(0);
}
