/* $Id: dspai.c,v 1.7 2001/08/07 03:03:40 balay Exp $*/

#include "petscmat.h"

/*
     MatDumpSPAI - Dumps a PETSc matrix to a file in an ASCII format 
  suitable for the SPAI code of Stephen Barnard to solve. This routine
  is simply here to allow testing of matrices directly with the SPAI 
  code, rather then through the PETSc interface.

*/
int MatDumpSPAI(Mat A,FILE *file)
{
  PetscScalar   *vals;
  int      i,j,ierr,*cols,n,size,nz;
  MPI_Comm comm;

  PetscObjectGetComm((PetscObject)A,&comm);
 
  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(1,"Only single processor dumps");

  ierr = MatGetSize(A,&n,&n);CHKERRQ(ierr);

  /* print the matrix */
  fprintf(file,"%d\n",n);
  for (i=0; i<n; i++) {
    ierr     = MatGetRow(A,i,&nz,&cols,&vals);CHKERRQ(ierr);
    for (j=0; j<nz; j++) {
      fprintf(file,"%d %d %16.14e\n",i+1,cols[j]+1,vals[j]);
    }
    ierr     = MatRestoreRow(A,i,&nz,&cols,&vals);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

int VecDumpSPAI(Vec b,FILE *file)
{
  int    n,i,ierr;
  PetscScalar *array;

  ierr = VecGetSize(b,&n);CHKERRQ(ierr);
  ierr = VecGetArray(b,&array);CHKERRQ(ierr);

  fprintf(file,"%d\n",n);
  for (i=0; i<n; i++) {
    fprintf(file,"%d %16.14e\n",i+1,array[i]);
  }

  PetscFunctionReturn(0);
}
