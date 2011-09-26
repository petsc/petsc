
static char help[] = "Reads a matrix from PETSc binary file. Use for view or investigating matrix data structure. \n\n"; 
/*
 Example: 
      ./ex16 -f <matrix file> -a_mat_view_draw -draw_pause -1
      ./ex16 -f <matrix file> -a_mat_view_info
 */

#include <petscmat.h>
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat                   A;          
  PetscViewer           fd;               /* viewer */
  char                  file[PETSC_MAX_PATH_LEN];     /* input file name */
  PetscErrorCode        ierr;
  PetscInt              m,n,rstart,rend;
  PetscBool             flg;
  PetscInt             row,ncols,j,nrows,nnz=0;
  const PetscInt       *cols;
  const PetscScalar    *vals;
  PetscReal            norm;
  PetscMPIInt          rank;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Determine files from which we read the linear systems. */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");

  /* Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file. */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /* Load the matrix; then destroy the viewer. */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Check zero rows */
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  //printf("zero rows:\n");
  nrows = 0;
  for (row=rstart; row<rend; row++){
    ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    nnz += ncols;
    norm = 0.0;
    for (j=0; j<ncols; j++){
      if (norm < PetscAbsScalar(vals[j])) norm = vals[j];
    }
    if (!norm) {
      //printf("  %d,",row); 
      nrows++;
    }
    ierr = MatRestoreRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
 
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n [%d] Matrix local size %d,%d; nnz %d, %g percent; No. of zero rows: %d\n",rank,m,n,nnz,100*PetscReal(nnz)/(m*n),nrows);
  
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

