/*$Id: ex31.c,v 1.17 2000/05/05 22:16:17 balay Exp bsmith $*/

static char help[] = 
"Tests binary I/O of matrices and illustrates user-defined event logging.\n\n";

#include "petscmat.h"

/* Note:  Most applications would not read and write the same matrix within
  the same program.  This example is intended only to demonstrate
  both input and output. */

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat     C;
  Scalar  v;
  int     i,j,I,J,ierr,Istart,Iend,N,m = 4,n = 4,rank,size;
  Viewer  viewer;
  int     MATRIX_GENERATE,MATRIX_READ;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  N = m*n;

  /* PART 1:  Generate matrix, then write it in binary format */

  ierr = PLogEventRegister(&MATRIX_GENERATE,"Generate Matrix",PETSC_NULL);CHKERRA(ierr);
  PLogEventBegin(MATRIX_GENERATE,0,0,0,0);

  /* Generate matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,&C);CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRA(ierr);
  for (I=Istart; I<Iend; I++) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if (i>0)   {J = I - n; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRA(ierr);}
    if (i<m-1) {J = I + n; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRA(ierr);}
    if (j>0)   {J = I - 1; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRA(ierr);}
    if (j<n-1) {J = I + 1; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRA(ierr);}
    v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,ADD_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"writing matrix in binary to matrix.dat ...\n");CHKERRA(ierr);
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",BINARY_CREATE,&viewer);CHKERRA(ierr);
  ierr = MatView(C,viewer);CHKERRA(ierr);
  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  PLogEventEnd(MATRIX_GENERATE,0,0,0,0);

  /* PART 2:  Read in matrix in binary format */

  /* All processors wait until test matrix has been dumped */
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRA(ierr);

  ierr = PLogEventRegister(&MATRIX_READ,"Read Matrix",PETSC_NULL);CHKERRA(ierr);
  PLogEventBegin(MATRIX_READ,0,0,0,0);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"reading matrix in binary from matrix.dat ...\n");CHKERRA(ierr);
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",BINARY_RDONLY,&viewer);CHKERRA(ierr);
  ierr = MatLoad(viewer,MATMPIAIJ,&C);CHKERRA(ierr);
  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  PLogEventEnd(MATRIX_READ,0,0,0,0);
  ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* Free data structures */
  ierr = MatDestroy(C);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}


