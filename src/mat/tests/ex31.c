
static char help[] = "Tests binary I/O of matrices and illustrates user-defined event logging.\n\n";

#include <petscmat.h>

/* Note:  Most applications would not read and write the same matrix within
  the same program.  This example is intended only to demonstrate
  both input and output. */

int main(int argc,char **args)
{
  Mat            C;
  PetscScalar    v;
  PetscInt       i,j,Ii,J,Istart,Iend,N,m = 4,n = 4;
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;
  PetscViewer    viewer;
#if defined(PETSC_USE_LOG)
  PetscLogEvent MATRIX_GENERATE,MATRIX_READ;
#endif

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  N    = m*n;

  /* PART 1:  Generate matrix, then write it in binary format */

  ierr = PetscLogEventRegister("Generate Matrix",0,&MATRIX_GENERATE);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MATRIX_GENERATE,0,0,0,0);CHKERRQ(ierr);

  /* Generate matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"writing matrix in binary to matrix.dat ...\n");CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(C,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MATRIX_GENERATE,0,0,0,0);CHKERRQ(ierr);

  /* PART 2:  Read in matrix in binary format */

  /* All processors wait until test matrix has been dumped */
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRMPI(ierr);

  ierr = PetscLogEventRegister("Read Matrix",0,&MATRIX_READ);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MATRIX_READ,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"reading matrix in binary from matrix.dat ...\n");CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatLoad(C,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MATRIX_READ,0,0,0,0);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free data structures */
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      filter: grep -v "MPI processes"

TEST*/
