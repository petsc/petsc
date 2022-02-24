
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
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  N    = m*n;

  /* PART 1:  Generate matrix, then write it in binary format */

  CHKERRQ(PetscLogEventRegister("Generate Matrix",0,&MATRIX_GENERATE));
  CHKERRQ(PetscLogEventBegin(MATRIX_GENERATE,0,0,0,0));

  /* Generate matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(MatGetOwnershipRange(C,&Istart,&Iend));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    v = 4.0; CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"writing matrix in binary to matrix.dat ...\n"));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_WRITE,&viewer));
  CHKERRQ(MatView(C,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscLogEventEnd(MATRIX_GENERATE,0,0,0,0));

  /* PART 2:  Read in matrix in binary format */

  /* All processors wait until test matrix has been dumped */
  CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));

  CHKERRQ(PetscLogEventRegister("Read Matrix",0,&MATRIX_READ));
  CHKERRQ(PetscLogEventBegin(MATRIX_READ,0,0,0,0));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"reading matrix in binary from matrix.dat ...\n"));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatLoad(C,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(PetscLogEventEnd(MATRIX_READ,0,0,0,0));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

  /* Free data structures */
  CHKERRQ(MatDestroy(&C));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      filter: grep -v "MPI processes"

TEST*/
