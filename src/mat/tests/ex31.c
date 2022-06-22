
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
  PetscViewer    viewer;
#if defined(PETSC_USE_LOG)
  PetscLogEvent MATRIX_GENERATE,MATRIX_READ;
#endif

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  N    = m*n;

  /* PART 1:  Generate matrix, then write it in binary format */

  PetscCall(PetscLogEventRegister("Generate Matrix",0,&MATRIX_GENERATE));
  PetscCall(PetscLogEventBegin(MATRIX_GENERATE,0,0,0,0));

  /* Generate matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  PetscCall(MatGetOwnershipRange(C,&Istart,&Iend));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    v = 4.0; PetscCall(MatSetValues(C,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"writing matrix in binary to matrix.dat ...\n"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_WRITE,&viewer));
  PetscCall(MatView(C,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscLogEventEnd(MATRIX_GENERATE,0,0,0,0));

  /* PART 2:  Read in matrix in binary format */

  /* All processors wait until test matrix has been dumped */
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  PetscCall(PetscLogEventRegister("Read Matrix",0,&MATRIX_READ));
  PetscCall(PetscLogEventBegin(MATRIX_READ,0,0,0,0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"reading matrix in binary from matrix.dat ...\n"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatLoad(C,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscLogEventEnd(MATRIX_READ,0,0,0,0));
  PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

  /* Free data structures */
  PetscCall(MatDestroy(&C));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      filter: grep -v " MPI process"

TEST*/
