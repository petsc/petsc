
static char help[] = "Tests the use of MatZeroRows() for parallel matrices.\n\
This example also tests the use of MatDuplicate() for both MPIAIJ and MPIBAIJ matrices";

#include <petscmat.h>

extern PetscErrorCode TestMatZeroRows_Basic(Mat,IS,PetscScalar);
extern PetscErrorCode TestMatZeroRows_with_no_allocation(Mat,IS,PetscScalar);

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       i,j,m = 3,n,Ii,J,Imax;
  PetscMPIInt    rank,size;
  PetscScalar    v,diag=-4.0;
  IS             is;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  /* create A Square matrix for the five point stencil,YET AGAIN*/
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  for (i=0; i<m; i++) {
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Create AN IS required by MatZeroRows() */
  Imax = n*rank; if (Imax>= n*m -m - 1) Imax = m*n - m - 1;
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,m,Imax,1,&is));

  CHKERRQ(TestMatZeroRows_Basic(A,is,0.0));
  CHKERRQ(TestMatZeroRows_Basic(A,is,diag));

  CHKERRQ(TestMatZeroRows_with_no_allocation(A,is,0.0));
  CHKERRQ(TestMatZeroRows_with_no_allocation(A,is,diag));

  CHKERRQ(MatDestroy(&A));

  /* Now Create a rectangular matrix with five point stencil (app)
   n+size is used so that this dimension is always divisible by size.
   This way, we can always use bs = size for any number of procs */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*(n+size)));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  for (i=0; i<m; i++) {
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n+size-1) {J = Ii + 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(TestMatZeroRows_Basic(A,is,0.0));
  CHKERRQ(TestMatZeroRows_Basic(A,is,diag));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(PetscFinalize());
  return 0;
}

PetscErrorCode TestMatZeroRows_Basic(Mat A,IS is,PetscScalar diag)
{
  Mat            B;
  PetscBool      keepnonzeropattern;

  /* Now copy A into B, and test it with MatZeroRows() */
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-keep_nonzero_pattern",&keepnonzeropattern));
  if (keepnonzeropattern) {
    CHKERRQ(MatSetOption(B,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE));
  }

  CHKERRQ(MatZeroRowsIS(B,is,diag,0,0));
  CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&B));
  return 0;
}

PetscErrorCode TestMatZeroRows_with_no_allocation(Mat A,IS is,PetscScalar diag)
{
  Mat            B;

  /* Now copy A into B, and test it with MatZeroRows() */
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
  /* Set this flag after assembly. This way, it affects only MatZeroRows() */
  CHKERRQ(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));

  CHKERRQ(MatZeroRowsIS(B,is,diag,0,0));
  CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&B));
  return 0;
}

/*TEST

   test:
      nsize: 2
      filter: grep -v "MPI processes"

   test:
      suffix: 2
      nsize: 3
      args: -mat_type mpibaij -mat_block_size 3
      filter: grep -v "MPI processes"

   test:
      suffix: 3
      nsize: 3
      args: -mat_type mpiaij -keep_nonzero_pattern
      filter: grep -v "MPI processes"

   test:
      suffix: 4
      nsize: 3
      args: -keep_nonzero_pattern -mat_type mpibaij -mat_block_size 3
      filter: grep -v "MPI processes"

TEST*/
