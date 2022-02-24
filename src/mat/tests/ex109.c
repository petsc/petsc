static char help[] = "Test MatMatMult() for AIJ and Dense matrices.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,B,C,D,AT;
  PetscInt       i,M,N,Istart,Iend,n=7,j,J,Ii,m=8,am,an;
  PetscScalar    v;
  PetscErrorCode ierr;
  PetscRandom    r;
  PetscBool      equal=PETSC_FALSE,flg;
  PetscReal      fill = 1.0,norm;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-fill",&fill,NULL));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&r));
  CHKERRQ(PetscRandomSetFromOptions(r));

  /* Create a aij matrix A */
  M    = N = m*n;
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatMPIAIJSetPreallocation(A,5,NULL,5,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(A,5,NULL));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  am   = Iend - Istart;
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    v = 4.0; CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Create a dense matrix B */
  CHKERRQ(MatGetLocalSize(A,&am,&an));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,an,PETSC_DECIDE,PETSC_DECIDE,M));
  CHKERRQ(MatSetType(B,MATDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(B,NULL));
  CHKERRQ(MatMPIDenseSetPreallocation(B,NULL));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetRandom(B,r));
  CHKERRQ(PetscRandomDestroy(&r));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Test reuse of user-provided dense C (unassembled) -- not recommended usage */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetType(C,MATDENSE));
  CHKERRQ(MatSetSizes(C,am,PETSC_DECIDE,PETSC_DECIDE,N));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(MatZeroEntries(C));
  CHKERRQ(MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C));
  CHKERRQ(MatNorm(C,NORM_INFINITY,&norm));
  CHKERRQ(MatDestroy(&C));

  /* Test C = A*B (aij*dense) */
  CHKERRQ(MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C));
  CHKERRQ(MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C));

  /* Test developer API */
  CHKERRQ(MatProductCreate(A,B,NULL,&D));
  CHKERRQ(MatProductSetType(D,MATPRODUCT_AB));
  CHKERRQ(MatProductSetAlgorithm(D,"default"));
  CHKERRQ(MatProductSetFill(D,fill));
  CHKERRQ(MatProductSetFromOptions(D));
  CHKERRQ(MatProductSymbolic(D));
  for (i=0; i<2; i++) {
    CHKERRQ(MatProductNumeric(D));
  }
  CHKERRQ(MatEqual(C,D,&equal));
  PetscCheckFalse(!equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"C != D");
  CHKERRQ(MatDestroy(&D));

  /* Test D = AT*B (transpose(aij)*dense) */
  CHKERRQ(MatCreateTranspose(A,&AT));
  CHKERRQ(MatMatMult(AT,B,MAT_INITIAL_MATRIX,fill,&D));
  CHKERRQ(MatMatMultEqual(AT,B,D,10,&equal));
  PetscCheckFalse(!equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != AT*B (transpose(aij)*dense)");
  CHKERRQ(MatDestroy(&D));
  CHKERRQ(MatDestroy(&AT));

  /* Test D = C*A (dense*aij) */
  CHKERRQ(MatMatMult(C,A,MAT_INITIAL_MATRIX,fill,&D));
  CHKERRQ(MatMatMult(C,A,MAT_REUSE_MATRIX,fill,&D));
  CHKERRQ(MatMatMultEqual(C,A,D,10,&equal));
  PetscCheckFalse(!equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != C*A (dense*aij)");
  CHKERRQ(MatDestroy(&D));

  /* Test D = A*C (aij*dense) */
  CHKERRQ(MatMatMult(A,C,MAT_INITIAL_MATRIX,fill,&D));
  CHKERRQ(MatMatMult(A,C,MAT_REUSE_MATRIX,fill,&D));
  CHKERRQ(MatMatMultEqual(A,C,D,10,&equal));
  PetscCheckFalse(!equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != A*C (aij*dense)");
  CHKERRQ(MatDestroy(&D));

  /* Test D = B*C (dense*dense) */
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  if (size == 1) {
    CHKERRQ(MatMatMult(B,C,MAT_INITIAL_MATRIX,fill,&D));
    CHKERRQ(MatMatMult(B,C,MAT_REUSE_MATRIX,fill,&D));
    CHKERRQ(MatMatMultEqual(B,C,D,10,&equal));
    PetscCheckFalse(!equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != B*C (dense*dense)");
    CHKERRQ(MatDestroy(&D));
  }

  /* Test D = B*C^T (dense*dense) */
  CHKERRQ(MatMatTransposeMult(B,C,MAT_INITIAL_MATRIX,fill,&D));
  CHKERRQ(MatMatTransposeMult(B,C,MAT_REUSE_MATRIX,fill,&D));
  CHKERRQ(MatMatTransposeMultEqual(B,C,D,10,&equal));
  PetscCheckFalse(!equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != B*C^T (dense*dense)");
  CHKERRQ(MatDestroy(&D));

  /* Test MatProductCreateWithMat() and reuse C and B for B = A*C */
  flg = PETSC_FALSE;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-test_userAPI",&flg));
  if (flg) {
    /* user driver */
    CHKERRQ(MatMatMult(A,C,MAT_REUSE_MATRIX,fill,&B));
  } else {
    /* clear internal data structures related with previous products to avoid circular references */
    CHKERRQ(MatProductClear(A));
    CHKERRQ(MatProductClear(B));
    CHKERRQ(MatProductClear(C));
    CHKERRQ(MatProductCreateWithMat(A,C,NULL,B));
    CHKERRQ(MatProductSetType(B,MATPRODUCT_AB));
    CHKERRQ(MatProductSetFromOptions(B));
    CHKERRQ(MatProductSymbolic(B));
    CHKERRQ(MatProductNumeric(B));
  }

  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&A));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -M 10 -N 10
      output_file: output/ex109.out

   test:
      suffix: 2
      nsize: 3
      output_file: output/ex109.out

   test:
      suffix: 3
      nsize: 2
      args: -matmattransmult_mpidense_mpidense_via cyclic
      output_file: output/ex109.out

   test:
      suffix: 4
      args: -test_userAPI
      output_file: output/ex109.out

   test:
      suffix: 5
      nsize: 3
      args: -test_userAPI
      output_file: output/ex109.out

TEST*/
