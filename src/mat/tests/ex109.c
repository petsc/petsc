static char help[] = "Test MatMatMult() for AIJ and Dense matrices.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,B,C,D,AT;
  PetscInt       i,M,N,Istart,Iend,n=7,j,J,Ii,m=8,am,an;
  PetscScalar    v;
  PetscRandom    r;
  PetscBool      equal=PETSC_FALSE,flg;
  PetscReal      fill = 1.0,norm;
  PetscMPIInt    size;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-fill",&fill,NULL));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&r));
  PetscCall(PetscRandomSetFromOptions(r));

  /* Create a aij matrix A */
  M    = N = m*n;
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  PetscCall(MatSetType(A,MATAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A,5,NULL,5,NULL));
  PetscCall(MatSeqAIJSetPreallocation(A,5,NULL));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  am   = Iend - Istart;
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    v = 4.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Create a dense matrix B */
  PetscCall(MatGetLocalSize(A,&am,&an));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,an,PETSC_DECIDE,PETSC_DECIDE,M));
  PetscCall(MatSetType(B,MATDENSE));
  PetscCall(MatSeqDenseSetPreallocation(B,NULL));
  PetscCall(MatMPIDenseSetPreallocation(B,NULL));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetRandom(B,r));
  PetscCall(PetscRandomDestroy(&r));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Test reuse of user-provided dense C (unassembled) -- not recommended usage */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetType(C,MATDENSE));
  PetscCall(MatSetSizes(C,am,PETSC_DECIDE,PETSC_DECIDE,N));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  PetscCall(MatZeroEntries(C));
  PetscCall(MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C));
  PetscCall(MatNorm(C,NORM_INFINITY,&norm));
  PetscCall(MatDestroy(&C));

  /* Test C = A*B (aij*dense) */
  PetscCall(MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C));
  PetscCall(MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C));

  /* Test developer API */
  PetscCall(MatProductCreate(A,B,NULL,&D));
  PetscCall(MatProductSetType(D,MATPRODUCT_AB));
  PetscCall(MatProductSetAlgorithm(D,"default"));
  PetscCall(MatProductSetFill(D,fill));
  PetscCall(MatProductSetFromOptions(D));
  PetscCall(MatProductSymbolic(D));
  for (i=0; i<2; i++) {
    PetscCall(MatProductNumeric(D));
  }
  PetscCall(MatEqual(C,D,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"C != D");
  PetscCall(MatDestroy(&D));

  /* Test D = AT*B (transpose(aij)*dense) */
  PetscCall(MatCreateTranspose(A,&AT));
  PetscCall(MatMatMult(AT,B,MAT_INITIAL_MATRIX,fill,&D));
  PetscCall(MatMatMultEqual(AT,B,D,10,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != AT*B (transpose(aij)*dense)");
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&AT));

  /* Test D = C*A (dense*aij) */
  PetscCall(MatMatMult(C,A,MAT_INITIAL_MATRIX,fill,&D));
  PetscCall(MatMatMult(C,A,MAT_REUSE_MATRIX,fill,&D));
  PetscCall(MatMatMultEqual(C,A,D,10,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != C*A (dense*aij)");
  PetscCall(MatDestroy(&D));

  /* Test D = A*C (aij*dense) */
  PetscCall(MatMatMult(A,C,MAT_INITIAL_MATRIX,fill,&D));
  PetscCall(MatMatMult(A,C,MAT_REUSE_MATRIX,fill,&D));
  PetscCall(MatMatMultEqual(A,C,D,10,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != A*C (aij*dense)");
  PetscCall(MatDestroy(&D));

  /* Test D = B*C (dense*dense) */
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  if (size == 1) {
    PetscCall(MatMatMult(B,C,MAT_INITIAL_MATRIX,fill,&D));
    PetscCall(MatMatMult(B,C,MAT_REUSE_MATRIX,fill,&D));
    PetscCall(MatMatMultEqual(B,C,D,10,&equal));
    PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != B*C (dense*dense)");
    PetscCall(MatDestroy(&D));
  }

  /* Test D = B*C^T (dense*dense) */
  PetscCall(MatMatTransposeMult(B,C,MAT_INITIAL_MATRIX,fill,&D));
  PetscCall(MatMatTransposeMult(B,C,MAT_REUSE_MATRIX,fill,&D));
  PetscCall(MatMatTransposeMultEqual(B,C,D,10,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != B*C^T (dense*dense)");
  PetscCall(MatDestroy(&D));

  /* Test MatProductCreateWithMat() and reuse C and B for B = A*C */
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(NULL,NULL,"-test_userAPI",&flg));
  if (flg) {
    /* user driver */
    PetscCall(MatMatMult(A,C,MAT_REUSE_MATRIX,fill,&B));
  } else {
    /* clear internal data structures related with previous products to avoid circular references */
    PetscCall(MatProductClear(A));
    PetscCall(MatProductClear(B));
    PetscCall(MatProductClear(C));
    PetscCall(MatProductCreateWithMat(A,C,NULL,B));
    PetscCall(MatProductSetType(B,MATPRODUCT_AB));
    PetscCall(MatProductSetFromOptions(B));
    PetscCall(MatProductSymbolic(B));
    PetscCall(MatProductNumeric(B));
  }

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
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
