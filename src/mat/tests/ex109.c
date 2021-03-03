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
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-fill",&fill,NULL);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);

  /* Create a aij matrix A */
  M    = N = m*n;
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  am   = Iend - Istart;
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create a dense matrix B */
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,an,PETSC_DECIDE,PETSC_DECIDE,M);CHKERRQ(ierr);
  ierr = MatSetType(B,MATDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(B,NULL);CHKERRQ(ierr);
  ierr = MatMPIDenseSetPreallocation(B,NULL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetRandom(B,r);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Test reuse of user-provided dense C (unassembled) -- not recommended usage */
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetType(C,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetSizes(C,am,PETSC_DECIDE,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatZeroEntries(C);CHKERRQ(ierr);
  ierr = MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
  ierr = MatNorm(C,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  /* Test C = A*B (aij*dense) */
  ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  ierr = MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);

  /* Test developer API */
  ierr = MatProductCreate(A,B,NULL,&D);CHKERRQ(ierr);
  ierr = MatProductSetType(D,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(D,"default");CHKERRQ(ierr);
  ierr = MatProductSetFill(D,fill);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(D);CHKERRQ(ierr);
  ierr = MatProductSymbolic(D);CHKERRQ(ierr);
  for (i=0; i<2; i++) {
    ierr = MatProductNumeric(D);CHKERRQ(ierr);
  }
  ierr = MatEqual(C,D,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"C != D");
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  /* Test D = AT*B (transpose(aij)*dense) */
  ierr = MatCreateTranspose(A,&AT);CHKERRQ(ierr);
  ierr = MatMatMult(AT,B,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr);
  ierr = MatMatMultEqual(AT,B,D,10,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != AT*B (transpose(aij)*dense)");
  ierr = MatDestroy(&D);CHKERRQ(ierr);
  ierr = MatDestroy(&AT);CHKERRQ(ierr);

  /* Test D = C*A (dense*aij) */
  ierr = MatMatMult(C,A,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr);
  ierr = MatMatMult(C,A,MAT_REUSE_MATRIX,fill,&D);CHKERRQ(ierr);
  ierr = MatMatMultEqual(C,A,D,10,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != C*A (dense*aij)");
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  /* Test D = A*C (aij*dense) */
  ierr = MatMatMult(A,C,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr);
  ierr = MatMatMult(A,C,MAT_REUSE_MATRIX,fill,&D);CHKERRQ(ierr);
  ierr = MatMatMultEqual(A,C,D,10,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != A*C (aij*dense)");
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  /* Test D = B*C (dense*dense) */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size == 1) {
    ierr = MatMatMult(B,C,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr);
    ierr = MatMatMult(B,C,MAT_REUSE_MATRIX,fill,&D);CHKERRQ(ierr);
    ierr = MatMatMultEqual(B,C,D,10,&equal);CHKERRQ(ierr);
    if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != B*C (dense*dense)");
    ierr = MatDestroy(&D);CHKERRQ(ierr);
  }

  /* Test D = B*C^T (dense*dense) */
  ierr = MatMatTransposeMult(B,C,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr);
  ierr = MatMatTransposeMult(B,C,MAT_REUSE_MATRIX,fill,&D);CHKERRQ(ierr);
  ierr = MatMatTransposeMultEqual(B,C,D,10,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D != B*C^T (dense*dense)");
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  /* Test MatProductCreateWithMat() and reuse C and B for B = A*C */
  flg = PETSC_FALSE;
  ierr = PetscOptionsHasName(NULL,NULL,"-test_userAPI",&flg);CHKERRQ(ierr);
  if (flg) {
    /* user driver */
    ierr = MatMatMult(A,C,MAT_REUSE_MATRIX,fill,&B);CHKERRQ(ierr);
  } else {
    /* clear internal data structures related with previous products to avoid circular references */
    ierr = MatProductClear(A);CHKERRQ(ierr);
    ierr = MatProductClear(B);CHKERRQ(ierr);
    ierr = MatProductClear(C);CHKERRQ(ierr);
    ierr = MatProductCreateWithMat(A,C,NULL,B);CHKERRQ(ierr);
    ierr = MatProductSetType(B,MATPRODUCT_AB);CHKERRQ(ierr);
    ierr = MatProductSetFromOptions(B);CHKERRQ(ierr);
    ierr = MatProductSymbolic(B);CHKERRQ(ierr);
    ierr = MatProductNumeric(B);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
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
