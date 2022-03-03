static char help[] = "Test MatMult() for Hermitian matrix.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,As;
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       i,j;
  PetscScalar    v,sigma2;
  PetscReal      h2,sigma1=100.0;
  PetscInt       dim,Ii,J,n = 3,rstart,rend;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-sigma1",&sigma1,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  dim  = n*n;

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  sigma2 = 10.0*PETSC_i;
  h2 = 1.0/((n+1)*(n+1));

  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  for (Ii=rstart; Ii<rend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0) {
      J = Ii-n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (i<n-1) {
      J = Ii+n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (j>0) {
      J = Ii-1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (j<n-1) {
      J = Ii+1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    v    = 4.0 - sigma1*h2;
    CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Check whether A is symmetric */
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-check_symmetric", &flg));
  if (flg) {
    CHKERRQ(MatIsSymmetric(A,0.0,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_USER,"A is not symmetric");
  }
  CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));

  /* make A complex Hermitian */
  Ii = 0; J = dim-1;
  if (Ii >= rstart && Ii < rend) {
    v    = sigma2*h2; /* RealPart(v) = 0.0 */
    CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    v    = -sigma2*h2;
    CHKERRQ(MatSetValues(A,1,&J,1,&Ii,&v,ADD_VALUES));
  }

  Ii = dim-2; J = dim-1;
  if (Ii >= rstart && Ii < rend) {
    v    = sigma2*h2; /* RealPart(v) = 0.0 */
    CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    v    = -sigma2*h2;
    CHKERRQ(MatSetValues(A,1,&J,1,&Ii,&v,ADD_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatViewFromOptions(A,NULL,"-disp_mat"));

  /* Check whether A is Hermitian, then set A->hermitian flag */
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-check_Hermitian", &flg));
  if (flg && size == 1) {
    CHKERRQ(MatIsHermitian(A,0.0,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_USER,"A is not Hermitian");
  }
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));

#if defined(PETSC_HAVE_SUPERLU_DIST)
  /* Test Cholesky factorization */
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-test_choleskyfactor", &flg));
  if (flg) {
    Mat      F;
    IS       perm,iperm;
    MatFactorInfo info;
    PetscInt nneg,nzero,npos;

    CHKERRQ(MatGetFactor(A,MATSOLVERSUPERLU_DIST,MAT_FACTOR_CHOLESKY,&F));
    CHKERRQ(MatGetOrdering(A,MATORDERINGND,&perm,&iperm));
    CHKERRQ(MatCholeskyFactorSymbolic(F,A,perm,&info));
    CHKERRQ(MatCholeskyFactorNumeric(F,A,&info));

    CHKERRQ(MatGetInertia(F,&nneg,&nzero,&npos));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," MatInertia: nneg: %" PetscInt_FMT ", nzero: %" PetscInt_FMT ", npos: %" PetscInt_FMT "\n",nneg,nzero,npos));
    CHKERRQ(MatDestroy(&F));
    CHKERRQ(ISDestroy(&perm));
    CHKERRQ(ISDestroy(&iperm));
  }
#endif

  /* Create a Hermitian matrix As in sbaij format */
  CHKERRQ(MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&As));
  CHKERRQ(MatViewFromOptions(As,NULL,"-disp_mat"));

  /* Test MatMult */
  CHKERRQ(MatMultEqual(A,As,10,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"MatMult not equal");
  CHKERRQ(MatMultAddEqual(A,As,10,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"MatMultAdd not equal");

  /* Free spaces */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&As));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: complex

   test:
      args: -n 1000
      output_file: output/ex127.out

   test:
      suffix: 2
      nsize: 3
      args: -n 1000
      output_file: output/ex127.out

   test:
      suffix: superlu_dist
      nsize: 3
      requires: superlu_dist
      args: -test_choleskyfactor -mat_superlu_dist_rowperm NOROWPERM
      output_file: output/ex127_superlu_dist.out
TEST*/
