static char help[] = "Test MatMult() for Hermitian matrix.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,As;
  PetscBool      flg;
  PetscMPIInt    size;
  PetscInt       i,j;
  PetscScalar    v,sigma2;
  PetscReal      h2,sigma1=100.0;
  PetscInt       dim,Ii,J,n = 3,rstart,rend;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-sigma1",&sigma1,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  dim  = n*n;

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim));
  PetscCall(MatSetType(A,MATAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  sigma2 = 10.0*PETSC_i;
  h2 = 1.0/((n+1)*(n+1));

  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  for (Ii=rstart; Ii<rend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0) {
      J = Ii-n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (i<n-1) {
      J = Ii+n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (j>0) {
      J = Ii-1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (j<n-1) {
      J = Ii+1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    v    = 4.0 - sigma1*h2;
    PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Check whether A is symmetric */
  PetscCall(PetscOptionsHasName(NULL,NULL, "-check_symmetric", &flg));
  if (flg) {
    PetscCall(MatIsSymmetric(A,0.0,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_USER,"A is not symmetric");
  }
  PetscCall(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));

  /* make A complex Hermitian */
  Ii = 0; J = dim-1;
  if (Ii >= rstart && Ii < rend) {
    v    = sigma2*h2; /* RealPart(v) = 0.0 */
    PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    v    = -sigma2*h2;
    PetscCall(MatSetValues(A,1,&J,1,&Ii,&v,ADD_VALUES));
  }

  Ii = dim-2; J = dim-1;
  if (Ii >= rstart && Ii < rend) {
    v    = sigma2*h2; /* RealPart(v) = 0.0 */
    PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    v    = -sigma2*h2;
    PetscCall(MatSetValues(A,1,&J,1,&Ii,&v,ADD_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(A,NULL,"-disp_mat"));

  /* Check whether A is Hermitian, then set A->hermitian flag */
  PetscCall(PetscOptionsHasName(NULL,NULL, "-check_Hermitian", &flg));
  if (flg && size == 1) {
    PetscCall(MatIsHermitian(A,0.0,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_USER,"A is not Hermitian");
  }
  PetscCall(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));

#if defined(PETSC_HAVE_SUPERLU_DIST)
  /* Test Cholesky factorization */
  PetscCall(PetscOptionsHasName(NULL,NULL, "-test_choleskyfactor", &flg));
  if (flg) {
    Mat      F;
    IS       perm,iperm;
    MatFactorInfo info;
    PetscInt nneg,nzero,npos;

    PetscCall(MatGetFactor(A,MATSOLVERSUPERLU_DIST,MAT_FACTOR_CHOLESKY,&F));
    PetscCall(MatGetOrdering(A,MATORDERINGND,&perm,&iperm));
    PetscCall(MatCholeskyFactorSymbolic(F,A,perm,&info));
    PetscCall(MatCholeskyFactorNumeric(F,A,&info));

    PetscCall(MatGetInertia(F,&nneg,&nzero,&npos));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," MatInertia: nneg: %" PetscInt_FMT ", nzero: %" PetscInt_FMT ", npos: %" PetscInt_FMT "\n",nneg,nzero,npos));
    PetscCall(MatDestroy(&F));
    PetscCall(ISDestroy(&perm));
    PetscCall(ISDestroy(&iperm));
  }
#endif

  /* Create a Hermitian matrix As in sbaij format */
  PetscCall(MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&As));
  PetscCall(MatViewFromOptions(As,NULL,"-disp_mat"));

  /* Test MatMult */
  PetscCall(MatMultEqual(A,As,10,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"MatMult not equal");
  PetscCall(MatMultAddEqual(A,As,10,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"MatMultAdd not equal");

  /* Free spaces */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&As));
  PetscCall(PetscFinalize());
  return 0;
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
