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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-sigma1",&sigma1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  dim  = n*n;

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  sigma2 = 10.0*PETSC_i;
  h2 = 1.0/((n+1)*(n+1));

  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (Ii=rstart; Ii<rend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0) {
      J = Ii-n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
    }
    if (i<n-1) {
      J = Ii+n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
    }
    if (j>0) {
      J = Ii-1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
    }
    if (j<n-1) {
      J = Ii+1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
    }
    v    = 4.0 - sigma1*h2;
    ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Check whether A is symmetric */
  ierr = PetscOptionsHasName(NULL,NULL, "-check_symmetric", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatIsSymmetric(A,0.0,&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_USER,"A is not symmetric");
  }
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

  /* make A complex Hermitian */
  Ii = 0; J = dim-1;
  if (Ii >= rstart && Ii < rend) {
    v    = sigma2*h2; /* RealPart(v) = 0.0 */
    ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
    v    = -sigma2*h2;
    ierr = MatSetValues(A,1,&J,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  Ii = dim-2; J = dim-1;
  if (Ii >= rstart && Ii < rend) {
    v    = sigma2*h2; /* RealPart(v) = 0.0 */
    ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
    v    = -sigma2*h2;
    ierr = MatSetValues(A,1,&J,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-disp_mat");CHKERRQ(ierr);

  /* Check whether A is Hermitian, then set A->hermitian flag */
  ierr = PetscOptionsHasName(NULL,NULL, "-check_Hermitian", &flg);CHKERRQ(ierr);
  if (flg && size == 1) {
    ierr = MatIsHermitian(A,0.0,&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_USER,"A is not Hermitian");
  }
  ierr = MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);

#if defined(PETSC_HAVE_SUPERLU_DIST)
  /* Test Cholesky factorization */
  ierr = PetscOptionsHasName(NULL,NULL, "-test_choleskyfactor", &flg);CHKERRQ(ierr);
  if (flg) {
    Mat      F;
    IS       perm,iperm;
    MatFactorInfo info;
    PetscInt nneg,nzero,npos;

    ierr = MatGetFactor(A,MATSOLVERSUPERLU_DIST,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
    ierr = MatGetOrdering(A,MATORDERINGND,&perm,&iperm);CHKERRQ(ierr);
    ierr = MatCholeskyFactorSymbolic(F,A,perm,&info);CHKERRQ(ierr);
    ierr = MatCholeskyFactorNumeric(F,A,&info);CHKERRQ(ierr);

    ierr = MatGetInertia(F,&nneg,&nzero,&npos);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," MatInertia: nneg: %" PetscInt_FMT ", nzero: %" PetscInt_FMT ", npos: %" PetscInt_FMT "\n",nneg,nzero,npos);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = ISDestroy(&perm);CHKERRQ(ierr);
    ierr = ISDestroy(&iperm);CHKERRQ(ierr);
  }
#endif

  /* Create a Hermitian matrix As in sbaij format */
  ierr = MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&As);CHKERRQ(ierr);
  ierr = MatViewFromOptions(As,NULL,"-disp_mat");CHKERRQ(ierr);

  /* Test MatMult */
  ierr = MatMultEqual(A,As,10,&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"MatMult not equal");
  ierr = MatMultAddEqual(A,As,10,&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"MatMultAdd not equal");

  /* Free spaces */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&As);CHKERRQ(ierr);
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
