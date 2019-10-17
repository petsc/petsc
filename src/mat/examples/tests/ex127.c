static char help[] = "Test MatMult() for Hermitian matrix.\n\n";
/*
  Example of usage
    ./ex127 -check_Hermitian -display_mat -display_vec
    mpiexec -n 2 ./ex127
*/

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,As;
  Vec            x,y,ys;
  PetscBool      flg,disp_mat=PETSC_FALSE,disp_vec=PETSC_FALSE;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       m,i,j;
  PetscScalar    v,sigma2;
  PetscRandom    rctx = NULL;
  PetscReal      h2,sigma1=100.0,norm;
  PetscInt       dim,Ii,J,n = 3,rstart,rend;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL, "-display_mat", &disp_mat);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL, "-display_vec", &disp_vec);CHKERRQ(ierr);

  ierr = PetscOptionsGetReal(NULL,NULL,"-sigma1",&sigma1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  dim  = n*n;

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-norandom",&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(rctx,0.0,PETSC_i);CHKERRQ(ierr);
    ierr = PetscRandomGetValue(rctx,&sigma2);CHKERRQ(ierr); /* RealPart(sigma2) == 0.0 */
  } else {
    sigma2 = 10.0*PETSC_i;
  }
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
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"A is not symmetric");
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

  /* Check whether A is Hermitian, then set A->hermitian flag */
  ierr = PetscOptionsHasName(NULL,NULL, "-check_Hermitian", &flg);CHKERRQ(ierr);
  if (flg && size == 1) {
    ierr = MatIsHermitian(A,0.0,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"A is not Hermitian");
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
    if (!rank) {
      ierr = PetscPrintf(PETSC_COMM_SELF," MatInertia: nneg: %D, nzero: %D, npos: %D\n",nneg,nzero,npos);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = ISDestroy(&perm);CHKERRQ(ierr);
    ierr = ISDestroy(&iperm);CHKERRQ(ierr);
  }
#endif

  /* Create a Hermitian matrix As in sbaij format */
  ierr = MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&As);CHKERRQ(ierr);
  if (disp_mat) {
    ierr = PetscPrintf(PETSC_COMM_WORLD," As:\n");CHKERRQ(ierr);
    ierr = MatView(As,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  if (rctx) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
  } else {
    ierr = VecSet(x,1.0);CHKERRQ(ierr);
  }

  /* Create vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&ys);CHKERRQ(ierr);

  /* Test MatMult */
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = MatMult(As,x,ys);CHKERRQ(ierr);
  if (disp_vec) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"y = A*x:\n");CHKERRQ(ierr);
    ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ys = As*x:\n");CHKERRQ(ierr);
    ierr = VecView(ys,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = VecAXPY(y,-1.0,ys);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_INFINITY,&norm);CHKERRQ(ierr);
  if (norm > 100.0*PETSC_MACHINE_EPSILON || disp_vec) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"|| A*x - As*x || = %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* Free spaces */
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&As);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&ys);CHKERRQ(ierr);
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
