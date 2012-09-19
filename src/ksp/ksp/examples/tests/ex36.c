static char help[] = "Test MatGetInertia() for Hermitian matrix. \n\n";
/*
  Example of usage
    ./ex36 -check_Hermitian -display_mat -display_vec
    mpiexec -n 2 ./ex36

  This example is modified from src/mat/examples/tests/ex127.c
*/

#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
  Mat            A,As;
  PetscBool      flg,disp_mat=PETSC_FALSE;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       i,j;
  PetscScalar    v,sigma2;
  PetscRandom    rctx;
  PetscReal      h2,sigma1=100.0;
  PetscInt       dim,Ii,J,n = 3,use_random,rstart,rend;
  KSP            ksp;
  PC             pc;
  Mat            F;
  PetscInt       nneg, nzero, npos;

  PetscInitialize(&argc,&args,(char *)0,help);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,1,"This example requires complex numbers");
#endif
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-display_mat", &disp_mat);CHKERRQ(ierr);

  ierr = PetscOptionsGetReal(PETSC_NULL,"-sigma1",&sigma1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  dim  = n*n;

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-norandom",&flg);CHKERRQ(ierr);
  if (flg) use_random = 0;
  else     use_random = 1;
  if (use_random) {
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
      J = Ii-n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (i<n-1) {
      J = Ii+n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j>0) {
      J = Ii-1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {
      J = Ii+1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    v = 4.0 - sigma1*h2;
    ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Check whether A is symmetric */
  ierr = PetscOptionsHasName(PETSC_NULL, "-check_symmetric", &flg);CHKERRQ(ierr);
  if (flg) {
    Mat Trans;
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX, &Trans);
    ierr = MatEqual(A, Trans, &flg);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"A is not symmetric");
    ierr = MatDestroy(&Trans);CHKERRQ(ierr);
  }
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

  /* make A complex Hermitian */
  Ii = 0; J = dim-1;
  if (Ii >= rstart && Ii < rend){
    v = sigma2*h2; /* RealPart(v) = 0.0 */
    ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
    v = -sigma2*h2;
    ierr = MatSetValues(A,1,&J,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  Ii = dim-2; J = dim-1;
  if (Ii >= rstart && Ii < rend){
  v = sigma2*h2; /* RealPart(v) = 0.0 */
  ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
  v = -sigma2*h2;
  ierr = MatSetValues(A,1,&J,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Check whether A is Hermitian */
  ierr = PetscOptionsHasName(PETSC_NULL, "-check_Hermitian", &flg);CHKERRQ(ierr);
  if (flg) {
    Mat Hermit;
    if (disp_mat){
      if (!rank) printf(" A:\n");
      ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = MatHermitianTranspose(A,MAT_INITIAL_MATRIX, &Hermit);
    if (disp_mat){
      if (!rank) printf(" A_Hermitian:\n");
      ierr = MatView(Hermit,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = MatEqual(A, Hermit, &flg);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"A is not Hermitian");
    ierr = MatDestroy(&Hermit);CHKERRQ(ierr);
  }
  ierr = MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);

  /* Create a Hermitian matrix As in sbaij format */
  ierr = MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&As);CHKERRQ(ierr);
  if (disp_mat){
    if (!rank) {ierr = PetscPrintf(PETSC_COMM_SELF," As:\n");CHKERRQ(ierr);}
    ierr = MatView(As,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Test MatGetInertia() */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,As,As,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc);CHKERRQ(ierr);

  ierr = PCSetUp(pc);CHKERRQ(ierr);
  ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
  ierr = MatGetInertia(F,&nneg,&nzero,&npos);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscPrintf(PETSC_COMM_SELF," MatInertia: nneg: %D, nzero: %D, npos: %D\n",nneg,nzero,npos);CHKERRQ(ierr);
  }

  /* Free spaces */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  if (use_random) {ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);}
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&As);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
