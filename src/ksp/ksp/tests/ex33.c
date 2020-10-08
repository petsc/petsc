static char help[] = "Test MatGetInertia().\n\n";

/*
  Examples of command line options:
  ./ex33 -sigma 2.0 -pc_factor_mat_solver_type mumps -mat_mumps_icntl_13 1 -mat_mumps_icntl_24 1
  ./ex33 -sigma <shift> -fA <matrix_file>
*/

#include <petscksp.h>
int main(int argc,char **args)
{
  Mat            A,B,F;
  PetscErrorCode ierr;
  KSP            ksp;
  PC             pc;
  PetscInt       N, n=10, m, Istart, Iend, II, J, i,j;
  PetscInt       nneg, nzero, npos;
  PetscScalar    v,sigma;
  PetscBool      flag,loadA=PETSC_FALSE,loadB=PETSC_FALSE;
  char           file[2][PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsGetString(NULL,NULL,"-fA",file[0],sizeof(file[0]),&loadA);CHKERRQ(ierr);
  if (loadA) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatLoad(A,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

    ierr = PetscOptionsGetString(NULL,NULL,"-fB",file[1],sizeof(file[1]),&loadB);CHKERRQ(ierr);
    if (loadB) {
      /* load B to get A = A + sigma*B */
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
      ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
      ierr = MatLoad(B,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
  }

  if (!loadA) { /* Matrix A is copied from slepc-3.0.0-p6/src/examples/ex13.c. */
    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag);CHKERRQ(ierr);
    if (!flag) m=n;
    N    = n*m;
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
    for (II=Istart; II<Iend; II++) {
      v = -1.0; i = II/n; j = II-i*n;
      if (i>0) { J=II-n; MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr); }
      if (i<m-1) { J=II+n; MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr); }
      if (j>0) { J=II-1; MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr); }
      if (j<n-1) { J=II+1; MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr); }
      v=4.0; MatSetValues(A,1,&II,1,&II,&v,INSERT_VALUES);CHKERRQ(ierr);

    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  /* ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  if (!loadB) {
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(B);CHKERRQ(ierr);
    ierr = MatSetUp(B);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

    for (II=Istart; II<Iend; II++) {
      /* v=4.0; MatSetValues(B,1,&II,1,&II,&v,INSERT_VALUES);CHKERRQ(ierr); */
      v=1.0; MatSetValues(B,1,&II,1,&II,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  /* ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* Set a shift: A = A - sigma*B */
  ierr = PetscOptionsGetScalar(NULL,NULL,"-sigma",&sigma,&flag);CHKERRQ(ierr);
  if (flag) {
    sigma = -1.0 * sigma;
    ierr  = MatAXPY(A,sigma,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); /* A <- A - sigma*B */
    /* ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  }

  /* Test MatGetInertia() */
  /* if A is symmetric, set its flag -- required by MatGetInertia() */
  ierr = MatIsSymmetric(A,0.0,&flag);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc);CHKERRQ(ierr);

  ierr = PCSetUp(pc);CHKERRQ(ierr);
  ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
  ierr = MatGetInertia(F,&nneg,&nzero,&npos);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  if (!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF," MatInertia: nneg: %D, nzero: %D, npos: %D\n",nneg,nzero,npos);CHKERRQ(ierr);
  }

  /* Destroy */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -sigma 2.0
      requires: !complex
      output_file: output/ex33.out

    test:
      suffix: mumps
      args: -sigma 2.0 -pc_factor_mat_solver_type mumps -mat_mumps_icntl_13 1 -mat_mumps_icntl_24 1
      requires: mumps !complex
      output_file: output/ex33.out

    test:
      suffix: mumps_2
      args: -sigma 2.0 -pc_factor_mat_solver_type mumps -mat_mumps_icntl_13 1 -mat_mumps_icntl_24 1
      requires: mumps !complex
      nsize: 3
      output_file: output/ex33.out

    test:
      suffix: mkl_pardiso
      args: -sigma 2.0 -pc_factor_mat_solver_type mkl_pardiso -mat_type sbaij
      requires: mkl_pardiso !complex
      output_file: output/ex33.out

    test:
      suffix: superlu_dist
      args: -sigma 2.0 -pc_factor_mat_solver_type superlu_dist -mat_superlu_dist_rowperm NOROWPERM
      requires: superlu_dist !complex
      output_file: output/ex33.out

    test:
      suffix: superlu_dist_2
      args: -sigma 2.0 -pc_factor_mat_solver_type superlu_dist -mat_superlu_dist_rowperm NOROWPERM
      requires: superlu_dist !complex
      nsize: 3
      output_file: output/ex33.out

TEST*/
