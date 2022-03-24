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
  KSP            ksp;
  PC             pc;
  PetscInt       N, n=10, m, Istart, Iend, II, J, i,j;
  PetscInt       nneg, nzero, npos;
  PetscScalar    v,sigma;
  PetscBool      flag,loadA=PETSC_FALSE,loadB=PETSC_FALSE;
  char           file[2][PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscMPIInt    rank;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fA",file[0],sizeof(file[0]),&loadA));
  if (loadA) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&viewer));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
    CHKERRQ(MatLoad(A,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));

    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fB",file[1],sizeof(file[1]),&loadB));
    if (loadB) {
      /* load B to get A = A + sigma*B */
      CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&viewer));
      CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
      CHKERRQ(MatLoad(B,viewer));
      CHKERRQ(PetscViewerDestroy(&viewer));
    }
  }

  if (!loadA) { /* Matrix A is copied from slepc-3.0.0-p6/src/examples/ex13.c. */
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
    if (!flag) m=n;
    N    = n*m;
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
    CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
    CHKERRQ(MatSetFromOptions(A));
    CHKERRQ(MatSetUp(A));

    CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
    for (II=Istart; II<Iend; II++) {
      v = -1.0; i = II/n; j = II-i*n;
      if (i>0) { J=II-n; CHKERRQ(MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES)); }
      if (i<m-1) { J=II+n; CHKERRQ(MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES)); }
      if (j>0) { J=II-1; CHKERRQ(MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES)); }
      if (j<n-1) { J=II+1; CHKERRQ(MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES)); }
      v=4.0; CHKERRQ(MatSetValues(A,1,&II,1,&II,&v,INSERT_VALUES));

    }
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  }
  /* CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD)); */

  if (!loadB) {
    CHKERRQ(MatGetLocalSize(A,&m,&n));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
    CHKERRQ(MatSetSizes(B,m,n,PETSC_DECIDE,PETSC_DECIDE));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatSetUp(B));
    CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));

    for (II=Istart; II<Iend; II++) {
      v=1.0; CHKERRQ(MatSetValues(B,1,&II,1,&II,&v,INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  /* CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD)); */

  /* Set a shift: A = A - sigma*B */
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-sigma",&sigma,&flag));
  if (flag) {
    sigma = -1.0 * sigma;
    CHKERRQ(MatAXPY(A,sigma,B,DIFFERENT_NONZERO_PATTERN)); /* A <- A - sigma*B */
    /* CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD)); */
  }

  /* Test MatGetInertia() */
  /* if A is symmetric, set its flag -- required by MatGetInertia() */
  CHKERRQ(MatIsSymmetric(A,0.0,&flag));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPPREONLY));
  CHKERRQ(KSPSetOperators(ksp,A,A));

  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCCHOLESKY));
  CHKERRQ(PCSetFromOptions(pc));

  CHKERRQ(PCSetUp(pc));
  CHKERRQ(PCFactorGetMatrix(pc,&F));
  CHKERRQ(MatGetInertia(F,&nneg,&nzero,&npos));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF," MatInertia: nneg: %D, nzero: %D, npos: %D\n",nneg,nzero,npos));
  }

  /* Destroy */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscFinalize());
  return 0;
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
