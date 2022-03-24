
static char help[] = "Solves a linear system in parallel with KSP and HMG.\n\
Input parameters include:\n\
  -view_exact_sol    : write exact solution vector to stdout\n\
  -m  <mesh_x>       : number of mesh points in x-direction\n\
  -n  <mesh_y>       : number of mesh points in y-direction\n\
  -bs                : number of variables on each mesh vertex \n\n";

/*
  Simple example is used to test PCHMG
*/
#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;    /* approx solution, RHS, exact solution */
  Mat            A;        /* linear system matrix */
  KSP            ksp;      /* linear solver context */
  PetscReal      norm;     /* norm of solution error */
  PetscInt       i,j,Ii,J,Istart,Iend,m = 8,n = 7,its,bs=1,II,JJ,jj;
  PetscBool      flg,test=PETSC_FALSE,reuse=PETSC_FALSE,viewexpl=PETSC_FALSE;
  PetscScalar    v;
  PC             pc;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_hmg_interface",&test,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_reuse_interpolation",&reuse,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-view_explicit_mat",&viewexpl,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n*bs,m*n*bs));
  CHKERRQ(MatSetBlockSize(A,bs));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatMPIAIJSetPreallocation(A,5,NULL,5,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(A,5,NULL));
#if defined(PETSC_HAVE_HYPRE)
  CHKERRQ(MatHYPRESetPreallocation(A,5,NULL,5,NULL));
#endif

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));

  for (Ii=Istart/bs; Ii<Iend/bs; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0) {
      J = Ii - n;
      for (jj=0; jj<bs; jj++) {
        II = Ii*bs + jj;
        JJ = J*bs + jj;
        CHKERRQ(MatSetValues(A,1,&II,1,&JJ,&v,ADD_VALUES));
      }
    }
    if (i<m-1) {
      J = Ii + n;
      for (jj=0; jj<bs; jj++) {
        II = Ii*bs + jj;
        JJ = J*bs + jj;
        CHKERRQ(MatSetValues(A,1,&II,1,&JJ,&v,ADD_VALUES));
      }
    }
    if (j>0) {
      J = Ii - 1;
      for (jj=0; jj<bs; jj++) {
        II = Ii*bs + jj;
        JJ = J*bs + jj;
        CHKERRQ(MatSetValues(A,1,&II,1,&JJ,&v,ADD_VALUES));
      }
    }
    if (j<n-1) {
      J = Ii + 1;
      for (jj=0; jj<bs; jj++) {
        II = Ii*bs + jj;
        JJ = J*bs + jj;
        CHKERRQ(MatSetValues(A,1,&II,1,&JJ,&v,ADD_VALUES));
      }
    }
    v = 4.0;
    for (jj=0; jj<bs; jj++) {
      II = Ii*bs + jj;
      CHKERRQ(MatSetValues(A,1,&II,1,&II,&v,ADD_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (viewexpl) {
    Mat E;
    CHKERRQ(MatComputeOperator(A,MATAIJ,&E));
    CHKERRQ(MatView(E,NULL));
    CHKERRQ(MatDestroy(&E));
  }

  CHKERRQ(MatCreateVecs(A,&u,NULL));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecDuplicate(u,&b));
  CHKERRQ(VecDuplicate(b,&x));

  CHKERRQ(VecSet(u,1.0));
  CHKERRQ(MatMult(A,u,b));

  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-view_exact_sol",&flg,NULL));
  if (flg) CHKERRQ(VecView(u,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetTolerances(ksp,1.e-2/((m+1)*(n+1)),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT));

  if (test) {
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetType(pc,PCHMG));
    CHKERRQ(PCHMGSetInnerPCType(pc,PCGAMG));
    CHKERRQ(PCHMGSetReuseInterpolation(pc,PETSC_TRUE));
    CHKERRQ(PCHMGSetUseSubspaceCoarsening(pc,PETSC_TRUE));
    CHKERRQ(PCHMGUseMatMAIJ(pc,PETSC_FALSE));
    CHKERRQ(PCHMGSetCoarseningComponent(pc,0));
  }

  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,b,x));

  if (reuse) {
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(KSPSolve(ksp,b,x));
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(KSPSolve(ksp,b,x));
    /* Make sparsity pattern different and reuse interpolation */
    CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
    CHKERRQ(MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES,PETSC_FALSE));
    CHKERRQ(MatGetSize(A,&m,NULL));
    n = 0;
    v = 0;
    m--;
    /* Connect the last element to the first element */
    CHKERRQ(MatSetValue(A,m,n,v,ADD_VALUES));
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(KSPSolve(ksp,b,x));
  }

  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",(double)norm,its));

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !single

   test:
      suffix: hypre
      nsize: 2
      requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type hypre

   test:
      suffix: hypre_bs4
      nsize: 2
      requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type hypre -bs 4 -pc_hmg_use_subspace_coarsening 1

   test:
      suffix: hypre_asm
      nsize: 2
      requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type hypre -bs 4 -pc_hmg_use_subspace_coarsening 1 -mg_levels_3_pc_type asm

   test:
      suffix: hypre_fieldsplit
      nsize: 2
      requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type hypre -bs 4 -mg_levels_4_pc_type fieldsplit

   test:
      suffix: gamg
      nsize: 2
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type gamg

   test:
      suffix: gamg_bs4
      nsize: 2
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type gamg -bs 4 -pc_hmg_use_subspace_coarsening 1

   test:
      suffix: gamg_asm
      nsize: 2
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type gamg -bs 4 -pc_hmg_use_subspace_coarsening 1 -mg_levels_1_pc_type asm

   test:
      suffix: gamg_fieldsplit
      nsize: 2
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type gamg -bs 4 -mg_levels_1_pc_type fieldsplit

   test:
      suffix: interface
      nsize: 2
      args: -ksp_monitor -ksp_rtol 1e-6 -test_hmg_interface 1 -bs 4

   test:
      suffix: reuse
      nsize: 2
      args: -ksp_monitor -ksp_rtol 1e-6   -pc_type hmg -pc_hmg_reuse_interpolation 1 -test_reuse_interpolation 1 -hmg_inner_pc_type gamg

   test:
      suffix: component
      nsize: 2
      args: -ksp_monitor -ksp_rtol 1e-6 -pc_type hmg -pc_hmg_coarsening_component 2  -pc_hmg_use_subspace_coarsening 1 -bs 4 -hmg_inner_pc_type gamg

   testset:
      output_file: output/ex4_expl.out
      nsize: {{1 2}}
      filter: grep -v "MPI processes" | grep -v " type:" | grep -v "Mat Object"
      args: -ksp_converged_reason -view_explicit_mat -pc_type none -ksp_type {{cg gmres}}
      test:
        suffix: expl_aij
        args: -mat_type aij
      test:
        suffix: expl_hypre
        requires: hypre
        args: -mat_type hypre

   test:
      suffix: hypre_device
      nsize: {{1 2}}
      requires: hypre defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -mat_type hypre -ksp_converged_reason -pc_type hypre -m 13 -n 17

   test:
      suffix: hypre_device_cusparse
      output_file: output/ex4_hypre_device.out
      nsize: {{1 2}}
      requires: hypre cuda defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -mat_type {{aij aijcusparse}} -vec_type cuda -ksp_converged_reason -pc_type hypre -m 13 -n 17

TEST*/
