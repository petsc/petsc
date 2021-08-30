
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
  PetscErrorCode ierr;
  PetscBool      flg,test=PETSC_FALSE,reuse=PETSC_FALSE,viewexpl=PETSC_FALSE;
  PetscScalar    v;
  PC             pc;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_hmg_interface",&test,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_reuse_interpolation",&reuse,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-view_explicit_mat",&viewexpl,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n*bs,m*n*bs);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A,bs);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = MatHYPRESetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
#endif

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  for (Ii=Istart/bs; Ii<Iend/bs; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0) {
      J = Ii - n;
      for (jj=0; jj<bs; jj++) {
        II = Ii*bs + jj;
        JJ = J*bs + jj;
        ierr = MatSetValues(A,1,&II,1,&JJ,&v,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    if (i<m-1) {
      J = Ii + n;
      for (jj=0; jj<bs; jj++) {
        II = Ii*bs + jj;
        JJ = J*bs + jj;
        ierr = MatSetValues(A,1,&II,1,&JJ,&v,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    if (j>0) {
      J = Ii - 1;
      for (jj=0; jj<bs; jj++) {
        II = Ii*bs + jj;
        JJ = J*bs + jj;
        ierr = MatSetValues(A,1,&II,1,&JJ,&v,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    if (j<n-1) {
      J = Ii + 1;
      for (jj=0; jj<bs; jj++) {
        II = Ii*bs + jj;
        JJ = J*bs + jj;
        ierr = MatSetValues(A,1,&II,1,&JJ,&v,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    v = 4.0;
    for (jj=0; jj<bs; jj++) {
      II = Ii*bs + jj;
      ierr = MatSetValues(A,1,&II,1,&II,&v,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (viewexpl) {
    Mat E;
    ierr = MatComputeOperator(A,MATAIJ,&E);CHKERRQ(ierr);
    ierr = MatView(E,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&E);CHKERRQ(ierr);
  }

  ierr = MatCreateVecs(A,&u,NULL);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  ierr = VecSet(u,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,u,b);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-view_exact_sol",&flg,NULL);CHKERRQ(ierr);
  if (flg) {ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-2/((m+1)*(n+1)),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  if (test) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHMG);CHKERRQ(ierr);
    ierr = PCHMGSetInnerPCType(pc,PCGAMG);CHKERRQ(ierr);
    ierr = PCHMGSetReuseInterpolation(pc,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PCHMGSetUseSubspaceCoarsening(pc,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PCHMGUseMatMAIJ(pc,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PCHMGSetCoarseningComponent(pc,0);CHKERRQ(ierr);
  }

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  if (reuse) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    /* Make sparsity pattern different and reuse interpolation */
    ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
    n = 0;
    v = 0;
    m--;
    /* Connect the last element to the first element */
    ierr = MatSetValue(A,m,n,v,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  }

  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",(double)norm,its);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
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
