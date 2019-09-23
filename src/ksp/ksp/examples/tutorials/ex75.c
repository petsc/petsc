#include <petsc.h>

static char help[] = "Solves a series of linear systems using KSPHPDDM.\n\n";

int main(int argc,char **args)
{
  Vec            x,b;        /* computed solution and RHS */
  Mat            A;          /* linear system matrix */
  KSP            ksp;        /* linear solver context */
  PetscInt       i,j,nmat = 10;
  PetscViewer    viewer;
  char           name[256];
  char           dir[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  ierr = PetscStrcpy(dir,".");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-load_dir",dir,sizeof(dir),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nmat",&nmat,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  for (i=0; i<nmat; i++) {
    j = i+400;
    ierr = PetscSNPrintf(name,sizeof(name),"%s/A_%d.dat",dir,j);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = MatLoad(A,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);CHKERRQ(ierr);
    ierr = PetscSNPrintf(name,sizeof(name),"%s/rhs_%d.dat",dir,j);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(b,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -nmat 1 -pc_type none -ksp_converged_reason -ksp_type {{gmres hpddm}shared ouput} -ksp_max_it 1000 -ksp_gmres_restart 1000 -ksp_rtol 1e-10 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR

   test:
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      suffix: 1_icc
      nsize: 1
      args: -nmat 1 -pc_type icc -ksp_converged_reason -ksp_type {{gmres hpddm}shared ouput} -ksp_max_it 1000 -ksp_gmres_restart 1000 -ksp_rtol 1e-10 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR

   test:
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      suffix: 2
      args: -nmat 3 -pc_type none -ksp_converged_reason -ksp_type hpddm -ksp_max_it 1000 -ksp_gmres_restart 40 -ksp_rtol 1e-10 -ksp_hpddm_krylov_method gcrodr -ksp_hpddm_recycle 20 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR

   test:
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      suffix: 2_icc
      nsize: 1
      args: -nmat 3 -pc_type icc -ksp_converged_reason -ksp_type hpddm -ksp_max_it 1000 -ksp_gmres_restart 40 -ksp_rtol 1e-10 -ksp_hpddm_krylov_method gcrodr -ksp_hpddm_recycle 20 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR

TEST*/
