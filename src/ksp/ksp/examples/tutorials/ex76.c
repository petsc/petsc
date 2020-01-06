#include <petscksp.h>

static char help[] = "Solves a linear system using PCHPDDM.\n\n";

int main(int argc,char **args)
{
  Vec            x,b;        /* computed solution and RHS */
  Mat            A,aux;      /* linear system matrix */
  KSP            ksp;        /* linear solver context */
  PC             pc;
  IS             is,sizes;
  const PetscInt *idx;
  PetscMPIInt    rank,size;
  char           dir[PETSC_MAX_PATH_LEN],name[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 4) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"This example requires 4 processes");
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&aux);CHKERRQ(ierr);
  ierr = ISCreate(PETSC_COMM_SELF,&is);CHKERRQ(ierr);
  ierr = PetscStrcpy(dir,".");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-load_dir",dir,sizeof(dir),NULL);CHKERRQ(ierr);
  /* loading matrices */
  ierr = PetscSNPrintf(name,sizeof(name),"%s/sizes_%d_%d.dat",dir,rank,size);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = ISCreate(PETSC_COMM_SELF,&sizes);CHKERRQ(ierr);
  ierr = ISLoad(sizes,viewer);CHKERRQ(ierr);
  ierr = ISGetIndices(sizes,&idx);CHKERRQ(ierr);
  ierr = MatSetSizes(A,idx[0],idx[1],idx[2],idx[3]);CHKERRQ(ierr);
  ierr = ISRestoreIndices(sizes,&idx);CHKERRQ(ierr);
  ierr = ISDestroy(&sizes);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name,sizeof(name),"%s/A.dat",dir);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name,sizeof(name),"%s/is_%d_%d.dat",dir,rank,size);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = ISLoad(is,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name,sizeof(name),"%s/Neumann_%d_%d.dat",dir,rank,size);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(aux,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  /* ready for testing */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)pc,"_PCHPDDM_Neumann_IS",(PetscObject)is);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)pc,"_PCHPDDM_Neumann_Mat",(PetscObject)aux);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatDestroy(&aux);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);
  ierr = VecSet(b,1.0);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      nsize: 4
      args: -ksp_rtol 1e-3 -ksp_converged_reason -pc_type {{bjacobi hpddm}shared output} -pc_hpddm_coarse_sub_pc_type lu -sub_pc_type lu -options_left no -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO

   test:
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      suffix: geneo
      nsize: 4
      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev {{5 10 15}separate output} -pc_hpddm_coarse_p {{1 2}shared output} -pc_hpddm_coarse_pc_type redundant -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO

   test:
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      suffix: fgmres_geneo_20_p_2
      nsize: 4
      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_type fgmres -pc_hpddm_coarse_mat_type {{baij sbaij}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO

   test:
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      suffix: fgmres_geneo_20_p_2_geneo
      nsize: 4
      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_levels_2_p 2 -pc_hpddm_levels_2_mat_type {{baij sbaij}shared output} -pc_hpddm_levels_2_eps_nev {{5 20}separate output} -pc_hpddm_levels_2_sub_pc_type cholesky -pc_hpddm_levels_2_ksp_type gmres -pc_hpddm_levels_2_ksp_converged_reason -ksp_type fgmres -pc_hpddm_coarse_mat_type {{baij sbaij}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO

TEST*/
