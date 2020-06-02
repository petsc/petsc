
static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load. For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscksp.h>
#include <petsclog.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       its;
  PetscLogStage  stage1,stage2;
  PetscReal      norm;
  Vec            x,b,u;
  Mat            A;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscBool      table = PETSC_FALSE,flg;
  KSP            ksp;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL,NULL,"-table",&table,NULL);CHKERRQ(ierr);

  /* Read matrix and RHS */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecLoad(b,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /*
   If the load matrix is larger then the vector, due to being padded
   to match the blocksize then create a new padded vector
  */
  {
    PetscInt    m,n,j,mvec,start,end,indx;
    Vec         tmp;
    PetscScalar *bold;

    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&tmp);CHKERRQ(ierr);
    ierr = VecSetSizes(tmp,m,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(tmp);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(b,&start,&end);CHKERRQ(ierr);
    ierr = VecGetLocalSize(b,&mvec);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bold);CHKERRQ(ierr);
    for (j=0; j<mvec; j++) {
      indx = start+j;
      ierr = VecSetValues(tmp,1,&indx,bold+j,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(b,&bold);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(tmp);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tmp);CHKERRQ(ierr);
    b    = tmp;
  }
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);

  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)A);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("mystage 1",&stage1);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage1);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)A);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("mystage 2",&stage2);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage2);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /* Show result */
  ierr = MatMult(A,x,u);CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  /*  matrix PC   KSP   Options       its    residual  */
  if (table) {
    char        *matrixname,kspinfo[120];
    PetscViewer viewer;
    ierr = PetscViewerStringOpen(PETSC_COMM_WORLD,kspinfo,sizeof(kspinfo),&viewer);CHKERRQ(ierr);
    ierr = KSPView(ksp,viewer);CHKERRQ(ierr);
    ierr = PetscStrrchr(file,'/',&matrixname);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%-8.8s %3D %2.0e %s \n",matrixname,its,norm,kspinfo);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm = %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* Cleanup */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -ksp_type preonly  -pc_type lu -options_left no  -f ${DATAFILESPATH}/matrices/arco1
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)

    test:
      suffix: 2
      args: -sub_pc_type ilu -options_left no  -f ${DATAFILESPATH}/matrices/arco1 -ksp_gmres_restart 100 -ksp_gmres_cgs_refinement_type refine_always -sub_ksp_type preonly -pc_type bjacobi -pc_bjacobi_blocks 8 -sub_pc_factor_in_place -ksp_monitor_short
      requires: datafilespath double  !complex !define(PETSC_USE_64BIT_INDICES)

    test:
      suffix: 7
      args: -ksp_gmres_cgs_refinement_type refine_always -pc_type asm -pc_asm_blocks 6 -f ${DATAFILESPATH}/matrices/small -matload_block_size 6  -ksp_monitor_short
      requires: datafilespath double  !complex !define(PETSC_USE_64BIT_INDICES)

    test:
      requires: double !complex !define(PETSC_USE_64BIT_INDICES)
      suffix: 3
      filter: sed -e "s/CONVERGED_RTOL/CONVERGED_ATOL/g"
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64 -pc_type none -ksp_type {{cg groppcg pipecg pipecgrr pipelcg pipeprcg cgne nash stcg gltr fcg pipefcg gmres pipefgmres fgmres lgmres dgmres pgmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl pipebcgs cgs tfqmr cr pipecr lsqr qcg bicg minres symmlq lcd gcr pipegcr cgls}} -ksp_max_it 20 -ksp_error_if_not_converged -ksp_converged_reason

    test:
      requires: double !complex !define(PETSC_USE_64BIT_INDICES)
      suffix: 3_maxits
      output_file: output/ex6_maxits.out
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64 -pc_type none -ksp_type {{chebyshev cg groppcg pipecg pipecgrr pipelcg pipeprcg cgne nash stcg gltr fcg pipefcg gmres pipefgmres fgmres lgmres dgmres pgmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl pipebcgs cgs tfqmr cr pipecr qcg bicg minres symmlq lcd gcr pipegcr cgls richardson}} -ksp_max_it 3 -ksp_error_if_not_converged -ksp_converged_maxits -ksp_converged_reason

TEST*/
