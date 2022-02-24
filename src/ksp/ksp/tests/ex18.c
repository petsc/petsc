
static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscmat.h>
#include <petscksp.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       its,m,n,mvec;
  PetscReal      norm;
  Vec            x,b,u;
  Mat            A;
  KSP            ksp;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage1;
#endif

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  /* Read matrix and RHS */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATSEQAIJ));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
  CHKERRQ(VecLoad(b,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  /*
     If the load matrix is larger then the vector, due to being padded
     to match the blocksize then create a new padded vector
  */
  CHKERRQ(MatGetSize(A,&m,&n));
  CHKERRQ(VecGetSize(b,&mvec));
  if (m > mvec) {
    Vec         tmp;
    PetscScalar *bold,*bnew;
    /* create a new vector b by padding the old one */
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&tmp));
    CHKERRQ(VecSetSizes(tmp,PETSC_DECIDE,m));
    CHKERRQ(VecSetFromOptions(tmp));
    CHKERRQ(VecGetArray(tmp,&bnew));
    CHKERRQ(VecGetArray(b,&bold));
    CHKERRQ(PetscArraycpy(bnew,bold,mvec));
    CHKERRQ(VecDestroy(&b));
    b    = tmp;
  }

  /* Set up solution */
  CHKERRQ(VecDuplicate(b,&x));
  CHKERRQ(VecDuplicate(b,&u));
  CHKERRQ(VecSet(x,0.0));

  /* Solve system */
  CHKERRQ(PetscLogStageRegister("Stage 1",&stage1));
  CHKERRQ(PetscLogStagePush(stage1));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,b,x));
  CHKERRQ(PetscLogStagePop());

  /* Show result */
  CHKERRQ(MatMult(A,x,u));
  CHKERRQ(VecAXPY(u,-1.0,b));
  CHKERRQ(VecNorm(u,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm));

  /* Cleanup */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(MatDestroy(&A));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -ksp_gmres_cgs_refinement_type refine_always -f  ${DATAFILESPATH}/matrices/arco1 -ksp_monitor_short
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
