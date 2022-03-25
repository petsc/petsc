
static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscmat.h>
#include <petscksp.h>

int main(int argc,char **args)
{
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));

  /* Read matrix and RHS */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetType(A,MATSEQAIJ));
  PetscCall(MatLoad(A,fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
  PetscCall(VecLoad(b,fd));
  PetscCall(PetscViewerDestroy(&fd));

  /*
     If the load matrix is larger then the vector, due to being padded
     to match the blocksize then create a new padded vector
  */
  PetscCall(MatGetSize(A,&m,&n));
  PetscCall(VecGetSize(b,&mvec));
  if (m > mvec) {
    Vec         tmp;
    PetscScalar *bold,*bnew;
    /* create a new vector b by padding the old one */
    PetscCall(VecCreate(PETSC_COMM_WORLD,&tmp));
    PetscCall(VecSetSizes(tmp,PETSC_DECIDE,m));
    PetscCall(VecSetFromOptions(tmp));
    PetscCall(VecGetArray(tmp,&bnew));
    PetscCall(VecGetArray(b,&bold));
    PetscCall(PetscArraycpy(bnew,bold,mvec));
    PetscCall(VecDestroy(&b));
    b    = tmp;
  }

  /* Set up solution */
  PetscCall(VecDuplicate(b,&x));
  PetscCall(VecDuplicate(b,&u));
  PetscCall(VecSet(x,0.0));

  /* Solve system */
  PetscCall(PetscLogStageRegister("Stage 1",&stage1));
  PetscCall(PetscLogStagePush(stage1));
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,b,x));
  PetscCall(PetscLogStagePop());

  /* Show result */
  PetscCall(MatMult(A,x,u));
  PetscCall(VecAXPY(u,-1.0,b));
  PetscCall(VecNorm(u,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm));

  /* Cleanup */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -ksp_gmres_cgs_refinement_type refine_always -f  ${DATAFILESPATH}/matrices/arco1 -ksp_monitor_short
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
