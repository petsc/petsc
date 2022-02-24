static char help[] = "Tests CG, MINRES and SYMMLQ on the symmetric indefinite matrices: afiro \n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscScalar    none = -1.0;
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;
  PetscInt       its,k;
  PetscReal      err_norm,res_norm;
  Vec            x,b,u,u_tmp;
  PC             pc;
  KSP            ksp;
  PetscViewer    view;
  char           filein[128];     /* input file name */

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Load the binary data file "filein". Set runtime option: -f filein */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n Load dataset ...\n"));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",filein,sizeof(filein),NULL));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filein,FILE_MODE_READ,&view));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetType(C,MATMPISBAIJ));
  CHKERRQ(MatLoad(C,view));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(VecLoad(b,view));
  CHKERRQ(VecLoad(u,view));
  CHKERRQ(PetscViewerDestroy(&view));
  /* CHKERRQ(VecView(b,VIEWER_STDOUT_WORLD)); */
  /* CHKERRQ(MatView(C,VIEWER_STDOUT_WORLD)); */

  CHKERRQ(VecDuplicate(u,&u_tmp));

  /* Check accuracy of the data */
  /*
  CHKERRQ(MatMult(C,u,u_tmp));
  CHKERRQ(VecAXPY(u_tmp,none,b));
  CHKERRQ(VecNorm(u_tmp,NORM_2,&res_norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Accuracy of the loading data: | b - A*u |_2 : %g \n",(double)res_norm));
  */

  /* Setup and solve for system */
  CHKERRQ(VecDuplicate(b,&x));
  for (k=0; k<3; k++) {
    if (k == 0) {                              /* CG  */
      CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
      CHKERRQ(KSPSetType(ksp,KSPCG));
      CHKERRQ(KSPSetOperators(ksp,C,C));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n CG: \n"));
    } else if (k == 1) {                       /* MINRES */
      CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
      CHKERRQ(KSPSetType(ksp,KSPMINRES));
      CHKERRQ(KSPSetOperators(ksp,C,C));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n MINRES: \n"));
    } else {                                 /* SYMMLQ */
      CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
      CHKERRQ(KSPSetOperators(ksp,C,C));
      CHKERRQ(KSPSetType(ksp,KSPSYMMLQ));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n SYMMLQ: \n"));
    }

    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetType(pc,PCNONE));

    /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
                         -pc_type jacobi -pc_jacobi_type rowmax
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization routines.
    */
    CHKERRQ(KSPSetFromOptions(ksp));

    /* Solve linear system; */
    CHKERRQ(KSPSolve(ksp,b,x));
    CHKERRQ(KSPGetIterationNumber(ksp,&its));

    /* Check error */
    CHKERRQ(VecCopy(u,u_tmp));
    CHKERRQ(VecAXPY(u_tmp,none,x));
    CHKERRQ(VecNorm(u_tmp,NORM_2,&err_norm));
    CHKERRQ(MatMult(C,x,u_tmp));
    CHKERRQ(VecAXPY(u_tmp,none,b));
    CHKERRQ(VecNorm(u_tmp,NORM_2,&res_norm));

    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3d\n",its));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm: %g;",(double)res_norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Error norm: %g.\n",(double)err_norm));

    CHKERRQ(KSPDestroy(&ksp));
  }

  /*
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
  */
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&u_tmp));
  CHKERRQ(MatDestroy(&C));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -f ${DATAFILESPATH}/matrices/indefinite/afiro -ksp_rtol 1.e-3
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
