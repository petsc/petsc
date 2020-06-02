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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* Load the binary data file "filein". Set runtime option: -f filein */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Load dataset ...\n");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",filein,sizeof(filein),NULL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filein,FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetType(C,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = MatLoad(C,view);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecLoad(b,view);CHKERRQ(ierr);
  ierr = VecLoad(u,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  /* ierr = VecView(b,VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  /* ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  ierr = VecDuplicate(u,&u_tmp);CHKERRQ(ierr);

  /* Check accuracy of the data */
  /*
  ierr = MatMult(C,u,u_tmp);CHKERRQ(ierr);
  ierr = VecAXPY(u_tmp,none,b);CHKERRQ(ierr);
  ierr = VecNorm(u_tmp,NORM_2,&res_norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Accuracy of the loading data: | b - A*u |_2 : %g \n",(double)res_norm);CHKERRQ(ierr);
  */

  /* Setup and solve for system */
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  for (k=0; k<3; k++) {
    if (k == 0) {                              /* CG  */
      ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,C,C);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n CG: \n");CHKERRQ(ierr);
    } else if (k == 1) {                       /* MINRES */
      ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPMINRES);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,C,C);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n MINRES: \n");CHKERRQ(ierr);
    } else {                                 /* SYMMLQ */
      ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,C,C);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPSYMMLQ);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n SYMMLQ: \n");CHKERRQ(ierr);
    }

    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);

    /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
                         -pc_type jacobi -pc_jacobi_type rowmax
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization routines.
    */
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

    /* Solve linear system; */
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

    /* Check error */
    ierr = VecCopy(u,u_tmp);CHKERRQ(ierr);
    ierr = VecAXPY(u_tmp,none,x);CHKERRQ(ierr);
    ierr = VecNorm(u_tmp,NORM_2,&err_norm);CHKERRQ(ierr);
    ierr = MatMult(C,x,u_tmp);CHKERRQ(ierr);
    ierr = VecAXPY(u_tmp,none,b);CHKERRQ(ierr);
    ierr = VecNorm(u_tmp,NORM_2,&res_norm);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3d\n",its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm: %g;",(double)res_norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Error norm: %g.\n",(double)err_norm);CHKERRQ(ierr);

    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  }

  /*
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
  */
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&u_tmp);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -f ${DATAFILESPATH}/matrices/indefinite/afiro -ksp_rtol 1.e-3
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)

TEST*/
