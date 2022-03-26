static char help[] = "Tests CG, MINRES and SYMMLQ on the symmetric indefinite matrices: afiro \n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscScalar    none = -1.0;
  PetscMPIInt    rank,size;
  PetscInt       its,k;
  PetscReal      err_norm,res_norm;
  Vec            x,b,u,u_tmp;
  PC             pc;
  KSP            ksp;
  PetscViewer    view;
  char           filein[128];     /* input file name */

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Load the binary data file "filein". Set runtime option: -f filein */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n Load dataset ...\n"));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",filein,sizeof(filein),NULL));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filein,FILE_MODE_READ,&view));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetType(C,MATMPISBAIJ));
  PetscCall(MatLoad(C,view));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecLoad(b,view));
  PetscCall(VecLoad(u,view));
  PetscCall(PetscViewerDestroy(&view));
  /* PetscCall(VecView(b,VIEWER_STDOUT_WORLD)); */
  /* PetscCall(MatView(C,VIEWER_STDOUT_WORLD)); */

  PetscCall(VecDuplicate(u,&u_tmp));

  /* Check accuracy of the data */
  /*
  PetscCall(MatMult(C,u,u_tmp));
  PetscCall(VecAXPY(u_tmp,none,b));
  PetscCall(VecNorm(u_tmp,NORM_2,&res_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Accuracy of the loading data: | b - A*u |_2 : %g \n",(double)res_norm));
  */

  /* Setup and solve for system */
  PetscCall(VecDuplicate(b,&x));
  for (k=0; k<3; k++) {
    if (k == 0) {                              /* CG  */
      PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
      PetscCall(KSPSetType(ksp,KSPCG));
      PetscCall(KSPSetOperators(ksp,C,C));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n CG: \n"));
    } else if (k == 1) {                       /* MINRES */
      PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
      PetscCall(KSPSetType(ksp,KSPMINRES));
      PetscCall(KSPSetOperators(ksp,C,C));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n MINRES: \n"));
    } else {                                 /* SYMMLQ */
      PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
      PetscCall(KSPSetOperators(ksp,C,C));
      PetscCall(KSPSetType(ksp,KSPSYMMLQ));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n SYMMLQ: \n"));
    }

    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetType(pc,PCNONE));

    /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
                         -pc_type jacobi -pc_jacobi_type rowmax
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization routines.
    */
    PetscCall(KSPSetFromOptions(ksp));

    /* Solve linear system; */
    PetscCall(KSPSolve(ksp,b,x));
    PetscCall(KSPGetIterationNumber(ksp,&its));

    /* Check error */
    PetscCall(VecCopy(u,u_tmp));
    PetscCall(VecAXPY(u_tmp,none,x));
    PetscCall(VecNorm(u_tmp,NORM_2,&err_norm));
    PetscCall(MatMult(C,x,u_tmp));
    PetscCall(VecAXPY(u_tmp,none,b));
    PetscCall(VecNorm(u_tmp,NORM_2,&res_norm));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3d\n",its));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Residual norm: %g;",(double)res_norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Error norm: %g.\n",(double)err_norm));

    PetscCall(KSPDestroy(&ksp));
  }

  /*
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
  */
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&u_tmp));
  PetscCall(MatDestroy(&C));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -f ${DATAFILESPATH}/matrices/indefinite/afiro -ksp_rtol 1.e-3
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
