
static char help[] = "Tests CG, MINRES and SYMMLQ on symmetric matrices with SBAIJ format. The preconditioner ICC only works on sequential SBAIJ format. \n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscScalar    v,none = -1.0;
  PetscInt       i,j,Ii,J,Istart,Iend,N,m = 4,n = 4,its,k;
  PetscMPIInt    size,rank;
  PetscReal      err_norm,res_norm;
  Vec            x,b,u,u_tmp;
  PC             pc;
  KSP            ksp;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  N    = m*n;

  /* Generate matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(MatGetOwnershipRange(C,&Istart,&Iend));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    v = 4.0; CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* a shift can make C indefinite. Preconditioners LU, ILU (for BAIJ format) and ICC may fail */
  /* CHKERRQ(MatShift(C,alpha)); */
  /* CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD)); */

  /* Setup and solve for system */
  /* Create vectors.  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,N));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(VecDuplicate(x,&u));
  CHKERRQ(VecDuplicate(x,&u_tmp));
  /* Set exact solution u; then compute right-hand-side vector b. */
  CHKERRQ(VecSet(u,1.0));
  CHKERRQ(MatMult(C,u,b));

  for (k=0; k<3; k++) {
    if (k == 0) {                              /* CG  */
      CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
      CHKERRQ(KSPSetOperators(ksp,C,C));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n CG: \n"));
      CHKERRQ(KSPSetType(ksp,KSPCG));
    } else if (k == 1) {                       /* MINRES */
      CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
      CHKERRQ(KSPSetOperators(ksp,C,C));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n MINRES: \n"));
      CHKERRQ(KSPSetType(ksp,KSPMINRES));
    } else {                                 /* SYMMLQ */
      CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
      CHKERRQ(KSPSetOperators(ksp,C,C));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n SYMMLQ: \n"));
      CHKERRQ(KSPSetType(ksp,KSPSYMMLQ));
    }
    CHKERRQ(KSPGetPC(ksp,&pc));
    /* CHKERRQ(PCSetType(pc,PCICC)); */
    CHKERRQ(PCSetType(pc,PCJACOBI));
    CHKERRQ(KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));

    /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
    */
    CHKERRQ(KSPSetFromOptions(ksp));

    /* Solve linear system; */
    CHKERRQ(KSPSetUp(ksp));
    CHKERRQ(KSPSolve(ksp,b,x));

    CHKERRQ(KSPGetIterationNumber(ksp,&its));
    /* Check error */
    CHKERRQ(VecCopy(u,u_tmp));
    CHKERRQ(VecAXPY(u_tmp,none,x));
    CHKERRQ(VecNorm(u_tmp,NORM_2,&err_norm));
    CHKERRQ(MatMult(C,x,u_tmp));
    CHKERRQ(VecAXPY(u_tmp,none,b));
    CHKERRQ(VecNorm(u_tmp,NORM_2,&res_norm));

    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g;",(double)res_norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Error norm %g.\n",(double)err_norm));
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

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -pc_type icc -mat_type seqsbaij -mat_ignore_lower_triangular

    test:
      suffix: 2
      args: -pc_type icc -pc_factor_levels 2  -mat_type seqsbaij -mat_ignore_lower_triangular

    test:
      suffix: 3
      nsize: 2
      args: -pc_type bjacobi -sub_pc_type icc  -mat_type mpisbaij -mat_ignore_lower_triangular -ksp_max_it 8

    test:
      suffix: 4
      nsize: 2
      args: -pc_type bjacobi -sub_pc_type icc -sub_pc_factor_levels 1 -mat_type mpisbaij -mat_ignore_lower_triangular

TEST*/
