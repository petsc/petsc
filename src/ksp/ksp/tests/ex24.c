
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  N    = m*n;

  /* Generate matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  PetscCall(MatGetOwnershipRange(C,&Istart,&Iend));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    v = 4.0; PetscCall(MatSetValues(C,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* a shift can make C indefinite. Preconditioners LU, ILU (for BAIJ format) and ICC may fail */
  /* PetscCall(MatShift(C,alpha)); */
  /* PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD)); */

  /* Setup and solve for system */
  /* Create vectors.  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,N));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&b));
  PetscCall(VecDuplicate(x,&u));
  PetscCall(VecDuplicate(x,&u_tmp));
  /* Set exact solution u; then compute right-hand-side vector b. */
  PetscCall(VecSet(u,1.0));
  PetscCall(MatMult(C,u,b));

  for (k=0; k<3; k++) {
    if (k == 0) {                              /* CG  */
      PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
      PetscCall(KSPSetOperators(ksp,C,C));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n CG: \n"));
      PetscCall(KSPSetType(ksp,KSPCG));
    } else if (k == 1) {                       /* MINRES */
      PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
      PetscCall(KSPSetOperators(ksp,C,C));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n MINRES: \n"));
      PetscCall(KSPSetType(ksp,KSPMINRES));
    } else {                                 /* SYMMLQ */
      PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
      PetscCall(KSPSetOperators(ksp,C,C));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n SYMMLQ: \n"));
      PetscCall(KSPSetType(ksp,KSPSYMMLQ));
    }
    PetscCall(KSPGetPC(ksp,&pc));
    /* PetscCall(PCSetType(pc,PCICC)); */
    PetscCall(PCSetType(pc,PCJACOBI));
    PetscCall(KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));

    /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
    */
    PetscCall(KSPSetFromOptions(ksp));

    /* Solve linear system; */
    PetscCall(KSPSetUp(ksp));
    PetscCall(KSPSolve(ksp,b,x));

    PetscCall(KSPGetIterationNumber(ksp,&its));
    /* Check error */
    PetscCall(VecCopy(u,u_tmp));
    PetscCall(VecAXPY(u_tmp,none,x));
    PetscCall(VecNorm(u_tmp,NORM_2,&err_norm));
    PetscCall(MatMult(C,x,u_tmp));
    PetscCall(VecAXPY(u_tmp,none,b));
    PetscCall(VecNorm(u_tmp,NORM_2,&res_norm));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g;",(double)res_norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Error norm %g.\n",(double)err_norm));
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
