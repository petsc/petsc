
static char help[] = "Tests CG, MINRES and SYMMLQ on symmetric matrices with SBAIJ format. The preconditioner ICC only works on sequential SBAIJ format. \n\n";

#include <petscksp.h>


int main(int argc,char **args)
{
  Mat            C;
  PetscScalar    v,none = -1.0;
  PetscInt       i,j,Ii,J,Istart,Iend,N,m = 4,n = 4,its,k;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscReal      err_norm,res_norm;
  Vec            x,b,u,u_tmp;
  PC             pc;
  KSP            ksp;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  N    = m*n;

  /* Generate matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* a shift can make C indefinite. Preconditioners LU, ILU (for BAIJ format) and ICC may fail */
  /* ierr = MatShift(C,alpha);CHKERRQ(ierr); */
  /* ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* Setup and solve for system */
  /* Create vectors.  */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u_tmp);CHKERRQ(ierr);
  /* Set exact solution u; then compute right-hand-side vector b. */
  ierr = VecSet(u,1.0);CHKERRQ(ierr);
  ierr = MatMult(C,u,b);CHKERRQ(ierr);

  for (k=0; k<3; k++) {
    if (k == 0) {                              /* CG  */
      ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,C,C);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n CG: \n");CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
    } else if (k == 1) {                       /* MINRES */
      ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,C,C);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n MINRES: \n");CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPMINRES);CHKERRQ(ierr);
    } else {                                 /* SYMMLQ */
      ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,C,C);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n SYMMLQ: \n");CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPSYMMLQ);CHKERRQ(ierr);
    }
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    /* ierr = PCSetType(pc,PCICC);CHKERRQ(ierr); */
    ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

    /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
    */
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

    /* Solve linear system; */
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    /* Check error */
    ierr = VecCopy(u,u_tmp);CHKERRQ(ierr);
    ierr = VecAXPY(u_tmp,none,x);CHKERRQ(ierr);
    ierr = VecNorm(u_tmp,NORM_2,&err_norm);CHKERRQ(ierr);
    ierr = MatMult(C,x,u_tmp);CHKERRQ(ierr);
    ierr = VecAXPY(u_tmp,none,b);CHKERRQ(ierr);
    ierr = VecNorm(u_tmp,NORM_2,&res_norm);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g;",(double)res_norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Error norm %g.\n",(double)err_norm);CHKERRQ(ierr);
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
