/*$Id: ex24.c,v 1.2 2000/09/23 14:56:18 balay Exp balay $*/

static char help[] = 
"Tests CG, MINRES and SYMMLQ on symmetric matrice with SBAIJ format. The preconditioner ICC only works on sequential SBAIJ format. \n\n";

#include "petscsles.h"


#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C;
  Scalar      v,none = -1.0;
  int         i,j,I,J,ierr,Istart,Iend,N,m = 4,n = 4,rank,size,its,k;
  double      err_norm,res_norm;
  Vec         x,b,u,u_tmp;
  PetscRandom r;
  SLES        sles;
  PC          pc;          
  KSP         ksp;  

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  N = m*n;


  /* Generate matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,&C);CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRA(ierr);
  for (I=Istart; I<Iend; I++) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if (i>0)   {J = I - n; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRA(ierr);}
    if (i<m-1) {J = I + n; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRA(ierr);}
    if (j>0)   {J = I - 1; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRA(ierr);}
    if (j<n-1) {J = I + 1; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRA(ierr);}
    v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,ADD_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* a shift can make C indefinite. Preconditioners LU, ILU (for BAIJ format) and ICC may fail */
  /* ierr = MatShift(&alpha, C); CHKERRA(ierr); */
  /* ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr); */

  /* PART 3: Setup and solve for system */
    
  /* Create vectors.  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,N,&x);CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&b);CHKERRA(ierr);
  ierr = VecDuplicate(x,&u);CHKERRA(ierr);
  ierr = VecDuplicate(x,&u_tmp);CHKERRA(ierr);

  /* Set exact solution u; then compute right-hand-side vector b. */   
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&r);CHKERRA(ierr);
  ierr = VecSetRandom(r,u);CHKERRA(ierr);
  ierr = PetscRandomDestroy(r); CHKERRA(ierr); 
  
  ierr = MatMult(C,u,b);CHKERRA(ierr); 

  for (k=0; k<3; k++){
    if (k == 0){                              /* CG  */
      ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
      ierr = SLESSetOperators(sles,C,C,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
      ierr = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n CG: \n");CHKERRA(ierr);
      ierr = KSPSetType(ksp,KSPCG);CHKERRA(ierr); 
    } else if (k == 1){                       /* MINRES */
      ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
      ierr = SLESSetOperators(sles,C,C,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
      ierr = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n MINRES: \n");CHKERRA(ierr);
      ierr = KSPSetType(ksp,KSPMINRES);CHKERRA(ierr); 
    } else {                                 /* SYMMLQ */
      ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
      ierr = SLESSetOperators(sles,C,C,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
      ierr = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n SYMMLQ: \n");CHKERRA(ierr);
      ierr = KSPSetType(ksp,KSPSYMMLQ);CHKERRA(ierr); 
    }

    ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
    /* ierr = PCSetType(pc,PCICC);CHKERRA(ierr); */
    ierr = PCSetType(pc,PCJACOBI);CHKERRA(ierr); 
    ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRA(ierr);

    /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    SLESSetFromOptions() is called _after_ any other customization
    routines.
    */
    ierr = SLESSetFromOptions(sles);CHKERRA(ierr);   

    /* Solve linear system; */ 
    ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);
   
  /* Check error */
    ierr = VecCopy(u,u_tmp); CHKERRA(ierr); 
    ierr = VecAXPY(&none,x,u_tmp);CHKERRA(ierr);
    ierr = VecNorm(u_tmp,NORM_2,&err_norm);CHKERRA(ierr);
    ierr = MatMult(C,x,u_tmp);CHKERRA(ierr);  
    ierr = VecAXPY(&none,b,u_tmp);CHKERRA(ierr);
    ierr = VecNorm(u_tmp,NORM_2,&res_norm);CHKERRA(ierr);
  
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3d\n",its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %A;",res_norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Error norm %A.\n",err_norm);CHKERRQ(ierr);

    ierr = SLESDestroy(sles);CHKERRA(ierr);
  }
   
  /* 
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
  */
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr); 
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(u_tmp);CHKERRA(ierr);  
  ierr = MatDestroy(C);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}


