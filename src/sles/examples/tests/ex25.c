/*$Id: ex25.c,v 1.3 2000/11/15 22:56:05 balay Exp $*/

static char help[] = 
"Tests CG, MINRES and SYMMLQ on the symmetric indefinite matrices: afiro and golan\n\
Runtime options: ex25 -fload ~petsc/matrices/indefinite/afiro -pc_type jacobi -pc_jacobi_rowmax\n\
See ~petsc/matrices/indefinite/readme \n\n";

#include "petscsles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C;
  Scalar      v,none = -1.0;
  int         i,j,ierr,Istart,Iend,N,rank,size,its,k;
  double      err_norm,res_norm;
  Vec         x,b,u,u_tmp;
  PetscRandom r;
  SLES        sles;
  PC          pc;          
  KSP         ksp;
  Viewer      view;
  char        filein[128];     /* input file name */

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  
  /* Load the binary data file "filein". Set runtime option: -fload filein */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Load dataset ...\n");CHKERRA(ierr);
  ierr = OptionsGetString(PETSC_NULL,"-fload",filein,127,PETSC_NULL);CHKERRA(ierr); 
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,filein,BINARY_RDONLY,&view);CHKERRA(ierr); 
  ierr = MatLoad(view,MATMPISBAIJ,&C);CHKERRA(ierr);
  ierr = VecLoad(view,&b);CHKERRA(ierr);
  ierr = VecLoad(view,&u);CHKERRA(ierr);
  ierr = ViewerDestroy(view);CHKERRA(ierr);
  /* ierr = VecView(b,VIEWER_STDOUT_WORLD);CHKERRA(ierr); */
  /* ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr); */

  ierr = VecDuplicate(u,&u_tmp);CHKERRA(ierr);

  /* Check accuracy of the data */ 
  /*
  ierr = MatMult(C,u,u_tmp);CHKERRA(ierr);
  ierr = VecAXPY(&none,b,u_tmp);CHKERRA(ierr);
  ierr = VecNorm(u_tmp,NORM_2,&res_norm);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Accuracy of the loading data: | b - A*u |_2 : %A \n",res_norm);CHKERRA(ierr); 
  */

  /* Setup and solve for system */
  ierr = VecDuplicate(b,&x);CHKERRA(ierr);
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
    ierr = PCSetType(pc,PCNONE);CHKERRA(ierr);  
    /* ierr = PCSetType(pc,PCJACOBI);CHKERRA(ierr); */
    ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRA(ierr);

    /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
                         -pc_type jacobi -pc_jacobi_rowmax
    These options will override those specified above as long as
    SLESSetFromOptions() is called _after_ any other customization routines.
    */
    ierr = SLESSetFromOptions(sles);CHKERRA(ierr);   

    /* Solve linear system; */ 
    ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);
   
  /* Check error */
    ierr = VecCopy(u,u_tmp);CHKERRA(ierr); 
    ierr = VecAXPY(&none,x,u_tmp);CHKERRA(ierr);
    ierr = VecNorm(u_tmp,NORM_2,&err_norm);CHKERRA(ierr);
    ierr = MatMult(C,x,u_tmp);CHKERRA(ierr);  
    ierr = VecAXPY(&none,b,u_tmp);CHKERRA(ierr);
    ierr = VecNorm(u_tmp,NORM_2,&res_norm);CHKERRA(ierr);
  
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3d\n",its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm: %A;",res_norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Error norm: %A.\n",err_norm);CHKERRQ(ierr);

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


