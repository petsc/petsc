#ifndef lint
static char vcid[] = "$Id: ex19.c,v 1.9 1996/04/12 20:28:07 curfman Exp curfman $";
#endif

static char help[] = "Illustrates the solution of 2 different linear systems\n\
with different linear solvers.  Also, this example illustrates the repeated\n\
solution of linear systems, while reusing matrix, vector, and solver data\n\
structures throughout the process.  Note the various stages of event logging\n\n";

#include "sles.h"
#include  <stdio.h>

int CheckError(Vec,Vec,Vec,int,int);
int MyKSPMonitor(KSP,int,double,void*);

int main(int argc,char **args)
{
  Vec    u, x1, b1, x2, b2;    /* vectors for systems #1 and #2 */
  Mat    C1, C2;               /* matrices for systems #1 and #2 */
  SLES   sles1, sles2;         /* SLES contexts for systems #1 and #2 */
  KSP    ksp1;                 /* KSP context for system #1 */
  Scalar v;
  int    I, J, ldim, ierr, low, high, iglobal, Istart, Iend, Istart2, Iend2;
  int    i, j, m = 3, n = 2, rank, size, its, flg, t, ntimes = 3, CHECK_ERROR = 0;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-t",&ntimes,&flg); CHKERRA(ierr);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  n = 2*size;

  /* Register various stages and events for logging */
  PLogStageRegister(0,"Prelim setup");
  PLogStageRegister(1,"Linear System 1");
  PLogStageRegister(2,"Linear System 2");
  PLogEventRegister(&CHECK_ERROR,"Check Error     ","Red:");

   /* Create data structures for first linear system */
  PLogStagePush(0);
  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&C1); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C1,&Istart,&Iend); CHKERRA(ierr);
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,m*n,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b1); CHKERRA(ierr);
  ierr = VecDuplicate(u,&x1); CHKERRA(ierr);
  ierr = SLESCreate(MPI_COMM_WORLD,&sles1); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles1); CHKERRA(ierr);
  ierr = SLESGetKSP(sles1,&ksp1); CHKERRA(ierr);

  /* Set user-defined monitoring routine for first linear system */
  ierr = OptionsHasName(PETSC_NULL,"-my_ksp_monitor",&flg); CHKERRA(ierr);
  if (flg) {ierr = KSPSetMonitor(ksp1,MyKSPMonitor,PETSC_NULL); CHKERRQ(ierr);}

  /* Create data structures for second linear system */
  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&C2); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C2,&Istart2,&Iend2); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b2); CHKERRA(ierr);
  ierr = VecDuplicate(u,&x2); CHKERRA(ierr);
  ierr = SLESCreate(MPI_COMM_WORLD,&sles2); CHKERRA(ierr);

  /* Set different options prefix for second linear system */
  ierr = SLESAppendOptionsPrefix(sles2,"s2_"); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles2); CHKERRA(ierr);

  /* Set exact solution vector */
  ierr = VecGetLocalSize(u,&ldim); CHKERRA(ierr);
  ierr = VecGetOwnershipRange(u,&low,&high); CHKERRA(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v = (Scalar)(i + 100*rank);
    ierr = VecSetValues(u,1,&iglobal,&v,ADD_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(u); CHKERRA(ierr);
  ierr = VecAssemblyEnd(u); CHKERRA(ierr);
  PLogFlops(2*ldim);
  PLogStagePop();

  /* Solve 2 different linear systems several times in succession */
  for (t=0; t<ntimes; t++) {

    /* Assemble first matrix */
    PLogStagePush(1);
    ierr = MatZeroEntries(C1); CHKERRA(ierr);
    for ( I=Istart; I<Iend; I++ ) { 
      v = -1.0; i = I/n; j = I - i*n;  
      if ( i>0 )   {J = I - n; MatSetValues(C1,1,&I,1,&J,&v,ADD_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C1,1,&I,1,&J,&v,ADD_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(C1,1,&I,1,&J,&v,ADD_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C1,1,&I,1,&J,&v,ADD_VALUES);}
      v = 4.0; MatSetValues(C1,1,&I,1,&I,&v,ADD_VALUES);
    }
    for ( I=Istart; I<Iend; I++ ) { /* Make matrix nonsymmetric */
      v = -1.0*(t+0.5); i = I/n;
      if ( i>0 )   {J = I - n; MatSetValues(C1,1,&I,1,&J,&v,ADD_VALUES);}
    }
    PLogFlops(2*(Istart-Iend));
    ierr = MatAssemblyBegin(C1,FINAL_ASSEMBLY); CHKERRA(ierr);
    ierr = MatAssemblyEnd(C1,FINAL_ASSEMBLY); CHKERRA(ierr);

    /* Indicate same nonzero structure of successive linear system matrices */
    ierr = MatSetOption(C1,NO_NEW_NONZERO_LOCATIONS); CHKERRA(ierr);

    /* Compute right-hand-side */
    ierr = MatMult(C1,u,b1); CHKERRA(ierr);

    /* Indicate same nonzero structure of successive preconditioner
       matrices by setting SAME_NONZERO_PATTERN below */
    ierr = SLESSetOperators(sles1,C1,C1,SAME_NONZERO_PATTERN); CHKERRA(ierr);

    /* Use the previous solution of linear system #1 as the initial guess
       for the next solve of linear system #1 */
    if (t>0) {
      ierr = KSPSetInitialGuessNonzero(ksp1); CHKERRA(ierr);
    }

    /* Solve first linear system */
    ierr = SLESSetUp(sles1,b1,x1); CHKERRA(ierr);
    ierr = SLESSolve(sles1,b1,x1,&its); CHKERRA(ierr);
    ierr = CheckError(u,x1,b1,its,CHECK_ERROR); CHKERRA(ierr); 

    /* Assemble and solve another system */
    PLogStagePop();
    PLogStagePush(2);

    /* Assemble second matrix */
    ierr = MatZeroEntries(C2); CHKERRA(ierr);
    for ( i=0; i<m; i++ ) { 
      for ( j=2*rank; j<2*rank+2; j++ ) {
        v = -1.0;  I = j + n*i;
        if ( i>0 )   {J = I - n; MatSetValues(C2,1,&I,1,&J,&v,ADD_VALUES);}
        if ( i<m-1 ) {J = I + n; MatSetValues(C2,1,&I,1,&J,&v,ADD_VALUES);}
        if ( j>0 )   {J = I - 1; MatSetValues(C2,1,&I,1,&J,&v,ADD_VALUES);}
        if ( j<n-1 ) {J = I + 1; MatSetValues(C2,1,&I,1,&J,&v,ADD_VALUES);}
        v = 6.0 + t*0.5; ierr = MatSetValues(C2,1,&I,1,&I,&v,ADD_VALUES); CHKERRA(ierr);
      }
    } 
    for ( I=Istart2; I<Iend2; I++ ) { /* Make matrix nonsymmetric */
      v = -1.0*(t+0.5); i = I/n;
      if ( i>0 )   {J = I - n; MatSetValues(C2,1,&I,1,&J,&v,ADD_VALUES);}
    }
    PLogFlops(2*(Istart-Iend));
    ierr = MatAssemblyBegin(C2,FINAL_ASSEMBLY); CHKERRA(ierr);
    ierr = MatAssemblyEnd(C2,FINAL_ASSEMBLY); CHKERRA(ierr); 

    /* Indicate same nonzero structure of successive linear system matrices */
    ierr = MatSetOption(C2,NO_NEW_NONZERO_LOCATIONS); CHKERRA(ierr);

    /* Compute right-hand-side */
    ierr = MatMult(C2,u,b2); CHKERRA(ierr);

    /* Indicate same nonzero structure of successive preconditioner
       matrices by setting SAME_NONZERO_PATTERN below */
    ierr = SLESSetOperators(sles2,C2,C2,SAME_NONZERO_PATTERN); CHKERRA(ierr);

    /* Solve second linear system */
    ierr = SLESSetUp(sles2,b2,x2); CHKERRA(ierr);
    ierr = SLESSolve(sles2,b2,x2,&its); CHKERRA(ierr);
    ierr = CheckError(u,x2,b2,its,CHECK_ERROR); CHKERRA(ierr); 
    PLogStagePop();
  }

  /* Free work space */
  ierr = SLESDestroy(sles1); CHKERRA(ierr); ierr = SLESDestroy(sles2); CHKERRA(ierr);
  ierr = VecDestroy(x1); CHKERRA(ierr);     ierr = VecDestroy(x2); CHKERRA(ierr);
  ierr = VecDestroy(b1); CHKERRA(ierr);     ierr = VecDestroy(b2); CHKERRA(ierr);
  ierr = MatDestroy(C1); CHKERRA(ierr);     ierr = MatDestroy(C2); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
/* ------------------------ Check error ------------------------- */
int CheckError(Vec u,Vec x,Vec b,int its,int CHECK_ERROR)
{
  Scalar none = -1.0;
  double norm;
  int    ierr;

  PLogEventBegin(CHECK_ERROR,u,x,b,0);
  ierr = VecCopy(x,b); CHKERRA(ierr);       /* use b as a work vector */
  ierr = VecAXPY(&none,u,b); CHKERRA(ierr);
  ierr = VecNorm(b,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    PetscPrintf(MPI_COMM_WORLD,"Norm of error %g, Iterations %d\n",norm,its);
  else 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error < 1.e-12, Iterations %d\n",its);
  PLogEventEnd(CHECK_ERROR,u,x,b,0);
  return 0;
}
/* ----------- User-defined KSP monitoring routine  -------------- */

int MyKSPMonitor(KSP ksp,int n,double rnorm,void *dummy)
{
  Vec      x;
  int      ierr;

  ierr = KSPBuildSolution(ksp,PETSC_NULL,&x); CHKERRQ(ierr);
  PetscPrintf(MPI_COMM_WORLD,"iteration %d solution vector:\n",n);
  ierr = VecView(x,STDOUT_VIEWER_WORLD); CHKERRQ(ierr);
  PetscPrintf(MPI_COMM_WORLD,"iteration %d KSP Residual norm %14.12e \n",n,rnorm);
  return 0;
}

