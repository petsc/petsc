#ifndef lint
static char vcid[] = "$Id: ex19.c,v 1.5 1996/04/01 19:05:47 curfman Exp curfman $";
#endif

static char help[] = "Illustrates the solution of 2 different linear systems\n\
with different linear solvers.  Also, this example illustrates the repeated\n\
solution of linear systems, while reusing matrix, vector, and solver data\n\
structures throughout the process.  Note the various stages of event logging.\n\n";

#include "sles.h"
#include  <stdio.h>

int CheckError(Vec,Vec,int,int);

int main(int argc,char **args)
{
  Mat    C, C2; 
  Scalar v;
  int    I, J, ldim, ierr, low, high, iglobal, Istart, Iend, Istart2, Iend2;
  int    i, j, m = 3, n = 2, rank, size, its, flg, t, ntimes = 3, CHECK_ERROR;
  Vec    u, x, b, x2, b2;
  SLES   sles, sles2;

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
  PLogEventRegister(&CHECK_ERROR,"Check Error     ","Red");

   /* Create data structures for first linear system */
  PLogStagePush(0);
  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&C); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend); CHKERRA(ierr);
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,m*n,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);

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
    ierr = MatZeroEntries(C); CHKERRA(ierr);
    for ( I=Istart; I<Iend; I++ ) { 
      v = -1.0; i = I/n; j = I - i*n;  
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
      v = 4.0; MatSetValues(C,1,&I,1,&I,&v,ADD_VALUES);
    }
    for ( I=Istart; I<Iend; I++ ) { /* Make matrix nonsymmetric */
      v = -1.0*(t+0.5); i = I/n;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
    }
    PLogFlops(2*(Istart-Iend));
    ierr = MatAssemblyBegin(C,FINAL_ASSEMBLY); CHKERRA(ierr);
    ierr = MatAssemblyEnd(C,FINAL_ASSEMBLY); CHKERRA(ierr);

    /* Indicate same nonzero structure of successive linear system matrices */
    ierr = MatSetOption(C,NO_NEW_NONZERO_LOCATIONS); CHKERRA(ierr);

    /* Compute right-hand-side */
    ierr = MatMult(C,u,b); CHKERRA(ierr);

    /* Indicate same nonzero structure of successive preconditioner
       matrices by setting SAME_NONZERO_PATTERN below */
    ierr = SLESSetOperators(sles,C,C,SAME_NONZERO_PATTERN); CHKERRA(ierr);

    /* Solve first linear system */
    ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
    ierr = SLESSetUp(sles,b,x); CHKERRA(ierr);
    ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
    ierr = CheckError(u,x,its,CHECK_ERROR); CHKERRA(ierr); 

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

    /* Indicate reuse of identical preconditioner matrix during successive
       iterations by setting SAME_PRECONDITIONER below */
    ierr = SLESSetOperators(sles2,C2,C2,SAME_PRECONDITIONER); CHKERRA(ierr);

    /* Solve second linear system */
    ierr = SLESSetUp(sles2,b2,x2); CHKERRA(ierr);
    ierr = SLESSolve(sles2,b2,x2,&its); CHKERRA(ierr);
    ierr = CheckError(u,x2,its,CHECK_ERROR); CHKERRA(ierr); 
    PLogStagePop();
  }

  /* Free work space */
  ierr = SLESDestroy(sles); CHKERRA(ierr); ierr = SLESDestroy(sles2); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);     ierr = VecDestroy(x2); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);     ierr = VecDestroy(b2); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);     ierr = MatDestroy(C2); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
/* ------------------------ Check error ------------------------- */
int CheckError(Vec u,Vec x,int its,int CHECK_ERROR)
{
  Scalar none = -1.0;
  double norm;
  int    ierr;

  PLogEventBegin(CHECK_ERROR,u,x,0,0);
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    PetscPrintf(MPI_COMM_WORLD,"Norm of error %g, Iterations %d\n",norm,its);
  else 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error < 1.e-12, Iterations %d\n",its);
  PLogEventEnd(CHECK_ERROR,u,x,0,0);
  return 0;
}
