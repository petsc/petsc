#ifndef lint
static char vcid[] = "$Id: ex8.c,v 1.55 1996/04/09 18:23:27 curfman Exp bsmith $";
#endif

static char help[] = "Tests MPI parallel linear solves with SLES.  The code\n\
illustrates repeated solution of linear systems with the same preconditioner\n\
method but different matrices (having the same nonzero structure).  Input\n\
arguments are\n\
  -m <size> : problem size\n\
  -mat_nonsym : use nonsymmetric matrix (default is symmetric)\n\n";

#include "sles.h"
#include  <stdio.h>

int main(int argc,char **args)
{
  Mat     C; 
  Scalar  v, none = -1.0;
  int     I, J, ldim, ierr, low, high, iglobal, Istart,Iend;
  int     i, j, m = 3, n = 2, rank, size, its, flg;
  Vec     x, u, b;
  SLES    sles;
  double  norm;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  n = 2*size;
  PLogStageRegister(0,"Original Solve");
  PLogStageRegister(1,"Second Solve");

  /* Create and assemble matrix */
  PLogStagePush(0);
  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&C); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend); CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
    if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
    if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
    if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
    v = 4.0; MatSetValues(C,1,&I,1,&I,&v,ADD_VALUES);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_nonsym",&flg); CHKERRA(ierr);
  if (flg) {
    for ( I=Istart; I<Iend; I++ ) { 
      v = -1.5; i = I/n;
      if ( i>1 )   {J = I-n-1; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Generate vectors */
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,m*n,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecGetLocalSize(x,&ldim); CHKERRA(ierr);
  ierr = VecGetOwnershipRange(x,&low,&high); CHKERRA(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v = (Scalar)(i + 100*rank);
    ierr = VecSetValues(u,1,&iglobal,&v,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(u); CHKERRA(ierr);
  ierr = VecAssemblyEnd(u); CHKERRA(ierr);
  
  /* Create SLES context; set operators and options; solve linear system */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,C,C,SAME_NONZERO_PATTERN); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  ierr = SLESSetUp(sles,b,x); CHKERRA(ierr);

  /* Compute right-hand-side */
  ierr = MatMult(C,u,b); CHKERRA(ierr);

  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
 
  /* Check error */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12) 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error %g, Iterations %d\n",norm,its);
  else 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error < 1.e-12, Iterations %d\n",its);

  /* Change matrix (keeping same nonzero structure) and solve again */
  PLogStagePop();
  PLogStagePush(1);
  ierr = MatSetOption(C,MAT_NO_NEW_NONZERO_LOCATIONS); CHKERRA(ierr);
  ierr = MatZeroEntries(C); CHKERRA(ierr);
  /* Fill matrix again */
  for ( i=0; i<m; i++ ) { 
    for ( j=2*rank; j<2*rank+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
      v = 6.0; ierr = MatSetValues(C,1,&I,1,&I,&v,ADD_VALUES);CHKERRA(ierr);
    }
  } 
  ierr = OptionsHasName(PETSC_NULL,"-mat_nonsym",&flg); CHKERRA(ierr);
  if (flg) {
    for ( I=Istart; I<Iend; I++ ) { 
      v = -1.5; i = I/n;
      if ( i>1 )   {J = I-n-1; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr); 

  /* Compute another right-hand-side; then solve */
  ierr = MatMult(C,u,b); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,C,C,SAME_NONZERO_PATTERN); CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /* Check error */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    PetscPrintf(MPI_COMM_WORLD,"Norm of error %g, Iterations %d\n",norm,its);
  else 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error < 1.e-12, Iterations %d\n",its);

  /* Free work space */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  PLogStagePop();

  PetscFinalize();
  return 0;
}


