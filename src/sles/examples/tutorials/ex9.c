#ifndef lint
static char vcid[] = "$Id: ex19.c,v 1.1 1996/03/26 17:48:33 curfman Exp curfman $";
#endif

static char help[] = "Illustrates the solution of 2 different linear systems\n\
with different linear solvers.  Also, the example illustrates the repeated solution\n\
of linear systems, while reusing matrix, vector, and solver data structures\n\
throughout the entire process. Note, in particular, the reuse of the preconditioner\n\
with the same preconditioner\n\
method but different matrices (having the same nonzero structure).  Input\n\
arguments are\n\
  -m <size> : problem size\n\
  -mat_nonsym : use nonsymmetric matrix (default is symmetric)\n\n";

#include "sles.h"
#include  <stdio.h>

int CheckError(Vec,Vec,int);

int main(int argc,char **args)
{
  Mat     C, C2; 
  Scalar  v;
  int     I, J, ldim, ierr, low, high, iglobal, Istart, Iend, Istart2, Iend2;
  int     i, j, m = 3, n = 2, rank, size, its, flg, t, ntimes = 2;
  Vec     u, x, b, x2, b2;
  SLES    sles, sles2;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-t",&ntimes,&flg); CHKERRA(ierr);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  n = 2*size;
  PLogStageRegister(0,"System 1");
  PLogStageRegister(1,"System 2");

  /* Create data structures for first linear system */
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
  ierr = SLESSetOptionsPrefix(sles2,"2"); CHKERRA(ierr);

  /* Set initial vectors */
  ierr = VecGetLocalSize(u,&ldim); CHKERRA(ierr);
  ierr = VecGetOwnershipRange(u,&low,&high); CHKERRA(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v = (Scalar)(i + 100*rank);
    ierr = VecSetValues(u,1,&iglobal,&v,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(u); CHKERRA(ierr);
  ierr = VecAssemblyEnd(u); CHKERRA(ierr);

  for (t=0; t<ntimes; t++) {

    /* Assemble first matrix */
    PLogStagePush(0);
    for ( I=Istart; I<Iend; I++ ) { 
      v = -1.0; i = I/n; j = I - i*n;  
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
      v = 4.0; MatSetValues(C,1,&I,1,&I,&v,ADD_VALUES);
   }
    for ( I=Istart; I<Iend; I++ ) { /* Make matrix nonsymmetric */
      v = -1.5; i = I/n;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
    }
    ierr = MatAssemblyBegin(C,FINAL_ASSEMBLY); CHKERRA(ierr);
    ierr = MatAssemblyEnd(C,FINAL_ASSEMBLY); CHKERRA(ierr);

    /* Compute right-hand-side */
    ierr = MatMult(C,u,b); CHKERRA(ierr);

    ierr = SLESSetOperators(sles,C,C,SAME_NONZERO_PATTERN); CHKERRA(ierr);
    ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
    ierr = SLESSetUp(sles,b,x); CHKERRA(ierr);
    ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
    ierr = CheckError(u,x,its); CHKERRA(ierr); 

    /* Solve another system */
    PLogStagePop();
    PLogStagePush(1);

    /* Assemble second matrix */
    for ( i=0; i<m; i++ ) { 
      for ( j=2*rank; j<2*rank+2; j++ ) {
        v = -1.0;  I = j + n*i;
        if ( i>0 )   {J = I - n; MatSetValues(C2,1,&I,1,&J,&v,INSERT_VALUES);}
        if ( i<m-1 ) {J = I + n; MatSetValues(C2,1,&I,1,&J,&v,INSERT_VALUES);}
        if ( j>0 )   {J = I - 1; MatSetValues(C2,1,&I,1,&J,&v,INSERT_VALUES);}
        if ( j<n-1 ) {J = I + 1; MatSetValues(C2,1,&I,1,&J,&v,INSERT_VALUES);}
        v = 6.0; ierr = MatSetValues(C2,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);
      }
    } 
    for ( I=Istart2; I<Iend2; I++ ) { /* Make matrix nonsymmetric */
      v = -1.5; i = I/n;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);}
    }
    ierr = MatAssemblyBegin(C2,FINAL_ASSEMBLY); CHKERRA(ierr);
    ierr = MatAssemblyEnd(C2,FINAL_ASSEMBLY); CHKERRA(ierr); 

    /* Compute right-hand-side */
    ierr = MatMult(C2,u,b2); CHKERRA(ierr);

    /* Solve second linear system */
    ierr = SLESSetOperators(sles2,C2,C2,SAME_NONZERO_PATTERN); CHKERRA(ierr);
    ierr = SLESSetFromOptions(sles2); CHKERRA(ierr);
    ierr = SLESSetUp(sles2,b2,x2); CHKERRA(ierr);
    ierr = SLESSolve(sles2,b2,x2,&its); CHKERRA(ierr);
    ierr = CheckError(u,x,its); CHKERRA(ierr); 
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
int CheckError(Vec u,Vec x,int its)
{
  Scalar none = -1.0;
  double norm;
  int    ierr;

  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    PetscPrintf(MPI_COMM_WORLD,"Norm of error %g, Iterations %d\n",norm,its);
  else 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error < 1.e-12, Iterations %d\n",its);
  return 0;
}
