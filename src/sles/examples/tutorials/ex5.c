
static char help[] = 
"This example tests MPI parallel linear solves with SLES.  The code\n\
illustrates repeated solution of linear systems with the same preconditioner\n\
method but different matrices (having the same nonzero structure).  Input\n\
arguments are\n\
  -m <size> : problem size\n\n";

#include "petsc.h"
#include "sles.h"
#include  <stdio.h>

extern int KSPMonitor_MPIRowbs(KSP,int,double,void *);

int main(int argc,char **args)
{
  Mat    C; 
  Scalar v, none = -1.0;
  int    I, J, ldim, ierr, low, high, iglobal;
  int    i, j, m = 3, n = 2, mytid, numtids, its;
  Vec    x, u, b;
  SLES   sles;
  double norm;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);
  OptionsGetInt(0,"-m",&m);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  n = 2*numtids;

  /* Generate matrix */
  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&C); CHKERRA(ierr);
  for ( i=0; i<m; i++ ) { 
    for ( j=2*mytid; j<2*mytid+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
      v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERTVALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Generate vectors */
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,m*n,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecGetLocalSize(x,&ldim); CHKERRA(ierr);
  ierr = VecGetOwnershipRange(x,&low,&high); CHKERRA(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v = (Scalar)(i + 100*mytid);
    ierr = VecSetValues(u,1,&iglobal,&v,INSERTVALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(u); CHKERRA(ierr);
  ierr = VecAssemblyEnd(u); CHKERRA(ierr);
  
  /* Compute right-hand-side */
  ierr = MatMult(C,u,b); CHKERRA(ierr);

  /* Solve linear system */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,C,C,MAT_SAME_NONZERO_PATTERN); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
  if (OptionsHasName(0,"-mat_rowbs")) {
    PC pc; KSP ksp; PCMethod pcmethod;
    ierr = SLESGetKSP(sles,&ksp); CHKERRA(ierr);
    ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);
    ierr = PCGetMethodFromContext(pc,&pcmethod); CHKERRA(ierr);
    if (pcmethod == PCICC) {
      ierr = KSPSetMonitor(ksp,KSPMonitor_MPIRowbs,(void *)C); CHKERRA(ierr);
    }
  }
#endif

  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
 
  /* Check error */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,&norm); CHKERRA(ierr);
  if (norm > 1.e-12) 
    MPIU_printf(MPI_COMM_WORLD,"Norm of error %g, Iterations %d\n",norm,its);
  else 
    MPIU_printf(MPI_COMM_WORLD,"Norm of error < 1.e-12, Iterations %d\n",its);

  MatView(C,SYNC_STDOUT_VIEWER);

  /* Change matrix (keeping same nonzero structure) and solve again */
  ierr = MatSetOption(C,NO_NEW_NONZERO_LOCATIONS); CHKERRA(ierr);
  ierr = MatZeroEntries(C); CHKERRA(ierr);
  /* Fill matrix again */
  for ( i=0; i<m; i++ ) { 
    for ( j=2*mytid; j<2*mytid+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
      v = 6.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERTVALUES); CHKERRA(ierr);
    }
  } 
  ierr = MatAssemblyBegin(C,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,FINAL_ASSEMBLY); CHKERRA(ierr); 

  /* Compute another right-hand-side; then solve */
  ierr = MatMult(C,u,b); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,C,C,MAT_SAME_NONZERO_PATTERN); CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /* Check error */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    MPIU_printf(MPI_COMM_WORLD,"Norm of error %g, Iterations %d\n",norm,its);
  else 
    MPIU_printf(MPI_COMM_WORLD,"Norm of error < 1.e-12, Iterations %d\n",its);

  /* Free work space */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}


