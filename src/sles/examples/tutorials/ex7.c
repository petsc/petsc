#ifndef lint
static char vcid[] = "$Id: ex5.c,v 1.7 1995/09/21 20:11:42 bsmith Exp bsmith $";
#endif

static char help[] = 
"This example illustrates use of the block Jacobi preconditioner for solving\n\
a linear system in parallel with SLES.  The code indicates the procedure for\n\
using different linear solvers on the individual blocks.\n\n";

#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  int       i, j, I, J, ierr, m = 3, n = 2;
  int       mytid, numtids, its, nlocal, first, Istart, Iend;
  Scalar    v, zero = 0.0, one = 1.0, none = -1.0;
  Vec       x, u, b;
  Mat       A; 
  SLES      sles, *subsles;
  PC        pc, subpc;
  KSP       subksp;
  double    norm;
  PCMethod  pcmethod;

  PetscInitialize(&argc,&args,0,0,help);
  OptionsGetInt(0,"-m",&m);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);  n = 2*numtids;

  /* Create and assemble matrix */
  ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,
                         0,0,0,0,&A); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 )   {J = I - n; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
    if ( i<m-1 ) {J = I + n; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
    if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
    if ( j<n-1 ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
    v = 4.0; MatSetValues(A,1,&I,1,&I,&v,ADD_VALUES);
  }
  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Create vectors for exact solution, approx solution, and RHS */
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,m*n,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = VecSet(&zero,x); CHKERRA(ierr);
  ierr = MatMult(A,u,b); CHKERRA(ierr);

  /* Create SLES context and set operators */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,ALLMAT_DIFFERENT_NONZERO_PATTERN);
         CHKERRA(ierr);

  /* Set default preconditioner */
  ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);
  ierr = PCSetMethod(pc,PCBJACOBI); CHKERRA(ierr);

  /* Set options */
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

  /* Set local solvers for Block Jacobi method.  Currently only 1 block 
     per processor is supported.  This code is intended as a simple
     illustration of setting different linear solvers for the individual 
     blocks.  These choices are obviously not recommended for solving this
     particular problem. */
  ierr = PCGetMethodFromContext(pc,&pcmethod); CHKERRA(ierr);
  if (pcmethod == PCBJACOBI) {
    /* Note that SLESSetUp() MUST be called before PCBJacobiGetSubSLES(). */
    ierr = SLESSetUp(sles,x,b); CHKERRA(ierr);
    ierr = PCBJacobiGetSubSLES(pc,&nlocal,&first,&subsles); CHKERRA(ierr);
    ierr = SLESGetPC(subsles[0],&subpc); CHKERRA(ierr);
    ierr = SLESGetKSP(subsles[0],&subksp); CHKERRA(ierr);
    if (mytid == 0) {
      ierr = PCSetMethod(subpc,PCILU); CHKERRA(ierr);
      ierr = KSPSetTolerances(subksp,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,
             PETSC_DEFAULT); CHKERRA(ierr);
    } else {
      ierr = PCSetMethod(subpc,PCJACOBI); CHKERRA(ierr);
      ierr = KSPSetTolerances(subksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,
             PETSC_DEFAULT); CHKERRA(ierr);
    }
  }
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
  if (!OptionsHasName(0,"-noslesview")) {
    ierr = SLESView(sles,STDOUT_VIEWER_WORLD); CHKERRA(ierr);
  }

  /* Check the error */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    MPIU_printf(MPI_COMM_WORLD,"Norm of error %g iterations %d\n",norm,its);
  else 
    MPIU_printf(MPI_COMM_WORLD,"Norm of error < 1.e-12 Iterations %d\n",its);

  /* Destroy work space */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
