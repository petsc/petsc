#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.26 1995/07/23 18:25:33 curfman Exp $";
#endif

static char help[] = 
"This example illustrates use of the block Jacobi preconditioner for solving\n\
a linear system in parallel with SLES. To test the parallel matrix assembly,\n\
the matrix is intentionally distributed across the processors differently\n\
from the way it is assembled.\n\n";

#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  int       i, j, I, J, ierr, m = 3, n = 2, mytid, numtids, its, nlocal, first;
  Scalar    v, zero = 0.0, one = 1.0, none = -1.0;
  Vec       x, u, b;
  Mat       A; 
  SLES      sles, *subsles;
  PC        pc, subpc;
  KSP       subksp;
  double    norm;
  PCMethod  pcmethod;
  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);
  OptionsGetInt(0,"-m",&m);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);  n = 2*numtids;

  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&A); CHKERRA(ierr);
  for ( i=0; i<m; i++ ) {   /* assemble matrix for the five point stencil */
    for ( j=2*mytid; j<2*mytid+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(A,1,&I,1,&J,&v,INSERTVALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(A,1,&I,1,&J,&v,INSERTVALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,INSERTVALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,INSERTVALUES);}
      v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERTVALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = VecCreate(MPI_COMM_WORLD,m*n,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecSet(&one,u); CHKERRA(ierr);ierr = VecSet(&zero,x); CHKERRA(ierr);
  ierr = MatMult(A,u,b); CHKERRA(ierr);

  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,ALLMAT_DIFFERENT_NONZERO_PATTERN);
  CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);
  ierr = PCSetMethod(pc,PCBJACOBI);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

  /* Set local solvers for Block Jacobi method.  Currently only 1 block 
     per processor is supported */
  ierr = PCGetMethodFromContext(pc,&pcmethod); CHKERRA(ierr);
  if (pcmethod == PCBJACOBI) {
    ierr = PCBJacobiGetSubSLES(pc,&nlocal,&first,&subsles);
    ierr = SLESGetPC(subsles[0],&subpc); CHKERRA(ierr);
    ierr = SLESGetKSP(subsles[0],&subksp); CHKERRA(ierr);
    if (mytid == 0) {
      ierr = PCSetMethod(subpc,PCILU); CHKERRA(ierr);
      ierr = KSPSetMethod(subksp,KSPCG); CHKERRA(ierr);
    } else {
      ierr = PCSetMethod(subpc,PCJACOBI); CHKERRA(ierr);
      ierr = KSPSetMethod(subksp,KSPBCGS); CHKERRA(ierr);
    }
  }
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
  ierr = SLESView(sles,SYNC_STDOUT_VIEWER); CHKERRA(ierr);

  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);   /* check the error */
  ierr = VecNorm(x,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    MPIU_printf(MPI_COMM_WORLD,"Norm of error %g iterations %d\n",norm,its);
  else 
    MPIU_printf(MPI_COMM_WORLD,"Norm of error < 1.e-12 Iterations %d\n",its);

  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
