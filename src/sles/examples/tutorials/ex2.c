static char help[] = 
"This example solves a linear system in parallel with SLES.  To test the\n\
parallel matrix assembly, the matrix is intentionally distributed across the\n\
processors differently from the way it is assembled.\n\n";

#include "sles.h"
#include "petsc.h"
#include <stdio.h>

int main(int argc,char **args)
{
  int       i, j, I, J, ierr, m = 3, n = 2, mytid, numtids, its;
  Scalar    v, zero = 0.0, one = 1.0, none = -1.0;
  Vec       x, u, b;                       Mat       A; 
  SLES      sles;                          double    norm;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);
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
  ierr = VecSet(&one,u); CHKERRA(ierr);     
  ierr = VecSet(&zero,x); CHKERRA(ierr);
  ierr = MatMult(A,u,b); CHKERRA(ierr);

  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,0); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);   /* check the error */
  ierr = VecNorm(x,&norm); CHKERRA(ierr);
  MPE_printf(MPI_COMM_WORLD,"Norm of error %g Number of iterations %d\n",norm,its);

  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
