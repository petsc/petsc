
static char help[] = 
"This example solves a linear system in parallel with SLES.  To test the\n\
parallel matrix assembly, the matrix is intentionally laid out across the\n\
processors differently from the way it is assembled.  Input arguments are:\n\
  -m <size> : problem size\n\n";

#include  <stdio.h>
#include "sles.h"
#include "petsc.h"

int main(int argc,char **args)
{
  Mat         C; 
  int         i,j, m = 3, n = 2, mytid,numtids,its;
  Scalar      v, zero = 0.0, one = 1.0, none = -1.0;
  int         I, J, ierr;
  Vec         x,u,b;
  SLES        sles;
  double      norm;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,"-m",&m);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  n = 2*numtids;

  /* create the matrix for the five point stencil, YET AGAIN*/
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

  ierr = VecCreate(MPI_COMM_WORLD,m*n,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = VecSet(&zero,x); CHKERRA(ierr);

  /* compute right hand side */
  ierr = MatMult(C,u,b); CHKERRA(ierr);

  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,C,C,0); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /* check error */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,&norm); CHKERRA(ierr);
  MPE_printf(MPI_COMM_WORLD,"Norm of error %g Number of iterations %d\n",norm,its);

  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
