#ifndef lint
static char vcid[] = "$Id: ex11.c,v 1.4 1995/07/23 18:25:13 curfman Exp bsmith $";
#endif

static char help[] = 
"This example tests the preconditioner PCSPAI\n\n";

#include "mat.h"
#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Mat     C, A; 
  int     i,j, m = 15, n = 17, mytid, numtids, low, high, iglobal;
  Scalar  v,  one = 1.0;
  int     its, I, J, ierr, nz, nzalloc, mem, ldim,Istart,Iend;
  Vec     u,b,x;
  SLES    sles;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stdout,help);
  OptionsGetInt(0,"-m",&m);
  OptionsGetInt(0,"-n",&n);

  /* create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&C); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend); CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
    if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
    if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
    if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
    v = 4.0; MatSetValues(C,1,&I,1,&I,&v,INSERTVALUES);
  }
  ierr = MatAssemblyBegin(C,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = VecCreate(MPI_COMM_WORLD,m*n,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&u); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = MatMult(C,u,b); CHKERRA(ierr);

  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,C,C,ALLMAT_DIFFERENT_NONZERO_PATTERN); 
  CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
