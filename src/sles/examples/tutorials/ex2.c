
static char help[] = "Tests MPI parallel AIJ solve with SLES. \n\
  This examples intentionally lays the matrix out across processors\n\
  differently then the way it is assembled, this is to test parallel\n\
  matrix assembly.\n";

#include "vec.h"
#include "mat.h"
#include "options.h"
#include  <stdio.h>
#include "sles.h"

int main(int argc,char **args)
{
  Mat         C; 
  int         i,j, m = 3, n = 2, mytid,numtids,its;
  Scalar      v, zero = 0.0,norm, one = 1.0, none = -1.0;
  int         I, J, ierr;
  Vec         x,u,b;
  SLES        sles;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,0,"-m",&m);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  n = 2*numtids;

  /* create the matrix for the five point stencil, YET AGAIN*/
  ierr = MatCreateInitialMatrix(m*n,m*n,&C); 
  CHKERRA(ierr);

  for ( i=0; i<m; i++ ) { 
    for ( j=2*mytid; j<2*mytid+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      v = 4.0; MatSetValues(C,1,&I,1,&I,&v,InsertValues);
    }
  }
  ierr = MatBeginAssembly(C); CHKERRA(ierr);
  ierr = MatEndAssembly(C); CHKERRA(ierr);

  ierr = VecCreateInitialVector(m*n,&u); CHKERRA(ierr);
  ierr = VecCreate(u,&b); CHKERRA(ierr);
  ierr = VecCreate(b,&x); CHKERRA(ierr);
  VecSet(&one,u); VecSet(&zero,x);

  /* compute right hand side */
  ierr = MatMult(C,u,b); CHKERRA(ierr);

  if ((ierr = SLESCreate(&sles))) SETERRA(ierr,0);
  if ((ierr = SLESSetOperators(sles,C,C,0))) SETERRA(ierr,0);
  if ((ierr = SLESSetFromOptions(sles))) SETERRA(ierr,0);
  if ((ierr = SLESSolve(sles,b,x,&its))) SETERRA(ierr,0);

  /* check error */
  if ((ierr = VecAXPY(&none,u,x))) SETERRA(ierr,0);
  if ((ierr = VecNorm(x,&norm))) SETERRA(ierr,0);
  MPE_printf(MPI_COMM_WORLD,"Norm of error %g Number of iterations %d\n",norm,its);

  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
