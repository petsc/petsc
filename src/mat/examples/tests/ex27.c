#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex27.c,v 1.7 1999/03/19 21:19:59 bsmith Exp bsmith $";
#endif

static char help[] = "Tests repeated use of assembly for matrices.\n\
 does nasty case where matrix must be rebuilt.\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C; 
  int         i,j, m = 5, n = 2, I, J, ierr, rank, size;
  Scalar      v;
  Vec         x,y;

  PetscInitialize(&argc,&args,(char *)0,help);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  n = 2*size;

  /* Create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&C); CHKERRA(ierr);
  for ( i=0; i<m; i++ ) { 
    for ( j=2*rank; j<2*rank+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  for ( i=0; i<m; i++ ) {
    for ( j=2*rank; j<2*rank+2; j++ ) {
      v = 1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      v = -4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES); CHKERRA(ierr);
    }
  }
  /* Introduce new nonzero that requires new construction for 
      matrix-vector product */
  if (rank) {
    I = rank-1; J = m*n-1;
    ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = MatView(C,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  /* Form a couple of vectors to test matrix-vector product */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m*n,&x); CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&y); CHKERRA(ierr);
  v = 1.0; ierr = VecSet(&v,x); CHKERRA(ierr);
  ierr = MatMult(C,x,y); CHKERRA(ierr);

  ierr = MatDestroy(C); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
