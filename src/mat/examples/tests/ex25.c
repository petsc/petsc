#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex25.c,v 1.6 1999/04/19 22:13:14 bsmith Exp balay $";
#endif

static char help[] = "Tests MatTranspose\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat     C, A; 
  Scalar  v;
  int     i, j, m = 4, n = 4, rank, size, I, J, ierr, Istart, Iend;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  
  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
         m*n,m*n,5,PETSC_NULL,5,PETSC_NULL,&C);CHKERRA(ierr);

  /* create the matrix for the five point stencil, YET AGAIN*/
  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = MatTranspose(C,&A);CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = MatDestroy(C);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
