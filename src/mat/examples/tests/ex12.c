#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex12.c,v 1.5 1997/09/22 15:24:34 balay Exp bsmith $";
#endif

static char help[] = "Tests the use of MatZeroRows() for parallel matrices.\n\n";

#include "mat.h"

int main(int argc,char **args)
{
  Mat         C;
  int         i,j, m = 3, n = 2, rank,size, I, J, ierr,Imax;
  Scalar      v;
  IS          is;

  PetscInitialize(&argc,&args,(char *)0,help);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  n = 2*size;

  /* create the matrix for the five point stencil, YET AGAIN*/
  ierr = MatCreate(PETSC_COMM_WORLD,m*n,m*n,&C); CHKERRA(ierr);
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

  Imax = n*rank; if (Imax>= n*m -m - 1) Imax = m*n - m - 1;
  ierr = ISCreateStride(PETSC_COMM_SELF,m,Imax,1,&is); CHKERRA(ierr);
  ierr = MatZeroRows(C,is,0); CHKERRA(ierr); 

  ierr = MatView(C,VIEWER_STDOUT_WORLD); CHKERRA(ierr); 
  
  ierr = ISDestroy(is); CHKERRA(ierr); 
  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
