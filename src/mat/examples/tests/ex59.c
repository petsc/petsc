#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex59.c,v 1.1 1997/10/21 04:00:45 bsmith Exp bsmith $";
#endif

static char help[] = "Tests MatGetSubmatrix() in parallel";

#include "mat.h"

int main(int argc,char **args)
{
  Mat         C, A;
  int         i,j, m = 3, n = 2, rank,size,I, J, ierr, rstart, rend;
  Scalar      v;
  IS          isrow,iscol;

  PetscInitialize(&argc,&args,(char *)0,help);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  n = 2*size;

  /* create the matrix for the five point stencil, YET AGAIN*/
  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
         m*n,m*n,5,PETSC_NULL,5,PETSC_NULL,&C); CHKERRA(ierr);
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
  ierr = MatView(C,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  /* 
     Generate a new matrix consisting of every second row and column of
   the original matrix
  */
  ierr = MatGetOwnershipRange(C,&rstart,&rend); CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,(rend-rstart)/2,rstart,2,&isrow);CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,(m*n)/2,0,2,&isrow);CHKERRA(ierr);
  ierr = MatGetSubMatrix(C,isrow,iscol,&A);CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_WORLD); CHKERRA(ierr); 

  ierr = MatDestroy(A); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
