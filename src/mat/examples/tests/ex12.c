#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex12.c,v 1.7 1999/02/11 22:07:31 balay Exp balay $";
#endif

static char help[] = "Tests the use of MatZeroRows() for parallel matrices.\n\n";

#include "mat.h"

extern int TestMatZeroRows_Basic(Mat,IS, Scalar *);
extern int TestMatZeroRows_with_no_allocation(Mat,IS, Scalar *);

int main(int argc,char **args)
{
  Mat         A;
  int         i,j, m = 3, n, rank,size, I, J, ierr,Imax;
  Scalar      v,*diag;
  IS          is;

  PetscInitialize(&argc,&args,(char *)0,help);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  n = 2*size;

  /* create A Square matrix for the five point stencil, YET AGAIN*/
  ierr = MatCreate(PETSC_COMM_WORLD,m*n,m*n,&A); CHKERRA(ierr);
  for ( i=0; i<m; i++ ) { 
    for ( j=2*rank; j<2*rank+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Create AN IS required by MatZeroRows() */
  Imax = n*rank; if (Imax>= n*m -m - 1) Imax = m*n - m - 1;
  ierr = ISCreateStride(PETSC_COMM_SELF,m,Imax,1,&is); CHKERRA(ierr);

  /* Now Create the DIAG required by MatZerows() */
  diag = (Scalar *) PetscMalloc((n+1)*sizeof(Scalar)); CHKPTRA(diag);
  for ( i=0; i<n; i++ ) diag[i] = -4.0;

  ierr = TestMatZeroRows_Basic(A,is,PETSC_NULL); CHKERRA(ierr);
  ierr = TestMatZeroRows_Basic(A,is,diag); CHKERRA(ierr);

  ierr = TestMatZeroRows_with_no_allocation(A,is,PETSC_NULL); CHKERRA(ierr);
  ierr = TestMatZeroRows_with_no_allocation(A,is,diag); CHKERRA(ierr);

  ierr = MatDestroy(A); CHKERRA(ierr);

  /* Now Create a rectangular matrix with five point stencil (app) */
  ierr = MatCreate(PETSC_COMM_WORLD,m*n,m*(n+1),&A); CHKERRA(ierr);
  for ( i=0; i<m; i++ ) { 
    for ( j=2*rank; j<2*rank+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j<n ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = TestMatZeroRows_Basic(A,is,PETSC_NULL); CHKERRA(ierr);
  ierr = TestMatZeroRows_Basic(A,is,diag); CHKERRA(ierr);

  ierr = MatDestroy(A); CHKERRA(ierr);
  ierr = ISDestroy(is); CHKERRQ(ierr); 
  PetscFree(diag);
  PetscFinalize();
  return 0;
}

int TestMatZeroRows_Basic(Mat A,IS is, Scalar *diag)
{
  Mat         B;
  int         ierr;

  /* Now copy A into B, and test it with MatZeroRows() */
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B); CHKERRQ(ierr);

  ierr = MatZeroRows(B,is,diag); CHKERRQ(ierr);
  ierr = MatView(B,VIEWER_STDOUT_WORLD); CHKERRQ(ierr); 
  ierr = MatDestroy(B); CHKERRQ(ierr);
  return 0;
}

int TestMatZeroRows_with_no_allocation(Mat A,IS is, Scalar *diag)
{
  Mat         B;
  int         ierr;

  /* Now copy A into B, and test it with MatZeroRows() */
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B); CHKERRQ(ierr);
  /* Set this flag after assembly. This way, it affects only MatZeroRows() */
  ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR); CHKERRQ(ierr);

  ierr = MatZeroRows(B,is,diag); CHKERRQ(ierr);
  ierr = MatView(B,VIEWER_STDOUT_WORLD); CHKERRQ(ierr); 
  ierr = MatDestroy(B); CHKERRQ(ierr);
  return 0;
}
