#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex12.c,v 1.16 1999/05/04 20:33:03 balay Exp bsmith $";
#endif

static char help[] = "Tests the use of MatZeroRows() for parallel matrices.\n\
This example also tests the use of MatDuplicate() for both MPIAIJ and MPIBAIJ matrices";

#include "mat.h"

extern int TestMatZeroRows_Basic(Mat,IS, Scalar *);
extern int TestMatZeroRows_with_no_allocation(Mat,IS, Scalar *);

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         A;
  int         i,j, m = 3, n, rank,size, I, J, ierr,Imax;
  Scalar      v,diag=-4.0;
  IS          is;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  n = 2*size;

  /* create A Square matrix for the five point stencil, YET AGAIN*/
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&A);CHKERRA(ierr);
  for ( i=0; i<m; i++ ) { 
    for ( j=2*rank; j<2*rank+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if ( i<m-1 ) {J = I + n; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if ( j>0 )   {J = I - 1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if ( j<n-1 ) {J = I + 1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* Create AN IS required by MatZeroRows() */
  Imax = n*rank; if (Imax>= n*m -m - 1) Imax = m*n - m - 1;
  ierr = ISCreateStride(PETSC_COMM_SELF,m,Imax,1,&is);CHKERRA(ierr);

  ierr = TestMatZeroRows_Basic(A,is,PETSC_NULL);CHKERRA(ierr);
  ierr = TestMatZeroRows_Basic(A,is,&diag);CHKERRA(ierr);

  ierr = TestMatZeroRows_with_no_allocation(A,is,PETSC_NULL);CHKERRA(ierr);
  ierr = TestMatZeroRows_with_no_allocation(A,is,&diag);CHKERRA(ierr);

  ierr = MatDestroy(A);CHKERRA(ierr);

  /* Now Create a rectangular matrix with five point stencil (app) 
   n+size is used so that this dimension is always divisible by size.
   This way, we can always use bs = size for any number of procs */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*(n+size),&A);CHKERRA(ierr);
  for ( i=0; i<m; i++ ) { 
    for ( j=2*rank; j<2*rank+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if ( i<m-1 ) {J = I + n; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if ( j>0 )   {J = I - 1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if ( j<n+size-1 ) {J = I + 1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = TestMatZeroRows_Basic(A,is,PETSC_NULL);CHKERRA(ierr);
  ierr = TestMatZeroRows_Basic(A,is,&diag);CHKERRA(ierr);

  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr); 
  PetscFinalize();
  return 0;
}

#undef __FUNC__
#define __FUNC__ "TestMatZeroRows_Basic"
int TestMatZeroRows_Basic(Mat A,IS is, Scalar *diag)
{
  Mat         B;
  int         ierr;

  /* Now copy A into B, and test it with MatZeroRows() */
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);

  ierr = MatZeroRows(B,is,diag);CHKERRQ(ierr);
  ierr = MatView(B,VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  ierr = MatDestroy(B);CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__
#define __FUNC__ "TestMatZeroRows_with_no_allocation"
int TestMatZeroRows_with_no_allocation(Mat A,IS is, Scalar *diag)
{
  Mat         B;
  int         ierr;

  /* Now copy A into B, and test it with MatZeroRows() */
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  /* Set this flag after assembly. This way, it affects only MatZeroRows() */
  ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR);CHKERRQ(ierr);

  ierr = MatZeroRows(B,is,diag);CHKERRQ(ierr);
  ierr = MatView(B,VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  ierr = MatDestroy(B);CHKERRQ(ierr);
  return 0;
}
