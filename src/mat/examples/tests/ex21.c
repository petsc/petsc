#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex21.c,v 1.6 1999/03/19 21:19:59 bsmith Exp bsmith $";
#endif

static char help[] = "Tests converting a parallel AIJ formatted matrix to the\n\
parallel Row format. This also tests MatGetRow() and MatRestoreRow()\n\
for the parallel case.";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C, A;
  int         i,j, m = 3, n = 2, rank,size,I, J, ierr, rstart, rend, nz, *idx;
  Scalar      v, *values;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
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
  ierr = ViewerPushFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_INFO,0);CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  ierr = ViewerPopFormat(VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = MatGetOwnershipRange(C,&rstart,&rend); CHKERRA(ierr);
  PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);
  for ( i=rstart; i<rend; i++ ) {
    ierr = MatGetRow(C,i,&nz,&idx,&values); CHKERRA(ierr);
    fprintf(stdout,"[%d] get row %d: ", rank, i);
    for ( j=0; j<nz; j++ ) {
#if defined(USE_PETSC_COMPLEX)
      fprintf(stdout,"%d %g  ",idx[j],PetscReal(values[j]));
#else
      fprintf(stdout,"%d %g  ",idx[j],values[j]);
#endif
    }
    fprintf(stdout,"\n"); fflush(stdout);
    ierr = MatRestoreRow(C,i,&nz,&idx,&values); CHKERRA(ierr);
  }
  fflush(stdout);
  PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);

  ierr = MatConvert(C,MATMPIAIJ,&A); CHKERRA(ierr);
  ierr = ViewerPushFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_INFO,0);CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_WORLD); CHKERRA(ierr); 
  ierr = ViewerPopFormat(VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_WORLD); CHKERRA(ierr); 

  ierr = MatDestroy(A); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
