#ifndef lint
static char vcid[] = "$Id: ex16.c,v 1.1 1997/01/16 21:31:21 bsmith Exp bsmith $";
#endif

static char help[] = "Tests MatGetArray().\n\n";

#include "mat.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Mat         A; 
  int         i, j, m = 3, n = 2, ierr,rstart,rend;
  Scalar      v,*array;

  PetscInitialize(&argc,&args,(char *)0,help);

  /*
      Create a parallel dense matrix shared by all processors 
  */
  ierr = MatCreateMPIDense(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,PETSC_NULL,&A);
         CHKERRA(ierr);

  /*
     Set values into the matrix 
  */
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      v = 1.0/(i+j+1); MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /*
       Print the matrix to the screen 
  */
  ierr = MatView(A,VIEWER_STDOUT_WORLD); CHKERRA(ierr);


  /*
      Print the local portion of the matrix to the screen
  */
  ierr = MatGetArray(A,&array); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend); CHKERRA(ierr);
  for ( i=rstart; i<rend; i++ ) {
    for ( j=0; j<n; j++ ) {
      PetscSynchronizedPrintf(MPI_COMM_WORLD,"%6.4e ",PetscReal(array[j*(rend-rstart)+i-rstart]));
    }
    PetscSynchronizedPrintf(MPI_COMM_WORLD,"\n");
  }
  PetscSynchronizedFlush(MPI_COMM_WORLD);
   
  /*
      Free the space used by the matrix
  */
  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
