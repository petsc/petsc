#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex59.c,v 1.6 1999/03/19 21:19:59 bsmith Exp balay $";
#endif

static char help[] = "Tests MatGetSubmatrix() in parallel";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C, A;
  int         i,j, m = 3, n = 2, rank,size, ierr, rstart, rend;
  Scalar      v;
  IS          isrow,iscol;

  PetscInitialize(&argc,&args,(char *)0,help);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  n = 2*size;

  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
         m*n,m*n,5,PETSC_NULL,5,PETSC_NULL,&C);CHKERRA(ierr);

  /*
        This is JUST to generate a nice test matrix, all processors fill up
    the entire matrix. This is not something one would ever do in practice.
  */
  for ( i=0; i<m*n; i++ ) { 
    for ( j=0; j<m*n; j++ ) {
      v = i + j; 
      ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* 
     Generate a new matrix consisting of every second row and column of
   the original matrix
  */
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRA(ierr);
  /* list the rows we want on THIS processor */
  ierr = ISCreateStride(PETSC_COMM_WORLD,(rend-rstart)/2,rstart,2,&isrow);CHKERRA(ierr);
  /* list ALL the columns we want */
  ierr = ISCreateStride(PETSC_COMM_WORLD,(m*n)/2,0,2,&iscol);CHKERRA(ierr);
  ierr = MatGetSubMatrix(C,isrow,iscol,PETSC_DECIDE,MAT_INITIAL_MATRIX,&A);CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_WORLD);CHKERRA(ierr); 

  ierr = MatGetSubMatrix(C,isrow,iscol,PETSC_DECIDE,MAT_REUSE_MATRIX,&A);CHKERRA(ierr); 
  ierr = MatView(A,VIEWER_STDOUT_WORLD);CHKERRA(ierr); 

  ierr = ISDestroy(isrow);CHKERRA(ierr);
  ierr = ISDestroy(iscol);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
