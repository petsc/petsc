#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex43.c,v 1.5 1998/12/03 04:01:49 bsmith Exp bsmith $";
#endif

static char help[] = "Saves a dense matrix in a dense format (binary).\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat     C;
  Scalar  v;
  int     i, j, ierr, m = 4, n = 4, rank, size,flg;
  Viewer  viewer;

  PetscInitialize(&argc,&args,(char *)0,help);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);

  /* PART 1:  Generate matrix, then write it in binary format */

  /* Generate matrix */
  ierr = MatCreateSeqDense(PETSC_COMM_WORLD,m,n,PETSC_NULL,&C); CHKERRA(ierr);
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      v = i*m+j;
      ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",BINARY_CREATE,&viewer);CHKERRA(ierr);
  ierr = ViewerSetFormat(viewer,VIEWER_FORMAT_BINARY_NATIVE,"Dummy"); CHKERRA(ierr);
  ierr = MatView(C,viewer); CHKERRA(ierr);
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}


