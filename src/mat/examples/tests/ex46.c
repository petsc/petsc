#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex46.c,v 1.6 1999/05/04 20:33:03 balay Exp bsmith $";
#endif

static char help[] = "Tests generating a nonsymmetric BlockSolve95 (MATMPIROWBS) matrix.\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat     C,A;
  Scalar  v;
  int     i, j, I, J, ierr, Istart, Iend, N, m = 4, n = 4, rank, size,flg;

  PetscInitialize(&argc,&args,0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);
  N = m*n;

  /* Generate matrix */
  ierr = MatCreateMPIRowbs(PETSC_COMM_WORLD,PETSC_DECIDE,N,0,0,0,&C);CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i >  0 )  {J = I - n; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
    if ( j >  0 )  {J = I - 1; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
    if ( I != 8) {v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);}
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = MatConvert(C,MATMPIAIJ,&A);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}


