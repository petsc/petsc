#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex46.c,v 1.5 1999/03/19 21:19:59 bsmith Exp balay $";
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
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);
  N = m*n;

  /* Generate matrix */
  ierr = MatCreateMPIRowbs(PETSC_COMM_WORLD,PETSC_DECIDE,N,0,0,0,&C);CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i >  0 )  {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j >  0 )  {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( I != 8) {v = 4.0; MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);}
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = MatConvert(C,MATMPIAIJ,&A);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}


