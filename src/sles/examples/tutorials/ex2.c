#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.36 1996/01/12 22:08:38 bsmith Exp curfman $";
#endif

static char help[] = "Solves a linear system in parallel with SLES.  To test the\n\
parallel matrix assembly, the matrix is intentionally distributed across the\n\
processors differently from the way it is assembled.\n\n";

#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  int       i, j, I, J, ierr, m = 3, n = 2, rank, size, its,flg;
  Scalar    v, zero = 0.0, one = 1.0, none = -1.0;
  Vec       x, u, b;                       Mat       A; 
  SLES      sles;                          double    norm;
  PetscInitialize(&argc,&args,0,0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);  n = 2*size;

  /* Create and assemble matrix */
  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&A); CHKERRA(ierr);
  for ( i=0; i<m; i++ ) {   /* assemble matrix for the five point stencil */
    for ( j=2*rank; j<2*rank+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; ierr=MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if ( i<m-1 ) {J = I + n; ierr=MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if ( j>0 )   {J = I - 1; ierr=MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if ( j<n-1 ) {J = I + 1; ierr=MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Create and set vectors */
  ierr = VecCreate(MPI_COMM_WORLD,m*n,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr); 
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = VecSet(&zero,x); CHKERRA(ierr);
  ierr = MatMult(A,u,b); CHKERRA(ierr);

  /* Create SLES context; set operators and options; solve linear system */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A, ALLMAT_DIFFERENT_NONZERO_PATTERN);
  CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /* Check error */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    MPIU_printf(MPI_COMM_WORLD,"Norm of error %g iterations %d\n",norm,its);
  else 
    MPIU_printf(MPI_COMM_WORLD,"Norm of error < 1.e-12 Iterations %d\n",its);

  /* Free work space */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
