
#ifndef lint
static char vcid[] = "$Id: ex6.c,v 1.8 1995/10/12 20:05:46 curfman Exp curfman $";
#endif

static char help[] = 
"Reads a PETSc matrix in from a file and solves linear system with it.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  Use mat.ex.binary for 5X5 5-pt. stencil\n\n";

#include "draw.h"
#include "mat.h"
#include "sles.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **args)
{
  int        ierr, its;
  double     time, norm;
  Scalar     zero = 0.0, none = -1.0;
  Vec        x, b, u;
  Mat        A;
  SLES       sles;
  char       file[128]; 
  Viewer     fd;

  PetscInitialize(&argc,&args,0,0,help);

/*   Read in matrix and RHS   */
  OptionsGetString(0,"-f",file,127);
  ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,file,BINARY_RDONLY,&fd); CHKERRA(ierr);
  ierr = MatLoad(fd,MATSEQAIJ,&A); CHKERRA(ierr);

  ierr = VecLoad(fd,&b); CHKERRA(ierr);
  ierr = ViewerDestroy(fd); CHKERRA(ierr);


/*   Set up solution   */
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecDuplicate(b,&u); CHKERRA(ierr);
  ierr = VecSet(&zero,x); CHKERRA(ierr);

/*   Solve system    */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A, ALLMAT_DIFFERENT_NONZERO_PATTERN); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  time = MPI_Wtime();
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
  time = MPI_Wtime()-time;


/*   Show result   */
  ierr = MatMult(A,x,u);
  ierr = VecAXPY(&none,b,u); CHKERRA(ierr);
  ierr = VecNorm(u,&norm); CHKERRA(ierr);
  MPIU_printf(MPI_COMM_WORLD,"Number of iterations = %3d\n",its);
  MPIU_printf(MPI_COMM_WORLD,"Residual norm = %10.4e\n",norm);
  MPIU_printf(MPI_COMM_WORLD,"Time for solve = %5.2f\n",time);

/*   cleanup   */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

