#ifndef lint
static char vcid[] = "$Id: ex6.c,v 1.11 1995/10/24 21:50:03 bsmith Exp curfman $";
#endif

static char help[] = 
"Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil,\n\
                    use the file petsc/src/mat/examples/matbinary.ex\n\n";

#include "draw.h"
#include "mat.h"
#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  int        ierr, its, set;
  double     time, norm;
  Scalar     zero = 0.0, none = -1.0;
  Vec        x, b, u;
  Mat        A;
  MatType    mtype;
  SLES       sles;
  char       file[128]; 
  Viewer     fd;

  PetscInitialize(&argc,&args,0,0,help);

  /* Read matrix and RHS */
  OptionsGetString(0,"-f",file,127);
  ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,file,BINARY_RDONLY,&fd); CHKERRA(ierr);
  mtype = MATSEQAIJ; /* default matrix type */
  ierr = MatGetFormatFromOptions(MPI_COMM_WORLD,&mtype,&set); CHKERRQ(ierr);
  ierr = MatLoad(fd,mtype,&A); CHKERRA(ierr);
  ierr = VecLoad(fd,&b); CHKERRA(ierr);
  ierr = ViewerDestroy(fd); CHKERRA(ierr);

  /* Set up solution */
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecDuplicate(b,&u); CHKERRA(ierr);
  ierr = VecSet(&zero,x); CHKERRA(ierr);

  /* Solve system */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,ALLMAT_DIFFERENT_NONZERO_PATTERN); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  time = MPI_Wtime();
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
  time = MPI_Wtime()-time;

  /* Show result */
  ierr = MatMult(A,x,u);
  ierr = VecAXPY(&none,b,u); CHKERRA(ierr);
  ierr = VecNorm(u,&norm); CHKERRA(ierr);
  MPIU_printf(MPI_COMM_WORLD,"Number of iterations = %3d\n",its);
  MPIU_printf(MPI_COMM_WORLD,"Residual norm = %10.4e\n",norm);
  MPIU_printf(MPI_COMM_WORLD,"Time for solve = %5.2f\n",time);

  /* Cleanup */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

