
#ifndef lint
static char vcid[] = "$Id: ex6.c,v 1.29 1996/03/01 20:44:42 balay Exp balay $";
#endif

static char help[] = 
"Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil,\n\
                    use the file petsc/src/mat/examples/matbinary.ex\n\n";

#include "draw.h"
#include "mat.h"
#include "sles.h"
#include "plog.h"
#include <stdio.h>

int main(int argc,char **args)
{
  int        ierr, its, set,flg;
  double     norm;
  Scalar     zero = 0.0, none = -1.0;
  Vec        x, b, u;
  Mat        A;
  MatType    mtype;
  SLES       sles;
  char       file[128]; 
  Viewer     fd;
  int        e1, e2, e3;
  /*   extern     int xyz(int *); */
  PetscInitialize(&argc,&args,0,0,help);

#if defined(PETSC_COMPLEX)
  SETERRA(1,"This example does not work with complex numbers");
#else

  /* Read matrix and RHS */
  ierr = OptionsGetString(PETSC_NULL,"-f",file,127,&flg); CHKERRA(ierr);
  ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,file,BINARY_RDONLY,&fd); CHKERRA(ierr);
  ierr = MatGetFormatFromOptions(MPI_COMM_WORLD,0,&mtype,&set); CHKERRQ(ierr);
  ierr = MatLoad(fd,mtype,&A); CHKERRA(ierr);
  ierr = VecLoad(fd,&b); CHKERRA(ierr);
  ierr = ViewerDestroy(fd); CHKERRA(ierr);

  /* Set up solution */
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecDuplicate(b,&u); CHKERRA(ierr);
  ierr = VecSet(&zero,x); CHKERRA(ierr);

  /* Solve system */
  PLogEventRegister(&e1,"SLES_Create    ", "black");
  PLogEventRegister(&e2,"Sles SetOper   ", "green");
  PLogEventRegister(&e3,"SLES_Options   ", "orange");
  PetscBarrier(A);
  
  PLogStagePush(1);
  PLogEventBegin(e1,sles,0,0,0);
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  PLogEventEnd(e1,sles,0,0,0);
  PLogEventBegin(e2,sles,0,0,0);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  PLogEventEnd(e2,sles,0,0,0);
  PLogEventBegin(e3,sles,0,0,0);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  PLogEventEnd(e3,sles,0,0,0);
  ierr = SLESSetUp(sles,b,x); CHKERRA(ierr);
  ierr = SLESSetUpOnBlocks(sles); CHKERRA(ierr);
  /*  ierr = xyz(&flg); CHKERRQ(ierr); */
  PLogStagePop();
  PetscBarrier(A);
  PLogStagePush(2);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
  PLogStagePop();

  /* Show result */
  ierr = MatMult(A,x,u);
  ierr = VecAXPY(&none,b,u); CHKERRA(ierr);
  ierr = VecNorm(u,NORM_2,&norm); CHKERRA(ierr);
  MPIU_printf(MPI_COMM_WORLD,"Number of iterations = %3d\n",its);
  if (norm < 1.e-10) {
    MPIU_printf(MPI_COMM_WORLD,"Residual norm < 1.e-10\n");
  } else {
    MPIU_printf(MPI_COMM_WORLD,"Residual norm = %10.4e\n",norm);
  }
  /* MPIU_printf(MPI_COMM_WORLD,"Time for solve = %5.2f seconds\n",time1); */

  /* Cleanup */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
  return 0;
#endif
}

