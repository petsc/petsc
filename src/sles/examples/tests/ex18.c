
#ifndef lint
static char vcid[] = "$Id: ex6.c,v 1.1 1996/02/11 23:25:35 bsmith Exp bsmith $";
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
  int        ierr, its, set,flg,m,n,mvec;
  double     time1, norm;
  Scalar     zero = 0.0, none = -1.0;
  Vec        x, b, u;
  Mat        A;
  MatType    mtype;
  SLES       sles;
  char       file[128]; 
  Viewer     fd;

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

/* 
   If the load matrix is larger then the vector, due to being padded 
   to match the blocksize then create a new padded vector
*/

  ierr = MatGetSize(A,&m,&n); CHKERRA(ierr);
  ierr = VecGetSize(b,&mvec); CHKERRA(ierr);
  if (m > mvec) {
    Vec    tmp;
    Scalar *bold,*bnew;
    /* create a new vector b by padding the old one */
    ierr = VecCreate(MPI_COMM_WORLD,m,&tmp); CHKERRA(ierr);
    ierr = VecGetArray(tmp,&bnew); CHKERRA(ierr);
    ierr = VecGetArray(b,&bold); CHKERRA(ierr);
    PetscMemcpy(bnew,bold,mvec*sizeof(Scalar)); CHKERRA(ierr);
    VecDestroy(b);
    b = tmp;
  }

/*
ierr = MatView(A,STDOUT_VIEWER_WORLD);
ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,"newfile",BINARY_CREATE,&fd);
CHKERRA(ierr);
ierr = MatView(A,fd); CHKERRA(ierr);
ierr = VecView(b,fd); CHKERRA(ierr);
ierr = ViewerDestroy(fd); CHKERRA(ierr);
*/


  /* Set up solution */
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecDuplicate(b,&u); CHKERRA(ierr);
  ierr = VecSet(&zero,x); CHKERRA(ierr);

  /* Solve system */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  time1 = PetscGetTime();
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
  time1 = PetscGetTime() - time1;

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

