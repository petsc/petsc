
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex18.c,v 1.10 1999/03/30 16:22:46 balay Exp balay $";
#endif

#if !defined(USE_PETSC_COMPLEX)

static char help[] = 
"Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil,\n\
                    use the file petsc/src/mat/examples/matbinary.ex\n\n";

#include "mat.h"
#include "sles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int        ierr, its, flg,m,n,mvec;
  PetscTruth set;
  PLogDouble time1,time2,time;
  double     norm;
  Scalar     zero = 0.0, none = -1.0;
  Vec        x, b, u;
  Mat        A;
  MatType    mtype;
  SLES       sles;
  char       file[128]; 
  Viewer     fd;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read matrix and RHS */
  ierr = OptionsGetString(PETSC_NULL,"-f",file,127,&flg);CHKERRA(ierr);
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatGetTypeFromOptions(PETSC_COMM_WORLD,0,&mtype,&set);CHKERRQ(ierr);
  ierr = MatLoad(fd,mtype,&A);CHKERRA(ierr);
  ierr = VecLoad(fd,&b);CHKERRA(ierr);
  ierr = ViewerDestroy(fd);CHKERRA(ierr);

  /* 
     If the load matrix is larger then the vector, due to being padded 
     to match the blocksize then create a new padded vector
  */
  ierr = MatGetSize(A,&m,&n);CHKERRA(ierr);
  ierr = VecGetSize(b,&mvec);CHKERRA(ierr);
  if (m > mvec) {
    Vec    tmp;
    Scalar *bold,*bnew;
    /* create a new vector b by padding the old one */
    ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m,&tmp);CHKERRA(ierr);
    ierr = VecSetFromOptions(tmp);CHKERRA(ierr);
    ierr = VecGetArray(tmp,&bnew);CHKERRA(ierr);
    ierr = VecGetArray(b,&bold);CHKERRA(ierr);
    ierr = PetscMemcpy(bnew,bold,mvec*sizeof(Scalar));CHKERRA(ierr);
    ierr = VecDestroy(b);CHKERRA(ierr);
    b = tmp;
  }

  /* Set up solution */
  ierr = VecDuplicate(b,&x);CHKERRA(ierr);
  ierr = VecDuplicate(b,&u);CHKERRA(ierr);
  ierr = VecSet(&zero,x);CHKERRA(ierr);

  /* Solve system */
  PLogStagePush(1);
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);
  ierr = PetscGetTime(&time1);CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);
  ierr = PetscGetTime(&time2);CHKERRA(ierr);
  time = time2 - time1;
  PLogStagePop();

  /* Show result */
  ierr = MatMult(A,x,u);
  ierr = VecAXPY(&none,b,u);CHKERRA(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3d\n",its);
  if (norm < 1.e-10) {
    PetscPrintf(PETSC_COMM_WORLD,"Residual norm < 1.e-10\n");
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"Residual norm = %10.4e\n",norm);
  }
  PetscPrintf(PETSC_COMM_WORLD,"Time for solve = %5.2f seconds\n",time); 

  /* Cleanup */
  ierr = SLESDestroy(sles);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

#else
#include <stdio.h>
int main(int argc,char **args)
{
  fprintf(stdout,"This example does not work for complex numbers.\n");
  return 0;
}
#endif
