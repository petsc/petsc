
#ifndef lint
static char vcid[] = "$Id: ex6.c,v 1.36 1996/03/26 15:07:22 bsmith Exp bsmith $";
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
  int        ierr, its, set, flg;
  double     norm,tsetup,tsolve;
  Scalar     zero = 0.0, none = -1.0;
  Vec        x, b, u;
  Mat        A;
  MatType    mtype;
  SLES       sles;
  char       file[128];
  Viewer     fd;
  PetscTruth table = PETSC_FALSE;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = OptionsHasName(PETSC_NULL,"-table",&flg);
  if (flg) table = PETSC_TRUE;

#if defined(PETSC_COMPLEX)
  SETERRA(1,"This example does not work with complex numbers");
#else

  /* Read matrix and RHS */
  ierr = OptionsGetString(PETSC_NULL,"-f",file,127,&flg); CHKERRA(ierr);
  if (!flg) SETERRA(1,"Must indicate binary file with the -f option");
  ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,file,BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatGetTypeFromOptions(MPI_COMM_WORLD,0,&mtype,&set); CHKERRQ(ierr);
  ierr = MatLoad(fd,mtype,&A); CHKERRA(ierr);
  ierr = VecLoad(fd,&b); CHKERRA(ierr);
  ierr = ViewerDestroy(fd); CHKERRA(ierr);

  /* 
     If the load matrix is larger then the vector, due to being padded 
     to match the blocksize then create a new padded vector
  */
  { 
    int m,n,mvec;
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
  }
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecDuplicate(b,&u); CHKERRA(ierr);

  /* Do solve once to bring in all instruction pages */
  /*
  ierr = VecSet(&zero,x); CHKERRA(ierr);
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  ierr = SLESSetUp(sles,b,x); CHKERRA(ierr);
  ierr = SLESSetUpOnBlocks(sles); CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
  */

  ierr = VecSet(&zero,x); CHKERRA(ierr);
  PetscBarrier(A);

  PLogStagePush(1);
  tsetup = PetscGetTime();  
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  ierr = SLESSetUp(sles,b,x); CHKERRA(ierr);
  ierr = SLESSetUpOnBlocks(sles); CHKERRA(ierr);
  tsetup = PetscGetTime() - tsetup;
  PLogStagePop();
  PetscBarrier(A);

  PLogStagePush(2);
  tsolve = PetscGetTime();
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
  tsolve = PetscGetTime() - tsolve;
  PLogStagePop();

  /* Show result */
  ierr = MatMult(A,x,u);
  ierr = VecAXPY(&none,b,u); CHKERRA(ierr);
  ierr = VecNorm(u,NORM_2,&norm); CHKERRA(ierr);
  /*  matrix PC   KSP   Options       its    residual setuptime solvetime  */
  if (table) {
    char   *matrixname, slesinfo[120];
    Viewer viewer;
    ViewerStringOpen(MPI_COMM_WORLD,slesinfo,120,&viewer);
    SLESView(sles,viewer);
    matrixname = PetscStrrchr(file,'/');
    PetscPrintf(MPI_COMM_WORLD,"%-8.8s %3d %2.0e %2.1e %2.1e %2.1e %s \n",
                matrixname,its,norm,tsetup+tsolve,tsetup,tsolve,slesinfo);
    ViewerDestroy(viewer);
  } else {
    PetscPrintf(MPI_COMM_WORLD,"Number of iterations = %3d\n",its);
    if (norm < 1.e-10) {
      PetscPrintf(MPI_COMM_WORLD,"Residual norm < 1.e-10\n");
    } else {
      PetscPrintf(MPI_COMM_WORLD,"Residual norm = %10.4e\n",norm);
    }
  }

  /* Cleanup */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
#endif
  return 0;
}

