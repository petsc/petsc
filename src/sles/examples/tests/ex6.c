/*$Id: ex6.c,v 1.61 2000/01/11 21:02:16 bsmith Exp balay $*/

static char help[] = 
"Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil,\n\
                    use the file petsc/src/mat/examples/matbinary.ex\n\n";

#include "petscsles.h"
#include "petsclog.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int        ierr,its;
  double     norm;
  PLogDouble tsetup1,tsetup2,tsetup,tsolve1,tsolve2,tsolve;
  Scalar     zero = 0.0,none = -1.0;
  Vec        x,b,u;
  Mat        A;
  MatType    mtype;
  SLES       sles;
  char       file[128];
  Viewer     fd;
  PetscTruth table,flg,set;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = OptionsHasName(PETSC_NULL,"-table",&table);CHKERRA(ierr);

#if defined(PETSC_USE_COMPLEX)
  SETERRA(1,0,"This example does not work with complex numbers");
#else

  /* Read matrix and RHS */
  ierr = OptionsGetString(PETSC_NULL,"-f",file,127,&flg);CHKERRA(ierr);
  if (!flg) SETERRA(1,0,"Must indicate binary file with the -f option");
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatGetTypeFromOptions(PETSC_COMM_WORLD,PETSC_NULL,&mtype,&set);CHKERRA(ierr);
  ierr = MatLoad(fd,mtype,&A);CHKERRA(ierr);
  ierr = VecLoad(fd,&b);CHKERRA(ierr);
  ierr = ViewerDestroy(fd);CHKERRA(ierr);

  /* 
   If the load matrix is larger then the vector, due to being padded 
   to match the blocksize then create a new padded vector
  */
  { 
    int    m,n,j,mvec,start,end,index;
    Vec    tmp;
    Scalar *bold;

    ierr = MatGetLocalSize(A,&m,&n);CHKERRA(ierr);
    ierr = VecCreateMPI(PETSC_COMM_WORLD,m,PETSC_DECIDE,&tmp);
    ierr = VecGetOwnershipRange(b,&start,&end);CHKERRA(ierr);
    ierr = VecGetLocalSize(b,&mvec);CHKERRA(ierr);
    ierr = VecGetArray(b,&bold);CHKERRA(ierr);
    for (j=0; j<mvec; j++) {
      index = start+j;
      ierr  = VecSetValues(tmp,1,&index,bold+j,INSERT_VALUES);CHKERRA(ierr);
    }
    ierr = VecRestoreArray(b,&bold);CHKERRA(ierr);
    ierr = VecDestroy(b);CHKERRA(ierr);
    ierr = VecAssemblyBegin(tmp);CHKERRA(ierr);
    ierr = VecAssemblyEnd(tmp);CHKERRA(ierr);
    b = tmp;
  }
  ierr = VecDuplicate(b,&x);CHKERRA(ierr);
  ierr = VecDuplicate(b,&u);CHKERRA(ierr);

  /* Do solve once to bring in all instruction pages */
  ierr = OptionsHasName(PETSC_NULL,"-preload",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = VecSet(&zero,x);CHKERRA(ierr);
    ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
    ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
    ierr = SLESSetFromOptions(sles);CHKERRA(ierr);
    ierr = SLESSetUp(sles,b,x);CHKERRA(ierr);
    ierr = SLESSetUpOnBlocks(sles);CHKERRA(ierr);
    ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);
  }

  ierr = VecSet(&zero,x);CHKERRA(ierr);
  ierr = PetscBarrier((PetscObject)A);CHKERRA(ierr);

  PLogStagePush(1);
  ierr = PetscGetTime(&tsetup1);CHKERRA(ierr);
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);
  ierr = SLESSetUp(sles,b,x);CHKERRA(ierr);
  ierr = SLESSetUpOnBlocks(sles);CHKERRA(ierr);
  ierr = PetscGetTime(&tsetup2);CHKERRA(ierr);
  tsetup = tsetup2 -tsetup1;
  PLogStagePop();
  ierr = PetscBarrier((PetscObject)A);CHKERRA(ierr);


  PLogStagePush(2);
  ierr = PetscGetTime(&tsolve1);CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);
  ierr = PetscGetTime(&tsolve2);CHKERRA(ierr);
  tsolve = tsolve2 - tsolve1;
  PLogStagePop();

  /* Show result */
  ierr = MatMult(A,x,u);
  ierr = VecAXPY(&none,b,u);CHKERRA(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRA(ierr);
  /*  matrix PC   KSP   Options       its    residual setuptime solvetime  */
  if (table) {
    char   *matrixname,slesinfo[120];
    Viewer viewer;
    ierr = ViewerStringOpen(PETSC_COMM_WORLD,slesinfo,120,&viewer);CHKERRA(ierr);
    ierr = SLESView(sles,viewer);CHKERRA(ierr);
    ierr = PetscStrrchr(file,'/',&matrixname);CHKERRA(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%-8.8s %3d %2.0e %2.1e %2.1e %2.1e %s \n",
                       matrixname,its,norm,tsetup+tsolve,tsetup,tsolve,slesinfo);CHKERRA(ierr);
    ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3d\n",its);CHKERRA(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm = %A\n",norm);CHKERRA(ierr);
  }

  /* Cleanup */
  ierr = SLESDestroy(sles);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);

  PetscFinalize();
#endif
  return 0;
}

