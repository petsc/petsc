
#ifndef lint
static char vcid[] = "$Id: ex21.c,v 1.2 1996/07/23 20:12:23 bsmith Exp curfman $";
#endif

static char help[] = 
"Reads a PETSc matrix and vector from a file and solves a linear system.\n\
This version first preloads and solves a small system, then loads \n\
another (larger) system and solves it as well.  This example illustrates\n\
preloading of instructions with the smaller system so that more accurate\n\
performance monitoring can be done with the larger one (that actually\n\
is the system of interest). Input parameters include\n\
  -f0 <input_file> : first file to load (small system)\n\
  -f1 <input_file> : second file to load (larger system)\n\n";

#include "draw.h"
#include "mat.h"
#include "sles.h"
#include "plog.h"
#include <stdio.h>

int main(int argc,char **args)
{
  int        ierr, its, set, flg, i;
  double     norm,tsetup,tsolve;
  Scalar     zero = 0.0, none = -1.0;
  Vec        x, b, u;
  Mat        A;
  MatType    mtype;
  SLES       sles;
  char       file[2][128];
  Viewer     fd;
  PetscTruth table = PETSC_FALSE;
  char       stagename[6][18];

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = OptionsHasName(PETSC_NULL,"-table",&flg);
  if (flg) table = PETSC_TRUE;

#if defined(PETSC_COMPLEX)
  SETERRA(1,"This example does not work with complex numbers");
#else

  /* Read matrix and RHS */
  ierr = OptionsGetString(PETSC_NULL,"-f0",file[0],127,&flg); CHKERRA(ierr);
  if (!flg) SETERRA(1,"Must indicate binary file with the -f0 option");
  ierr = OptionsGetString(PETSC_NULL,"-f1",file[1],127,&flg); CHKERRA(ierr);
  if (!flg) SETERRA(1,"Must indicate binary file with the -f1 option");

  for ( i=0; i<2; i++ ) {

    PLogStagePush(3*i);
    sprintf(stagename[3*i],"Load System %d",i);
    PLogStageRegister(3*i,stagename[3*i]);
    ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,file[i],BINARY_RDONLY,&fd);
         CHKERRA(ierr);
    ierr = MatGetTypeFromOptions(MPI_COMM_WORLD,0,&mtype,&set); CHKERRQ(ierr);
    ierr = MatLoad(fd,mtype,&A); CHKERRA(ierr);
    ierr = VecLoad(fd,&b); CHKERRA(ierr);
    ierr = ViewerDestroy(fd); CHKERRA(ierr);

    /* 
     If the load matrix is larger then the vector, due to being padded 
     to match the blocksize then create a new padded vector
    */
    /* 
     If the load matrix is larger then the vector, due to being padded 
     to match the blocksize then create a new padded vector
    */
    { 
      int    m,n,j,mvec,start,end,index;
      Vec    tmp;
      Scalar *bold;
      /* create a new vector b by padding the old one */
      ierr = MatGetLocalSize(A,&m,&n); CHKERRA(ierr);
      ierr = VecCreateMPI(MPI_COMM_WORLD,m,PETSC_DECIDE,&tmp);
      ierr = VecGetOwnershipRange(b,&start,&end); CHKERRA(ierr);
      ierr = VecGetLocalSize(b,&mvec); CHKERRA(ierr);
      ierr = VecGetArray(b,&bold); CHKERRA(ierr);
      for (j=0; j<mvec; j++ ) {
        index = start+j;
        ierr  = VecSetValues(tmp,1,&index,bold+j,INSERT_VALUES); CHKERRA(ierr);
      }
      ierr = VecRestoreArray(b,&bold); CHKERRA(ierr);
      ierr = VecDestroy(b); CHKERRA(ierr);
      ierr = VecAssemblyBegin(tmp); CHKERRA(ierr);
      ierr = VecAssemblyEnd(tmp); CHKERRA(ierr);
      b = tmp;
    }
    ierr = VecDuplicate(b,&x); CHKERRA(ierr);
    ierr = VecDuplicate(b,&u); CHKERRA(ierr);
    ierr = VecSet(&zero,x); CHKERRA(ierr);
    PetscBarrier(A);

    PLogStagePush(3*i+1);
    sprintf(stagename[3*i+1],"SLESSetUp %d",i);
    PLogStageRegister(3*i+1,stagename[3*i+1]);
    tsetup = PetscGetTime();  
    ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
    ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
    ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
    ierr = SLESSetUp(sles,b,x); CHKERRA(ierr);
    ierr = SLESSetUpOnBlocks(sles); CHKERRA(ierr);
    tsetup = PetscGetTime() - tsetup;
    PLogStagePop();
    PetscBarrier(A);


    PLogStagePush(3*i+2);
    sprintf(stagename[3*i+2],"SLESSolve %d",i);
    PLogStageRegister(3*i+2,stagename[3*i+2]);
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
      matrixname = PetscStrrchr(file[i],'/');
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

  }
  PetscFinalize();
#endif
  return 0;
}

