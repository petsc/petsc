/*$Id: ex10.c,v 1.33 1999/10/24 14:03:24 bsmith Exp bsmith $*/

static char help[] = 
"Reads a PETSc matrix and vector from a file and solves a linear system.\n\
This version first preloads and solves a small system, then loads \n\
another (larger) system and solves it as well.  This example illustrates\n\
preloading of instructions with the smaller system so that more accurate\n\
performance monitoring can be done with the larger one (that actually\n\
is the system of interest).  See the 'Performance Hints' chapter of the\n\
users manual for a discussion of preloading.  Input parameters include\n\
  -f0 <input_file> : first file to load (small system)\n\
  -f1 <input_file> : second file to load (larger system)\n\n\
  -trans  : solve transpose system instead\n\n";

/*T
   Concepts: SLES^Solving a linear system - loading a binary matrix and vector;
   Concepts: PLog^Profiling multiple stages of code;
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
   Routines: SLESSolve(); SLESSetUp(); SLESSetUpOnBlocks(); SLESView();
   Routines: PLogStageRegister(); PLogStagePush(); PLogStagePop(); PLogFlops();
   Routines: PetscBarrier(); PetscGetTime();
   Routines: MatGetTypeFromOptions(); MatLoad(); VecLoad();
   Routines: ViewerBinaryOpen(); ViewerStringOpen(); ViewerDestroy();
   Processors: n
T*/

/* 
  Include "sles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
*/
#include "sles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  SLES       sles;             /* linear solver context */
  MatType    mtype;            /* matrix format */
  Mat        A;                /* matrix */
  Vec        x, b, u;          /* approx solution, RHS, exact solution */
  Viewer     fd;               /* viewer */
  char       file[2][128];     /* input file name */
  char       stagename[6][16]; /* names of profiling stages */
  PetscTruth table,set,flg,trans;
  int        ierr, its, i,loops  = 2;
  double     norm;
  PLogDouble tsetup,tsetup1,tsetup2,tsolve,tsolve1,tsolve2;
  Scalar     zero = 0.0, none = -1.0;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = OptionsHasName(PETSC_NULL,"-table",&table);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-trans",&trans);;CHKERRA(ierr);

  /* 
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = OptionsGetString(PETSC_NULL,"-f0",file[0],127,&flg);CHKERRA(ierr);
  if (!flg) SETERRA(1,0,"Must indicate binary file with the -f0 option");
  ierr = OptionsGetString(PETSC_NULL,"-f1",file[1],127,&flg);CHKERRA(ierr);
  if (!flg) {loops = 1;} /* don't bother with second system */

  /* -----------------------------------------------------------
                  Beginning of linear solver loop
     ----------------------------------------------------------- */
  /* 
     Loop through the linear solve 2 times.  
      - The intention here is to preload and solve a small system;
        then load another (larger) system and solve it as well.
        This process preloads the instructions with the smaller
        system so that more accurate performance monitoring (via
        -log_summary) can be done with the larger one (that actually
        is the system of interest). 
  */
  for ( i=0; i<loops; i++ ) {

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                           Load system i
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Begin profiling next stage
    */
    PLogStagePush(3*i);
    sprintf(stagename[3*i],"Load System %d",i);
    PLogStageRegister(3*i,stagename[3*i]);

    /* 
       Open binary file.  Note that we use BINARY_RDONLY to indicate
       reading from this file.
    */
    ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file[i],BINARY_RDONLY,&fd);CHKERRA(ierr);

    /* 
       Determine matrix format to be used (specified at runtime).
       See the manpage for MatLoad() for available formats.
    */
    ierr = MatGetTypeFromOptions(PETSC_COMM_WORLD,0,&mtype,&set);CHKERRQ(ierr);

    /*
       Load the matrix and vector; then destroy the viewer.
    */
    ierr = MatLoad(fd,mtype,&A);CHKERRA(ierr);
    ierr = VecLoad(fd,&b);CHKERRA(ierr);
    ierr = ViewerDestroy(fd);CHKERRA(ierr);

    /* 
       If the loaded matrix is larger than the vector (due to being padded 
       to match the block size of the system), then create a new padded vector.
    */
    { 
      int    m,n,j,mvec,start,end,index;
      Vec    tmp;
      Scalar *bold;

      /* Create a new vector b by padding the old one */
      ierr = MatGetLocalSize(A,&m,&n);CHKERRA(ierr);
      ierr = VecCreateMPI(PETSC_COMM_WORLD,m,PETSC_DECIDE,&tmp);
      ierr = VecGetOwnershipRange(b,&start,&end);CHKERRA(ierr);
      ierr = VecGetLocalSize(b,&mvec);CHKERRA(ierr);
      ierr = VecGetArray(b,&bold);CHKERRA(ierr);
      for (j=0; j<mvec; j++ ) {
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
    ierr = VecSet(&zero,x);CHKERRA(ierr);

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                      Setup solve for system i
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Conclude profiling last stage; begin profiling next stage.
    */
    /*    PLogStagePop(); */
    ierr = PetscBarrier((PetscObject)A);CHKERRA(ierr);
    PLogStagePush(3*i+1);
    sprintf(stagename[3*i+1],"SLESSetUp %d",i);
    PLogStageRegister(3*i+1,stagename[3*i+1]);

    /*
       We also explicitly time this stage via PetscGetTime()
    */
    ierr = PetscGetTime(&tsetup1);CHKERRA(ierr);

    /*
       Create linear solver; set operators; set runtime options.
    */
    ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
    ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
    ierr = SLESSetFromOptions(sles);CHKERRA(ierr);

    /* 
       Here we explicitly call SLESSetUp() and SLESSetUpOnBlocks() to
       enable more precise profiling of setting up the preconditioner.
       These calls are optional, since both will be called within
       SLESSolve() if they haven't been called already.
    */
    ierr = SLESSetUp(sles,b,x);CHKERRA(ierr);
    ierr = SLESSetUpOnBlocks(sles);CHKERRA(ierr);
    ierr = PetscGetTime(&tsetup2);CHKERRA(ierr);
    tsetup = tsetup2 - tsetup1;

    /*
       Conclude profiling this stage
    */
    PLogStagePop();

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                           Solve system i
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Begin profiling next stage
    */
    ierr = PetscBarrier((PetscObject)A);CHKERRA(ierr);
    PLogStagePush(3*i+2);
    sprintf(stagename[3*i+2],"SLESSolve %d",i);
    PLogStageRegister(3*i+2,stagename[3*i+2]);

    /*
       Solve linear system; we also explicitly time this stage.
    */
    ierr = PetscGetTime(&tsolve1);CHKERRA(ierr);
    if (trans) {
      ierr = SLESSolveTrans(sles,b,x,&its);CHKERRA(ierr);
    } else {
      ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);
    }
    ierr = PetscGetTime(&tsolve2);CHKERRA(ierr);
    tsolve = tsolve2 - tsolve1;

   /* 
       Conclude profiling this stage
    */
    PLogStagePop();

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
            Check error, print output, free data structures.
            This stage is not profiled separately.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* 
       Check error
    */
    if (trans) {
      ierr = MatMultTrans(A,x,u);CHKERRA(ierr);
    } else {
      ierr = MatMult(A,x,u);CHKERRA(ierr);
    }
    ierr = VecAXPY(&none,b,u);CHKERRA(ierr);
    ierr = VecNorm(u,NORM_2,&norm);CHKERRA(ierr);

    /*
       Write output (optinally using table for solver details).
        - PetscPrintf() handles output for multiprocessor jobs 
          by printing from only one processor in the communicator.
        - SLESView() prints information about the linear solver.
    */
    if (table) {
      char   *matrixname, slesinfo[120];
      Viewer viewer;

      /*
         Open a string viewer; then write info to it.
      */
      ierr = ViewerStringOpen(PETSC_COMM_WORLD,slesinfo,120,&viewer);CHKERRA(ierr);
      ierr = SLESView(sles,viewer);CHKERRA(ierr);
      ierr = PetscStrrchr(file[i],'/',&matrixname);CHKERRA(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%-8.8s %3d %2.0e %2.1e %2.1e %2.1e %s \n",
                matrixname,its,norm,tsetup+tsolve,tsetup,tsolve,slesinfo);CHKERRQ(ierr);

      /*
         Destroy the viewer
      */
      ierr = ViewerDestroy(viewer);CHKERRA(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3d\n",its);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %A\n",norm);CHKERRQ(ierr);
    }
    /* 
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
    */
    ierr = MatDestroy(A);CHKERRA(ierr); ierr = VecDestroy(b);CHKERRA(ierr);
    ierr = VecDestroy(u);CHKERRA(ierr); ierr = VecDestroy(x);CHKERRA(ierr);
    ierr = SLESDestroy(sles);CHKERRA(ierr); 
  }
  /* -----------------------------------------------------------
                      End of linear solver loop
     ----------------------------------------------------------- */

  PetscFinalize();
  return 0;
}

