/*$Id: ex7.c,v 1.12 2000/09/28 21:13:39 bsmith Exp bsmith $*/

static char help[] = 
"Reads a PETSc matrix and vector from a file and solves a linear system.\n\
 Tests inplace factorization for SeqBAIJ. Input parameters include\n\
  -f0 <input_file> : first file to load (small system)\n\n";

/*T
   Concepts: SLES^solving a linear system
   Concepts: PetscLog^profiling multiple stages of code;
   Processors: n
T*/

/* 
  Include "petscsles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include "petscsles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  SLES       sles;             /* linear solver context */
  Mat        A,B;                /* matrix */
  Vec        x,b,u;          /* approx solution, RHS, exact solution */
  PetscViewer     fd;               /* viewer */
  char       file[2][128];     /* input file name */
  int        ierr,its;
  PetscTruth flg;
  double     norm;
  Scalar     zero = 0.0,none = -1.0;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* 
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f0",file[0],127,&flg);CHKERRA(ierr);
  if (!flg) SETERRA(1,"Must indicate binary file with the -f0 option");


  /* 
       Open binary file.  Note that we use PETSC_BINARY_RDONLY to indicate
       reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],PETSC_BINARY_RDONLY,&fd);CHKERRA(ierr);

  /*
       Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatLoad(fd,MATSEQBAIJ,&A);CHKERRA(ierr);
  ierr = MatConvert(A,MATSAME,&B);CHKERRQ(ierr);
  ierr = VecLoad(fd,&b);CHKERRA(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRA(ierr);

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
  ierr = VecSet(&zero,x);CHKERRA(ierr);

  /*
      Create linear solver; set operators; set runtime options.
  */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,B,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);

  /* 
       Here we explicitly call SLESSetUp() and SLESSetUpOnBlocks() to
       enable more precise profiling of setting up the preconditioner.
       These calls are optional, since both will be called within
       SLESSolve() if they haven't been called already.
  */
  ierr = SLESSetUp(sles,b,x);CHKERRA(ierr);
  ierr = SLESSetUpOnBlocks(sles);CHKERRA(ierr);

  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);

  /*
            Check error, print output, free data structures.
            This stage is not profiled separately.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Check error
  */
  ierr = MatMult(A,x,u);
  ierr = VecAXPY(&none,b,u);CHKERRA(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRA(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3d\n",its);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm = %A\n",norm);CHKERRA(ierr);

  /* 
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
  */
  ierr = MatDestroy(A);CHKERRA(ierr); 
  ierr = MatDestroy(B);CHKERRA(ierr); 
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr); ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = SLESDestroy(sles);CHKERRA(ierr); 


  PetscFinalize();
  return 0;
}

