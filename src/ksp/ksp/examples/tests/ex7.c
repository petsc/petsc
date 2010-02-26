
static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
 Tests inplace factorization for SeqBAIJ. Input parameters include\n\
  -f0 <input_file> : first file to load (small system)\n\n";

/*T
   Concepts: KSP^solving a linear system
   Concepts: PetscLog^profiling multiple stages of code;
   Processors: n
T*/

/* 
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include "petscksp.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  KSP            ksp;             /* linear solver context */
  Mat            A,B;                /* matrix */
  Vec            x,b,u;          /* approx solution, RHS, exact solution */
  PetscViewer    fd;               /* viewer */
  char           file[2][PETSC_MAX_PATH_LEN];     /* input file name */
  PetscErrorCode ierr;
  PetscInt       its;
  PetscTruth     flg;
  PetscReal      norm;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* 
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f0",file[0],PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,"Must indicate binary file with the -f0 option");


  /* 
       Open binary file.  Note that we use FILE_MODE_READ to indicate
       reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /*
       Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatLoad(fd,MATSEQBAIJ,&A);CHKERRQ(ierr);
  ierr = MatConvert(A,MATSAME,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = VecLoad(fd,PETSC_NULL,&b);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);

  /* 
       If the loaded matrix is larger than the vector (due to being padded 
       to match the block size of the system), then create a new padded vector.
  */
  { 
      PetscInt     m,n,j,mvec,start,end,idx;
      Vec          tmp;
      PetscScalar *bold;

      /* Create a new vector b by padding the old one */
      ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
      ierr = VecCreate(PETSC_COMM_WORLD,&tmp);CHKERRQ(ierr);
      ierr = VecSetSizes(tmp,m,PETSC_DECIDE);CHKERRQ(ierr);
      ierr = VecSetFromOptions(tmp);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(b,&start,&end);CHKERRQ(ierr);
      ierr = VecGetLocalSize(b,&mvec);CHKERRQ(ierr);
      ierr = VecGetArray(b,&bold);CHKERRQ(ierr);
      for (j=0; j<mvec; j++) {
        idx = start+j;
        ierr  = VecSetValues(tmp,1,&idx,bold+j,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(b,&bold);CHKERRQ(ierr);
      ierr = VecDestroy(b);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(tmp);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(tmp);CHKERRQ(ierr);
      b = tmp;
    }
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);

  /*
      Create linear solver; set operators; set runtime options.
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* 
       Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
       enable more precise profiling of setting up the preconditioner.
       These calls are optional, since both will be called within
       KSPSolve() if they haven't been called already.
  */
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /*
            Check error, print output, free data structures.
            This stage is not profiled separately.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Check error
  */
  ierr = MatMult(A,x,u);
  ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm = %A\n",norm);CHKERRQ(ierr);

  /* 
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
  */
  ierr = MatDestroy(A);CHKERRQ(ierr); 
  ierr = MatDestroy(B);CHKERRQ(ierr); 
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr); ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = KSPDestroy(ksp);CHKERRQ(ierr); 


  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

