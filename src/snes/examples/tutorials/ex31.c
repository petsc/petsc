
static char help[] = "\n\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DA^using distributed arrays;
   Concepts: multicomponent
   Processors: n
T*/

/* ------------------------------------------------------------------------


  ------------------------------------------------------------------------- */

/* 
   Include "petscda.h" so that we can use distributed arrays (DAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers 
*/
#include "petscsnes.h"
#include "petscda.h"
#include "petscdmmg.h"

/* 
   User-defined routines and data structures
*/
typedef struct {
  PetscScalar h,uh;
} Field;

extern PetscErrorCode FormInitialGuess(DMMG,Vec);
extern PetscErrorCode FormFunctionLocal(DALocalInfo*,Field*,Field
*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;               /* multilevel grid structure */
  PetscInt       mx;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  DA             da;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;


  PreLoadBegin(PETSC_TRUE,"SetUp");
    ierr = DMMGCreate(comm,2,0,&dmmg);CHKERRQ(ierr);

    /*
      Create distributed array multigrid object (DMMG) to manage parallel grid and vectors
      for principal unknowns (x) and governing residuals (f)
    */
    ierr = DACreate1d(comm,DA_XPERIODIC,-4,2,1,0,&da);CHKERRQ(ierr);
    ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);

    ierr = DAGetInfo(DMMGGetDA(dmmg),0,&mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
    /* 
     Problem parameters 
    */
    ierr = DASetFieldName(DMMGGetDA(dmmg),0,"h");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),1,"uh");CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create user context, set problem data, create vector data structures.
       Also, compute the initial guess.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create nonlinear solver context

       Process adiC(36): FormFunctionLocal 
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMMGSetSNESLocal(dmmg,FormFunctionLocal,0,ad_FormFunctionLocal,admf_FormFunctionLocal);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve the nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMMGSetInitialGuess(dmmg,FormInitialGuess);CHKERRQ(ierr);

  PreLoadStage("Solve");
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  PreLoadEnd();

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */


#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitialGuess(DMMG dmmg,Vec X)
{
  DA             da = (DA)dmmg->dm;
  PetscInt       i,mx,xs,xm;
  PetscErrorCode ierr;
  PetscReal      dx;
  Field          *x;


  ierr = DAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx  = 1.0/(mx-1);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
  */
  ierr = DAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
     Initial condition is motionless fluid and equilibrium temperature
  */
  for (i=xs; i<xs+xm; i++) {
    x[i].h     = 0.0;
    x[i].uh    = 0.0;
  }

  /*
     Restore vector
  */
  ierr = DAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  return 0;
}
 
PetscErrorCode FormFunctionLocal(DALocalInfo *info,Field *x,Field *f,void *ptr)
 {
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      hx,dhx;

  PetscFunctionBegin;

  /* 
     Define mesh intervals ratios for uniform grid.
     [Note: FD formulae below are normalized by multiplying through by
     local volume element to obtain coefficients O(1) in two dimensions.]
  */
  dhx = (PetscReal)(info->mx-1);
  hx = 1.0/dhx;                 

  for (i=info->xs; i<info->xs+info->xm; i++) {
    f[i].h   = 0.0;
    f[i].uh  = 0.0;
  }

  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  ierr = PetscLogFlops(0.0*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

