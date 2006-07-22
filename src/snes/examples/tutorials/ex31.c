
static char help[] = "\n\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DA^using distributed arrays;
   Concepts: multicomponent
   Processors: n
T*/

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

extern PetscErrorCode FormInitialGuessLocal(DALocalInfo*,Field[]);
extern PetscErrorCode FormFunctionLocal(DALocalInfo*,Field*,Field*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;               /* multilevel grid structure */
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

      The problem is actually not periodic; but we declare it as periodic so that we have
      two ghost points at each end where we can put "ghost" boundary conditions.
    */
    ierr = DACreate1d(comm,DA_XPERIODIC,-6,2,2,0,&da);CHKERRQ(ierr);
    ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);

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
    ierr = DMMGSetInitialGuessLocal(dmmg,(PetscErrorCode (*)(DMMG,void*)) FormInitialGuessLocal);CHKERRQ(ierr);

  PreLoadStage("Solve");
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 
    ierr = DMMGSetInitialGuess(dmmg,PETSC_NULL);CHKERRQ(ierr);

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
#define __FUNCT__ "FormInitialGuessLocal"
/* 
   FormInitialGuessLocal - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitialGuessLocal(DALocalInfo* info,Field x[])
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscReal      dhx,hx;

  PetscFunctionBegin;
  dhx = (PetscReal)(info->mx-1);
  hx = 1.0/dhx;                 

  for (i=info->xs; i<info->xs+info->xm; i++) {
    x[i].h   = 22.3;
    x[i].uh  = 0.0;
  }
  PetscFunctionReturn(0);
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

  /* 
     Put the ghost boundary conditions at the left and right end; we have the available space because
     we declared the problem with periodic boundary conditions and two sets of ghost points.

     This is actually a staggered grid with the h's living on cell vertices and uh on cell centers,
     the "extra" x[mx-1].uh is set to zero.
       
  */
  if (info->xs = 0) {
    x[-2].uh  = x[-1].uh = 0.0;
    x[-2].h   = x[-1].h  = x[0].h;  
  } else if (info->xs+>info->xm == mx) {
    x[mx+1].uh = x[mx].uh = x[mx-1].uh  = 0.0;
    x[mx+1].h = x[mx].h  = x[mx-1].h;  
  }

  for (i=info->xs; i<info->xs+info->xm; i++) {
    f[i].h   = x[i].h;
    f[i].uh  = x[i].uh;
  }
  PetscFunctionReturn(0);
} 

