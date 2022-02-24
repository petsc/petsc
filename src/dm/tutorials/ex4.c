
static char help[] = "Demonstrates various vector routines for DMDA.\n\n";

/*T
   Concepts: mathematical functions
   Processors: n
T*/

/*
  Include "petscpf.h" so that we can use pf functions and "petscdmda.h" so
 we can use the PETSc distributed arrays
*/

#include <petscpf.h>
#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode myfunction(void *ctx,PetscInt n,const PetscScalar *xy,PetscScalar *u)
{
  PetscInt i;

  PetscFunctionBeginUser;
  for (i=0; i<n; i++) {
    u[2*i]   = xy[2*i];
    u[2*i+1] = xy[2*i+1];
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Vec            u,xy;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       m = 10, n = 10, dof = 2;
  PF             pf;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,m,n,PETSC_DECIDE,PETSC_DECIDE,dof,1,0,0,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0));
  CHKERRQ(DMCreateGlobalVector(da,&u));
  CHKERRQ(DMGetCoordinates(da,&xy));

  CHKERRQ(DMDACreatePF(da,&pf));
  CHKERRQ(PFSet(pf,myfunction,0,0,0,0));
  CHKERRQ(PFSetFromOptions(pf));

  CHKERRQ(PFApplyVec(pf,xy,u));

  CHKERRQ(VecView(u,PETSC_VIEWER_DRAW_WORLD));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(PFDestroy(&pf));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
