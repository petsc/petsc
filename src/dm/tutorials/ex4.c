
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
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,m,n,PETSC_DECIDE,PETSC_DECIDE,dof,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&xy);CHKERRQ(ierr);

  ierr = DMDACreatePF(da,&pf);CHKERRQ(ierr);
  ierr = PFSet(pf,myfunction,0,0,0,0);CHKERRQ(ierr);
  ierr = PFSetFromOptions(pf);CHKERRQ(ierr);

  ierr = PFApplyVec(pf,xy,u);CHKERRQ(ierr);

  ierr = VecView(u,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PFDestroy(&pf);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
