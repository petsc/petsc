
static char help[] = "Tests DMDA with variable multiple degrees of freedom per node.\n\n";

/*
   This code only compiles with gcc, since it is not ANSI C
*/

#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode doit(DM da,Vec global)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,M,N,dof;

  ierr = DMDAGetInfo(da,0,&M,&N,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  {
    struct {PetscScalar inside[dof];} **mystruct;
    ierr = DMDAVecGetArrayRead(da,global,(void*) &mystruct);CHKERRQ(ierr);
    for (i=0; i<N; i++) {
      for (j=0; j<M; j++) {
        for (k=0; k<dof; k++) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,"%D %D %g\n",i,j,(double)mystruct[i][j].inside[0]);CHKERRQ(ierr);

          mystruct[i][j].inside[1] = 2.1;
        }
      }
    }
    ierr = DMDAVecRestoreArrayRead(da,global,(void*) &mystruct);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscInt       dof = 2,M = 3,N = 3,m = PETSC_DECIDE,n = PETSC_DECIDE;
  PetscErrorCode ierr;
  DM             da;
  Vec            global,local;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,0,"-dof",&dof,0);CHKERRQ(ierr);
  /* Create distributed array and get vectors */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,dof,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);

  ierr = doit(da,global);CHKERRQ(ierr);

  ierr = VecView(global,0);CHKERRQ(ierr);

  /* Free memory */
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

