
static char help[] = "Tests DMDALocalToGlocal() for dof > 1\n\n";

#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       M = 6,N = 5,m = PETSC_DECIDE,n = PETSC_DECIDE,i,j,is,js,in,jen;
  PetscErrorCode ierr;
  DM             da;
  Vec            local,global;
  PetscScalar    ***l;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /* Create distributed array and get vectors */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,3,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&is,&js,0,&in,&jen,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,local,&l);CHKERRQ(ierr);
  for (i=is; i<is+in; i++) {
    for (j=js; j<js+jen; j++) {
      l[j][i][0] = 3*(i + j*M);
      l[j][i][1] = 3*(i + j*M) + 1;
      l[j][i][2] = 3*(i + j*M) + 2;
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da,local,&l);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,local,ADD_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,local,ADD_VALUES,global);CHKERRQ(ierr);

  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free memory */
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
