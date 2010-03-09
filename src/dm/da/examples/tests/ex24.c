
static char help[] = "Tests DALocalToGlocal() for dof > 1\n\n";

#include "petscda.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       M = 6,N = 5,m = PETSC_DECIDE,n = PETSC_DECIDE,i,j,is,js,in,jen;
  PetscErrorCode ierr;
  DA             da;
  Vec            local,global;
  PetscScalar    ***l;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /* Create distributed array and get vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,M,N,m,n,3,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRQ(ierr);

  ierr = DAGetCorners(da,&is,&js,0,&in,&jen,0);CHKERRQ(ierr);
  ierr = DAVecGetArrayDOF(da,local,&l);CHKERRQ(ierr);
  for (i=is; i<is+in; i++) {
    for (j=js; j<js+jen; j++) {
      l[j][i][0] = 3*(i + j*M);
      l[j][i][1] = 3*(i + j*M) + 1;
      l[j][i][2] = 3*(i + j*M) + 2;
    }
  }
  ierr = DAVecRestoreArrayDOF(da,local,&l);CHKERRQ(ierr);
  ierr = DALocalToGlobal(da,local,ADD_VALUES,global);CHKERRQ(ierr);

  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free memory */
  ierr = VecDestroy(local);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
