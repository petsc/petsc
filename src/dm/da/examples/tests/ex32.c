#include "petscda.h"

#undef __FUNCT__  
#define __FUNCT__ "CompareGhostedCoords"
static PetscErrorCode CompareGhostedCoords(Vec gc1,Vec gc2)
{
  PetscErrorCode ierr;
  PetscReal      nrm,gnrm;
  Vec            tmp;

  PetscFunctionBegin;
  ierr = VecDuplicate(gc1,&tmp);CHKERRQ(ierr);
  ierr = VecWAXPY(tmp,-1.0,gc1,gc2);CHKERRQ(ierr);
  ierr = VecNorm(tmp,NORM_INFINITY,&nrm);CHKERRQ(ierr);
  ierr = PetscGlobalMax(&nrm,&gnrm,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"norm of difference of ghosted coordinates %8.2e\n",gnrm);CHKERRQ(ierr);
  ierr = VecDestroy(tmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TestQ2Q1DA"
static PetscErrorCode TestQ2Q1DA( void )
{
  DA             Q2_da,Q1_da,cda;
  PetscInt       mx,my,mz;
  Vec            coords,gcoords,gcoords2;
  PetscErrorCode ierr;

  mx=7;
  my=11;
  mz=13;
  ierr=DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,mx,my,mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,2,0,0,0,&Q2_da);CHKERRQ(ierr);
  ierr = DASetUniformCoordinates(Q2_da,-1.0,1.0,-2.0,2.0,-3.0,3.0);CHKERRQ(ierr);
  ierr = DAGetCoordinates(Q2_da,&coords);CHKERRQ(ierr);
  ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,mx,my,mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,1,0,0,0,&Q1_da);CHKERRQ(ierr);
  ierr = DASetCoordinates(Q1_da,coords);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);

  /* Get ghost coordinates one way */
  ierr = DAGetGhostedCoordinates(Q1_da,&gcoords);CHKERRQ(ierr);

  /* And another */
  ierr = DAGetCoordinates(Q1_da,&coords);CHKERRQ(ierr);
  ierr = DAGetCoordinateDA(Q1_da,&cda);CHKERRQ(ierr);
  ierr = DAGetLocalVector(cda,&gcoords2);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(cda,coords,INSERT_VALUES,gcoords2);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(cda,coords,INSERT_VALUES,gcoords2);CHKERRQ(ierr);

  ierr = CompareGhostedCoords(gcoords,gcoords2);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(cda,&gcoords2);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);

  ierr = VecScale(coords,10.0);CHKERRQ(ierr);
  ierr = VecScale(gcoords,10.0);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(Q1_da,&gcoords2);CHKERRQ(ierr);
  ierr = CompareGhostedCoords(gcoords,gcoords2);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  ierr = VecDestroy(gcoords2);CHKERRQ(ierr);

  ierr = VecDestroy(gcoords);CHKERRQ(ierr);
  ierr = DADestroy(Q2_da);CHKERRQ(ierr);
  ierr = DADestroy(Q1_da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,0);
  ierr = TestQ2Q1DA();CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
