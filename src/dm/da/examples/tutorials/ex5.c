
static char help[] = "Tests DAGetElements() and VecView() contour plotting for 2d DAs.\n\n";

#include "petscda.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       M = 10,N = 8,m = PETSC_DECIDE,n = PETSC_DECIDE,ne,i;
  const PetscInt *e;
  PetscErrorCode ierr;
  PetscTruth     flg = PETSC_FALSE;
  DA             da;
  PetscViewer    viewer;
  Vec            local,global;
  PetscScalar    value;
  DAPeriodicType ptype = DA_NONPERIODIC;
  DAStencilType  stype = DA_STENCIL_BOX;
  PetscScalar    *lv;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer);CHKERRQ(ierr);

  /* Read options */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-star_stencil",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) stype = DA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,ptype,stype,M,N,m,n,1,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRQ(ierr);

  value = -3.0;
  ierr = VecSet(global,value);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  ierr = DAGetElements(da,&ne,&e);CHKERRQ(ierr);
  ierr = VecGetArray(local,&lv);CHKERRQ(ierr);
  for (i=0; i<ne; i++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"i %D e[3*i] %D %D %D\n",i,e[3*i],e[3*i+1],e[3*i+2]);
    lv[e[3*i]] = i; 
  }
  ierr = VecRestoreArray(local,&lv);CHKERRQ(ierr);
  ierr = DARestoreElements(da,&ne,&e);CHKERRQ(ierr);

  ierr = DALocalToGlobal(da,local,ADD_VALUES,global);CHKERRQ(ierr);

  ierr = DAView(da,viewer);CHKERRQ(ierr);
  ierr = VecView(global,viewer);CHKERRQ(ierr);

  /* Free memory */
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(local);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
