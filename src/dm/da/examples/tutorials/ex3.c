/*$Id: ex2.c,v 1.15 2001/08/07 03:04:45 balay Exp $*/

static char help[] = "Tests DAGetInterpolation for nonuniform DA coordinates.\n\n";

#include "petscda.h"
#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "SetCoordinates"
int SetCoordinates(DA da)
{
  int         ierr,i,start,m;
  Vec         gc,global;
  PetscScalar *coors;

  PetscFunctionBegin;
  ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(da,&gc);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,gc,&coors);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&start,0,0,&m,0,0);CHKERRQ(ierr);
  for (i=start+1; i<start+m-1; i++) {
    if (i % 2) {
      coors[i] = coors[i-1] + .1*(coors[i+1] - coors[i-1]);
    }
  }
  ierr = DAVecRestoreArray(da,gc,&coors);CHKERRQ(ierr);
  ierr = DAGetCoordinates(da,&global);CHKERRQ(ierr);
  ierr = DALocalToGlobal(da,gc,INSERT_VALUES,global);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int            M = 10,N = 8,m = PETSC_DECIDE,n = PETSC_DECIDE,ierr;
  DA             dac,daf;
  DAPeriodicType ptype = DA_NONPERIODIC;
  DAStencilType  stype = DA_STENCIL_BOX;
  Mat            Auniform;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /* Read options */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  /* Create distributed array and get vectors */
  ierr = DACreate1d(PETSC_COMM_WORLD,ptype,M,1,1,PETSC_NULL,&dac);CHKERRQ(ierr);
  ierr = DARefine(dac,PETSC_COMM_WORLD,&daf);CHKERRQ(ierr);

  ierr = DASetUniformCoordinates(dac,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = SetCoordinates(daf);CHKERRQ(ierr);

  ierr = DAGetInterpolation(dac,daf,&Auniform,0);CHKERRQ(ierr);


  /* Free memory */
  ierr = DADestroy(dac);CHKERRQ(ierr);
  ierr = DADestroy(daf);CHKERRQ(ierr);
  ierr = MatDestroy(Auniform);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
