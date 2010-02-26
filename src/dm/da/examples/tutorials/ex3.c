
static char help[] = "Tests DAGetInterpolation for nonuniform DA coordinates.\n\n";

#include "petscda.h"

#undef __FUNCT__
#define __FUNCT__ "SetCoordinates1d"
PetscErrorCode SetCoordinates1d(DA da)
{
  PetscErrorCode ierr;
  PetscInt       i,start,m;
  Vec            gc,global;
  PetscScalar    *coors;
  DA             cda;

  PetscFunctionBegin;
  ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(da,&gc);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DAGetCorners(cda,&start,0,0,&m,0,0);CHKERRQ(ierr);
  for (i=start; i<start+m; i++) {
    if (i % 2) {
      coors[i] = coors[i-1] + .1*(coors[i+1] - coors[i-1]);
    }
  }
  ierr = DAVecRestoreArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DAGetCoordinates(da,&global);CHKERRQ(ierr);
  ierr = DALocalToGlobal(cda,gc,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = VecDestroy(gc);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetCoordinates2d"
PetscErrorCode SetCoordinates2d(DA da)
{
  PetscErrorCode ierr;
  PetscInt       i,j,mstart,m,nstart,n;
  Vec            gc,global;
  DACoor2d       **coors;
  DA             cda;

  PetscFunctionBegin;
  ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(da,&gc);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DAGetCorners(cda,&mstart,&nstart,0,&m,&n,0);CHKERRQ(ierr);
  for (i=mstart; i<mstart+m; i++) {
    for (j=nstart; j<nstart+n; j++) {
      if (i % 2) {
        coors[j][i].x = coors[j][i-1].x + .1*(coors[j][i+1].x - coors[j][i-1].x);
      }
      if (j % 2) {
        coors[j][i].y = coors[j-1][i].y + .3*(coors[j+1][i].y - coors[j-1][i].y);
      }
    }
  }
  ierr = DAVecRestoreArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DAGetCoordinates(da,&global);CHKERRQ(ierr);
  ierr = DALocalToGlobal(cda,gc,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = VecDestroy(gc);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetCoordinates3d"
PetscErrorCode SetCoordinates3d(DA da)
{
  PetscErrorCode ierr;
  PetscInt       i,j,mstart,m,nstart,n,pstart,p,k;
  Vec            gc,global;
  DACoor3d       ***coors;
  DA             cda;

  PetscFunctionBegin;
  ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(da,&gc);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DAGetCorners(cda,&mstart,&nstart,&pstart,&m,&n,&p);CHKERRQ(ierr);
  for (i=mstart; i<mstart+m; i++) {
    for (j=nstart; j<nstart+n; j++) {
      for (k=pstart; k<pstart+p; k++) {
	if (i % 2) {
	  coors[k][j][i].x = coors[k][j][i-1].x + .1*(coors[k][j][i+1].x - coors[k][j][i-1].x);
	}
	if (j % 2) {
	  coors[k][j][i].y = coors[k][j-1][i].y + .3*(coors[k][j+1][i].y - coors[k][j-1][i].y);
	}
	if (k % 2) {
	  coors[k][j][i].z = coors[k-1][j][i].z + .4*(coors[k+1][j][i].z - coors[k-1][j][i].z);
	}
      }
    }
  }
  ierr = DAVecRestoreArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DAGetCoordinates(da,&global);CHKERRQ(ierr);
  ierr = DALocalToGlobal(cda,gc,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = VecDestroy(gc);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       M = 5,N = 4,P = 3, m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE,dim = 1;
  PetscErrorCode ierr;
  DA             dac,daf;
  DAPeriodicType ptype = DA_NONPERIODIC;
  DAStencilType  stype = DA_STENCIL_BOX;
  Mat            A;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /* Read options */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-P",&P,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dim",&dim,PETSC_NULL);CHKERRQ(ierr);

  /* Create distributed array and get vectors */
  if (dim == 1) {
    ierr = DACreate1d(PETSC_COMM_WORLD,ptype,M,1,1,PETSC_NULL,&dac);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = DACreate2d(PETSC_COMM_WORLD,ptype,stype,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&dac);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = DACreate3d(PETSC_COMM_WORLD,ptype,stype,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&dac);CHKERRQ(ierr);
  }

  ierr = DARefine(dac,PETSC_COMM_WORLD,&daf);CHKERRQ(ierr);

  ierr = DASetUniformCoordinates(dac,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  if (dim == 1) {
    ierr = SetCoordinates1d(daf);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = SetCoordinates2d(daf);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = SetCoordinates3d(daf);CHKERRQ(ierr);
  }
  ierr = DAGetInterpolation(dac,daf,&A,0);CHKERRQ(ierr);


  /* Free memory */
  ierr = DADestroy(dac);CHKERRQ(ierr);
  ierr = DADestroy(daf);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
