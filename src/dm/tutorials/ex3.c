
static char help[] = "Tests DMCreateInterpolation() for nonuniform DMDA coordinates.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode SetCoordinates1d(DM da)
{
  PetscErrorCode ierr;
  PetscInt       i,start,m;
  Vec            local,global;
  PetscScalar    *coors,*coorslocal;
  DM             cda;

  PetscFunctionBeginUser;
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&global);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&local);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,global,&coors);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda,local,&coorslocal);CHKERRQ(ierr);
  ierr = DMDAGetCorners(cda,&start,0,0,&m,0,0);CHKERRQ(ierr);
  for (i=start; i<start+m; i++) {
    if (i % 2) {
      coors[i] = coorslocal[i-1] + .1*(coorslocal[i+1] - coorslocal[i-1]);
    }
  }
  ierr = DMDAVecRestoreArray(cda,global,&coors);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cda,local,&coorslocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(cda,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(cda,global,INSERT_VALUES,local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetCoordinates2d(DM da)
{
  PetscErrorCode ierr;
  PetscInt       i,j,mstart,m,nstart,n;
  Vec            local,global;
  DMDACoor2d     **coors,**coorslocal;
  DM             cda;

  PetscFunctionBeginUser;
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&global);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&local);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,global,&coors);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda,local,&coorslocal);CHKERRQ(ierr);
  ierr = DMDAGetCorners(cda,&mstart,&nstart,0,&m,&n,0);CHKERRQ(ierr);
  for (i=mstart; i<mstart+m; i++) {
    for (j=nstart; j<nstart+n; j++) {
      if (i % 2) {
        coors[j][i].x = coorslocal[j][i-1].x + .1*(coorslocal[j][i+1].x - coorslocal[j][i-1].x);
      }
      if (j % 2) {
        coors[j][i].y = coorslocal[j-1][i].y + .3*(coorslocal[j+1][i].y - coorslocal[j-1][i].y);
      }
    }
  }
  ierr = DMDAVecRestoreArray(cda,global,&coors);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cda,local,&coorslocal);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(cda,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(cda,global,INSERT_VALUES,local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetCoordinates3d(DM da)
{
  PetscErrorCode ierr;
  PetscInt       i,j,mstart,m,nstart,n,pstart,p,k;
  Vec            local,global;
  DMDACoor3d     ***coors,***coorslocal;
  DM             cda;

  PetscFunctionBeginUser;
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&global);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&local);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,global,&coors);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda,local,&coorslocal);CHKERRQ(ierr);
  ierr = DMDAGetCorners(cda,&mstart,&nstart,&pstart,&m,&n,&p);CHKERRQ(ierr);
  for (i=mstart; i<mstart+m; i++) {
    for (j=nstart; j<nstart+n; j++) {
      for (k=pstart; k<pstart+p; k++) {
        if (i % 2) {
          coors[k][j][i].x = coorslocal[k][j][i-1].x + .1*(coorslocal[k][j][i+1].x - coorslocal[k][j][i-1].x);
        }
        if (j % 2) {
          coors[k][j][i].y = coorslocal[k][j-1][i].y + .3*(coorslocal[k][j+1][i].y - coorslocal[k][j-1][i].y);
        }
        if (k % 2) {
          coors[k][j][i].z = coorslocal[k-1][j][i].z + .4*(coorslocal[k+1][j][i].z - coorslocal[k-1][j][i].z);
        }
      }
    }
  }
  ierr = DMDAVecRestoreArray(cda,global,&coors);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cda,local,&coorslocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(cda,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(cda,global,INSERT_VALUES,local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscInt         M = 5,N = 4,P = 3, m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE,dim = 1;
  PetscErrorCode   ierr;
  DM               dac,daf;
  DMBoundaryType   bx    = DM_BOUNDARY_NONE,by=DM_BOUNDARY_NONE,bz=DM_BOUNDARY_NONE;
  DMDAStencilType  stype = DMDA_STENCIL_BOX;
  Mat              A;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-P",&P,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);

  /* Create distributed array and get vectors */
  if (dim == 1) {
    ierr = DMDACreate1d(PETSC_COMM_WORLD,bx,M,1,1,NULL,&dac);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&dac);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stype,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&dac);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"dim must be 1,2, or 3");
  ierr = DMSetFromOptions(dac);CHKERRQ(ierr);
  ierr = DMSetUp(dac);CHKERRQ(ierr);

  ierr = DMRefine(dac,PETSC_COMM_WORLD,&daf);CHKERRQ(ierr);

  ierr = DMDASetUniformCoordinates(dac,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  if (dim == 1) {
    ierr = SetCoordinates1d(daf);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = SetCoordinates2d(daf);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = SetCoordinates3d(daf);CHKERRQ(ierr);
  }
  ierr = DMCreateInterpolation(dac,daf,&A,0);CHKERRQ(ierr);

  /* Free memory */
  ierr = DMDestroy(&dac);CHKERRQ(ierr);
  ierr = DMDestroy(&daf);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3
      args: -mat_view

   test:
      suffix: 2
      nsize: 3
      args: -mat_view -dim 2

   test:
      suffix: 3
      nsize: 3
      args: -mat_view -dim 3

TEST*/
