/* -*- Mode: C++; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */

static char help[] = "Test VTK Rectilinear grid (.vtr) viewer support\n\n";

#include <petscdm.h>
#include <petscdmda.h>

/*
  Write 3D DMDA vector with coordinates in VTK VTR format

*/
PetscErrorCode test_3d(const char filename[])
{
  MPI_Comm          comm = MPI_COMM_WORLD;
  const PetscInt    M=10, N=15, P=30, dof=1, sw=1;
  const PetscScalar Lx=1.0, Ly=1.0, Lz=1.0;
  DM                da;
  Vec               v;
  PetscViewer       view;
  DMDALocalInfo     info;
  PetscScalar       ***va;
  PetscInt          i,j,k;
  PetscErrorCode    ierr;

  ierr = DMDACreate3d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,sw,NULL,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  ierr = DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,Lz);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&v);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,v,&va);CHKERRQ(ierr);
  for (k=info.zs; k<info.zs+info.zm; k++) {
    for (j=info.ys; j<info.ys+info.ym; j++) {
      for (i=info.xs; i<info.xs+info.xm; i++) {
        PetscScalar x = (Lx*i)/M;
        PetscScalar y = (Ly*j)/N;
        PetscScalar z = (Lz*k)/P;
        va[k][j][i] = PetscPowScalarInt(x-0.5*Lx,2)+PetscPowScalarInt(y-0.5*Ly,2)+PetscPowScalarInt(z-0.5*Lz,2);
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,v,&va);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = VecView(v,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  return 0;
}


/*
  Write 2D DMDA vector with coordinates in VTK VTR format

*/
PetscErrorCode test_2d(const char filename[])
{
  MPI_Comm          comm = MPI_COMM_WORLD;
  const PetscInt    M=10, N=20, dof=1, sw=1;
  const PetscScalar Lx=1.0, Ly=1.0, Lz=1.0;
  DM                da;
  Vec               v;
  PetscViewer       view;
  DMDALocalInfo     info;
  PetscScalar       **va;
  PetscInt          i,j;
  PetscErrorCode    ierr;

  ierr = DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, M,N,PETSC_DECIDE,PETSC_DECIDE,dof,sw,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,Lz);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&v);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,v,&va);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      PetscScalar x = (Lx*i)/M;
      PetscScalar y = (Ly*j)/N;
      va[j][i] = PetscPowScalarInt(x-0.5*Lx,2)+PetscPowScalarInt(y-0.5*Ly,2);
    }
  }
  ierr = DMDAVecRestoreArray(da,v,&va);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = VecView(v,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  return 0;
}


/*
  Write 2D DMDA vector without coordinates in VTK VTR format

*/
PetscErrorCode test_2d_nocoord(const char filename[])
{
  MPI_Comm          comm = MPI_COMM_WORLD;
  const PetscInt    M=10, N=20, dof=1, sw=1;
  const PetscScalar Lx=1.0, Ly=1.0;
  DM                da;
  Vec               v;
  PetscViewer       view;
  DMDALocalInfo     info;
  PetscScalar       **va;
  PetscInt          i,j;
  PetscErrorCode    ierr;

  ierr = DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,sw,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&v);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,v,&va);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      PetscScalar x = (Lx*i)/M;
      PetscScalar y = (Ly*j)/N;
      va[j][i] = PetscPowScalarInt(x-0.5*Lx,2)+PetscPowScalarInt(y-0.5*Ly,2);
    }
  }
  ierr = DMDAVecRestoreArray(da,v,&va);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = VecView(v,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  return 0;
}


/*
  Write 3D DMDA vector without coordinates in VTK VTR format

*/
PetscErrorCode test_3d_nocoord(const char filename[])
{
  MPI_Comm          comm = MPI_COMM_WORLD;
  const PetscInt    M=10, N=20, P=30, dof=1, sw=1;
  const PetscScalar Lx=1.0, Ly=1.0, Lz=1.0;
  DM                da;
  Vec               v;
  PetscViewer       view;
  DMDALocalInfo     info;
  PetscScalar       ***va;
  PetscInt          i,j,k;
  PetscErrorCode    ierr;

  ierr = DMDACreate3d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,sw,NULL,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&v);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,v,&va);CHKERRQ(ierr);
  for (k=info.zs; k<info.zs+info.zm; k++) {
    for (j=info.ys; j<info.ys+info.ym; j++) {
      for (i=info.xs; i<info.xs+info.xm; i++) {
        PetscScalar x = (Lx*i)/M;
        PetscScalar y = (Ly*j)/N;
        PetscScalar z = (Lz*k)/P;
        va[k][j][i] = PetscPowScalarInt(x-0.5*Lx,2)+PetscPowScalarInt(y-0.5*Ly,2)+PetscPowScalarInt(z-0.5*Lz,2);
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,v,&va);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = VecView(v,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  return 0;
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  
  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  ierr = test_3d("3d.vtr");CHKERRQ(ierr);
  ierr = test_2d("2d.vtr");CHKERRQ(ierr);
  ierr = test_2d_nocoord("2d_nocoord.vtr");CHKERRQ(ierr);
  ierr = test_3d_nocoord("3d_nocoord.vtr");CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   build:
      requires: !complex

   test:
      nsize: 2

TEST*/
