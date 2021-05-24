static char help[] = "Test VTK structured (.vts)  and rectilinear (.vtr) viewer support with multi-dof DMDAs.\n\
                      Supply the -namefields flag to test with field names.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

/* Helper function to name DMDA fields */
PetscErrorCode NameFields(DM da,PetscInt dof)
{
  PetscErrorCode ierr;
  PetscInt       c;

  PetscFunctionBeginUser;
  for (c=0; c<dof; ++c) {
    char fieldname[256];
    ierr = PetscSNPrintf(fieldname,sizeof(fieldname),"field_%D",c);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,c,fieldname);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  Write 3D DMDA vector with coordinates in VTK format
*/
PetscErrorCode test_3d(const char filename[],PetscInt dof,PetscBool namefields)
{
  MPI_Comm          comm = MPI_COMM_WORLD;
  const PetscInt    M=10,N=15,P=30,sw=1;
  const PetscScalar Lx=1.0,Ly=1.0,Lz=1.0;
  DM                da;
  Vec               v;
  PetscViewer       view;
  DMDALocalInfo     info;
  PetscScalar       ****va;
  PetscInt          i,j,k,c;
  PetscErrorCode    ierr;

  ierr = DMDACreate3d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,sw,NULL,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  if (namefields) {ierr = NameFields(da,dof);CHKERRQ(ierr);}

  ierr = DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,Lz);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&v);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,v,&va);CHKERRQ(ierr);
  for (k=info.zs; k<info.zs+info.zm; k++) {
    for (j=info.ys; j<info.ys+info.ym; j++) {
      for (i=info.xs; i<info.xs+info.xm; i++) {
        const PetscScalar x = (Lx*i)/M;
        const PetscScalar y = (Ly*j)/N;
        const PetscScalar z = (Lz*k)/P;
        for (c=0; c<dof; ++ c) {
        va[k][j][i][c] = PetscPowScalarInt(x-0.5*Lx,2)+PetscPowScalarInt(y-0.5*Ly,2)+PetscPowScalarInt(z-0.5*Lz,2) + 10.0*c;
        }
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da,v,&va);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = VecView(v,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  return 0;
}

/*
  Write 2D DMDA vector with coordinates in VTK format
*/
PetscErrorCode test_2d(const char filename[],PetscInt dof,PetscBool namefields)
{
  MPI_Comm          comm = MPI_COMM_WORLD;
  const PetscInt    M=10,N=20,sw=1;
  const PetscScalar Lx=1.0,Ly=1.0,Lz=1.0;
  DM                da;
  Vec               v;
  PetscViewer       view;
  DMDALocalInfo     info;
  PetscScalar       ***va;
  PetscInt          i,j,c;
  PetscErrorCode    ierr;

  ierr = DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, M,N,PETSC_DECIDE,PETSC_DECIDE,dof,sw,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  if (namefields) {ierr = NameFields(da,dof);CHKERRQ(ierr);}
  ierr = DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,Lz);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&v);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,v,&va);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      const PetscScalar x = (Lx*i)/M;
      const PetscScalar y = (Ly*j)/N;
      for (c=0; c<dof; ++c) {
        va[j][i][c] = PetscPowScalarInt(x-0.5*Lx,2)+PetscPowScalarInt(y-0.5*Ly,2) + 10.0*c;
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da,v,&va);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = VecView(v,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  return 0;
}

/*
  Write a scalar and a vector field from two compatible 3d DMDAs
*/
PetscErrorCode test_3d_compat(const char filename[],PetscInt dof,PetscBool namefields)
{
  MPI_Comm          comm = MPI_COMM_WORLD;
  const PetscInt    M=10,N=15,P=30,sw=1;
  const PetscScalar Lx=1.0,Ly=1.0,Lz=1.0;
  DM                da,daVector;
  Vec               v,vVector;
  PetscViewer       view;
  DMDALocalInfo     info;
  PetscScalar       ***va,****vVectora;
  PetscInt          i,j,k,c;
  PetscErrorCode    ierr;

  ierr = DMDACreate3d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,/* dof:*/1,sw,NULL,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  if (namefields) {ierr = NameFields(da,1);CHKERRQ(ierr);}

  ierr = DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,Lz);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDACreateCompatibleDMDA(da,dof,&daVector);CHKERRQ(ierr);
  if (namefields) {ierr = NameFields(daVector,dof);CHKERRQ(ierr);}
  ierr = DMCreateGlobalVector(da,&v);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(daVector,&vVector);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,v,&va);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(daVector,vVector,&vVectora);CHKERRQ(ierr);
  for (k=info.zs; k<info.zs+info.zm; k++) {
    for (j=info.ys; j<info.ys+info.ym; j++) {
      for (i=info.xs; i<info.xs+info.xm; i++) {
        const PetscScalar x = (Lx*i)/M;
        const PetscScalar y = (Ly*j)/N;
        const PetscScalar z = (Lz*k)/P;
        va[k][j][i] = PetscPowScalarInt(x-0.5*Lx,2)+PetscPowScalarInt(y-0.5*Ly,2)+PetscPowScalarInt(z-0.5*Lz,2);
        for (c=0; c<dof; ++c) {
          vVectora[k][j][i][c] = PetscPowScalarInt(x-0.5*Lx,2)+PetscPowScalarInt(y-0.5*Ly,2)+PetscPowScalarInt(z-0.5*Lz,2) + 10.0*c;
        }
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,v,&va);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(da,v,&vVectora);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = VecView(v,view);CHKERRQ(ierr);
  ierr = VecView(vVector,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&vVector);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = DMDestroy(&daVector);CHKERRQ(ierr);
  return 0;
}

/*
  Write a scalar and a vector field from two compatible 2d DMDAs
*/
PetscErrorCode test_2d_compat(const char filename[],PetscInt dof,PetscBool namefields)
{
  MPI_Comm          comm = MPI_COMM_WORLD;
  const PetscInt    M=10,N=20,sw=1;
  const PetscScalar Lx=1.0,Ly=1.0,Lz=1.0;
  DM                da, daVector;
  Vec               v,vVector;
  PetscViewer       view;
  DMDALocalInfo     info;
  PetscScalar       **va,***vVectora;
  PetscInt          i,j,c;
  PetscErrorCode    ierr;

  ierr = DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, M,N,PETSC_DECIDE,PETSC_DECIDE,/* dof:*/ 1,sw,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  if (namefields) {ierr = NameFields(da,1);CHKERRQ(ierr);}
  ierr = DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,Lz);CHKERRQ(ierr);
  ierr = DMDACreateCompatibleDMDA(da,dof,&daVector);CHKERRQ(ierr);
  if (namefields) {ierr = NameFields(daVector,dof);CHKERRQ(ierr);}
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&v);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(daVector,&vVector);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,v,&va);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(daVector,vVector,&vVectora);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      const PetscScalar x = (Lx*i)/M;
      const PetscScalar y = (Ly*j)/N;
      va[j][i] = PetscPowScalarInt(x-0.5*Lx,2)+PetscPowScalarInt(y-0.5*Ly,2);
      for (c=0; c<dof; ++c) {
        vVectora[j][i][c] = PetscPowScalarInt(x-0.5*Lx,2)+PetscPowScalarInt(y-0.5*Ly,2) + 10.0*c;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,v,&va);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(daVector,vVector,&vVectora);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = VecView(v,view);CHKERRQ(ierr);
  ierr = VecView(vVector,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&vVector);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = DMDestroy(&daVector);CHKERRQ(ierr);
  return 0;
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  PetscInt       dof;
  PetscBool      namefields;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  dof = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  namefields = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-namefields",&namefields,NULL);CHKERRQ(ierr);
  ierr = test_3d("3d.vtr",dof,namefields);CHKERRQ(ierr);
  ierr = test_2d("2d.vtr",dof,namefields);CHKERRQ(ierr);
  ierr = test_3d_compat("3d_compat.vtr",dof,namefields);CHKERRQ(ierr);
  ierr = test_2d_compat("2d_compat.vtr",dof,namefields);CHKERRQ(ierr);
  ierr = test_3d("3d.vts",dof,namefields);CHKERRQ(ierr);
  ierr = test_2d("2d.vts",dof,namefields);CHKERRQ(ierr);
  ierr = test_3d_compat("3d_compat.vts",dof,namefields);CHKERRQ(ierr);
  ierr = test_2d_compat("2d_compat.vts",dof,namefields);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: 1
      nsize: 2
      args: -dof 2

   test:
      suffix: 2
      nsize: 2
      args: -dof 2

   test:
      suffix: 3
      nsize: 2
      args: -dof 3

   test:
      suffix: 4
      nsize: 1
      args: -dof 2 -namefields

   test:
      suffix: 5
      nsize: 2
      args: -dof 4 -namefields

TEST*/
