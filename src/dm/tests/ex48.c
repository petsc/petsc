static char help[] = "Test VTK structured (.vts)  and rectilinear (.vtr) viewer support with multi-dof DMDAs.\n\
                      Supply the -namefields flag to test with field names.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

/* Helper function to name DMDA fields */
PetscErrorCode NameFields(DM da,PetscInt dof)
{
  PetscInt       c;

  PetscFunctionBeginUser;
  for (c=0; c<dof; ++c) {
    char fieldname[256];
    PetscCall(PetscSNPrintf(fieldname,sizeof(fieldname),"field_%" PetscInt_FMT,c));
    PetscCall(DMDASetFieldName(da,c,fieldname));
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

  PetscCall(DMDACreate3d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,sw,NULL,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  if (namefields) PetscCall(NameFields(da,dof));

  PetscCall(DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,Lz));
  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMCreateGlobalVector(da,&v));
  PetscCall(DMDAVecGetArrayDOF(da,v,&va));
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
  PetscCall(DMDAVecRestoreArrayDOF(da,v,&va));
  PetscCall(PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&view));
  PetscCall(VecView(v,view));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(VecDestroy(&v));
  PetscCall(DMDestroy(&da));
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

  PetscCall(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, M,N,PETSC_DECIDE,PETSC_DECIDE,dof,sw,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  if (namefields) PetscCall(NameFields(da,dof));
  PetscCall(DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,Lz));
  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMCreateGlobalVector(da,&v));
  PetscCall(DMDAVecGetArrayDOF(da,v,&va));
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      const PetscScalar x = (Lx*i)/M;
      const PetscScalar y = (Ly*j)/N;
      for (c=0; c<dof; ++c) {
        va[j][i][c] = PetscPowScalarInt(x-0.5*Lx,2)+PetscPowScalarInt(y-0.5*Ly,2) + 10.0*c;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(da,v,&va));
  PetscCall(PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&view));
  PetscCall(VecView(v,view));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(VecDestroy(&v));
  PetscCall(DMDestroy(&da));
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

  PetscCall(DMDACreate3d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,/* dof:*/1,sw,NULL,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  if (namefields) PetscCall(NameFields(da,1));

  PetscCall(DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,Lz));
  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMDACreateCompatibleDMDA(da,dof,&daVector));
  if (namefields) PetscCall(NameFields(daVector,dof));
  PetscCall(DMCreateGlobalVector(da,&v));
  PetscCall(DMCreateGlobalVector(daVector,&vVector));
  PetscCall(DMDAVecGetArray(da,v,&va));
  PetscCall(DMDAVecGetArrayDOF(daVector,vVector,&vVectora));
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
  PetscCall(DMDAVecRestoreArray(da,v,&va));
  PetscCall(DMDAVecRestoreArrayDOF(da,v,&vVectora));
  PetscCall(PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&view));
  PetscCall(VecView(v,view));
  PetscCall(VecView(vVector,view));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&vVector));
  PetscCall(DMDestroy(&da));
  PetscCall(DMDestroy(&daVector));
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

  PetscCall(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, M,N,PETSC_DECIDE,PETSC_DECIDE,/* dof:*/ 1,sw,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  if (namefields) PetscCall(NameFields(da,1));
  PetscCall(DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,Lz));
  PetscCall(DMDACreateCompatibleDMDA(da,dof,&daVector));
  if (namefields) PetscCall(NameFields(daVector,dof));
  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMCreateGlobalVector(da,&v));
  PetscCall(DMCreateGlobalVector(daVector,&vVector));
  PetscCall(DMDAVecGetArray(da,v,&va));
  PetscCall(DMDAVecGetArrayDOF(daVector,vVector,&vVectora));
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
  PetscCall(DMDAVecRestoreArray(da,v,&va));
  PetscCall(DMDAVecRestoreArrayDOF(daVector,vVector,&vVectora));
  PetscCall(PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&view));
  PetscCall(VecView(v,view));
  PetscCall(VecView(vVector,view));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&vVector));
  PetscCall(DMDestroy(&da));
  PetscCall(DMDestroy(&daVector));
  return 0;
}

int main(int argc, char *argv[])
{
  PetscInt       dof;
  PetscBool      namefields;

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  dof = 2;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  namefields = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-namefields",&namefields,NULL));
  PetscCall(test_3d("3d.vtr",dof,namefields));
  PetscCall(test_2d("2d.vtr",dof,namefields));
  PetscCall(test_3d_compat("3d_compat.vtr",dof,namefields));
  PetscCall(test_2d_compat("2d_compat.vtr",dof,namefields));
  PetscCall(test_3d("3d.vts",dof,namefields));
  PetscCall(test_2d("2d.vts",dof,namefields));
  PetscCall(test_3d_compat("3d_compat.vts",dof,namefields));
  PetscCall(test_2d_compat("2d_compat.vts",dof,namefields));
  PetscCall(PetscFinalize());
  return 0;
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
