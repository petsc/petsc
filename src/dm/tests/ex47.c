static char help[] = "Test VTK structured grid (.vts) viewer support\n\n";

#include <petscdm.h>
#include <petscdmda.h>

/*
  Write 3D DMDA vector with coordinates in VTK .vts format

*/
PetscErrorCode test_3d(const char filename[])
{
  MPI_Comm          comm = MPI_COMM_WORLD;
  const PetscInt    M = 10, N = 15, P = 30, dof = 1, sw = 1;
  const PetscScalar Lx = 1.0, Ly = 1.0, Lz = 1.0;
  DM                da;
  Vec               v;
  PetscViewer       view;
  DMDALocalInfo     info;
  PetscScalar    ***va;
  PetscInt          i, j, k;

  PetscCall(DMDACreate3d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, N, P, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, sw, NULL, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMDASetUniformCoordinates(da, 0.0, Lx, 0.0, Ly, 0.0, Lz));
  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(DMCreateGlobalVector(da, &v));
  PetscCall(DMDAVecGetArray(da, v, &va));
  for (k = info.zs; k < info.zs + info.zm; k++) {
    for (j = info.ys; j < info.ys + info.ym; j++) {
      for (i = info.xs; i < info.xs + info.xm; i++) {
        PetscScalar x = (Lx * i) / M;
        PetscScalar y = (Ly * j) / N;
        PetscScalar z = (Lz * k) / P;
        va[k][j][i]   = PetscPowScalarInt(x - 0.5 * Lx, 2) + PetscPowScalarInt(y - 0.5 * Ly, 2) + PetscPowScalarInt(z - 0.5 * Lz, 2);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da, v, &va));
  PetscCall(PetscViewerVTKOpen(comm, filename, FILE_MODE_WRITE, &view));
  PetscCall(VecView(v, view));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(VecDestroy(&v));
  PetscCall(DMDestroy(&da));
  return PETSC_SUCCESS;
}

/*
  Write 2D DMDA vector with coordinates in VTK .vts format

*/
PetscErrorCode test_2d(const char filename[])
{
  MPI_Comm          comm = MPI_COMM_WORLD;
  const PetscInt    M = 10, N = 20, dof = 1, sw = 1;
  const PetscScalar Lx = 1.0, Ly = 1.0, Lz = 1.0;
  DM                da;
  Vec               v;
  PetscViewer       view;
  DMDALocalInfo     info;
  PetscScalar     **va;
  PetscInt          i, j;

  PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, N, PETSC_DECIDE, PETSC_DECIDE, dof, sw, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, Lx, 0.0, Ly, 0.0, Lz));
  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(DMCreateGlobalVector(da, &v));
  PetscCall(DMDAVecGetArray(da, v, &va));
  for (j = info.ys; j < info.ys + info.ym; j++) {
    for (i = info.xs; i < info.xs + info.xm; i++) {
      PetscScalar x = (Lx * i) / M;
      PetscScalar y = (Ly * j) / N;
      va[j][i]      = PetscPowScalarInt(x - 0.5 * Lx, 2) + PetscPowScalarInt(y - 0.5 * Ly, 2);
    }
  }
  PetscCall(DMDAVecRestoreArray(da, v, &va));
  PetscCall(PetscViewerVTKOpen(comm, filename, FILE_MODE_WRITE, &view));
  PetscCall(VecView(v, view));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(VecDestroy(&v));
  PetscCall(DMDestroy(&da));
  return PETSC_SUCCESS;
}

/*
  Write 2D DMDA vector without coordinates in VTK .vts format

*/
PetscErrorCode test_2d_nocoord(const char filename[])
{
  MPI_Comm          comm = MPI_COMM_WORLD;
  const PetscInt    M = 10, N = 20, dof = 1, sw = 1;
  const PetscScalar Lx = 1.0, Ly = 1.0;
  DM                da;
  Vec               v;
  PetscViewer       view;
  DMDALocalInfo     info;
  PetscScalar     **va;
  PetscInt          i, j;

  PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, N, PETSC_DECIDE, PETSC_DECIDE, dof, sw, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(DMCreateGlobalVector(da, &v));
  PetscCall(DMDAVecGetArray(da, v, &va));
  for (j = info.ys; j < info.ys + info.ym; j++) {
    for (i = info.xs; i < info.xs + info.xm; i++) {
      PetscScalar x = (Lx * i) / M;
      PetscScalar y = (Ly * j) / N;
      va[j][i]      = PetscPowScalarInt(x - 0.5 * Lx, 2) + PetscPowScalarInt(y - 0.5 * Ly, 2);
    }
  }
  PetscCall(DMDAVecRestoreArray(da, v, &va));
  PetscCall(PetscViewerVTKOpen(comm, filename, FILE_MODE_WRITE, &view));
  PetscCall(VecView(v, view));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(VecDestroy(&v));
  PetscCall(DMDestroy(&da));
  return PETSC_SUCCESS;
}

/*
  Write 3D DMDA vector without coordinates in VTK .vts format

*/
PetscErrorCode test_3d_nocoord(const char filename[])
{
  MPI_Comm          comm = MPI_COMM_WORLD;
  const PetscInt    M = 10, N = 20, P = 30, dof = 1, sw = 1;
  const PetscScalar Lx = 1.0, Ly = 1.0, Lz = 1.0;
  DM                da;
  Vec               v;
  PetscViewer       view;
  DMDALocalInfo     info;
  PetscScalar    ***va;
  PetscInt          i, j, k;

  PetscCall(DMDACreate3d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, N, P, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, sw, NULL, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(DMCreateGlobalVector(da, &v));
  PetscCall(DMDAVecGetArray(da, v, &va));
  for (k = info.zs; k < info.zs + info.zm; k++) {
    for (j = info.ys; j < info.ys + info.ym; j++) {
      for (i = info.xs; i < info.xs + info.xm; i++) {
        PetscScalar x = (Lx * i) / M;
        PetscScalar y = (Ly * j) / N;
        PetscScalar z = (Lz * k) / P;
        va[k][j][i]   = PetscPowScalarInt(x - 0.5 * Lx, 2) + PetscPowScalarInt(y - 0.5 * Ly, 2) + PetscPowScalarInt(z - 0.5 * Lz, 2);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da, v, &va));
  PetscCall(PetscViewerVTKOpen(comm, filename, FILE_MODE_WRITE, &view));
  PetscCall(VecView(v, view));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(VecDestroy(&v));
  PetscCall(DMDestroy(&da));
  return PETSC_SUCCESS;
}

int main(int argc, char *argv[])
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  PetscCall(test_3d("3d.vts"));
  PetscCall(test_2d("2d.vts"));
  PetscCall(test_2d_nocoord("2d_nocoord.vts"));
  PetscCall(test_3d_nocoord("3d_nocoord.vts"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      nsize: 2

TEST*/
