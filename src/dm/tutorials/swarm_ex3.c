
static char help[] = "Tests DMSwarm with DMShell\n\n";

#include <petscsf.h>
#include <petscdm.h>
#include <petscdmshell.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include <petsc/private/dmimpl.h>

PetscErrorCode _DMLocatePoints_DMDARegular_IS(DM dm, Vec pos, IS *iscell)
{
  PetscInt           p, n, bs, npoints, si, sj, milocal, mjlocal, mx, my;
  DM                 dmregular;
  PetscInt          *cellidx;
  const PetscScalar *coor;
  PetscReal          dx, dy;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(VecGetLocalSize(pos, &n));
  PetscCall(VecGetBlockSize(pos, &bs));
  npoints = n / bs;

  PetscCall(PetscMalloc1(npoints, &cellidx));
  PetscCall(DMGetApplicationContext(dm, &dmregular));
  PetscCall(DMDAGetCorners(dmregular, &si, &sj, NULL, &milocal, &mjlocal, NULL));
  PetscCall(DMDAGetInfo(dmregular, NULL, &mx, &my, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));

  dx = 2.0 / ((PetscReal)mx);
  dy = 2.0 / ((PetscReal)my);

  PetscCall(VecGetArrayRead(pos, &coor));
  for (p = 0; p < npoints; p++) {
    PetscReal coorx, coory;
    PetscInt  mi, mj;

    coorx = PetscRealPart(coor[2 * p]);
    coory = PetscRealPart(coor[2 * p + 1]);

    mi = (PetscInt)((coorx - (-1.0)) / dx);
    mj = (PetscInt)((coory - (-1.0)) / dy);

    cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;

    if ((mj >= sj) && (mj < sj + mjlocal)) {
      if ((mi >= si) && (mi < si + milocal)) cellidx[p] = (mi - si) + (mj - sj) * milocal;
    }
    if (coorx < -1.0) cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (coorx > 1.0) cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (coory < -1.0) cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (coory > 1.0) cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
  }
  PetscCall(VecRestoreArrayRead(pos, &coor));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, npoints, cellidx, PETSC_OWN_POINTER, iscell));
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocatePoints_DMDARegular(DM dm, Vec pos, DMPointLocationType ltype, PetscSF cellSF)
{
  IS              iscell;
  PetscSFNode    *cells;
  PetscInt        p, bs, npoints, nfound;
  const PetscInt *boxCells;

  PetscFunctionBegin;
  PetscCall(_DMLocatePoints_DMDARegular_IS(dm, pos, &iscell));
  PetscCall(VecGetLocalSize(pos, &npoints));
  PetscCall(VecGetBlockSize(pos, &bs));
  npoints = npoints / bs;

  PetscCall(PetscMalloc1(npoints, &cells));
  PetscCall(ISGetIndices(iscell, &boxCells));

  for (p = 0; p < npoints; p++) {
    cells[p].rank  = 0;
    cells[p].index = DMLOCATEPOINT_POINT_NOT_FOUND;
    cells[p].index = boxCells[p];
  }
  PetscCall(ISRestoreIndices(iscell, &boxCells));
  PetscCall(ISDestroy(&iscell));
  nfound = npoints;
  PetscCall(PetscSFSetGraph(cellSF, npoints, nfound, NULL, PETSC_OWN_POINTER, cells, PETSC_OWN_POINTER));
  PetscCall(ISDestroy(&iscell));
  PetscFunctionReturn(0);
}

PetscErrorCode DMGetNeighbors_DMDARegular(DM dm, PetscInt *nneighbors, const PetscMPIInt **neighbors)
{
  DM dmregular;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(dm, &dmregular));
  PetscCall(DMGetNeighbors(dmregular, nneighbors, neighbors));
  PetscFunctionReturn(0);
}

PetscErrorCode SwarmViewGP(DM dms, const char prefix[])
{
  PetscReal  *array;
  PetscInt   *iarray;
  PetscInt    npoints, p, bs;
  FILE       *fp;
  char        name[PETSC_MAX_PATH_LEN];
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN - 1, "%s-rank%d.gp", prefix, rank));
  fp = fopen(name, "w");
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open file %s", name);
  PetscCall(DMSwarmGetLocalSize(dms, &npoints));
  PetscCall(DMSwarmGetField(dms, DMSwarmPICField_coor, &bs, NULL, (void **)&array));
  PetscCall(DMSwarmGetField(dms, "itag", NULL, NULL, (void **)&iarray));
  for (p = 0; p < npoints; p++) fprintf(fp, "%+1.4e %+1.4e %1.4e\n", array[2 * p], array[2 * p + 1], (double)iarray[p]);
  PetscCall(DMSwarmRestoreField(dms, "itag", NULL, NULL, (void **)&iarray));
  PetscCall(DMSwarmRestoreField(dms, DMSwarmPICField_coor, &bs, NULL, (void **)&array));
  fclose(fp);
  PetscFunctionReturn(0);
}

/*
 Create a DMShell and attach a regularly spaced DMDA for point location
 Override methods for point location
*/
PetscErrorCode ex3_1(void)
{
  DM          dms, dmcell, dmregular;
  PetscMPIInt rank;
  PetscInt    p, bs, nlocal, overlap, mx, tk;
  PetscReal   dx;
  PetscReal  *array, dt;
  PetscInt   *iarray;
  PetscRandom rand;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Create a regularly spaced DMDA */
  mx      = 40;
  overlap = 0;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, mx, mx, PETSC_DECIDE, PETSC_DECIDE, 1, overlap, NULL, NULL, &dmregular));
  PetscCall(DMSetFromOptions(dmregular));
  PetscCall(DMSetUp(dmregular));

  dx = 2.0 / ((PetscReal)mx);
  PetscCall(DMDASetUniformCoordinates(dmregular, -1.0 + 0.5 * dx, 1.0 - 0.5 * dx, -1.0 + 0.5 * dx, 1.0 - 0.5 * dx, -1.0, 1.0));

  /* Create a DMShell for point location purposes */
  PetscCall(DMShellCreate(PETSC_COMM_WORLD, &dmcell));
  PetscCall(DMSetApplicationContext(dmcell, dmregular));
  dmcell->ops->locatepoints = DMLocatePoints_DMDARegular;
  dmcell->ops->getneighbors = DMGetNeighbors_DMDARegular;

  /* Create the swarm */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dms));
  PetscCall(DMSetType(dms, DMSWARM));
  PetscCall(DMSetDimension(dms, 2));
  PetscCall(PetscObjectSetName((PetscObject)dms, "Particles"));

  PetscCall(DMSwarmSetType(dms, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(dms, dmcell));

  PetscCall(DMSwarmRegisterPetscDatatypeField(dms, "itag", 1, PETSC_INT));
  PetscCall(DMSwarmFinalizeFieldRegister(dms));
  {
    PetscInt           si, sj, milocal, mjlocal;
    const PetscScalar *LA_coors;
    Vec                coors;
    PetscInt           cnt;

    PetscCall(DMDAGetCorners(dmregular, &si, &sj, NULL, &milocal, &mjlocal, NULL));
    PetscCall(DMGetCoordinates(dmregular, &coors));
    PetscCall(VecGetArrayRead(coors, &LA_coors));
    PetscCall(DMSwarmSetLocalSizes(dms, milocal * mjlocal, 4));
    PetscCall(DMSwarmGetLocalSize(dms, &nlocal));
    PetscCall(DMSwarmGetField(dms, DMSwarmPICField_coor, &bs, NULL, (void **)&array));
    cnt = 0;
    PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rand));
    PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
    for (p = 0; p < nlocal; p++) {
      PetscReal px, py, rx, ry, r2;

      PetscCall(PetscRandomGetValueReal(rand, &rx));
      PetscCall(PetscRandomGetValueReal(rand, &ry));

      px = PetscRealPart(LA_coors[2 * p + 0]) + 0.1 * rx * dx;
      py = PetscRealPart(LA_coors[2 * p + 1]) + 0.1 * ry * dx;

      r2 = px * px + py * py;
      if (r2 < 0.75 * 0.75) {
        array[bs * cnt + 0] = px;
        array[bs * cnt + 1] = py;
        cnt++;
      }
    }
    PetscCall(PetscRandomDestroy(&rand));
    PetscCall(DMSwarmRestoreField(dms, DMSwarmPICField_coor, &bs, NULL, (void **)&array));
    PetscCall(VecRestoreArrayRead(coors, &LA_coors));
    PetscCall(DMSwarmSetLocalSizes(dms, cnt, 4));

    PetscCall(DMSwarmGetLocalSize(dms, &nlocal));
    PetscCall(DMSwarmGetField(dms, "itag", &bs, NULL, (void **)&iarray));
    for (p = 0; p < nlocal; p++) iarray[p] = (PetscInt)rank;
    PetscCall(DMSwarmRestoreField(dms, "itag", &bs, NULL, (void **)&iarray));
  }

  PetscCall(DMView(dms, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(SwarmViewGP(dms, "step0"));

  dt = 0.1;
  for (tk = 1; tk < 20; tk++) {
    char prefix[PETSC_MAX_PATH_LEN];
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %" PetscInt_FMT " \n", tk));
    /* push points */
    PetscCall(DMSwarmGetLocalSize(dms, &nlocal));
    PetscCall(DMSwarmGetField(dms, DMSwarmPICField_coor, &bs, NULL, (void **)&array));
    for (p = 0; p < nlocal; p++) {
      PetscReal cx, cy, vx, vy;

      cx = array[2 * p];
      cy = array[2 * p + 1];
      vx = cy;
      vy = -cx;

      array[2 * p] += dt * vx;
      array[2 * p + 1] += dt * vy;
    }
    PetscCall(DMSwarmRestoreField(dms, DMSwarmPICField_coor, &bs, NULL, (void **)&array));

    /* migrate points */
    PetscCall(DMSwarmMigrate(dms, PETSC_TRUE));
    /* view points */
    PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN - 1, "step%" PetscInt_FMT, tk));
    /* should use the regular SwarmView() api, not one for a particular type */
    PetscCall(SwarmViewGP(dms, prefix));
  }
  PetscCall(DMDestroy(&dmregular));
  PetscCall(DMDestroy(&dmcell));
  PetscCall(DMDestroy(&dms));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(ex3_1());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: double !complex

   test:
      filter: grep -v atomic
      filter_output: grep -v atomic
TEST*/
