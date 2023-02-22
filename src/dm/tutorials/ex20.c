
static char help[] = "DMSwarm-PIC demonstator of inserting points into a cell DM \n\
Options: \n\
-mode {0,1} : 0 ==> DMDA, 1 ==> DMPLEX cell DM \n\
-dim {2,3}  : spatial dimension\n";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>

PetscErrorCode pic_insert_DMDA(PetscInt dim)
{
  DM        celldm = NULL, swarm;
  PetscInt  dof, stencil_width;
  PetscReal min[3], max[3];
  PetscInt  ndir[3];

  PetscFunctionBegin;
  /* Create the background cell DM */
  dof           = 1;
  stencil_width = 1;
  if (dim == 2) PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 25, 13, PETSC_DECIDE, PETSC_DECIDE, dof, stencil_width, NULL, NULL, &celldm));
  if (dim == 3) PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 25, 13, 19, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, stencil_width, NULL, NULL, NULL, &celldm));

  PetscCall(DMDASetElementType(celldm, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(celldm));
  PetscCall(DMSetUp(celldm));

  PetscCall(DMDASetUniformCoordinates(celldm, 0.0, 2.0, 0.0, 1.0, 0.0, 1.5));

  /* Create the DMSwarm */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &swarm));
  PetscCall(PetscObjectSetName((PetscObject)swarm, "Swarm"));
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));

  /* Configure swarm to be of type PIC */
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(swarm, celldm));

  /* Register two scalar fields within the DMSwarm */
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm, "viscosity", 1, PETSC_DOUBLE));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm, "density", 1, PETSC_DOUBLE));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));

  /* Set initial local sizes of the DMSwarm with a buffer length of zero */
  PetscCall(DMSwarmSetLocalSizes(swarm, 4, 0));

  /* Insert swarm coordinates cell-wise */
  PetscCall(DMSwarmInsertPointsUsingCellDM(swarm, DMSWARMPIC_LAYOUT_REGULAR, 3));
  min[0]  = 0.5;
  max[0]  = 0.7;
  min[1]  = 0.5;
  max[1]  = 0.8;
  min[2]  = 0.5;
  max[2]  = 0.9;
  ndir[0] = ndir[1] = ndir[2] = 30;
  PetscCall(DMSwarmSetPointsUniformCoordinates(swarm, min, max, ndir, ADD_VALUES));

  /* This should be dispatched from a regular DMView() */
  PetscCall(DMSwarmViewXDMF(swarm, "ex20.xmf"));
  PetscCall(DMView(celldm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMView(swarm, PETSC_VIEWER_STDOUT_WORLD));

  {
    PetscInt    npoints, *list;
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(DMSwarmSortGetAccess(swarm));
    PetscCall(DMSwarmSortGetNumberOfPointsPerCell(swarm, 0, &npoints));
    PetscCall(DMSwarmSortGetPointsPerCell(swarm, rank, &npoints, &list));
    PetscCall(PetscFree(list));
    PetscCall(DMSwarmSortRestoreAccess(swarm));
  }
  PetscCall(DMSwarmMigrate(swarm, PETSC_FALSE));
  PetscCall(DMDestroy(&celldm));
  PetscCall(DMDestroy(&swarm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode pic_insert_DMPLEX_with_cell_list(PetscInt dim)
{
  DM          celldm = NULL, swarm, distributedMesh = NULL;
  const char *fieldnames[] = {"viscosity"};

  PetscFunctionBegin;
  /* Create the background cell DM */
  if (dim == 2) {
    PetscInt   cells_per_dim[2], nx[2];
    PetscInt   n_tricells;
    PetscInt   n_trivert;
    PetscInt  *tricells;
    PetscReal *trivert, dx, dy;
    PetscInt   ii, jj, cnt;

    cells_per_dim[0] = 4;
    cells_per_dim[1] = 4;
    n_tricells       = cells_per_dim[0] * cells_per_dim[1] * 2;
    nx[0]            = cells_per_dim[0] + 1;
    nx[1]            = cells_per_dim[1] + 1;
    n_trivert        = nx[0] * nx[1];

    PetscCall(PetscMalloc1(n_tricells * 3, &tricells));
    PetscCall(PetscMalloc1(nx[0] * nx[1] * 2, &trivert));

    /* verts */
    cnt = 0;
    dx  = 2.0 / ((PetscReal)cells_per_dim[0]);
    dy  = 1.0 / ((PetscReal)cells_per_dim[1]);
    for (jj = 0; jj < nx[1]; jj++) {
      for (ii = 0; ii < nx[0]; ii++) {
        trivert[2 * cnt + 0] = 0.0 + ii * dx;
        trivert[2 * cnt + 1] = 0.0 + jj * dy;
        cnt++;
      }
    }

    /* connectivity */
    cnt = 0;
    for (jj = 0; jj < cells_per_dim[1]; jj++) {
      for (ii = 0; ii < cells_per_dim[0]; ii++) {
        PetscInt idx, idx0, idx1, idx2, idx3;

        idx  = (ii) + (jj)*nx[0];
        idx0 = idx;
        idx1 = idx0 + 1;
        idx2 = idx1 + nx[0];
        idx3 = idx0 + nx[0];

        tricells[3 * cnt + 0] = idx0;
        tricells[3 * cnt + 1] = idx1;
        tricells[3 * cnt + 2] = idx2;
        cnt++;

        tricells[3 * cnt + 0] = idx0;
        tricells[3 * cnt + 1] = idx2;
        tricells[3 * cnt + 2] = idx3;
        cnt++;
      }
    }
    PetscCall(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, n_tricells, n_trivert, 3, PETSC_TRUE, tricells, dim, trivert, &celldm));
    PetscCall(PetscFree(trivert));
    PetscCall(PetscFree(tricells));
  }
  PetscCheck(dim != 3, PETSC_COMM_WORLD, PETSC_ERR_SUP, "Only 2D PLEX example supported");

  /* Distribute mesh over processes */
  PetscCall(DMPlexDistribute(celldm, 0, NULL, &distributedMesh));
  if (distributedMesh) {
    PetscCall(DMDestroy(&celldm));
    celldm = distributedMesh;
  }
  PetscCall(PetscObjectSetName((PetscObject)celldm, "Cells"));
  PetscCall(DMSetFromOptions(celldm));
  {
    PetscInt     numComp[] = {1};
    PetscInt     numDof[]  = {1, 0, 0}; /* vert, edge, cell */
    PetscInt     numBC     = 0;
    PetscSection section;

    PetscCall(DMPlexCreateSection(celldm, NULL, numComp, numDof, numBC, NULL, NULL, NULL, NULL, &section));
    PetscCall(DMSetLocalSection(celldm, section));
    PetscCall(PetscSectionDestroy(&section));
  }
  PetscCall(DMSetUp(celldm));
  {
    PetscViewer viewer;

    PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
    PetscCall(PetscViewerSetType(viewer, PETSCVIEWERVTK));
    PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
    PetscCall(PetscViewerFileSetName(viewer, "ex20plex.vtk"));
    PetscCall(DMView(celldm, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Create the DMSwarm */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &swarm));
  PetscCall(PetscObjectSetName((PetscObject)swarm, "Swarm"));
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));

  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(swarm, celldm));

  /* Register two scalar fields within the DMSwarm */
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm, "viscosity", 1, PETSC_DOUBLE));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm, "density", 1, PETSC_DOUBLE));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));

  /* Set initial local sizes of the DMSwarm with a buffer length of zero */
  PetscCall(DMSwarmSetLocalSizes(swarm, 4, 0));

  /* Insert swarm coordinates cell-wise */
  PetscCall(DMSwarmInsertPointsUsingCellDM(swarm, DMSWARMPIC_LAYOUT_SUBDIVISION, 2));
  PetscCall(DMSwarmViewFieldsXDMF(swarm, "ex20.xmf", 1, fieldnames));
  PetscCall(DMView(celldm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMView(swarm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&celldm));
  PetscCall(DMDestroy(&swarm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode pic_insert_DMPLEX(PetscBool is_simplex, PetscInt dim)
{
  DM          celldm, swarm, distributedMesh = NULL;
  const char *fieldnames[] = {"viscosity", "DMSwarm_rank"};

  PetscFunctionBegin;

  /* Create the background cell DM */
  {
    PetscInt faces[3] = {4, 2, 4};
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, is_simplex, faces, NULL, NULL, NULL, PETSC_TRUE, &celldm));
  }

  /* Distribute mesh over processes */
  PetscCall(DMPlexDistribute(celldm, 0, NULL, &distributedMesh));
  if (distributedMesh) {
    PetscCall(DMDestroy(&celldm));
    celldm = distributedMesh;
  }
  PetscCall(PetscObjectSetName((PetscObject)celldm, "Cells"));
  PetscCall(DMSetFromOptions(celldm));
  {
    PetscInt     numComp[] = {1};
    PetscInt     numDof[]  = {1, 0, 0}; /* vert, edge, cell */
    PetscInt     numBC     = 0;
    PetscSection section;

    PetscCall(DMPlexCreateSection(celldm, NULL, numComp, numDof, numBC, NULL, NULL, NULL, NULL, &section));
    PetscCall(DMSetLocalSection(celldm, section));
    PetscCall(PetscSectionDestroy(&section));
  }
  PetscCall(DMSetUp(celldm));
  {
    PetscViewer viewer;

    PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
    PetscCall(PetscViewerSetType(viewer, PETSCVIEWERVTK));
    PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
    PetscCall(PetscViewerFileSetName(viewer, "ex20plex.vtk"));
    PetscCall(DMView(celldm, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(DMCreate(PETSC_COMM_WORLD, &swarm));
  PetscCall(PetscObjectSetName((PetscObject)swarm, "Swarm"));
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, dim));

  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(swarm, celldm));

  /* Register two scalar fields within the DMSwarm */
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm, "viscosity", 1, PETSC_DOUBLE));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm, "density", 1, PETSC_DOUBLE));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));

  /* Set initial local sizes of the DMSwarm with a buffer length of zero */
  PetscCall(DMSwarmSetLocalSizes(swarm, 4, 0));

  /* Insert swarm coordinates cell-wise */
  PetscCall(DMSwarmInsertPointsUsingCellDM(swarm, DMSWARMPIC_LAYOUT_GAUSS, 3));
  PetscCall(DMSwarmViewFieldsXDMF(swarm, "ex20.xmf", 2, fieldnames));
  PetscCall(DMView(celldm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMView(swarm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&celldm));
  PetscCall(DMDestroy(&swarm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscInt mode = 0;
  PetscInt dim  = 2;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mode", &mode, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  switch (mode) {
  case 0:
    PetscCall(pic_insert_DMDA(dim));
    break;
  case 1:
    /* tri / tet */
    PetscCall(pic_insert_DMPLEX(PETSC_TRUE, dim));
    break;
  case 2:
    /* quad / hex */
    PetscCall(pic_insert_DMPLEX(PETSC_FALSE, dim));
    break;
  default:
    PetscCall(pic_insert_DMDA(dim));
    break;
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args:
      requires: !complex double
      filter: grep -v atomic
      filter_output: grep -v atomic

   test:
      suffix: 2
      requires: triangle double !complex
      args: -mode 1
      filter: grep -v atomic
      filter_output: grep -v atomic

TEST*/
