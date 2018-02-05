#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode DMPlexCreateHDF5VizFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *newdm)
{
  PetscViewer     viewer;
  DM              dm;
  Vec             coordinates;
  IS              cells;
  PetscInt        dim, spatialDim, N, numCells, numVertices, numCorners, bs;
  //hid_t           fileId;
  PetscMPIInt     rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERHDF5);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);

  /* Read topology */
  ierr = PetscViewerHDF5ReadAttribute(viewer, "/viz/topology/cells", "cell_dim", PETSC_INT, (void *) &dim);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "/viz/topology/cells", "cell_corners", PETSC_INT, (void *) &numCorners);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/viz/topology");CHKERRQ(ierr);
  ierr = ISCreate(comm, &cells);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) cells, "cells");CHKERRQ(ierr);
  ierr = ISLoad(cells, viewer);CHKERRQ(ierr);
  ierr = ISGetLocalSize(cells, &numCells);CHKERRQ(ierr);
  ierr = ISGetBlockSize(cells, &bs);CHKERRQ(ierr);
  if (bs != numCorners) SETERRQ(comm, PETSC_ERR_FILE_UNEXPECTED, "cell_corners and 2-dimension of cells not equal");
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  numCells /= numCorners;

  /* Read geometry */
  ierr = PetscViewerHDF5PushGroup(viewer, "/geometry");CHKERRQ(ierr);
  ierr = VecCreate(comm, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "vertices");CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadSizes(viewer, "vertices", &spatialDim, &N);CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, PETSC_DECIDE, N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, spatialDim);CHKERRQ(ierr);
  ierr = VecLoad(coordinates, viewer);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &numVertices);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &spatialDim);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  numVertices /= spatialDim;

  /* TODO: check that maximum in cells is <= number of vertices */

  {
    const PetscReal *coordinates_arr;
    const PetscInt *cells_arr;

    ierr = VecGetArrayRead(coordinates, &coordinates_arr);CHKERRQ(ierr);
    ierr = ISGetIndices(cells, &cells_arr);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellListParallel(comm, dim, numCells, numVertices, numCorners, interpolate, cells_arr, spatialDim, coordinates_arr, NULL, &dm);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(coordinates, &coordinates_arr);CHKERRQ(ierr);
    ierr = ISRestoreIndices(cells, &cells_arr);CHKERRQ(ierr);

  }
  ierr = ISDestroy(&cells);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);

  /* scale coordinates - unlike in DMPlexLoad_HDF5_Internal, this can only be done after DM is populated */
  {
    PetscReal lengthScale;

    ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
    ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
    ierr = VecScale(coordinates, 1.0/lengthScale);CHKERRQ(ierr);
  }

  /* TODO: read labels */

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  *newdm = dm;
  PetscFunctionReturn(0);
}
#endif
