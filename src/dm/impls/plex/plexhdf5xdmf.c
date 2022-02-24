#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsclayouthdf5.h>

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode SplitPath_Private(char path[], char name[])
{
  char *tmp;

  PetscFunctionBegin;
  CHKERRQ(PetscStrrchr(path,'/',&tmp));
  CHKERRQ(PetscStrcpy(name,tmp));
  if (tmp != path) {
    /* '/' found, name is substring of path after last occurence of '/'. */
    /* Trim the '/name' part from path just by inserting null character. */
    tmp--;
    *tmp = '\0';
  } else {
    /* '/' not found, name = path, path = "/". */
    CHKERRQ(PetscStrcpy(path,"/"));
  }
  PetscFunctionReturn(0);
}

/*
  - invert (involute) cells of some types according to XDMF/VTK numbering of vertices in a cells
  - cell type is identified using the number of vertices
*/
static PetscErrorCode DMPlexInvertCells_XDMF_Private(DM dm)
{
  PetscInt       dim, *cones, cHeight, cStart, cEnd, p;
  PetscSection   cs;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  if (dim != 3) PetscFunctionReturn(0);
  CHKERRQ(DMPlexGetCones(dm, &cones));
  CHKERRQ(DMPlexGetConeSection(dm, &cs));
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cHeight));
  CHKERRQ(DMPlexGetHeightStratum(dm, cHeight, &cStart, &cEnd));
  for (p=cStart; p<cEnd; p++) {
    PetscInt numCorners, o;

    CHKERRQ(PetscSectionGetDof(cs, p, &numCorners));
    CHKERRQ(PetscSectionGetOffset(cs, p, &o));
    switch (numCorners) {
      case 4: CHKERRQ(DMPlexInvertCell(DM_POLYTOPE_TETRAHEDRON,&cones[o])); break;
      case 6: CHKERRQ(DMPlexInvertCell(DM_POLYTOPE_TRI_PRISM,&cones[o])); break;
      case 8: CHKERRQ(DMPlexInvertCell(DM_POLYTOPE_HEXAHEDRON,&cones[o])); break;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexLoad_HDF5_Xdmf_Internal(DM dm, PetscViewer viewer)
{
  Vec             coordinates;
  IS              cells;
  PetscInt        spatialDim, topoDim = -1, numCells, numVertices, NVertices, numCorners;
  PetscMPIInt     rank;
  MPI_Comm        comm;
  PetscErrorCode  ierr;
  char            topo_path[PETSC_MAX_PATH_LEN]="/viz/topology/cells", topo_name[PETSC_MAX_PATH_LEN];
  char            geom_path[PETSC_MAX_PATH_LEN]="/geometry/vertices",  geom_name[PETSC_MAX_PATH_LEN];
  PetscBool       seq = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)dm),((PetscObject)dm)->prefix,"DMPlex HDF5/XDMF Loader Options","PetscViewer");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-dm_plex_hdf5_topology_path","HDF5 path of topology dataset",NULL,topo_path,topo_path,sizeof(topo_path),NULL));
  CHKERRQ(PetscOptionsString("-dm_plex_hdf5_geometry_path","HDF5 path to geometry dataset",NULL,geom_path,geom_path,sizeof(geom_path),NULL));
  CHKERRQ(PetscOptionsBool("-dm_plex_hdf5_force_sequential","force sequential loading",NULL,seq,&seq,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(SplitPath_Private(topo_path, topo_name));
  CHKERRQ(SplitPath_Private(geom_path, geom_name));
  CHKERRQ(PetscInfo(dm, "Topology group %s, name %s\n", topo_path, topo_name));
  CHKERRQ(PetscInfo(dm, "Geometry group %s, name %s\n", geom_path, geom_name));

  /* Read topology */
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, topo_path));
  CHKERRQ(ISCreate(comm, &cells));
  CHKERRQ(PetscObjectSetName((PetscObject) cells, topo_name));
  if (seq) {
    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, topo_name, NULL, &numCells));
    CHKERRQ(PetscLayoutSetSize(cells->map, numCells));
    numCells = rank == 0 ? numCells : 0;
    CHKERRQ(PetscLayoutSetLocalSize(cells->map, numCells));
  }
  CHKERRQ(ISLoad(cells, viewer));
  CHKERRQ(ISGetLocalSize(cells, &numCells));
  CHKERRQ(ISGetBlockSize(cells, &numCorners));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, topo_name, "cell_dim", PETSC_INT, &topoDim, &topoDim));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  numCells /= numCorners;

  /* Read geometry */
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, geom_path));
  CHKERRQ(VecCreate(comm, &coordinates));
  CHKERRQ(PetscObjectSetName((PetscObject) coordinates, geom_name));
  if (seq) {
    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, geom_name, NULL, &numVertices));
    CHKERRQ(PetscLayoutSetSize(coordinates->map, numVertices));
    numVertices = rank == 0 ? numVertices : 0;
    CHKERRQ(PetscLayoutSetLocalSize(coordinates->map, numVertices));
  }
  CHKERRQ(VecLoad(coordinates, viewer));
  CHKERRQ(VecGetLocalSize(coordinates, &numVertices));
  CHKERRQ(VecGetSize(coordinates, &NVertices));
  CHKERRQ(VecGetBlockSize(coordinates, &spatialDim));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  numVertices /= spatialDim;
  NVertices /= spatialDim;

  CHKERRQ(PetscInfo(NULL, "Loaded mesh dimensions: numCells %D numCorners %D numVertices %D spatialDim %D\n", numCells, numCorners, numVertices, spatialDim));
  {
    const PetscScalar *coordinates_arr;
    PetscReal         *coordinates_arr_real;
    const PetscInt    *cells_arr;
    PetscSF           sfVert = NULL;
    PetscInt          i;

    CHKERRQ(VecGetArrayRead(coordinates, &coordinates_arr));
    CHKERRQ(ISGetIndices(cells, &cells_arr));

    if (PetscDefined(USE_COMPLEX)) {
      /* convert to real numbers if PetscScalar is complex */
      /*TODO More systematic would be to change all the function arguments to PetscScalar */
      CHKERRQ(PetscMalloc1(numVertices*spatialDim, &coordinates_arr_real));
      for (i = 0; i < numVertices*spatialDim; ++i) {
        coordinates_arr_real[i] = PetscRealPart(coordinates_arr[i]);
        if (PetscUnlikelyDebug(PetscImaginaryPart(coordinates_arr[i]))) {
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vector of coordinates contains complex numbers but only real vectors are currently supported.");
        }
      }
    } else coordinates_arr_real = (PetscReal*)coordinates_arr;

    CHKERRQ(DMSetDimension(dm, topoDim < 0 ? spatialDim : topoDim));
    CHKERRQ(DMPlexBuildFromCellListParallel(dm, numCells, numVertices, NVertices, numCorners, cells_arr, &sfVert, NULL));
    CHKERRQ(DMPlexInvertCells_XDMF_Private(dm));
    CHKERRQ(DMPlexBuildCoordinatesFromCellListParallel(dm, spatialDim, sfVert, coordinates_arr_real));
    CHKERRQ(VecRestoreArrayRead(coordinates, &coordinates_arr));
    CHKERRQ(ISRestoreIndices(cells, &cells_arr));
    CHKERRQ(PetscSFDestroy(&sfVert));
    if (PetscDefined(USE_COMPLEX)) CHKERRQ(PetscFree(coordinates_arr_real));
  }
  CHKERRQ(ISDestroy(&cells));
  CHKERRQ(VecDestroy(&coordinates));

  /* scale coordinates - unlike in DMPlexLoad_HDF5_Internal, this can only be done after DM is populated */
  {
    PetscReal lengthScale;

    CHKERRQ(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
    CHKERRQ(DMGetCoordinates(dm, &coordinates));
    CHKERRQ(VecScale(coordinates, 1.0/lengthScale));
  }

  /* Read Labels */
  /* TODO: this probably does not work as elements get permuted */
  /* CHKERRQ(DMPlexLabelsLoad_HDF5_Internal(dm, viewer)); */
  PetscFunctionReturn(0);
}
#endif
