#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsclayouthdf5.h>

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode SplitPath_Private(char path[], char name[])
{
  char *tmp;

  PetscFunctionBegin;
  PetscCall(PetscStrrchr(path,'/',&tmp));
  PetscCall(PetscStrcpy(name,tmp));
  if (tmp != path) {
    /* '/' found, name is substring of path after last occurence of '/'. */
    /* Trim the '/name' part from path just by inserting null character. */
    tmp--;
    *tmp = '\0';
  } else {
    /* '/' not found, name = path, path = "/". */
    PetscCall(PetscStrcpy(path,"/"));
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
  PetscCall(DMGetDimension(dm, &dim));
  if (dim != 3) PetscFunctionReturn(0);
  PetscCall(DMPlexGetCones(dm, &cones));
  PetscCall(DMPlexGetConeSection(dm, &cs));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cHeight));
  PetscCall(DMPlexGetHeightStratum(dm, cHeight, &cStart, &cEnd));
  for (p=cStart; p<cEnd; p++) {
    PetscInt numCorners, o;

    PetscCall(PetscSectionGetDof(cs, p, &numCorners));
    PetscCall(PetscSectionGetOffset(cs, p, &o));
    switch (numCorners) {
      case 4: PetscCall(DMPlexInvertCell(DM_POLYTOPE_TETRAHEDRON,&cones[o])); break;
      case 6: PetscCall(DMPlexInvertCell(DM_POLYTOPE_TRI_PRISM,&cones[o])); break;
      case 8: PetscCall(DMPlexInvertCell(DM_POLYTOPE_HEXAHEDRON,&cones[o])); break;
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
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)dm),((PetscObject)dm)->prefix,"DMPlex HDF5/XDMF Loader Options","PetscViewer");PetscCall(ierr);
  PetscCall(PetscOptionsString("-dm_plex_hdf5_topology_path","HDF5 path of topology dataset",NULL,topo_path,topo_path,sizeof(topo_path),NULL));
  PetscCall(PetscOptionsString("-dm_plex_hdf5_geometry_path","HDF5 path to geometry dataset",NULL,geom_path,geom_path,sizeof(geom_path),NULL));
  PetscCall(PetscOptionsBool("-dm_plex_hdf5_force_sequential","force sequential loading",NULL,seq,&seq,NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);

  PetscCall(SplitPath_Private(topo_path, topo_name));
  PetscCall(SplitPath_Private(geom_path, geom_name));
  PetscCall(PetscInfo(dm, "Topology group %s, name %s\n", topo_path, topo_name));
  PetscCall(PetscInfo(dm, "Geometry group %s, name %s\n", geom_path, geom_name));

  /* Read topology */
  PetscCall(PetscViewerHDF5PushGroup(viewer, topo_path));
  PetscCall(ISCreate(comm, &cells));
  PetscCall(PetscObjectSetName((PetscObject) cells, topo_name));
  if (seq) {
    PetscCall(PetscViewerHDF5ReadSizes(viewer, topo_name, NULL, &numCells));
    PetscCall(PetscLayoutSetSize(cells->map, numCells));
    numCells = rank == 0 ? numCells : 0;
    PetscCall(PetscLayoutSetLocalSize(cells->map, numCells));
  }
  PetscCall(ISLoad(cells, viewer));
  PetscCall(ISGetLocalSize(cells, &numCells));
  PetscCall(ISGetBlockSize(cells, &numCorners));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, topo_name, "cell_dim", PETSC_INT, &topoDim, &topoDim));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  numCells /= numCorners;

  /* Read geometry */
  PetscCall(PetscViewerHDF5PushGroup(viewer, geom_path));
  PetscCall(VecCreate(comm, &coordinates));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, geom_name));
  if (seq) {
    PetscCall(PetscViewerHDF5ReadSizes(viewer, geom_name, NULL, &numVertices));
    PetscCall(PetscLayoutSetSize(coordinates->map, numVertices));
    numVertices = rank == 0 ? numVertices : 0;
    PetscCall(PetscLayoutSetLocalSize(coordinates->map, numVertices));
  }
  PetscCall(VecLoad(coordinates, viewer));
  PetscCall(VecGetLocalSize(coordinates, &numVertices));
  PetscCall(VecGetSize(coordinates, &NVertices));
  PetscCall(VecGetBlockSize(coordinates, &spatialDim));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  numVertices /= spatialDim;
  NVertices /= spatialDim;

  PetscCall(PetscInfo(NULL, "Loaded mesh dimensions: numCells %D numCorners %D numVertices %D spatialDim %D\n", numCells, numCorners, numVertices, spatialDim));
  {
    const PetscScalar *coordinates_arr;
    PetscReal         *coordinates_arr_real;
    const PetscInt    *cells_arr;
    PetscSF           sfVert = NULL;
    PetscInt          i;

    PetscCall(VecGetArrayRead(coordinates, &coordinates_arr));
    PetscCall(ISGetIndices(cells, &cells_arr));

    if (PetscDefined(USE_COMPLEX)) {
      /* convert to real numbers if PetscScalar is complex */
      /*TODO More systematic would be to change all the function arguments to PetscScalar */
      PetscCall(PetscMalloc1(numVertices*spatialDim, &coordinates_arr_real));
      for (i = 0; i < numVertices*spatialDim; ++i) {
        coordinates_arr_real[i] = PetscRealPart(coordinates_arr[i]);
        if (PetscUnlikelyDebug(PetscImaginaryPart(coordinates_arr[i]))) {
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vector of coordinates contains complex numbers but only real vectors are currently supported.");
        }
      }
    } else coordinates_arr_real = (PetscReal*)coordinates_arr;

    PetscCall(DMSetDimension(dm, topoDim < 0 ? spatialDim : topoDim));
    PetscCall(DMPlexBuildFromCellListParallel(dm, numCells, numVertices, NVertices, numCorners, cells_arr, &sfVert, NULL));
    PetscCall(DMPlexInvertCells_XDMF_Private(dm));
    PetscCall(DMPlexBuildCoordinatesFromCellListParallel(dm, spatialDim, sfVert, coordinates_arr_real));
    PetscCall(VecRestoreArrayRead(coordinates, &coordinates_arr));
    PetscCall(ISRestoreIndices(cells, &cells_arr));
    PetscCall(PetscSFDestroy(&sfVert));
    if (PetscDefined(USE_COMPLEX)) PetscCall(PetscFree(coordinates_arr_real));
  }
  PetscCall(ISDestroy(&cells));
  PetscCall(VecDestroy(&coordinates));

  /* scale coordinates - unlike in DMPlexLoad_HDF5_Internal, this can only be done after DM is populated */
  {
    PetscReal lengthScale;

    PetscCall(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
    PetscCall(DMGetCoordinates(dm, &coordinates));
    PetscCall(VecScale(coordinates, 1.0/lengthScale));
  }

  /* Read Labels */
  /* TODO: this probably does not work as elements get permuted */
  /* PetscCall(DMPlexLabelsLoad_HDF5_Internal(dm, viewer)); */
  PetscFunctionReturn(0);
}
#endif
