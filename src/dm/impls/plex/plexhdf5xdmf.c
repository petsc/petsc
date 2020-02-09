#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsclayouthdf5.h>

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode SplitPath_Private(char path[], char name[])
{
  char *tmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrrchr(path,'/',&tmp);CHKERRQ(ierr);
  ierr = PetscStrcpy(name,tmp);CHKERRQ(ierr);
  if (tmp != path) {
    /* '/' found, name is substring of path after last occurence of '/'. */
    /* Trim the '/name' part from path just by inserting null character. */
    tmp--;
    *tmp = '\0';
  } else {
    /* '/' not found, name = path, path = "/". */
    ierr = PetscStrcpy(path,"/");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexLoad_HDF5_Xdmf_Internal(DM dm, PetscViewer viewer)
{
  Vec             coordinates;
  IS              cells;
  PetscInt        spatialDim, numCells, numVertices, numCorners;
  PetscMPIInt     rank;
  MPI_Comm        comm;
  PetscErrorCode  ierr;
  char            topo_path[PETSC_MAX_PATH_LEN]="/viz/topology/cells", topo_name[PETSC_MAX_PATH_LEN];
  char            geom_path[PETSC_MAX_PATH_LEN]="/geometry/vertices",  geom_name[PETSC_MAX_PATH_LEN];
  PetscBool       seq = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)dm),((PetscObject)dm)->prefix,"DMPlex HDF5/XDMF Loader Options","PetscViewer");CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_plex_hdf5_topology_path","HDF5 path of topology dataset",NULL,topo_path,topo_path,sizeof(topo_path),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_plex_hdf5_geometry_path","HDF5 path to geometry dataset",NULL,geom_path,geom_path,sizeof(geom_path),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool(  "-dm_plex_hdf5_force_sequential","force sequential loading",NULL,seq,&seq,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = SplitPath_Private(topo_path, topo_name);CHKERRQ(ierr);
  ierr = SplitPath_Private(geom_path, geom_name);CHKERRQ(ierr);
  ierr = PetscInfo2(dm, "Topology group %s, name %s\n", topo_path, topo_name);CHKERRQ(ierr);
  ierr = PetscInfo2(dm, "Geometry group %s, name %s\n", geom_path, geom_name);CHKERRQ(ierr);

  /* Read topology */
  ierr = PetscViewerHDF5PushGroup(viewer, topo_path);CHKERRQ(ierr);
  ierr = ISCreate(comm, &cells);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) cells, topo_name);CHKERRQ(ierr);
  if (seq) {
    ierr = PetscViewerHDF5ReadSizes(viewer, topo_name, NULL, &numCells);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(cells->map, numCells);CHKERRQ(ierr);
    numCells = !rank ? numCells : 0;
    ierr = PetscLayoutSetLocalSize(cells->map, numCells);CHKERRQ(ierr);
  }
  ierr = ISLoad(cells, viewer);CHKERRQ(ierr);
  ierr = ISGetLocalSize(cells, &numCells);CHKERRQ(ierr);
  ierr = ISGetBlockSize(cells, &numCorners);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  numCells /= numCorners;

  /* Read geometry */
  ierr = PetscViewerHDF5PushGroup(viewer, geom_path);CHKERRQ(ierr);
  ierr = VecCreate(comm, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, geom_name);CHKERRQ(ierr);
  if (seq) {
    ierr = PetscViewerHDF5ReadSizes(viewer, geom_name, NULL, &numVertices);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(coordinates->map, numVertices);CHKERRQ(ierr);
    numVertices = !rank ? numVertices : 0;
    ierr = PetscLayoutSetLocalSize(coordinates->map, numVertices);CHKERRQ(ierr);
  }
  ierr = VecLoad(coordinates, viewer);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &numVertices);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &spatialDim);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  numVertices /= spatialDim;

  ierr = PetscInfo4(NULL, "Loaded mesh dimensions: numCells %d numCorners %d numVertices %d spatialDim %d\n", numCells, numCorners, numVertices, spatialDim);CHKERRQ(ierr);

  if (PetscDefined(USE_DEBUG)) {
    /* Check that maximum index referred in cells is in line with global number of vertices */
    PetscInt max1, max2;
    ierr = ISGetMinMax(cells, NULL, &max1);CHKERRQ(ierr);
    ierr = VecGetSize(coordinates, &max2);CHKERRQ(ierr);
    max2 /= spatialDim; max2--;
    if (max1 > max2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "maximum index in cells = %d > %d = total number of vertices - 1", max1, max2);
  }

  {
    const PetscScalar *coordinates_arr;
    PetscReal         *coordinates_arr_real;
    const PetscInt    *cells_arr;
    PetscSF           sfVert=NULL;
    PetscInt          i;

    ierr = VecGetArrayRead(coordinates, &coordinates_arr);CHKERRQ(ierr);
    ierr = ISGetIndices(cells, &cells_arr);CHKERRQ(ierr);

    if (PetscDefined(USE_COMPLEX)) {
      /* convert to real numbers if PetscScalar is complex */
      /*TODO More systematic would be to change all the function arguments to PetscScalar */
      ierr = PetscMalloc1(numVertices*spatialDim, &coordinates_arr_real);CHKERRQ(ierr);
      for (i = 0; i < numVertices*spatialDim; ++i) {
        coordinates_arr_real[i] = PetscRealPart(coordinates_arr[i]);
        if (PetscUnlikelyDebug(PetscImaginaryPart(coordinates_arr[i]))) {
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vector of coordinates contains complex numbers but only real vectors are currently supported.");
        }
      }
    } else coordinates_arr_real = (PetscReal*)coordinates_arr;

    ierr = DMSetDimension(dm, spatialDim);CHKERRQ(ierr);
    ierr = DMPlexBuildFromCellList_Parallel_Internal(dm, spatialDim, numCells, numVertices, numCorners, cells_arr, PETSC_TRUE, &sfVert);CHKERRQ(ierr);
    ierr = DMPlexBuildCoordinates_Parallel_Internal( dm, spatialDim, numCells, numVertices, sfVert, coordinates_arr_real);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(coordinates, &coordinates_arr);CHKERRQ(ierr);
    ierr = ISRestoreIndices(cells, &cells_arr);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfVert);CHKERRQ(ierr);
    if (PetscDefined(USE_COMPLEX)) {ierr = PetscFree(coordinates_arr_real);CHKERRQ(ierr);}
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

  /* Read Labels */
  /* TODO: this probably does not work as elements get permuted */
  /* ierr = DMPlexLoadLabels_HDF5_Internal(dm, viewer);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}
#endif
