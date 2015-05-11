#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#if defined(PETSC_HAVE_MED)
#include <med.h>
#endif

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateMedFromFile"
/*@C
  DMPlexCreateMedFromFile - Create a DMPlex mesh from a (Salome-)Med file.

+ comm        - The MPI communicator
. filename    - Name of the .med file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://www.code-aster.org/outils/med/html/index.html

  Level: beginner

.seealso: DMPlexCreateFromFile(), DMPlexCreateGmsh(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateMedFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscMPIInt     rank;
  PetscInt        i, nstep, ngeo, fileID, cellID, facetID, spaceDim, meshDim;
  PetscInt        numVertices = 0, numCells = 0, numFacets = 0, numCorners, numFacetCorners;
  PetscInt       *cellList, *facetList, *facetIDs;
  char           *axisname, *unitname, meshname[MED_NAME_SIZE+1], geotypename[MED_NAME_SIZE+1];
  char            meshdescription[MED_COMMENT_SIZE+1], dtunit[MED_SNAME_SIZE+1];
  PetscScalar    *coordinates = NULL;
#if defined(PETSC_HAVE_MED)
  med_sorting_type sortingtype;
  med_mesh_type   meshtype;
  med_axis_type   axistype;
  med_bool        coordinatechangement, geotransformation;
  med_geometry_type geotype[2];
#endif
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MED)
  fileID = MEDfileOpen(filename, MED_ACC_RDONLY);
  if (fileID < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Unable to open .med mesh file: %s", filename);
  if (MEDnMesh(fileID) < 1) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "No meshes found in .med mesh file: %s", filename);
  spaceDim = MEDmeshnAxis(fileID, 1);
  if (spaceDim < 1) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh of unknown space dimension found in .med mesh file: %s", filename);

  /* Read general mesh information */
  ierr = PetscMalloc1(MED_SNAME_SIZE*spaceDim+1, &axisname);CHKERRQ(ierr);
  ierr = PetscMalloc1(MED_SNAME_SIZE*spaceDim+1, &unitname);CHKERRQ(ierr);
  ierr = MEDmeshInfo(fileID, 1, meshname, &spaceDim, &meshDim, &meshtype, meshdescription,
                     dtunit, &sortingtype, &nstep, &axistype, axisname, unitname);CHKERRQ(ierr);
  ierr = PetscFree(axisname);
  ierr = PetscFree(unitname);

  if (!rank) {
    /* Read mesh coordinates */
    numVertices = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_NODE, MED_NO_GEOTYPE,
                                 MED_COORDINATE, MED_NO_CMODE,&coordinatechangement, &geotransformation);
    if (numVertices < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "No nodes found in .med mesh file: %s", filename);
    ierr = PetscMalloc1(numVertices*spaceDim, &coordinates);CHKERRQ(ierr);
    ierr = MEDmeshNodeCoordinateRd(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_FULL_INTERLACE, coordinates);CHKERRQ(ierr);
  }
  /* Read the types of entity sets in the mesh */
  ngeo = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL,MED_GEO_ALL, MED_CONNECTIVITY,
                        MED_NODAL, &coordinatechangement, &geotransformation);
  if (ngeo < 1) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "No cells found in .med mesh file: %s", filename);
  if (ngeo > 2) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Currently no support for hybrid meshes in .med mesh file: %s", filename);
  ierr = MEDmeshEntityInfo(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, 1, geotypename, &(geotype[0]));CHKERRQ(ierr);
  if (ngeo > 1) {ierr = MEDmeshEntityInfo(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, 2, geotypename, &(geotype[1]));CHKERRQ(ierr);}
  else geotype[1] = 0;
  /* Determine topological dim and set ID for cells */
  cellID = geotype[0]/100 > geotype[1]/100 ? 0 : 1;
  facetID = geotype[0]/100 > geotype[1]/100 ? 1 : 0;
  meshDim = geotype[cellID] / 100;
  numCorners = geotype[cellID] % 100;

  if (!rank) {
    /* Read cell connectivity */
    numCells = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[cellID], MED_CONNECTIVITY,
                              MED_NODAL,&coordinatechangement, &geotransformation);
    if (numCells < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "No cells found in .med mesh file: %s", filename);
    ierr = PetscMalloc1(numCells*numCorners, &cellList);
    ierr = MEDmeshElementConnectivityRd(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[cellID],
                                        MED_NODAL, MED_FULL_INTERLACE, cellList);CHKERRQ(ierr);
    for (i = 0; i < numCells*numCorners; i++) cellList[i]--; /* Correct entity counting */
  }
  /* Generate the DM */
  ierr = DMPlexCreateFromCellList(comm, meshDim, numCells, numVertices, numCorners, interpolate, cellList, spaceDim, coordinates, dm);CHKERRQ(ierr);

  if (!rank) {
    /* Read facet connectivity */
    if (ngeo > 1) {
      numFacets = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[facetID],
                                 MED_CONNECTIVITY, MED_NODAL,&coordinatechangement, &geotransformation);
    }
    if (numFacets > 0) {
      PetscInt c, f, vStart, joinSize, vertices[8];
      const PetscInt *join;
      ierr = DMPlexGetDepthStratum(*dm, 0, &vStart, NULL);CHKERRQ(ierr);
      numFacetCorners = geotype[facetID] % 100;
      ierr = PetscMalloc1(numFacets*numFacetCorners, &facetList);
      ierr = PetscCalloc1(numFacets, &facetIDs);
      ierr = MEDmeshElementConnectivityRd(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[facetID], MED_NODAL, MED_FULL_INTERLACE, facetList);CHKERRQ(ierr);
      for (i = 0; i < numFacets*numFacetCorners; i++) facetList[i]--; /* Correct entity counting */
      ierr = MEDmeshEntityFamilyNumberRd(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[facetID], facetIDs);CHKERRQ(ierr);
      /* Identify marked facets via vertex joins */
      for (f = 0; f < numFacets; ++f) {
        for (c = 0; c < numFacetCorners; ++c) vertices[c] = vStart + facetList[f*numFacetCorners+c];
        ierr = DMPlexGetFullJoin(*dm, numFacetCorners, (const PetscInt*)vertices, &joinSize, &join);CHKERRQ(ierr);
        if (joinSize != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not determine Plex facet for element %d", f);
        ierr = DMSetLabelValue(*dm, "Face Sets", join[0], facetIDs[f]);CHKERRQ(ierr);
        ierr = DMPlexRestoreJoin(*dm, numFacetCorners, (const PetscInt*)vertices, &joinSize, &join);CHKERRQ(ierr);
      }
      ierr = PetscFree(facetList);
      ierr = PetscFree(facetIDs);
    }
  }
  ierr = MEDfileClose(fileID);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscFree(coordinates);CHKERRQ(ierr);
    ierr = PetscFree(cellList);CHKERRQ(ierr);
  }
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires Med mesh reader support. Reconfigure using --download-med");
#endif
  PetscFunctionReturn(0);
}
