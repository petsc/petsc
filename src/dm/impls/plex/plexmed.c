#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/

#if defined(PETSC_HAVE_MED)
  #include <med.h>
#endif

/*@C
  DMPlexCreateMedFromFile - Create a `DMPLEX` mesh from a (Salome-)Med file.

  Collective

+ comm        - The MPI communicator
. filename    - Name of the .med file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The `DM` object representing the mesh

  Level: beginner

  Reference:
. * -  https://www.salome-platform.org/user-section/about/med, http://docs.salome-platform.org/latest/MED_index.html

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMPlexCreateFromFile()`, `DMPlexCreateGmsh()`, `DMPlexCreate()`
@*/
PetscErrorCode DMPlexCreateMedFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscMPIInt rank, size;
#if defined(PETSC_HAVE_MED)
  med_idt           fileID;
  PetscInt          i, ngeo, spaceDim, meshDim;
  PetscInt          numVertices = 0, numCells = 0, c, numCorners, numCellsLocal, numVerticesLocal;
  med_int          *medCellList;
  PetscInt         *cellList;
  med_float        *coordinates = NULL;
  PetscReal        *vertcoords  = NULL;
  PetscLayout       vLayout, cLayout;
  const PetscInt   *vrange, *crange;
  PetscSF           sfVertices;
  char             *axisname, *unitname, meshname[MED_NAME_SIZE + 1], geotypename[MED_NAME_SIZE + 1];
  char              meshdescription[MED_COMMENT_SIZE + 1], dtunit[MED_SNAME_SIZE + 1];
  med_sorting_type  sortingtype;
  med_mesh_type     meshtype;
  med_axis_type     axistype;
  med_bool          coordinatechangement, geotransformation, hdfok, medok;
  med_geometry_type geotype[2], cellID, facetID;
  med_filter        vfilter = MED_FILTER_INIT;
  med_filter        cfilter = MED_FILTER_INIT;
  med_err           mederr;
  med_int           major, minor, release;
#endif

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
#if defined(PETSC_HAVE_MED)
  mederr = MEDfileCompatibility(filename, &hdfok, &medok);
  PetscCheck(!mederr, comm, PETSC_ERR_ARG_WRONG, "Cannot determine MED file compatibility: %s", filename);
  PetscCheck(hdfok, comm, PETSC_ERR_ARG_WRONG, "Not a compatible HDF format: %s", filename);
  PetscCheck(medok, comm, PETSC_ERR_ARG_WRONG, "Not a compatible MED format: %s", filename);

  fileID = MEDfileOpen(filename, MED_ACC_RDONLY);
  PetscCheck(fileID >= 0, comm, PETSC_ERR_ARG_WRONG, "Unable to open .med mesh file: %s", filename);
  mederr = MEDfileNumVersionRd(fileID, &major, &minor, &release);
  PetscCheck(MEDnMesh(fileID) >= 1, comm, PETSC_ERR_ARG_WRONG, "No meshes found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  spaceDim = MEDmeshnAxis(fileID, 1);
  PetscCheck(spaceDim >= 1, comm, PETSC_ERR_ARG_WRONG, "Mesh of unknown space dimension found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  /* Read general mesh information */
  PetscCall(PetscMalloc1(MED_SNAME_SIZE * spaceDim + 1, &axisname));
  PetscCall(PetscMalloc1(MED_SNAME_SIZE * spaceDim + 1, &unitname));
  {
    med_int medMeshDim, medNstep;
    med_int medSpaceDim = spaceDim;

    PetscCallExternal(MEDmeshInfo, fileID, 1, meshname, &medSpaceDim, &medMeshDim, &meshtype, meshdescription, dtunit, &sortingtype, &medNstep, &axistype, axisname, unitname);
    spaceDim = medSpaceDim;
    meshDim  = medMeshDim;
  }
  PetscCall(PetscFree(axisname));
  PetscCall(PetscFree(unitname));
  /* Partition mesh coordinates */
  numVertices = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_NODE, MED_NO_GEOTYPE, MED_COORDINATE, MED_NO_CMODE, &coordinatechangement, &geotransformation);
  PetscCall(PetscLayoutCreate(comm, &vLayout));
  PetscCall(PetscLayoutSetSize(vLayout, numVertices));
  PetscCall(PetscLayoutSetBlockSize(vLayout, 1));
  PetscCall(PetscLayoutSetUp(vLayout));
  PetscCall(PetscLayoutGetRanges(vLayout, &vrange));
  numVerticesLocal = vrange[rank + 1] - vrange[rank];
  PetscCallExternal(MEDfilterBlockOfEntityCr, fileID, numVertices, 1, spaceDim, MED_ALL_CONSTITUENT, MED_FULL_INTERLACE, MED_COMPACT_STMODE, MED_NO_PROFILE, vrange[rank] + 1, 1, numVerticesLocal, 1, 1, &vfilter);
  /* Read mesh coordinates */
  PetscCheck(numVertices >= 0, comm, PETSC_ERR_ARG_WRONG, "No nodes found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  PetscCall(PetscMalloc1(numVerticesLocal * spaceDim, &coordinates));
  PetscCallExternal(MEDmeshNodeCoordinateAdvancedRd, fileID, meshname, MED_NO_DT, MED_NO_IT, &vfilter, coordinates);
  /* Read the types of entity sets in the mesh */
  ngeo = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, MED_GEO_ALL, MED_CONNECTIVITY, MED_NODAL, &coordinatechangement, &geotransformation);
  PetscCheck(ngeo >= 1, comm, PETSC_ERR_ARG_WRONG, "No cells found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  PetscCheck(ngeo <= 2, comm, PETSC_ERR_ARG_WRONG, "Currently no support for hybrid meshes in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  PetscCallExternal(MEDmeshEntityInfo, fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, 1, geotypename, &(geotype[0]));
  if (ngeo > 1) PetscCallExternal(MEDmeshEntityInfo, fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, 2, geotypename, &(geotype[1]));
  else geotype[1] = 0;
  /* Determine topological dim and set ID for cells */
  cellID     = geotype[0] / 100 > geotype[1] / 100 ? 0 : 1;
  facetID    = geotype[0] / 100 > geotype[1] / 100 ? 1 : 0;
  meshDim    = geotype[cellID] / 100;
  numCorners = geotype[cellID] % 100;
  /* Partition cells */
  numCells = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[cellID], MED_CONNECTIVITY, MED_NODAL, &coordinatechangement, &geotransformation);
  PetscCall(PetscLayoutCreate(comm, &cLayout));
  PetscCall(PetscLayoutSetSize(cLayout, numCells));
  PetscCall(PetscLayoutSetBlockSize(cLayout, 1));
  PetscCall(PetscLayoutSetUp(cLayout));
  PetscCall(PetscLayoutGetRanges(cLayout, &crange));
  numCellsLocal = crange[rank + 1] - crange[rank];
  PetscCallExternal(MEDfilterBlockOfEntityCr, fileID, numCells, 1, numCorners, MED_ALL_CONSTITUENT, MED_FULL_INTERLACE, MED_COMPACT_STMODE, MED_NO_PROFILE, crange[rank] + 1, 1, numCellsLocal, 1, 1, &cfilter);
  /* Read cell connectivity */
  PetscCheck(numCells >= 0, comm, PETSC_ERR_ARG_WRONG, "No cells found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  PetscCall(PetscMalloc1(numCellsLocal * numCorners, &medCellList));
  PetscCallExternal(MEDmeshElementConnectivityAdvancedRd, fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[cellID], MED_NODAL, &cfilter, medCellList);
  PetscCheck(sizeof(med_int) <= sizeof(PetscInt), comm, PETSC_ERR_ARG_SIZ, "Size of PetscInt %zd less than  size of med_int %zd. Reconfigure PETSc --with-64-bit-indices=1", sizeof(PetscInt), sizeof(med_int));
  PetscCall(PetscMalloc1(numCellsLocal * numCorners, &cellList));
  for (i = 0; i < numCellsLocal * numCorners; i++) { cellList[i] = ((PetscInt)medCellList[i]) - 1; /* Correct entity counting */ }
  PetscCall(PetscFree(medCellList));
  /* Generate the DM */
  if (sizeof(med_float) == sizeof(PetscReal)) {
    vertcoords = (PetscReal *)coordinates;
  } else {
    PetscCall(PetscMalloc1(numVerticesLocal * spaceDim, &vertcoords));
    for (i = 0; i < numVerticesLocal * spaceDim; i++) vertcoords[i] = (PetscReal)coordinates[i];
  }
  /* Account for cell inversion */
  for (c = 0; c < numCellsLocal; ++c) {
    PetscInt *pcone = &cellList[c * numCorners];

    if (meshDim == 3) {
      /* Hexahedra are inverted */
      if (numCorners == 8) {
        PetscInt tmp = pcone[4 + 1];
        pcone[4 + 1] = pcone[4 + 3];
        pcone[4 + 3] = tmp;
      }
    }
  }
  PetscCall(DMPlexCreateFromCellListParallelPetsc(comm, meshDim, numCellsLocal, numVerticesLocal, numVertices, numCorners, interpolate, cellList, spaceDim, vertcoords, &sfVertices, NULL, dm));
  if (sizeof(med_float) == sizeof(PetscReal)) {
    vertcoords = NULL;
  } else {
    PetscCall(PetscFree(vertcoords));
  }
  if (ngeo > 1) {
    PetscInt        numFacets = 0, numFacetsLocal, numFacetCorners, numFacetsRendezvous;
    PetscInt        c, f, v, vStart, joinSize, vertices[8];
    PetscInt       *facetList, *facetListRendezvous, *facetIDs, *facetIDsRendezvous, *facetListRemote, *facetIDsRemote;
    const PetscInt *frange, *join;
    PetscLayout     fLayout;
    med_filter      ffilter = MED_FILTER_INIT, fidfilter = MED_FILTER_INIT;
    PetscSection    facetSectionRemote, facetSectionIDsRemote;
    /* Partition facets */
    numFacets       = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[facetID], MED_CONNECTIVITY, MED_NODAL, &coordinatechangement, &geotransformation);
    numFacetCorners = geotype[facetID] % 100;
    PetscCall(PetscLayoutCreate(comm, &fLayout));
    PetscCall(PetscLayoutSetSize(fLayout, numFacets));
    PetscCall(PetscLayoutSetBlockSize(fLayout, 1));
    PetscCall(PetscLayoutSetUp(fLayout));
    PetscCall(PetscLayoutGetRanges(fLayout, &frange));
    numFacetsLocal = frange[rank + 1] - frange[rank];
    PetscCallExternal(MEDfilterBlockOfEntityCr, fileID, numFacets, 1, numFacetCorners, MED_ALL_CONSTITUENT, MED_FULL_INTERLACE, MED_COMPACT_STMODE, MED_NO_PROFILE, frange[rank] + 1, 1, numFacetsLocal, 1, 1, &ffilter);
    PetscCallExternal(MEDfilterBlockOfEntityCr, fileID, numFacets, 1, 1, MED_ALL_CONSTITUENT, MED_FULL_INTERLACE, MED_COMPACT_STMODE, MED_NO_PROFILE, frange[rank] + 1, 1, numFacetsLocal, 1, 1, &fidfilter);
    PetscCall(DMPlexGetDepthStratum(*dm, 0, &vStart, NULL));
    PetscCall(PetscMalloc1(numFacetsLocal, &facetIDs));
    PetscCall(PetscMalloc1(numFacetsLocal * numFacetCorners, &facetList));

    /* Read facet connectivity */
    {
      med_int *medFacetList;

      PetscCall(PetscMalloc1(numFacetsLocal * numFacetCorners, &medFacetList));
      PetscCallExternal(MEDmeshElementConnectivityAdvancedRd, fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[facetID], MED_NODAL, &ffilter, medFacetList);
      for (i = 0; i < numFacetsLocal * numFacetCorners; i++) { facetList[i] = ((PetscInt)medFacetList[i]) - 1; /* Correct entity counting */ }
      PetscCall(PetscFree(medFacetList));
    }

    /* Read facet IDs */
    {
      med_int *medFacetIDs;

      PetscCall(PetscMalloc1(numFacetsLocal, &medFacetIDs));
      PetscCallExternal(MEDmeshEntityAttributeAdvancedRd, fileID, meshname, MED_FAMILY_NUMBER, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[facetID], &fidfilter, medFacetIDs);
      for (i = 0; i < numFacetsLocal; i++) facetIDs[i] = (PetscInt)medFacetIDs[i];
      PetscCall(PetscFree(medFacetIDs));
    }

    /* Send facets and IDs to a rendezvous partition that is based on the initial vertex partitioning. */
    {
      PetscInt           r;
      DMLabel            lblFacetRendezvous, lblFacetMigration;
      PetscSection       facetSection, facetSectionRendezvous;
      PetscSF            sfProcess, sfFacetMigration;
      const PetscSFNode *remoteVertices;
      PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Facet Rendezvous", &lblFacetRendezvous));
      PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Facet Migration", &lblFacetMigration));
      PetscCall(PetscSFGetGraph(sfVertices, NULL, NULL, NULL, &remoteVertices));
      for (f = 0; f < numFacetsLocal; f++) {
        for (v = 0; v < numFacetCorners; v++) {
          /* Find vertex owner on rendezvous partition and mark in label */
          const PetscInt vertex = facetList[f * numFacetCorners + v];
          r                     = rank;
          while (vrange[r] > vertex) r--;
          while (vrange[r + 1] < vertex) r++;
          PetscCall(DMLabelSetValue(lblFacetRendezvous, f, r));
        }
      }
      /* Build a global process SF */
      PetscCall(PetscSFCreate(comm, &sfProcess));
      PetscCall(PetscSFSetGraphWithPattern(sfProcess, NULL, PETSCSF_PATTERN_ALLTOALL));
      PetscCall(PetscObjectSetName((PetscObject)sfProcess, "Process SF"));
      /* Convert facet rendezvous label into SF for migration */
      PetscCall(DMPlexPartitionLabelInvert(*dm, lblFacetRendezvous, sfProcess, lblFacetMigration));
      PetscCall(DMPlexPartitionLabelCreateSF(*dm, lblFacetMigration, &sfFacetMigration));
      /* Migrate facet connectivity data */
      PetscCall(PetscSectionCreate(comm, &facetSection));
      PetscCall(PetscSectionSetChart(facetSection, 0, numFacetsLocal));
      for (f = 0; f < numFacetsLocal; f++) PetscCall(PetscSectionSetDof(facetSection, f, numFacetCorners));
      PetscCall(PetscSectionSetUp(facetSection));
      PetscCall(PetscSectionCreate(comm, &facetSectionRendezvous));
      PetscCall(DMPlexDistributeData(*dm, sfFacetMigration, facetSection, MPIU_INT, facetList, facetSectionRendezvous, (void **)&facetListRendezvous));
      /* Migrate facet IDs */
      PetscCall(PetscSFGetGraph(sfFacetMigration, NULL, &numFacetsRendezvous, NULL, NULL));
      PetscCall(PetscMalloc1(numFacetsRendezvous, &facetIDsRendezvous));
      PetscCall(PetscSFBcastBegin(sfFacetMigration, MPIU_INT, facetIDs, facetIDsRendezvous, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sfFacetMigration, MPIU_INT, facetIDs, facetIDsRendezvous, MPI_REPLACE));
      /* Clean up */
      PetscCall(DMLabelDestroy(&lblFacetRendezvous));
      PetscCall(DMLabelDestroy(&lblFacetMigration));
      PetscCall(PetscSFDestroy(&sfProcess));
      PetscCall(PetscSFDestroy(&sfFacetMigration));
      PetscCall(PetscSectionDestroy(&facetSection));
      PetscCall(PetscSectionDestroy(&facetSectionRendezvous));
    }

    /* On the rendevouz partition we build a vertex-wise section/array of facets and IDs. */
    {
      PetscInt               sizeVertexFacets, offset, sizeFacetIDsRemote;
      PetscInt              *vertexFacets, *vertexIdx, *vertexFacetIDs;
      PetscSection           facetSectionVertices, facetSectionIDs;
      ISLocalToGlobalMapping ltogVertexNumbering;
      PetscCall(PetscSectionCreate(comm, &facetSectionVertices));
      PetscCall(PetscSectionSetChart(facetSectionVertices, 0, numVerticesLocal));
      PetscCall(PetscSectionCreate(comm, &facetSectionIDs));
      PetscCall(PetscSectionSetChart(facetSectionIDs, 0, numVerticesLocal));
      for (f = 0; f < numFacetsRendezvous * numFacetCorners; f++) {
        const PetscInt vertex = facetListRendezvous[f];
        if (vrange[rank] <= vertex && vertex < vrange[rank + 1]) {
          PetscCall(PetscSectionAddDof(facetSectionIDs, vertex - vrange[rank], 1));
          PetscCall(PetscSectionAddDof(facetSectionVertices, vertex - vrange[rank], numFacetCorners));
        }
      }
      PetscCall(PetscSectionSetUp(facetSectionVertices));
      PetscCall(PetscSectionSetUp(facetSectionIDs));
      PetscCall(PetscSectionGetStorageSize(facetSectionVertices, &sizeVertexFacets));
      PetscCall(PetscSectionGetStorageSize(facetSectionVertices, &sizeFacetIDsRemote));
      PetscCall(PetscMalloc1(sizeVertexFacets, &vertexFacets));
      PetscCall(PetscMalloc1(sizeFacetIDsRemote, &vertexFacetIDs));
      PetscCall(PetscCalloc1(numVerticesLocal, &vertexIdx));
      for (f = 0; f < numFacetsRendezvous; f++) {
        for (c = 0; c < numFacetCorners; c++) {
          const PetscInt vertex = facetListRendezvous[f * numFacetCorners + c];
          if (vrange[rank] <= vertex && vertex < vrange[rank + 1]) {
            /* Flip facet connectivities and IDs to a vertex-wise layout */
            PetscCall(PetscSectionGetOffset(facetSectionVertices, vertex - vrange[rank], &offset));
            offset += vertexIdx[vertex - vrange[rank]] * numFacetCorners;
            PetscCall(PetscArraycpy(&(vertexFacets[offset]), &(facetListRendezvous[f * numFacetCorners]), numFacetCorners));
            PetscCall(PetscSectionGetOffset(facetSectionIDs, vertex - vrange[rank], &offset));
            offset += vertexIdx[vertex - vrange[rank]];
            vertexFacetIDs[offset] = facetIDsRendezvous[f];
            vertexIdx[vertex - vrange[rank]]++;
          }
        }
      }
      /* Distribute the vertex-wise facet connectivities over the vertexSF */
      PetscCall(PetscSectionCreate(comm, &facetSectionRemote));
      PetscCall(DMPlexDistributeData(*dm, sfVertices, facetSectionVertices, MPIU_INT, vertexFacets, facetSectionRemote, (void **)&facetListRemote));
      PetscCall(PetscSectionCreate(comm, &facetSectionIDsRemote));
      PetscCall(DMPlexDistributeData(*dm, sfVertices, facetSectionIDs, MPIU_INT, vertexFacetIDs, facetSectionIDsRemote, (void **)&facetIDsRemote));
      /* Convert facet connectivities to local vertex numbering */
      PetscCall(PetscSectionGetStorageSize(facetSectionRemote, &sizeVertexFacets));
      PetscCall(ISLocalToGlobalMappingCreateSF(sfVertices, vrange[rank], &ltogVertexNumbering));
      PetscCall(ISGlobalToLocalMappingApplyBlock(ltogVertexNumbering, IS_GTOLM_MASK, sizeVertexFacets, facetListRemote, NULL, facetListRemote));
      /* Clean up */
      PetscCall(PetscFree(vertexFacets));
      PetscCall(PetscFree(vertexIdx));
      PetscCall(PetscFree(vertexFacetIDs));
      PetscCall(PetscSectionDestroy(&facetSectionVertices));
      PetscCall(PetscSectionDestroy(&facetSectionIDs));
      PetscCall(ISLocalToGlobalMappingDestroy(&ltogVertexNumbering));
    }
    {
      PetscInt  offset, dof;
      DMLabel   lblFaceSets;
      PetscBool verticesLocal;
      /* Identify and mark facets locally with facet joins */
      PetscCall(DMCreateLabel(*dm, "Face Sets"));
      PetscCall(DMGetLabel(*dm, "Face Sets", &lblFaceSets));
      /* We need to set a new default value here, since -1 is a legitimate facet ID */
      PetscCall(DMLabelSetDefaultValue(lblFaceSets, -666666666));
      for (v = 0; v < numVerticesLocal; v++) {
        PetscCall(PetscSectionGetOffset(facetSectionRemote, v, &offset));
        PetscCall(PetscSectionGetDof(facetSectionRemote, v, &dof));
        for (f = 0; f < dof; f += numFacetCorners) {
          for (verticesLocal = PETSC_TRUE, c = 0; c < numFacetCorners; ++c) {
            if (facetListRemote[offset + f + c] < 0) {
              verticesLocal = PETSC_FALSE;
              break;
            }
            vertices[c] = vStart + facetListRemote[offset + f + c];
          }
          if (verticesLocal) {
            PetscCall(DMPlexGetFullJoin(*dm, numFacetCorners, (const PetscInt *)vertices, &joinSize, &join));
            if (joinSize == 1) PetscCall(DMLabelSetValue(lblFaceSets, join[0], facetIDsRemote[(offset + f) / numFacetCorners]));
            PetscCall(DMPlexRestoreJoin(*dm, numFacetCorners, (const PetscInt *)vertices, &joinSize, &join));
          }
        }
      }
    }
    PetscCall(PetscFree(facetList));
    PetscCall(PetscFree(facetListRendezvous));
    PetscCall(PetscFree(facetListRemote));
    PetscCall(PetscFree(facetIDs));
    PetscCall(PetscFree(facetIDsRendezvous));
    PetscCall(PetscFree(facetIDsRemote));
    PetscCall(PetscLayoutDestroy(&fLayout));
    PetscCall(PetscSectionDestroy(&facetSectionRemote));
    PetscCall(PetscSectionDestroy(&facetSectionIDsRemote));
  }
  PetscCallExternal(MEDfileClose, fileID);
  PetscCall(PetscFree(coordinates));
  PetscCall(PetscFree(cellList));
  PetscCall(PetscLayoutDestroy(&vLayout));
  PetscCall(PetscLayoutDestroy(&cLayout));
  PetscCall(PetscSFDestroy(&sfVertices));
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires Med mesh reader support. Reconfigure using --download-med");
#endif
}
