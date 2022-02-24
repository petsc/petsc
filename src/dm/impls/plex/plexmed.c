#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#if defined(PETSC_HAVE_MED)
#include <med.h>
#endif

/*@C
  DMPlexCreateMedFromFile - Create a DMPlex mesh from a (Salome-)Med file.

+ comm        - The MPI communicator
. filename    - Name of the .med file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: https://www.salome-platform.org/user-section/about/med, http://docs.salome-platform.org/latest/MED_index.html

  Level: beginner

.seealso: DMPlexCreateFromFile(), DMPlexCreateGmsh(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateMedFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscMPIInt     rank, size;
#if defined(PETSC_HAVE_MED)
  med_idt         fileID;
  PetscInt        i, ngeo, spaceDim, meshDim;
  PetscInt        numVertices = 0, numCells = 0, c, numCorners, numCellsLocal, numVerticesLocal;
  med_int        *medCellList;
  PetscInt       *cellList;
  med_float      *coordinates = NULL;
  PetscReal      *vertcoords = NULL;
  PetscLayout     vLayout, cLayout;
  const PetscInt *vrange, *crange;
  PetscSF         sfVertices;
  char           *axisname, *unitname, meshname[MED_NAME_SIZE+1], geotypename[MED_NAME_SIZE+1];
  char            meshdescription[MED_COMMENT_SIZE+1], dtunit[MED_SNAME_SIZE+1];
  med_sorting_type sortingtype;
  med_mesh_type   meshtype;
  med_axis_type   axistype;
  med_bool        coordinatechangement, geotransformation, hdfok, medok;
  med_geometry_type geotype[2], cellID, facetID;
  med_filter      vfilter = MED_FILTER_INIT;
  med_filter      cfilter = MED_FILTER_INIT;
  med_err         mederr;
  med_int         major, minor, release;
#endif

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
#if defined(PETSC_HAVE_MED)
  mederr = MEDfileCompatibility(filename, &hdfok, &medok);
  PetscCheck(!mederr,comm, PETSC_ERR_ARG_WRONG, "Cannot determine MED file compatibility: %s", filename);
  PetscCheck(hdfok,comm, PETSC_ERR_ARG_WRONG, "Not a compatible HDF format: %s", filename);
  PetscCheck(medok,comm, PETSC_ERR_ARG_WRONG, "Not a compatible MED format: %s", filename);

  fileID = MEDfileOpen(filename, MED_ACC_RDONLY);
  PetscCheck(fileID >= 0,comm, PETSC_ERR_ARG_WRONG, "Unable to open .med mesh file: %s", filename);
  mederr = MEDfileNumVersionRd(fileID, &major, &minor, &release);
  PetscCheck(MEDnMesh (fileID) >= 1,comm, PETSC_ERR_ARG_WRONG, "No meshes found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  spaceDim = MEDmeshnAxis(fileID, 1);
  PetscCheck(spaceDim >= 1,comm, PETSC_ERR_ARG_WRONG, "Mesh of unknown space dimension found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  /* Read general mesh information */
  CHKERRQ(PetscMalloc1(MED_SNAME_SIZE*spaceDim+1, &axisname));
  CHKERRQ(PetscMalloc1(MED_SNAME_SIZE*spaceDim+1, &unitname));
  {
    med_int medMeshDim, medNstep;
    med_int medSpaceDim = spaceDim;

    PetscStackCallStandard(MEDmeshInfo,fileID, 1, meshname, &medSpaceDim, &medMeshDim, &meshtype, meshdescription,dtunit, &sortingtype, &medNstep, &axistype, axisname, unitname);
    spaceDim = medSpaceDim;
    meshDim  = medMeshDim;
  }
  CHKERRQ(PetscFree(axisname));
  CHKERRQ(PetscFree(unitname));
  /* Partition mesh coordinates */
  numVertices = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_NODE, MED_NO_GEOTYPE,
                               MED_COORDINATE, MED_NO_CMODE,&coordinatechangement, &geotransformation);
  CHKERRQ(PetscLayoutCreate(comm, &vLayout));
  CHKERRQ(PetscLayoutSetSize(vLayout, numVertices));
  CHKERRQ(PetscLayoutSetBlockSize(vLayout, 1));
  CHKERRQ(PetscLayoutSetUp(vLayout));
  CHKERRQ(PetscLayoutGetRanges(vLayout, &vrange));
  numVerticesLocal = vrange[rank+1]-vrange[rank];
  PetscStackCallStandard(MEDfilterBlockOfEntityCr,fileID, numVertices, 1, spaceDim, MED_ALL_CONSTITUENT, MED_FULL_INTERLACE, MED_COMPACT_STMODE,MED_NO_PROFILE, vrange[rank]+1, 1, numVerticesLocal, 1, 1, &vfilter);
  /* Read mesh coordinates */
  PetscCheck(numVertices >= 0,comm, PETSC_ERR_ARG_WRONG, "No nodes found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  CHKERRQ(PetscMalloc1(numVerticesLocal*spaceDim, &coordinates));
  PetscStackCallStandard(MEDmeshNodeCoordinateAdvancedRd,fileID, meshname, MED_NO_DT, MED_NO_IT, &vfilter, coordinates);
  /* Read the types of entity sets in the mesh */
  ngeo = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL,MED_GEO_ALL, MED_CONNECTIVITY,
                        MED_NODAL, &coordinatechangement, &geotransformation);
  PetscCheck(ngeo >= 1,comm, PETSC_ERR_ARG_WRONG, "No cells found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  PetscCheck(ngeo <= 2,comm, PETSC_ERR_ARG_WRONG, "Currently no support for hybrid meshes in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  PetscStackCallStandard(MEDmeshEntityInfo,fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, 1, geotypename, &(geotype[0]));
  if (ngeo > 1) PetscStackCallStandard(MEDmeshEntityInfo,fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, 2, geotypename, &(geotype[1]));
  else geotype[1] = 0;
  /* Determine topological dim and set ID for cells */
  cellID = geotype[0]/100 > geotype[1]/100 ? 0 : 1;
  facetID = geotype[0]/100 > geotype[1]/100 ? 1 : 0;
  meshDim = geotype[cellID] / 100;
  numCorners = geotype[cellID] % 100;
  /* Partition cells */
  numCells = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[cellID],
                            MED_CONNECTIVITY, MED_NODAL,&coordinatechangement, &geotransformation);
  CHKERRQ(PetscLayoutCreate(comm, &cLayout));
  CHKERRQ(PetscLayoutSetSize(cLayout, numCells));
  CHKERRQ(PetscLayoutSetBlockSize(cLayout, 1));
  CHKERRQ(PetscLayoutSetUp(cLayout));
  CHKERRQ(PetscLayoutGetRanges(cLayout, &crange));
  numCellsLocal = crange[rank+1]-crange[rank];
  PetscStackCallStandard(MEDfilterBlockOfEntityCr,fileID, numCells, 1, numCorners, MED_ALL_CONSTITUENT, MED_FULL_INTERLACE, MED_COMPACT_STMODE,MED_NO_PROFILE, crange[rank]+1, 1, numCellsLocal, 1, 1, &cfilter);
  /* Read cell connectivity */
  PetscCheck(numCells >= 0,comm, PETSC_ERR_ARG_WRONG, "No cells found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  CHKERRQ(PetscMalloc1(numCellsLocal*numCorners, &medCellList));
  PetscStackCallStandard(MEDmeshElementConnectivityAdvancedRd,fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[cellID],MED_NODAL, &cfilter, medCellList);
  PetscCheck(sizeof(med_int) <= sizeof(PetscInt),comm, PETSC_ERR_ARG_SIZ, "Size of PetscInt %zd less than  size of med_int %zd. Reconfigure PETSc --with-64-bit-indices=1", sizeof(PetscInt), sizeof(med_int));
  CHKERRQ(PetscMalloc1(numCellsLocal*numCorners, &cellList));
  for (i = 0; i < numCellsLocal*numCorners; i++) {
    cellList[i] = ((PetscInt) medCellList[i]) - 1; /* Correct entity counting */
  }
  CHKERRQ(PetscFree(medCellList));
  /* Generate the DM */
  if (sizeof(med_float) == sizeof(PetscReal)) {
    vertcoords = (PetscReal *) coordinates;
  } else {
    CHKERRQ(PetscMalloc1(numVerticesLocal*spaceDim, &vertcoords));
    for (i = 0; i < numVerticesLocal*spaceDim; i++) vertcoords[i] = (PetscReal) coordinates[i];
  }
  /* Account for cell inversion */
  for (c = 0; c < numCellsLocal; ++c) {
    PetscInt *pcone = &cellList[c*numCorners];

    if (meshDim == 3) {
      /* Hexahedra are inverted */
      if (numCorners == 8) {
        PetscInt tmp = pcone[4+1];
        pcone[4+1] = pcone[4+3];
        pcone[4+3] = tmp;
      }
    }
  }
  CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, meshDim, numCellsLocal, numVerticesLocal, numVertices, numCorners, interpolate, cellList, spaceDim, vertcoords, &sfVertices, NULL, dm));
  if (sizeof(med_float) == sizeof(PetscReal)) {
    vertcoords = NULL;
  } else {
    CHKERRQ(PetscFree(vertcoords));
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
    numFacets = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[facetID],
                               MED_CONNECTIVITY, MED_NODAL,&coordinatechangement, &geotransformation);
    numFacetCorners = geotype[facetID] % 100;
    CHKERRQ(PetscLayoutCreate(comm, &fLayout));
    CHKERRQ(PetscLayoutSetSize(fLayout, numFacets));
    CHKERRQ(PetscLayoutSetBlockSize(fLayout, 1));
    CHKERRQ(PetscLayoutSetUp(fLayout));
    CHKERRQ(PetscLayoutGetRanges(fLayout, &frange));
    numFacetsLocal = frange[rank+1]-frange[rank];
    PetscStackCallStandard(MEDfilterBlockOfEntityCr,fileID, numFacets, 1, numFacetCorners, MED_ALL_CONSTITUENT, MED_FULL_INTERLACE, MED_COMPACT_STMODE,MED_NO_PROFILE, frange[rank]+1, 1, numFacetsLocal, 1, 1, &ffilter);
    PetscStackCallStandard(MEDfilterBlockOfEntityCr,fileID, numFacets, 1, 1, MED_ALL_CONSTITUENT, MED_FULL_INTERLACE, MED_COMPACT_STMODE,MED_NO_PROFILE, frange[rank]+1, 1, numFacetsLocal, 1, 1, &fidfilter);
    CHKERRQ(DMPlexGetDepthStratum(*dm, 0, &vStart, NULL));
    CHKERRQ(PetscMalloc1(numFacetsLocal, &facetIDs));
    CHKERRQ(PetscMalloc1(numFacetsLocal*numFacetCorners, &facetList));

    /* Read facet connectivity */
    {
      med_int *medFacetList;

      CHKERRQ(PetscMalloc1(numFacetsLocal*numFacetCorners, &medFacetList));
      PetscStackCallStandard(MEDmeshElementConnectivityAdvancedRd,fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[facetID], MED_NODAL, &ffilter, medFacetList);
      for (i = 0; i < numFacetsLocal*numFacetCorners; i++) {
        facetList[i] = ((PetscInt) medFacetList[i]) - 1 ; /* Correct entity counting */
      }
      CHKERRQ(PetscFree(medFacetList));
    }

    /* Read facet IDs */
    {
      med_int *medFacetIDs;

      CHKERRQ(PetscMalloc1(numFacetsLocal, &medFacetIDs));
      PetscStackCallStandard(MEDmeshEntityAttributeAdvancedRd,fileID, meshname, MED_FAMILY_NUMBER, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[facetID], &fidfilter, medFacetIDs);
      for (i = 0; i < numFacetsLocal; i++) {
        facetIDs[i] = (PetscInt) medFacetIDs[i];
      }
      CHKERRQ(PetscFree(medFacetIDs));
    }

    /* Send facets and IDs to a rendezvous partition that is based on the initial vertex partitioning. */
    {
      PetscInt           r;
      DMLabel            lblFacetRendezvous, lblFacetMigration;
      PetscSection       facetSection, facetSectionRendezvous;
      PetscSF            sfProcess, sfFacetMigration;
      const PetscSFNode *remoteVertices;
      CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "Facet Rendezvous", &lblFacetRendezvous));
      CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "Facet Migration", &lblFacetMigration));
      CHKERRQ(PetscSFGetGraph(sfVertices, NULL, NULL, NULL, &remoteVertices));
      for (f = 0; f < numFacetsLocal; f++) {
        for (v = 0; v < numFacetCorners; v++) {
          /* Find vertex owner on rendezvous partition and mark in label */
          const PetscInt vertex = facetList[f*numFacetCorners+v];
          r = rank; while (vrange[r] > vertex) r--; while (vrange[r + 1] < vertex) r++;
          CHKERRQ(DMLabelSetValue(lblFacetRendezvous, f, r));
        }
      }
      /* Build a global process SF */
      CHKERRQ(PetscSFCreate(comm,&sfProcess));
      CHKERRQ(PetscSFSetGraphWithPattern(sfProcess,NULL,PETSCSF_PATTERN_ALLTOALL));
      CHKERRQ(PetscObjectSetName((PetscObject) sfProcess, "Process SF"));
      /* Convert facet rendezvous label into SF for migration */
      CHKERRQ(DMPlexPartitionLabelInvert(*dm, lblFacetRendezvous, sfProcess, lblFacetMigration));
      CHKERRQ(DMPlexPartitionLabelCreateSF(*dm, lblFacetMigration, &sfFacetMigration));
      /* Migrate facet connectivity data */
      CHKERRQ(PetscSectionCreate(comm, &facetSection));
      CHKERRQ(PetscSectionSetChart(facetSection, 0, numFacetsLocal));
      for (f = 0; f < numFacetsLocal; f++) CHKERRQ(PetscSectionSetDof(facetSection, f, numFacetCorners));
      CHKERRQ(PetscSectionSetUp(facetSection));
      CHKERRQ(PetscSectionCreate(comm, &facetSectionRendezvous));
      CHKERRQ(DMPlexDistributeData(*dm, sfFacetMigration, facetSection, MPIU_INT, facetList, facetSectionRendezvous, (void**) &facetListRendezvous));
      /* Migrate facet IDs */
      CHKERRQ(PetscSFGetGraph(sfFacetMigration, NULL, &numFacetsRendezvous, NULL, NULL));
      CHKERRQ(PetscMalloc1(numFacetsRendezvous, &facetIDsRendezvous));
      CHKERRQ(PetscSFBcastBegin(sfFacetMigration, MPIU_INT, facetIDs, facetIDsRendezvous,MPI_REPLACE));
      CHKERRQ(PetscSFBcastEnd(sfFacetMigration, MPIU_INT, facetIDs, facetIDsRendezvous,MPI_REPLACE));
      /* Clean up */
      CHKERRQ(DMLabelDestroy(&lblFacetRendezvous));
      CHKERRQ(DMLabelDestroy(&lblFacetMigration));
      CHKERRQ(PetscSFDestroy(&sfProcess));
      CHKERRQ(PetscSFDestroy(&sfFacetMigration));
      CHKERRQ(PetscSectionDestroy(&facetSection));
      CHKERRQ(PetscSectionDestroy(&facetSectionRendezvous));
    }

    /* On the rendevouz partition we build a vertex-wise section/array of facets and IDs. */
    {
      PetscInt               sizeVertexFacets, offset, sizeFacetIDsRemote;
      PetscInt              *vertexFacets, *vertexIdx, *vertexFacetIDs;
      PetscSection           facetSectionVertices, facetSectionIDs;
      ISLocalToGlobalMapping ltogVertexNumbering;
      CHKERRQ(PetscSectionCreate(comm, &facetSectionVertices));
      CHKERRQ(PetscSectionSetChart(facetSectionVertices, 0, numVerticesLocal));
      CHKERRQ(PetscSectionCreate(comm, &facetSectionIDs));
      CHKERRQ(PetscSectionSetChart(facetSectionIDs, 0, numVerticesLocal));
      for (f = 0; f < numFacetsRendezvous*numFacetCorners; f++) {
        const PetscInt vertex = facetListRendezvous[f];
        if (vrange[rank] <= vertex && vertex < vrange[rank+1]) {
          CHKERRQ(PetscSectionAddDof(facetSectionIDs, vertex-vrange[rank], 1));
          CHKERRQ(PetscSectionAddDof(facetSectionVertices, vertex-vrange[rank], numFacetCorners));
        }
      }
      CHKERRQ(PetscSectionSetUp(facetSectionVertices));
      CHKERRQ(PetscSectionSetUp(facetSectionIDs));
      CHKERRQ(PetscSectionGetStorageSize(facetSectionVertices, &sizeVertexFacets));
      CHKERRQ(PetscSectionGetStorageSize(facetSectionVertices, &sizeFacetIDsRemote));
      CHKERRQ(PetscMalloc1(sizeVertexFacets, &vertexFacets));
      CHKERRQ(PetscMalloc1(sizeFacetIDsRemote, &vertexFacetIDs));
      CHKERRQ(PetscCalloc1(numVerticesLocal, &vertexIdx));
      for (f = 0; f < numFacetsRendezvous; f++) {
        for (c = 0; c < numFacetCorners; c++) {
          const PetscInt vertex = facetListRendezvous[f*numFacetCorners+c];
          if (vrange[rank] <= vertex && vertex < vrange[rank+1]) {
            /* Flip facet connectivities and IDs to a vertex-wise layout */
            CHKERRQ(PetscSectionGetOffset(facetSectionVertices, vertex-vrange[rank], &offset));
            offset += vertexIdx[vertex-vrange[rank]] * numFacetCorners;
            CHKERRQ(PetscArraycpy(&(vertexFacets[offset]), &(facetListRendezvous[f*numFacetCorners]), numFacetCorners));
            CHKERRQ(PetscSectionGetOffset(facetSectionIDs, vertex-vrange[rank], &offset));
            offset += vertexIdx[vertex-vrange[rank]];
            vertexFacetIDs[offset] = facetIDsRendezvous[f];
            vertexIdx[vertex-vrange[rank]]++;
          }
        }
      }
      /* Distribute the vertex-wise facet connectivities over the vertexSF */
      CHKERRQ(PetscSectionCreate(comm, &facetSectionRemote));
      CHKERRQ(DMPlexDistributeData(*dm, sfVertices, facetSectionVertices, MPIU_INT, vertexFacets, facetSectionRemote, (void**) &facetListRemote));
      CHKERRQ(PetscSectionCreate(comm, &facetSectionIDsRemote));
      CHKERRQ(DMPlexDistributeData(*dm, sfVertices, facetSectionIDs, MPIU_INT, vertexFacetIDs, facetSectionIDsRemote, (void**) &facetIDsRemote));
      /* Convert facet connectivities to local vertex numbering */
      CHKERRQ(PetscSectionGetStorageSize(facetSectionRemote, &sizeVertexFacets));
      CHKERRQ(ISLocalToGlobalMappingCreateSF(sfVertices, vrange[rank], &ltogVertexNumbering));
      CHKERRQ(ISGlobalToLocalMappingApplyBlock(ltogVertexNumbering, IS_GTOLM_MASK, sizeVertexFacets, facetListRemote, NULL, facetListRemote));
      /* Clean up */
      CHKERRQ(PetscFree(vertexFacets));
      CHKERRQ(PetscFree(vertexIdx));
      CHKERRQ(PetscFree(vertexFacetIDs));
      CHKERRQ(PetscSectionDestroy(&facetSectionVertices));
      CHKERRQ(PetscSectionDestroy(&facetSectionIDs));
      CHKERRQ(ISLocalToGlobalMappingDestroy(&ltogVertexNumbering));
    }
    {
      PetscInt offset, dof;
      DMLabel lblFaceSets;
      PetscBool verticesLocal;
      /* Identify and mark facets locally with facet joins */
      CHKERRQ(DMCreateLabel(*dm, "Face Sets"));
      CHKERRQ(DMGetLabel(*dm, "Face Sets", &lblFaceSets));
      /* We need to set a new default value here, since -1 is a legitimate facet ID */
      CHKERRQ(DMLabelSetDefaultValue(lblFaceSets, -666666666));
      for (v = 0; v < numVerticesLocal; v++) {
        CHKERRQ(PetscSectionGetOffset(facetSectionRemote, v, &offset));
        CHKERRQ(PetscSectionGetDof(facetSectionRemote, v, &dof));
        for (f = 0; f < dof; f += numFacetCorners) {
          for (verticesLocal = PETSC_TRUE, c = 0; c < numFacetCorners; ++c) {
            if (facetListRemote[offset+f+c] < 0) {verticesLocal = PETSC_FALSE; break;}
            vertices[c] = vStart + facetListRemote[offset+f+c];
          }
          if (verticesLocal) {
            CHKERRQ(DMPlexGetFullJoin(*dm, numFacetCorners, (const PetscInt*)vertices, &joinSize, &join));
            if (joinSize == 1) {
              CHKERRQ(DMLabelSetValue(lblFaceSets, join[0], facetIDsRemote[(offset+f) / numFacetCorners]));
            }
            CHKERRQ(DMPlexRestoreJoin(*dm, numFacetCorners, (const PetscInt*)vertices, &joinSize, &join));
          }
        }
      }
    }
    CHKERRQ(PetscFree(facetList));
    CHKERRQ(PetscFree(facetListRendezvous));
    CHKERRQ(PetscFree(facetListRemote));
    CHKERRQ(PetscFree(facetIDs));
    CHKERRQ(PetscFree(facetIDsRendezvous));
    CHKERRQ(PetscFree(facetIDsRemote));
    CHKERRQ(PetscLayoutDestroy(&fLayout));
    CHKERRQ(PetscSectionDestroy(&facetSectionRemote));
    CHKERRQ(PetscSectionDestroy(&facetSectionIDsRemote));
  }
  CHKERRQ(MEDfileClose(fileID));
  CHKERRQ(PetscFree(coordinates));
  CHKERRQ(PetscFree(cellList));
  CHKERRQ(PetscLayoutDestroy(&vLayout));
  CHKERRQ(PetscLayoutDestroy(&cLayout));
  CHKERRQ(PetscSFDestroy(&sfVertices));
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires Med mesh reader support. Reconfigure using --download-med");
#endif
}
