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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MED)
  mederr = MEDfileCompatibility(filename, &hdfok, &medok);
  if (mederr) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Cannot determine MED file compatibility: %s", filename);
  if (!hdfok) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Not a compatible HDF format: %s", filename);
  if (!medok) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Not a compatible MED format: %s", filename);

  fileID = MEDfileOpen(filename, MED_ACC_RDONLY);
  if (fileID < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Unable to open .med mesh file: %s", filename);
  mederr = MEDfileNumVersionRd(fileID, &major, &minor, &release);
  if (MEDnMesh (fileID) < 1) SETERRQ4(comm, PETSC_ERR_ARG_WRONG, "No meshes found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  spaceDim = MEDmeshnAxis(fileID, 1);
  if (spaceDim < 1) SETERRQ4(comm, PETSC_ERR_ARG_WRONG, "Mesh of unknown space dimension found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  /* Read general mesh information */
  ierr = PetscMalloc1(MED_SNAME_SIZE*spaceDim+1, &axisname);CHKERRQ(ierr);
  ierr = PetscMalloc1(MED_SNAME_SIZE*spaceDim+1, &unitname);CHKERRQ(ierr);
  {
    med_int medMeshDim, medNstep;
    med_int medSpaceDim = spaceDim;

    ierr = MEDmeshInfo(fileID, 1, meshname, &medSpaceDim, &medMeshDim, &meshtype, meshdescription,
                       dtunit, &sortingtype, &medNstep, &axistype, axisname, unitname);CHKERRQ(ierr);
    spaceDim = medSpaceDim;
    meshDim  = medMeshDim;
  }
  ierr = PetscFree(axisname);
  ierr = PetscFree(unitname);
  /* Partition mesh coordinates */
  numVertices = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_NODE, MED_NO_GEOTYPE,
                               MED_COORDINATE, MED_NO_CMODE,&coordinatechangement, &geotransformation);
  ierr = PetscLayoutCreate(comm, &vLayout);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(vLayout, numVertices);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(vLayout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(vLayout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRanges(vLayout, &vrange);CHKERRQ(ierr);
  numVerticesLocal = vrange[rank+1]-vrange[rank];
  ierr = MEDfilterBlockOfEntityCr(fileID, numVertices, 1, spaceDim, MED_ALL_CONSTITUENT, MED_FULL_INTERLACE, MED_COMPACT_STMODE,
                                  MED_NO_PROFILE, vrange[rank]+1, 1, numVerticesLocal, 1, 1, &vfilter);CHKERRQ(ierr);
  /* Read mesh coordinates */
  if (numVertices < 0) SETERRQ4(comm, PETSC_ERR_ARG_WRONG, "No nodes found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  ierr = PetscMalloc1(numVerticesLocal*spaceDim, &coordinates);CHKERRQ(ierr);
  ierr = MEDmeshNodeCoordinateAdvancedRd(fileID, meshname, MED_NO_DT, MED_NO_IT, &vfilter, coordinates);CHKERRQ(ierr);
  /* Read the types of entity sets in the mesh */
  ngeo = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL,MED_GEO_ALL, MED_CONNECTIVITY,
                        MED_NODAL, &coordinatechangement, &geotransformation);
  if (ngeo < 1) SETERRQ4(comm, PETSC_ERR_ARG_WRONG, "No cells found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  if (ngeo > 2) SETERRQ4(comm, PETSC_ERR_ARG_WRONG, "Currently no support for hybrid meshes in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  ierr = MEDmeshEntityInfo(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, 1, geotypename, &(geotype[0]));CHKERRQ(ierr);
  if (ngeo > 1) {ierr = MEDmeshEntityInfo(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, 2, geotypename, &(geotype[1]));CHKERRQ(ierr);}
  else geotype[1] = 0;
  /* Determine topological dim and set ID for cells */
  cellID = geotype[0]/100 > geotype[1]/100 ? 0 : 1;
  facetID = geotype[0]/100 > geotype[1]/100 ? 1 : 0;
  meshDim = geotype[cellID] / 100;
  numCorners = geotype[cellID] % 100;
  /* Partition cells */
  numCells = MEDmeshnEntity(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[cellID],
                            MED_CONNECTIVITY, MED_NODAL,&coordinatechangement, &geotransformation);
  ierr = PetscLayoutCreate(comm, &cLayout);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(cLayout, numCells);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(cLayout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(cLayout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRanges(cLayout, &crange);CHKERRQ(ierr);
  numCellsLocal = crange[rank+1]-crange[rank];
  ierr = MEDfilterBlockOfEntityCr(fileID, numCells, 1, numCorners, MED_ALL_CONSTITUENT, MED_FULL_INTERLACE, MED_COMPACT_STMODE,
                                  MED_NO_PROFILE, crange[rank]+1, 1, numCellsLocal, 1, 1, &cfilter);CHKERRQ(ierr);
  /* Read cell connectivity */
  if (numCells < 0) SETERRQ4(comm, PETSC_ERR_ARG_WRONG, "No cells found in .med v%d.%d.%d mesh file: %s", major, minor, release, filename);
  ierr = PetscMalloc1(numCellsLocal*numCorners, &medCellList);CHKERRQ(ierr);
  ierr = MEDmeshElementConnectivityAdvancedRd(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[cellID],
                                              MED_NODAL, &cfilter, medCellList);CHKERRQ(ierr);
  if (sizeof(med_int) > sizeof(PetscInt)) SETERRQ2(comm, PETSC_ERR_ARG_SIZ, "Size of PetscInt %zd less than  size of med_int %zd. Reconfigure PETSc --with-64-bit-indices=1", sizeof(PetscInt), sizeof(med_int));
  ierr = PetscMalloc1(numCellsLocal*numCorners, &cellList);CHKERRQ(ierr);
  for (i = 0; i < numCellsLocal*numCorners; i++) {
    cellList[i] = ((PetscInt) medCellList[i]) - 1; /* Correct entity counting */
  }
  ierr = PetscFree(medCellList);CHKERRQ(ierr);
  /* Generate the DM */
  if (sizeof(med_float) == sizeof(PetscReal)) {
    vertcoords = (PetscReal *) coordinates;
  } else {
    ierr = PetscMalloc1(numVerticesLocal*spaceDim, &vertcoords);CHKERRQ(ierr);
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
  ierr = DMPlexCreateFromCellListParallelPetsc(comm, meshDim, numCellsLocal, numVerticesLocal, numVertices, numCorners, interpolate, cellList, spaceDim, vertcoords, &sfVertices, dm);CHKERRQ(ierr);
  if (sizeof(med_float) == sizeof(PetscReal)) {
    vertcoords = NULL;
  } else {
    ierr = PetscFree(vertcoords);CHKERRQ(ierr);
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
    ierr = PetscLayoutCreate(comm, &fLayout);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(fLayout, numFacets);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(fLayout, 1);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(fLayout);CHKERRQ(ierr);
    ierr = PetscLayoutGetRanges(fLayout, &frange);CHKERRQ(ierr);
    numFacetsLocal = frange[rank+1]-frange[rank];
    ierr = MEDfilterBlockOfEntityCr(fileID, numFacets, 1, numFacetCorners, MED_ALL_CONSTITUENT, MED_FULL_INTERLACE, MED_COMPACT_STMODE,
                                    MED_NO_PROFILE, frange[rank]+1, 1, numFacetsLocal, 1, 1, &ffilter);CHKERRQ(ierr);
    ierr = MEDfilterBlockOfEntityCr(fileID, numFacets, 1, 1, MED_ALL_CONSTITUENT, MED_FULL_INTERLACE, MED_COMPACT_STMODE,
                                    MED_NO_PROFILE, frange[rank]+1, 1, numFacetsLocal, 1, 1, &fidfilter);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(*dm, 0, &vStart, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(numFacetsLocal, &facetIDs);CHKERRQ(ierr);
    ierr = PetscMalloc1(numFacetsLocal*numFacetCorners, &facetList);CHKERRQ(ierr);

    /* Read facet connectivity */
    {
      med_int *medFacetList;

      ierr = PetscMalloc1(numFacetsLocal*numFacetCorners, &medFacetList);CHKERRQ(ierr);
      ierr = MEDmeshElementConnectivityAdvancedRd(fileID, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[facetID], MED_NODAL, &ffilter, medFacetList);CHKERRQ(ierr);
      for (i = 0; i < numFacetsLocal*numFacetCorners; i++) {
        facetList[i] = ((PetscInt) medFacetList[i]) - 1 ; /* Correct entity counting */
      }
      ierr = PetscFree(medFacetList);
    }

    /* Read facet IDs */
    {
      med_int *medFacetIDs;

      ierr = PetscMalloc1(numFacetsLocal, &medFacetIDs);CHKERRQ(ierr);
      ierr = MEDmeshEntityAttributeAdvancedRd(fileID, meshname, MED_FAMILY_NUMBER, MED_NO_DT, MED_NO_IT, MED_CELL, geotype[facetID], &fidfilter, medFacetIDs);CHKERRQ(ierr);
      for (i = 0; i < numFacetsLocal; i++) {
        facetIDs[i] = (PetscInt) medFacetIDs[i];
      }
      ierr = PetscFree(medFacetIDs);CHKERRQ(ierr);
    }

    /* Send facets and IDs to a rendezvous partition that is based on the initial vertex partitioning. */
    {
      PetscInt           r;
      DMLabel            lblFacetRendezvous, lblFacetMigration;
      PetscSection       facetSection, facetSectionRendezvous;
      PetscSF            sfProcess, sfFacetMigration;
      const PetscSFNode *remoteVertices;
      ierr = DMLabelCreate(PETSC_COMM_SELF, "Facet Rendezvous", &lblFacetRendezvous);CHKERRQ(ierr);
      ierr = DMLabelCreate(PETSC_COMM_SELF, "Facet Migration", &lblFacetMigration);CHKERRQ(ierr);
      ierr = PetscSFGetGraph(sfVertices, NULL, NULL, NULL, &remoteVertices);CHKERRQ(ierr);
      for (f = 0; f < numFacetsLocal; f++) {
        for (v = 0; v < numFacetCorners; v++) {
          /* Find vertex owner on rendezvous partition and mark in label */
          const PetscInt vertex = facetList[f*numFacetCorners+v];
          r = rank; while (vrange[r] > vertex) r--; while (vrange[r + 1] < vertex) r++;
          ierr = DMLabelSetValue(lblFacetRendezvous, f, r);CHKERRQ(ierr);
        }
      }
      /* Build a global process SF */
      ierr = PetscSFCreate(comm,&sfProcess);CHKERRQ(ierr);
      ierr = PetscSFSetGraphWithPattern(sfProcess,NULL,PETSCSF_PATTERN_ALLTOALL);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) sfProcess, "Process SF");CHKERRQ(ierr);
      /* Convert facet rendezvous label into SF for migration */
      ierr = DMPlexPartitionLabelInvert(*dm, lblFacetRendezvous, sfProcess, lblFacetMigration);CHKERRQ(ierr);
      ierr = DMPlexPartitionLabelCreateSF(*dm, lblFacetMigration, &sfFacetMigration);CHKERRQ(ierr);
      /* Migrate facet connectivity data */
      ierr = PetscSectionCreate(comm, &facetSection);CHKERRQ(ierr);
      ierr = PetscSectionSetChart(facetSection, 0, numFacetsLocal);CHKERRQ(ierr);
      for (f = 0; f < numFacetsLocal; f++) {ierr = PetscSectionSetDof(facetSection, f, numFacetCorners);CHKERRQ(ierr);}
      ierr = PetscSectionSetUp(facetSection);CHKERRQ(ierr);
      ierr = PetscSectionCreate(comm, &facetSectionRendezvous);CHKERRQ(ierr);
      ierr = DMPlexDistributeData(*dm, sfFacetMigration, facetSection, MPIU_INT, facetList, facetSectionRendezvous, (void**) &facetListRendezvous);CHKERRQ(ierr);
      /* Migrate facet IDs */
      ierr = PetscSFGetGraph(sfFacetMigration, NULL, &numFacetsRendezvous, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscMalloc1(numFacetsRendezvous, &facetIDsRendezvous);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sfFacetMigration, MPIU_INT, facetIDs, facetIDsRendezvous);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sfFacetMigration, MPIU_INT, facetIDs, facetIDsRendezvous);CHKERRQ(ierr);
      /* Clean up */
      ierr = DMLabelDestroy(&lblFacetRendezvous);CHKERRQ(ierr);
      ierr = DMLabelDestroy(&lblFacetMigration);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&sfProcess);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&sfFacetMigration);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&facetSection);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&facetSectionRendezvous);CHKERRQ(ierr);
    }

    /* On the rendevouz partition we build a vertex-wise section/array of facets and IDs. */
    {
      PetscInt               sizeVertexFacets, offset, sizeFacetIDsRemote;
      PetscInt              *vertexFacets, *vertexIdx, *vertexFacetIDs;
      PetscSection           facetSectionVertices, facetSectionIDs;
      ISLocalToGlobalMapping ltogVertexNumbering;
      ierr = PetscSectionCreate(comm, &facetSectionVertices);CHKERRQ(ierr);
      ierr = PetscSectionSetChart(facetSectionVertices, 0, numVerticesLocal);CHKERRQ(ierr);
      ierr = PetscSectionCreate(comm, &facetSectionIDs);CHKERRQ(ierr);
      ierr = PetscSectionSetChart(facetSectionIDs, 0, numVerticesLocal);CHKERRQ(ierr);
      for (f = 0; f < numFacetsRendezvous*numFacetCorners; f++) {
        const PetscInt vertex = facetListRendezvous[f];
        if (vrange[rank] <= vertex && vertex < vrange[rank+1]) {
          ierr = PetscSectionAddDof(facetSectionIDs, vertex-vrange[rank], 1);CHKERRQ(ierr);
          ierr = PetscSectionAddDof(facetSectionVertices, vertex-vrange[rank], numFacetCorners);CHKERRQ(ierr);
        }
      }
      ierr = PetscSectionSetUp(facetSectionVertices);CHKERRQ(ierr);
      ierr = PetscSectionSetUp(facetSectionIDs);CHKERRQ(ierr);
      ierr = PetscSectionGetStorageSize(facetSectionVertices, &sizeVertexFacets);CHKERRQ(ierr);
      ierr = PetscSectionGetStorageSize(facetSectionVertices, &sizeFacetIDsRemote);CHKERRQ(ierr);
      ierr = PetscMalloc1(sizeVertexFacets, &vertexFacets);CHKERRQ(ierr);
      ierr = PetscMalloc1(sizeFacetIDsRemote, &vertexFacetIDs);CHKERRQ(ierr);
      ierr = PetscCalloc1(numVerticesLocal, &vertexIdx);CHKERRQ(ierr);
      for (f = 0; f < numFacetsRendezvous; f++) {
        for (c = 0; c < numFacetCorners; c++) {
          const PetscInt vertex = facetListRendezvous[f*numFacetCorners+c];
          if (vrange[rank] <= vertex && vertex < vrange[rank+1]) {
            /* Flip facet connectivities and IDs to a vertex-wise layout */
            ierr = PetscSectionGetOffset(facetSectionVertices, vertex-vrange[rank], &offset);
            offset += vertexIdx[vertex-vrange[rank]] * numFacetCorners;
            ierr = PetscArraycpy(&(vertexFacets[offset]), &(facetListRendezvous[f*numFacetCorners]), numFacetCorners);CHKERRQ(ierr);
            ierr = PetscSectionGetOffset(facetSectionIDs, vertex-vrange[rank], &offset);
            offset += vertexIdx[vertex-vrange[rank]];
            vertexFacetIDs[offset] = facetIDsRendezvous[f];
            vertexIdx[vertex-vrange[rank]]++;
          }
        }
      }
      /* Distribute the vertex-wise facet connectivities over the vertexSF */
      ierr = PetscSectionCreate(comm, &facetSectionRemote);CHKERRQ(ierr);
      ierr = DMPlexDistributeData(*dm, sfVertices, facetSectionVertices, MPIU_INT, vertexFacets, facetSectionRemote, (void**) &facetListRemote);
      ierr = PetscSectionCreate(comm, &facetSectionIDsRemote);CHKERRQ(ierr);
      ierr = DMPlexDistributeData(*dm, sfVertices, facetSectionIDs, MPIU_INT, vertexFacetIDs, facetSectionIDsRemote, (void**) &facetIDsRemote);
      /* Convert facet connectivities to local vertex numbering */
      ierr = PetscSectionGetStorageSize(facetSectionRemote, &sizeVertexFacets);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreateSF(sfVertices, vrange[rank], &ltogVertexNumbering);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApplyBlock(ltogVertexNumbering, IS_GTOLM_MASK, sizeVertexFacets, facetListRemote, NULL, facetListRemote);CHKERRQ(ierr);
      /* Clean up */
      ierr = PetscFree(vertexFacets);
      ierr = PetscFree(vertexIdx);
      ierr = PetscFree(vertexFacetIDs);
      ierr = PetscSectionDestroy(&facetSectionVertices);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&facetSectionIDs);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&ltogVertexNumbering);CHKERRQ(ierr);
    }
    {
      PetscInt offset, dof;
      DMLabel lblFaceSets;
      PetscBool verticesLocal;
      /* Identify and mark facets locally with facet joins */
      ierr = DMCreateLabel(*dm, "Face Sets");CHKERRQ(ierr);
      ierr = DMGetLabel(*dm, "Face Sets", &lblFaceSets);CHKERRQ(ierr);
      /* We need to set a new default value here, since -1 is a legitimate facet ID */
      ierr = DMLabelSetDefaultValue(lblFaceSets, -666666666);
      for (v = 0; v < numVerticesLocal; v++) {
        ierr = PetscSectionGetOffset(facetSectionRemote, v, &offset);
        ierr = PetscSectionGetDof(facetSectionRemote, v, &dof);
        for (f = 0; f < dof; f += numFacetCorners) {
          for (verticesLocal = PETSC_TRUE, c = 0; c < numFacetCorners; ++c) {
            if (facetListRemote[offset+f+c] < 0) {verticesLocal = PETSC_FALSE; break;}
            vertices[c] = vStart + facetListRemote[offset+f+c];
          }
          if (verticesLocal) {
            ierr = DMPlexGetFullJoin(*dm, numFacetCorners, (const PetscInt*)vertices, &joinSize, &join);CHKERRQ(ierr);
            if (joinSize == 1) {
              ierr = DMLabelSetValue(lblFaceSets, join[0], facetIDsRemote[(offset+f) / numFacetCorners]);CHKERRQ(ierr);
            }
            ierr = DMPlexRestoreJoin(*dm, numFacetCorners, (const PetscInt*)vertices, &joinSize, &join);CHKERRQ(ierr);
          }
        }
      }
    }
    ierr = PetscFree(facetList);CHKERRQ(ierr);
    ierr = PetscFree(facetListRendezvous);CHKERRQ(ierr);
    ierr = PetscFree(facetListRemote);CHKERRQ(ierr);
    ierr = PetscFree(facetIDs);CHKERRQ(ierr);
    ierr = PetscFree(facetIDsRendezvous);CHKERRQ(ierr);
    ierr = PetscFree(facetIDsRemote);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&fLayout);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&facetSectionRemote);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&facetSectionIDsRemote);CHKERRQ(ierr);
  }
  ierr = MEDfileClose(fileID);CHKERRQ(ierr);
  ierr = PetscFree(coordinates);CHKERRQ(ierr);
  ierr = PetscFree(cellList);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&vLayout);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&cLayout);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfVertices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires Med mesh reader support. Reconfigure using --download-med");
#endif
}
