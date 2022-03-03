#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#undef I /* Very old CGNS stupidly uses I as a variable, which fails when using complex. Curse you idiot package managers */
#if defined(PETSC_HAVE_CGNS)
#include <cgnslib.h>
#include <cgns_io.h>
#endif
#if !defined(CGNS_ENUMT)
#define CGNS_ENUMT(a) a
#endif
#if !defined(CGNS_ENUMV)
#define CGNS_ENUMV(a) a
#endif

#define CHKERRCGNS(ierr) \
do { \
  int _cgns_ier = (ierr); \
  PetscCheck(!_cgns_ier,PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS error %d %s",_cgns_ier,cg_get_error()); \
} while (0)

/*@C
  DMPlexCreateCGNS - Create a DMPlex mesh from a CGNS file.

  Collective

  Input Parameters:
+ comm  - The MPI communicator
. filename - The name of the CGNS file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://www.grc.nasa.gov/WWW/cgns/CGNS_docs_current/index.html

  Level: beginner

.seealso: DMPlexCreate(), DMPlexCreateCGNS(), DMPlexCreateExodus()
@*/
PetscErrorCode DMPlexCreateCGNSFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscMPIInt    rank;
#if defined(PETSC_HAVE_CGNS)
  int cgid = -1;
#endif

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 2);
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
#if defined(PETSC_HAVE_CGNS)
  if (rank == 0) {
    CHKERRCGNS(cg_open(filename, CG_MODE_READ, &cgid));
    PetscCheckFalse(cgid <= 0,PETSC_COMM_SELF, PETSC_ERR_LIB, "cg_open(\"%s\",...) did not return a valid file ID", filename);
  }
  CHKERRQ(DMPlexCreateCGNS(comm, cgid, interpolate, dm));
  if (rank == 0) CHKERRCGNS(cg_close(cgid));
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "Loading meshes requires CGNS support. Reconfigure using --with-cgns-dir");
#endif
}

/*@
  DMPlexCreateCGNS - Create a DMPlex mesh from a CGNS file ID.

  Collective

  Input Parameters:
+ comm  - The MPI communicator
. cgid - The CG id associated with a file and obtained using cg_open
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://www.grc.nasa.gov/WWW/cgns/CGNS_docs_current/index.html

  Level: beginner

.seealso: DMPlexCreate(), DMPlexCreateExodus()
@*/
PetscErrorCode DMPlexCreateCGNS(MPI_Comm comm, PetscInt cgid, PetscBool interpolate, DM *dm)
{
#if defined(PETSC_HAVE_CGNS)
  PetscMPIInt    num_proc, rank;
  DM             cdm;
  DMLabel        label;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt      *cellStart, *vertStart, v;
  PetscInt       labelIdRange[2], labelId;
  PetscErrorCode ierr;
  /* Read from file */
  char basename[CGIO_MAX_NAME_LENGTH+1];
  char buffer[CGIO_MAX_NAME_LENGTH+1];
  int  dim    = 0, physDim = 0, coordDim =0, numVertices = 0, numCells = 0;
  int  nzones = 0;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CGNS)
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &num_proc));
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));

  /* Open CGNS II file and read basic information on rank 0, then broadcast to all processors */
  if (rank == 0) {
    int nbases, z;

    CHKERRCGNS(cg_nbases(cgid, &nbases));
    PetscCheckFalse(nbases > 1,PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a single base, not %d",nbases);
    CHKERRCGNS(cg_base_read(cgid, 1, basename, &dim, &physDim));
    CHKERRCGNS(cg_nzones(cgid, 1, &nzones));
    CHKERRQ(PetscCalloc2(nzones+1, &cellStart, nzones+1, &vertStart));
    for (z = 1; z <= nzones; ++z) {
      cgsize_t sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */

      CHKERRCGNS(cg_zone_read(cgid, 1, z, buffer, sizes));
      numVertices += sizes[0];
      numCells    += sizes[1];
      cellStart[z] += sizes[1] + cellStart[z-1];
      vertStart[z] += sizes[0] + vertStart[z-1];
    }
    for (z = 1; z <= nzones; ++z) {
      vertStart[z] += numCells;
    }
    coordDim = dim;
  }
  CHKERRMPI(MPI_Bcast(basename, CGIO_MAX_NAME_LENGTH+1, MPI_CHAR, 0, comm));
  CHKERRMPI(MPI_Bcast(&dim, 1, MPI_INT, 0, comm));
  CHKERRMPI(MPI_Bcast(&coordDim, 1, MPI_INT, 0, comm));
  CHKERRMPI(MPI_Bcast(&nzones, 1, MPI_INT, 0, comm));

  CHKERRQ(PetscObjectSetName((PetscObject) *dm, basename));
  CHKERRQ(DMSetDimension(*dm, dim));
  CHKERRQ(DMCreateLabel(*dm, "celltype"));
  CHKERRQ(DMPlexSetChart(*dm, 0, numCells+numVertices));

  /* Read zone information */
  if (rank == 0) {
    int z, c, c_loc;

    /* Read the cell set connectivity table and build mesh topology
       CGNS standard requires that cells in a zone be numbered sequentially and be pairwise disjoint. */
    /* First set sizes */
    for (z = 1, c = 0; z <= nzones; ++z) {
      CGNS_ENUMT(ZoneType_t)    zonetype;
      int                       nsections;
      CGNS_ENUMT(ElementType_t) cellType;
      cgsize_t                  start, end;
      int                       nbndry, parentFlag;
      PetscInt                  numCorners;
      DMPolytopeType            ctype;

      CHKERRCGNS(cg_zone_type(cgid, 1, z, &zonetype));
      PetscCheckFalse(zonetype == CGNS_ENUMV(Structured),PETSC_COMM_SELF,PETSC_ERR_LIB,"Can only handle Unstructured zones for CGNS");
      CHKERRCGNS(cg_nsections(cgid, 1, z, &nsections));
      PetscCheckFalse(nsections > 1,PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a single section, not %d",nsections);
      CHKERRCGNS(cg_section_read(cgid, 1, z, 1, buffer, &cellType, &start, &end, &nbndry, &parentFlag));
      /* This alone is reason enough to bludgeon every single CGNDS developer, this must be what they describe as the "idiocy of crowds" */
      if (cellType == CGNS_ENUMV(MIXED)) {
        cgsize_t elementDataSize, *elements;
        PetscInt off;

        CHKERRCGNS(cg_ElementDataSize(cgid, 1, z, 1, &elementDataSize));
        CHKERRQ(PetscMalloc1(elementDataSize, &elements));
        CHKERRCGNS(cg_poly_elements_read(cgid, 1, z, 1, elements, NULL, NULL));
        for (c_loc = start, off = 0; c_loc <= end; ++c_loc, ++c) {
          switch (elements[off]) {
          case CGNS_ENUMV(BAR_2):   numCorners = 2; ctype = DM_POLYTOPE_SEGMENT;       break;
          case CGNS_ENUMV(TRI_3):   numCorners = 3; ctype = DM_POLYTOPE_TRIANGLE;      break;
          case CGNS_ENUMV(QUAD_4):  numCorners = 4; ctype = DM_POLYTOPE_QUADRILATERAL; break;
          case CGNS_ENUMV(TETRA_4): numCorners = 4; ctype = DM_POLYTOPE_TETRAHEDRON;   break;
          case CGNS_ENUMV(PENTA_6): numCorners = 6; ctype = DM_POLYTOPE_TRI_PRISM;     break;
          case CGNS_ENUMV(HEXA_8):  numCorners = 8; ctype = DM_POLYTOPE_HEXAHEDRON;    break;
          default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) elements[off]);
          }
          CHKERRQ(DMPlexSetConeSize(*dm, c, numCorners));
          CHKERRQ(DMPlexSetCellType(*dm, c, ctype));
          off += numCorners+1;
        }
        CHKERRQ(PetscFree(elements));
      } else {
        switch (cellType) {
        case CGNS_ENUMV(BAR_2):   numCorners = 2; ctype = DM_POLYTOPE_SEGMENT;       break;
        case CGNS_ENUMV(TRI_3):   numCorners = 3; ctype = DM_POLYTOPE_TRIANGLE;      break;
        case CGNS_ENUMV(QUAD_4):  numCorners = 4; ctype = DM_POLYTOPE_QUADRILATERAL; break;
        case CGNS_ENUMV(TETRA_4): numCorners = 4; ctype = DM_POLYTOPE_TETRAHEDRON;   break;
        case CGNS_ENUMV(PENTA_6): numCorners = 6; ctype = DM_POLYTOPE_TRI_PRISM;     break;
        case CGNS_ENUMV(HEXA_8):  numCorners = 8; ctype = DM_POLYTOPE_HEXAHEDRON;    break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) cellType);
        }
        for (c_loc = start; c_loc <= end; ++c_loc, ++c) {
          CHKERRQ(DMPlexSetConeSize(*dm, c, numCorners));
          CHKERRQ(DMPlexSetCellType(*dm, c, ctype));
        }
      }
    }
    for (v = numCells; v < numCells+numVertices; ++v) {
      CHKERRQ(DMPlexSetCellType(*dm, v, DM_POLYTOPE_POINT));
    }
  }

  CHKERRQ(DMSetUp(*dm));

  CHKERRQ(DMCreateLabel(*dm, "zone"));
  if (rank == 0) {
    int z, c, c_loc, v_loc;

    CHKERRQ(DMGetLabel(*dm, "zone", &label));
    for (z = 1, c = 0; z <= nzones; ++z) {
      CGNS_ENUMT(ElementType_t)   cellType;
      cgsize_t                    elementDataSize, *elements, start, end;
      int                          nbndry, parentFlag;
      PetscInt                    *cone, numc, numCorners, maxCorners = 27;

      CHKERRCGNS(cg_section_read(cgid, 1, z, 1, buffer, &cellType, &start, &end, &nbndry, &parentFlag));
      numc = end - start;
      /* This alone is reason enough to bludgeon every single CGNDS developer, this must be what they describe as the "idiocy of crowds" */
      CHKERRCGNS(cg_ElementDataSize(cgid, 1, z, 1, &elementDataSize));
      CHKERRQ(PetscMalloc2(elementDataSize,&elements,maxCorners,&cone));
      CHKERRCGNS(cg_poly_elements_read(cgid, 1, z, 1, elements, NULL, NULL));
      if (cellType == CGNS_ENUMV(MIXED)) {
        /* CGNS uses Fortran-based indexing, DMPlex uses C-style and numbers cell first then vertices. */
        for (c_loc = 0, v = 0; c_loc <= numc; ++c_loc, ++c) {
          switch (elements[v]) {
          case CGNS_ENUMV(BAR_2):   numCorners = 2; break;
          case CGNS_ENUMV(TRI_3):   numCorners = 3; break;
          case CGNS_ENUMV(QUAD_4):  numCorners = 4; break;
          case CGNS_ENUMV(TETRA_4): numCorners = 4; break;
          case CGNS_ENUMV(PENTA_6): numCorners = 6; break;
          case CGNS_ENUMV(HEXA_8):  numCorners = 8; break;
          default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) elements[v]);
          }
          ++v;
          for (v_loc = 0; v_loc < numCorners; ++v_loc, ++v) {
            cone[v_loc] = elements[v]+numCells-1;
          }
          CHKERRQ(DMPlexReorderCell(*dm, c, cone));
          CHKERRQ(DMPlexSetCone(*dm, c, cone));
          CHKERRQ(DMLabelSetValue(label, c, z));
        }
      } else {
        switch (cellType) {
        case CGNS_ENUMV(BAR_2):   numCorners = 2; break;
        case CGNS_ENUMV(TRI_3):   numCorners = 3; break;
        case CGNS_ENUMV(QUAD_4):  numCorners = 4; break;
        case CGNS_ENUMV(TETRA_4): numCorners = 4; break;
        case CGNS_ENUMV(PENTA_6): numCorners = 6; break;
        case CGNS_ENUMV(HEXA_8):  numCorners = 8; break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) cellType);
        }
        /* CGNS uses Fortran-based indexing, DMPlex uses C-style and numbers cell first then vertices. */
        for (c_loc = 0, v = 0; c_loc <= numc; ++c_loc, ++c) {
          for (v_loc = 0; v_loc < numCorners; ++v_loc, ++v) {
            cone[v_loc] = elements[v]+numCells-1;
          }
          CHKERRQ(DMPlexReorderCell(*dm, c, cone));
          CHKERRQ(DMPlexSetCone(*dm, c, cone));
          CHKERRQ(DMLabelSetValue(label, c, z));
        }
      }
      CHKERRQ(PetscFree2(elements,cone));
    }
  }

  CHKERRQ(DMPlexSymmetrize(*dm));
  CHKERRQ(DMPlexStratify(*dm));
  if (interpolate) {
    DM idm;

    CHKERRQ(DMPlexInterpolate(*dm, &idm));
    CHKERRQ(DMDestroy(dm));
    *dm  = idm;
  }

  /* Read coordinates */
  CHKERRQ(DMSetCoordinateDim(*dm, coordDim));
  CHKERRQ(DMGetCoordinateDM(*dm, &cdm));
  CHKERRQ(DMGetLocalSection(cdm, &coordSection));
  CHKERRQ(PetscSectionSetNumFields(coordSection, 1));
  CHKERRQ(PetscSectionSetFieldComponents(coordSection, 0, coordDim));
  CHKERRQ(PetscSectionSetChart(coordSection, numCells, numCells + numVertices));
  for (v = numCells; v < numCells+numVertices; ++v) {
    CHKERRQ(PetscSectionSetDof(coordSection, v, dim));
    CHKERRQ(PetscSectionSetFieldDof(coordSection, v, 0, coordDim));
  }
  CHKERRQ(PetscSectionSetUp(coordSection));

  CHKERRQ(DMCreateLocalVector(cdm, &coordinates));
  CHKERRQ(VecGetArray(coordinates, &coords));
  if (rank == 0) {
    PetscInt off = 0;
    float   *x[3];
    int      z, d;

    CHKERRQ(PetscMalloc3(numVertices,&x[0],numVertices,&x[1],numVertices,&x[2]));
    for (z = 1; z <= nzones; ++z) {
      CGNS_ENUMT(DataType_t) datatype;
      cgsize_t               sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */
      cgsize_t               range_min[3] = {1, 1, 1};
      cgsize_t               range_max[3] = {1, 1, 1};
      int                    ngrids, ncoords;

      CHKERRCGNS(cg_zone_read(cgid, 1, z, buffer, sizes));
      range_max[0] = sizes[0];
      CHKERRCGNS(cg_ngrids(cgid, 1, z, &ngrids));
      PetscCheckFalse(ngrids > 1,PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a single grid, not %d",ngrids);
      CHKERRCGNS(cg_ncoords(cgid, 1, z, &ncoords));
      PetscCheckFalse(ncoords != coordDim,PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a coordinate array for each dimension, not %d",ncoords);
      for (d = 0; d < coordDim; ++d) {
        CHKERRCGNS(cg_coord_info(cgid, 1, z, 1+d, &datatype, buffer));
        CHKERRCGNS(cg_coord_read(cgid, 1, z, buffer, CGNS_ENUMV(RealSingle), range_min, range_max, x[d]));
      }
      if (coordDim >= 1) {
        for (v = 0; v < sizes[0]; ++v) coords[(v+off)*coordDim+0] = x[0][v];
      }
      if (coordDim >= 2) {
        for (v = 0; v < sizes[0]; ++v) coords[(v+off)*coordDim+1] = x[1][v];
      }
      if (coordDim >= 3) {
        for (v = 0; v < sizes[0]; ++v) coords[(v+off)*coordDim+2] = x[2][v];
      }
      off += sizes[0];
    }
    CHKERRQ(PetscFree3(x[0],x[1],x[2]));
  }
  CHKERRQ(VecRestoreArray(coordinates, &coords));

  CHKERRQ(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  CHKERRQ(VecSetBlockSize(coordinates, coordDim));
  CHKERRQ(DMSetCoordinatesLocal(*dm, coordinates));
  CHKERRQ(VecDestroy(&coordinates));

  /* Read boundary conditions */
  CHKERRQ(DMGetNumLabels(*dm, &labelIdRange[0]));
  if (rank == 0) {
    CGNS_ENUMT(BCType_t)        bctype;
    CGNS_ENUMT(DataType_t)      datatype;
    CGNS_ENUMT(PointSetType_t)  pointtype;
    cgsize_t                    *points;
    PetscReal                   *normals;
    int                         normal[3];
    char                        *bcname = buffer;
    cgsize_t                    npoints, nnormals;
    int                         z, nbc, bc, c, ndatasets;

    for (z = 1; z <= nzones; ++z) {
      CHKERRCGNS(cg_nbocos(cgid, 1, z, &nbc));
      for (bc = 1; bc <= nbc; ++bc) {
        CHKERRCGNS(cg_boco_info(cgid, 1, z, bc, bcname, &bctype, &pointtype, &npoints, normal, &nnormals, &datatype, &ndatasets));
        CHKERRQ(DMCreateLabel(*dm, bcname));
        CHKERRQ(DMGetLabel(*dm, bcname, &label));
        CHKERRQ(PetscMalloc2(npoints, &points, nnormals, &normals));
        CHKERRCGNS(cg_boco_read(cgid, 1, z, bc, points, (void *) normals));
        if (pointtype == CGNS_ENUMV(ElementRange)) {
          /* Range of cells: assuming half-open interval since the documentation sucks */
          for (c = points[0]; c < points[1]; ++c) {
            CHKERRQ(DMLabelSetValue(label, c - cellStart[z-1], 1));
          }
        } else if (pointtype == CGNS_ENUMV(ElementList)) {
          /* List of cells */
          for (c = 0; c < npoints; ++c) {
            CHKERRQ(DMLabelSetValue(label, points[c] - cellStart[z-1], 1));
          }
        } else if (pointtype == CGNS_ENUMV(PointRange)) {
          CGNS_ENUMT(GridLocation_t) gridloc;

          /* List of points: Oh please, someone get the CGNS developers away from a computer. This is unconscionable. */
          CHKERRCGNS(cg_goto(cgid, 1, "Zone_t", z, "BC_t", bc, "end"));
          CHKERRCGNS(cg_gridlocation_read(&gridloc));
          /* Range of points: assuming half-open interval since the documentation sucks */
          for (c = points[0]; c < points[1]; ++c) {
            if (gridloc == CGNS_ENUMV(Vertex)) CHKERRQ(DMLabelSetValue(label, c - vertStart[z-1], 1));
            else                               CHKERRQ(DMLabelSetValue(label, c - cellStart[z-1], 1));
          }
        } else if (pointtype == CGNS_ENUMV(PointList)) {
          CGNS_ENUMT(GridLocation_t) gridloc;

          /* List of points: Oh please, someone get the CGNS developers away from a computer. This is unconscionable. */
          CHKERRCGNS(cg_goto(cgid, 1, "Zone_t", z, "BC_t", bc, "end"));
          CHKERRCGNS(cg_gridlocation_read(&gridloc));
          for (c = 0; c < npoints; ++c) {
            if (gridloc == CGNS_ENUMV(Vertex)) CHKERRQ(DMLabelSetValue(label, points[c] - vertStart[z-1], 1));
            else                               CHKERRQ(DMLabelSetValue(label, points[c] - cellStart[z-1], 1));
          }
        } else SETERRQ(comm, PETSC_ERR_SUP, "Unsupported point set type %d", (int) pointtype);
        CHKERRQ(PetscFree2(points, normals));
      }
    }
    CHKERRQ(PetscFree2(cellStart, vertStart));
  }
  CHKERRQ(DMGetNumLabels(*dm, &labelIdRange[1]));
  CHKERRMPI(MPI_Bcast(labelIdRange, 2, MPIU_INT, 0, comm));

  /* Create BC labels at all processes */
  for (labelId = labelIdRange[0]; labelId < labelIdRange[1]; ++labelId) {
    char *labelName = buffer;
    size_t len = sizeof(buffer);
    const char *locName;

    if (rank == 0) {
      CHKERRQ(DMGetLabelByNum(*dm, labelId, &label));
      CHKERRQ(PetscObjectGetName((PetscObject)label, &locName));
      CHKERRQ(PetscStrncpy(labelName, locName, len));
    }
    CHKERRMPI(MPI_Bcast(labelName, (PetscMPIInt)len, MPIU_INT, 0, comm));
    CHKERRMPI(DMCreateLabel(*dm, labelName));
  }
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires CGNS support. Reconfigure using --with-cgns-dir");
#endif
}
