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
  if (PetscUnlikely(_cgns_ier)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS error %d %s",_cgns_ier,cg_get_error()); \
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
  PetscErrorCode ierr;
#if defined(PETSC_HAVE_CGNS)
  int cgid = -1;
#endif

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 2);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
#if defined(PETSC_HAVE_CGNS)
  if (!rank) {
    ierr = cg_open(filename, CG_MODE_READ, &cgid);CHKERRCGNS(ierr);
    if (cgid <= 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "cg_open(\"%s\",...) did not return a valid file ID", filename);
  }
  ierr = DMPlexCreateCGNS(comm, cgid, interpolate, dm);CHKERRQ(ierr);
  if (!rank) {ierr = cg_close(cgid);CHKERRCGNS(ierr);}
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
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &num_proc);CHKERRMPI(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);

  /* Open CGNS II file and read basic information on rank 0, then broadcast to all processors */
  if (!rank) {
    int nbases, z;

    ierr = cg_nbases(cgid, &nbases);CHKERRCGNS(ierr);
    if (nbases > 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a single base, not %d\n",nbases);
    ierr = cg_base_read(cgid, 1, basename, &dim, &physDim);CHKERRCGNS(ierr);
    ierr = cg_nzones(cgid, 1, &nzones);CHKERRCGNS(ierr);
    ierr = PetscCalloc2(nzones+1, &cellStart, nzones+1, &vertStart);CHKERRQ(ierr);
    for (z = 1; z <= nzones; ++z) {
      cgsize_t sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */

      ierr = cg_zone_read(cgid, 1, z, buffer, sizes);CHKERRCGNS(ierr);
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
  ierr = MPI_Bcast(basename, CGIO_MAX_NAME_LENGTH+1, MPI_CHAR, 0, comm);CHKERRMPI(ierr);
  ierr = MPI_Bcast(&dim, 1, MPI_INT, 0, comm);CHKERRMPI(ierr);
  ierr = MPI_Bcast(&coordDim, 1, MPI_INT, 0, comm);CHKERRMPI(ierr);
  ierr = MPI_Bcast(&nzones, 1, MPI_INT, 0, comm);CHKERRMPI(ierr);

  ierr = PetscObjectSetName((PetscObject) *dm, basename);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMCreateLabel(*dm, "celltype");CHKERRQ(ierr);
  ierr = DMPlexSetChart(*dm, 0, numCells+numVertices);CHKERRQ(ierr);

  /* Read zone information */
  if (!rank) {
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

      ierr = cg_zone_type(cgid, 1, z, &zonetype);CHKERRCGNS(ierr);
      if (zonetype == CGNS_ENUMV(Structured)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Can only handle Unstructured zones for CGNS");
      ierr = cg_nsections(cgid, 1, z, &nsections);CHKERRCGNS(ierr);
      if (nsections > 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a single section, not %d\n",nsections);
      ierr = cg_section_read(cgid, 1, z, 1, buffer, &cellType, &start, &end, &nbndry, &parentFlag);CHKERRCGNS(ierr);
      /* This alone is reason enough to bludgeon every single CGNDS developer, this must be what they describe as the "idiocy of crowds" */
      if (cellType == CGNS_ENUMV(MIXED)) {
        cgsize_t elementDataSize, *elements;
        PetscInt off;

        ierr = cg_ElementDataSize(cgid, 1, z, 1, &elementDataSize);CHKERRCGNS(ierr);
        ierr = PetscMalloc1(elementDataSize, &elements);CHKERRQ(ierr);
        ierr = cg_poly_elements_read(cgid, 1, z, 1, elements, NULL, NULL);CHKERRCGNS(ierr);
        for (c_loc = start, off = 0; c_loc <= end; ++c_loc, ++c) {
          switch (elements[off]) {
          case CGNS_ENUMV(BAR_2):   numCorners = 2; ctype = DM_POLYTOPE_SEGMENT;       break;
          case CGNS_ENUMV(TRI_3):   numCorners = 3; ctype = DM_POLYTOPE_TRIANGLE;      break;
          case CGNS_ENUMV(QUAD_4):  numCorners = 4; ctype = DM_POLYTOPE_QUADRILATERAL; break;
          case CGNS_ENUMV(TETRA_4): numCorners = 4; ctype = DM_POLYTOPE_TETRAHEDRON;   break;
          case CGNS_ENUMV(PENTA_6): numCorners = 6; ctype = DM_POLYTOPE_TRI_PRISM;     break;
          case CGNS_ENUMV(HEXA_8):  numCorners = 8; ctype = DM_POLYTOPE_HEXAHEDRON;    break;
          default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) elements[off]);
          }
          ierr = DMPlexSetConeSize(*dm, c, numCorners);CHKERRQ(ierr);
          ierr = DMPlexSetCellType(*dm, c, ctype);CHKERRQ(ierr);
          off += numCorners+1;
        }
        ierr = PetscFree(elements);CHKERRQ(ierr);
      } else {
        switch (cellType) {
        case CGNS_ENUMV(BAR_2):   numCorners = 2; ctype = DM_POLYTOPE_SEGMENT;       break;
        case CGNS_ENUMV(TRI_3):   numCorners = 3; ctype = DM_POLYTOPE_TRIANGLE;      break;
        case CGNS_ENUMV(QUAD_4):  numCorners = 4; ctype = DM_POLYTOPE_QUADRILATERAL; break;
        case CGNS_ENUMV(TETRA_4): numCorners = 4; ctype = DM_POLYTOPE_TETRAHEDRON;   break;
        case CGNS_ENUMV(PENTA_6): numCorners = 6; ctype = DM_POLYTOPE_TRI_PRISM;     break;
        case CGNS_ENUMV(HEXA_8):  numCorners = 8; ctype = DM_POLYTOPE_HEXAHEDRON;    break;
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) cellType);
        }
        for (c_loc = start; c_loc <= end; ++c_loc, ++c) {
          ierr = DMPlexSetConeSize(*dm, c, numCorners);CHKERRQ(ierr);
          ierr = DMPlexSetCellType(*dm, c, ctype);CHKERRQ(ierr);
        }
      }
    }
    for (v = numCells; v < numCells+numVertices; ++v) {
      ierr = DMPlexSetCellType(*dm, v, DM_POLYTOPE_POINT);CHKERRQ(ierr);
    }
  }

  ierr = DMSetUp(*dm);CHKERRQ(ierr);

  ierr = DMCreateLabel(*dm, "zone");CHKERRQ(ierr);
  if (!rank) {
    int z, c, c_loc, v_loc;

    ierr = DMGetLabel(*dm, "zone", &label);CHKERRQ(ierr);
    for (z = 1, c = 0; z <= nzones; ++z) {
      CGNS_ENUMT(ElementType_t)   cellType;
      cgsize_t                    elementDataSize, *elements, start, end;
      int                          nbndry, parentFlag;
      PetscInt                    *cone, numc, numCorners, maxCorners = 27;

      ierr = cg_section_read(cgid, 1, z, 1, buffer, &cellType, &start, &end, &nbndry, &parentFlag);CHKERRCGNS(ierr);
      numc = end - start;
      /* This alone is reason enough to bludgeon every single CGNDS developer, this must be what they describe as the "idiocy of crowds" */
      ierr = cg_ElementDataSize(cgid, 1, z, 1, &elementDataSize);CHKERRCGNS(ierr);
      ierr = PetscMalloc2(elementDataSize,&elements,maxCorners,&cone);CHKERRQ(ierr);
      ierr = cg_poly_elements_read(cgid, 1, z, 1, elements, NULL, NULL);CHKERRCGNS(ierr);
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
          default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) elements[v]);
          }
          ++v;
          for (v_loc = 0; v_loc < numCorners; ++v_loc, ++v) {
            cone[v_loc] = elements[v]+numCells-1;
          }
          ierr = DMPlexReorderCell(*dm, c, cone);CHKERRQ(ierr);
          ierr = DMPlexSetCone(*dm, c, cone);CHKERRQ(ierr);
          ierr = DMLabelSetValue(label, c, z);CHKERRQ(ierr);
        }
      } else {
        switch (cellType) {
        case CGNS_ENUMV(BAR_2):   numCorners = 2; break;
        case CGNS_ENUMV(TRI_3):   numCorners = 3; break;
        case CGNS_ENUMV(QUAD_4):  numCorners = 4; break;
        case CGNS_ENUMV(TETRA_4): numCorners = 4; break;
        case CGNS_ENUMV(PENTA_6): numCorners = 6; break;
        case CGNS_ENUMV(HEXA_8):  numCorners = 8; break;
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) cellType);
        }
        /* CGNS uses Fortran-based indexing, DMPlex uses C-style and numbers cell first then vertices. */
        for (c_loc = 0, v = 0; c_loc <= numc; ++c_loc, ++c) {
          for (v_loc = 0; v_loc < numCorners; ++v_loc, ++v) {
            cone[v_loc] = elements[v]+numCells-1;
          }
          ierr = DMPlexReorderCell(*dm, c, cone);CHKERRQ(ierr);
          ierr = DMPlexSetCone(*dm, c, cone);CHKERRQ(ierr);
          ierr = DMLabelSetValue(label, c, z);CHKERRQ(ierr);
        }
      }
      ierr = PetscFree2(elements,cone);CHKERRQ(ierr);
    }
  }

  ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  if (interpolate) {
    DM idm;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }

  /* Read coordinates */
  ierr = DMSetCoordinateDim(*dm, coordDim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(*dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, coordDim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numCells, numCells + numVertices);CHKERRQ(ierr);
  for (v = numCells; v < numCells+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, coordDim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(cdm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  if (!rank) {
    PetscInt off = 0;
    float   *x[3];
    int      z, d;

    ierr = PetscMalloc3(numVertices,&x[0],numVertices,&x[1],numVertices,&x[2]);CHKERRQ(ierr);
    for (z = 1; z <= nzones; ++z) {
      CGNS_ENUMT(DataType_t) datatype;
      cgsize_t               sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */
      cgsize_t               range_min[3] = {1, 1, 1};
      cgsize_t               range_max[3] = {1, 1, 1};
      int                    ngrids, ncoords;

      ierr = cg_zone_read(cgid, 1, z, buffer, sizes);CHKERRCGNS(ierr);
      range_max[0] = sizes[0];
      ierr = cg_ngrids(cgid, 1, z, &ngrids);CHKERRCGNS(ierr);
      if (ngrids > 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a single grid, not %d\n",ngrids);
      ierr = cg_ncoords(cgid, 1, z, &ncoords);CHKERRCGNS(ierr);
      if (ncoords != coordDim) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a coordinate array for each dimension, not %d\n",ncoords);
      for (d = 0; d < coordDim; ++d) {
        ierr = cg_coord_info(cgid, 1, z, 1+d, &datatype, buffer);CHKERRCGNS(ierr);
        ierr = cg_coord_read(cgid, 1, z, buffer, CGNS_ENUMV(RealSingle), range_min, range_max, x[d]);CHKERRCGNS(ierr);
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
    ierr = PetscFree3(x[0],x[1],x[2]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, coordDim);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);

  /* Read boundary conditions */
  ierr = DMGetNumLabels(*dm, &labelIdRange[0]);CHKERRQ(ierr);
  if (!rank) {
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
      ierr = cg_nbocos(cgid, 1, z, &nbc);CHKERRCGNS(ierr);
      for (bc = 1; bc <= nbc; ++bc) {
        ierr = cg_boco_info(cgid, 1, z, bc, bcname, &bctype, &pointtype, &npoints, normal, &nnormals, &datatype, &ndatasets);CHKERRCGNS(ierr);
        ierr = DMCreateLabel(*dm, bcname);CHKERRQ(ierr);
        ierr = DMGetLabel(*dm, bcname, &label);CHKERRQ(ierr);
        ierr = PetscMalloc2(npoints, &points, nnormals, &normals);CHKERRQ(ierr);
        ierr = cg_boco_read(cgid, 1, z, bc, points, (void *) normals);CHKERRCGNS(ierr);
        if (pointtype == CGNS_ENUMV(ElementRange)) {
          /* Range of cells: assuming half-open interval since the documentation sucks */
          for (c = points[0]; c < points[1]; ++c) {
            ierr = DMLabelSetValue(label, c - cellStart[z-1], 1);CHKERRQ(ierr);
          }
        } else if (pointtype == CGNS_ENUMV(ElementList)) {
          /* List of cells */
          for (c = 0; c < npoints; ++c) {
            ierr = DMLabelSetValue(label, points[c] - cellStart[z-1], 1);CHKERRQ(ierr);
          }
        } else if (pointtype == CGNS_ENUMV(PointRange)) {
          CGNS_ENUMT(GridLocation_t) gridloc;

          /* List of points: Oh please, someone get the CGNS developers away from a computer. This is unconscionable. */
          ierr = cg_goto(cgid, 1, "Zone_t", z, "BC_t", bc, "end");CHKERRCGNS(ierr);
          ierr = cg_gridlocation_read(&gridloc);CHKERRCGNS(ierr);
          /* Range of points: assuming half-open interval since the documentation sucks */
          for (c = points[0]; c < points[1]; ++c) {
            if (gridloc == CGNS_ENUMV(Vertex)) {ierr = DMLabelSetValue(label, c - vertStart[z-1], 1);CHKERRQ(ierr);}
            else                               {ierr = DMLabelSetValue(label, c - cellStart[z-1], 1);CHKERRQ(ierr);}
          }
        } else if (pointtype == CGNS_ENUMV(PointList)) {
          CGNS_ENUMT(GridLocation_t) gridloc;

          /* List of points: Oh please, someone get the CGNS developers away from a computer. This is unconscionable. */
          ierr = cg_goto(cgid, 1, "Zone_t", z, "BC_t", bc, "end");CHKERRCGNS(ierr);
          ierr = cg_gridlocation_read(&gridloc);CHKERRCGNS(ierr);
          for (c = 0; c < npoints; ++c) {
            if (gridloc == CGNS_ENUMV(Vertex)) {ierr = DMLabelSetValue(label, points[c] - vertStart[z-1], 1);CHKERRQ(ierr);}
            else                               {ierr = DMLabelSetValue(label, points[c] - cellStart[z-1], 1);CHKERRQ(ierr);}
          }
        } else SETERRQ1(comm, PETSC_ERR_SUP, "Unsupported point set type %d", (int) pointtype);
        ierr = PetscFree2(points, normals);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree2(cellStart, vertStart);CHKERRQ(ierr);
  }
  ierr = DMGetNumLabels(*dm, &labelIdRange[1]);CHKERRQ(ierr);
  ierr = MPI_Bcast(labelIdRange, 2, MPIU_INT, 0, comm);CHKERRMPI(ierr);

  /* Create BC labels at all processes */
  for (labelId = labelIdRange[0]; labelId < labelIdRange[1]; ++labelId) {
    char *labelName = buffer;
    size_t len = sizeof(buffer);
    const char *locName;

    if (!rank) {
      ierr = DMGetLabelByNum(*dm, labelId, &label);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject)label, &locName);CHKERRQ(ierr);
      ierr = PetscStrncpy(labelName, locName, len);CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(labelName, (PetscMPIInt)len, MPIU_INT, 0, comm);CHKERRMPI(ierr);
    ierr = DMCreateLabel(*dm, labelName);CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires CGNS support. Reconfigure using --with-cgns-dir");
#endif
}
