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
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CGNS)
  if (!rank) {
    ierr = cg_open(filename, CG_MODE_READ, &cgid);CHKERRQ(ierr);
    if (cgid <= 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "cg_open(\"%s\",...) did not return a valid file ID", filename);
  }
  ierr = DMPlexCreateCGNS(comm, cgid, interpolate, dm);CHKERRQ(ierr);
  if (!rank) {ierr = cg_close(cgid);CHKERRQ(ierr);}
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
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt      *cellStart, *vertStart;
  PetscInt       coordSize, v;
  PetscErrorCode ierr;
  /* Read from file */
  char basename[CGIO_MAX_NAME_LENGTH+1];
  char buffer[CGIO_MAX_NAME_LENGTH+1];
  int  dim    = 0, physDim = 0, numVertices = 0, numCells = 0;
  int  nzones = 0;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CGNS)
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &num_proc);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  /* Open CGNS II file and read basic informations on rank 0, then broadcast to all processors */
  if (!rank) {
    int nbases, z;

    ierr = cg_nbases(cgid, &nbases);CHKERRQ(ierr);
    if (nbases > 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a single base, not %d\n",nbases);
    ierr = cg_base_read(cgid, 1, basename, &dim, &physDim);CHKERRQ(ierr);
    ierr = cg_nzones(cgid, 1, &nzones);CHKERRQ(ierr);
    ierr = PetscCalloc2(nzones+1, &cellStart, nzones+1, &vertStart);CHKERRQ(ierr);
    for (z = 1; z <= nzones; ++z) {
      cgsize_t sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */

      ierr = cg_zone_read(cgid, 1, z, buffer, sizes);CHKERRQ(ierr);
      numVertices += sizes[0];
      numCells    += sizes[1];
      cellStart[z] += sizes[1] + cellStart[z-1];
      vertStart[z] += sizes[0] + vertStart[z-1];
    }
    for (z = 1; z <= nzones; ++z) {
      vertStart[z] += numCells;
    }
  }
  ierr = MPI_Bcast(basename, CGIO_MAX_NAME_LENGTH+1, MPI_CHAR, 0, comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(&dim, 1, MPI_INT, 0, comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(&nzones, 1, MPI_INT, 0, comm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, basename);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMPlexSetChart(*dm, 0, numCells+numVertices);CHKERRQ(ierr);

  /* Read zone information */
  if (!rank) {
    int z, c, c_loc, v, v_loc;

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

      ierr = cg_zone_type(cgid, 1, z, &zonetype);CHKERRQ(ierr);
      if (zonetype == CGNS_ENUMV(Structured)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Can only handle Unstructured zones for CGNS");
      ierr = cg_nsections(cgid, 1, z, &nsections);CHKERRQ(ierr);
      if (nsections > 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a single section, not %d\n",nsections);
      ierr = cg_section_read(cgid, 1, z, 1, buffer, &cellType, &start, &end, &nbndry, &parentFlag);CHKERRQ(ierr);
      /* This alone is reason enough to bludgeon every single CGNDS developer, this must be what they describe as the "idiocy of crowds" */
      if (cellType == CGNS_ENUMV(MIXED)) {
        cgsize_t elementDataSize, *elements;
        PetscInt off;

        ierr = cg_ElementDataSize(cgid, 1, z, 1, &elementDataSize);CHKERRQ(ierr);
        ierr = PetscMalloc1(elementDataSize, &elements);CHKERRQ(ierr);
        ierr = cg_elements_read(cgid, 1, z, 1, elements, NULL);CHKERRQ(ierr);
        for (c_loc = start, off = 0; c_loc <= end; ++c_loc, ++c) {
          switch (elements[off]) {
          case CGNS_ENUMV(TRI_3):   numCorners = 3;break;
          case CGNS_ENUMV(QUAD_4):  numCorners = 4;break;
          case CGNS_ENUMV(TETRA_4): numCorners = 4;break;
          case CGNS_ENUMV(HEXA_8):  numCorners = 8;break;
          default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) elements[off]);
          }
          ierr = DMPlexSetConeSize(*dm, c, numCorners);CHKERRQ(ierr);
          off += numCorners+1;
        }
        ierr = PetscFree(elements);CHKERRQ(ierr);
      } else {
        switch (cellType) {
        case CGNS_ENUMV(TRI_3):   numCorners = 3;break;
        case CGNS_ENUMV(QUAD_4):  numCorners = 4;break;
        case CGNS_ENUMV(TETRA_4): numCorners = 4;break;
        case CGNS_ENUMV(HEXA_8):  numCorners = 8;break;
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) cellType);
        }
        for (c_loc = start; c_loc <= end; ++c_loc, ++c) {
          ierr = DMPlexSetConeSize(*dm, c, numCorners);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMSetUp(*dm);CHKERRQ(ierr);
    for (z = 1, c = 0; z <= nzones; ++z) {
      CGNS_ENUMT(ElementType_t) cellType;
      cgsize_t                    *elements, elementDataSize, start, end;
      int                          nbndry, parentFlag;
      PetscInt                    *cone, numc, numCorners, maxCorners = 27;

      ierr = cg_section_read(cgid, 1, z, 1, buffer, &cellType, &start, &end, &nbndry, &parentFlag);CHKERRQ(ierr);
      numc = end - start;
      /* This alone is reason enough to bludgeon every single CGNDS developer, this must be what they describe as the "idiocy of crowds" */
      ierr = cg_ElementDataSize(cgid, 1, z, 1, &elementDataSize);CHKERRQ(ierr);
      ierr = PetscMalloc2(elementDataSize,&elements,maxCorners,&cone);CHKERRQ(ierr);
      ierr = cg_elements_read(cgid, 1, z, 1, elements, NULL);CHKERRQ(ierr);
      if (cellType == CGNS_ENUMV(MIXED)) {
        /* CGNS uses Fortran-based indexing, DMPlex uses C-style and numbers cell first then vertices. */
        for (c_loc = 0, v = 0; c_loc <= numc; ++c_loc, ++c) {
          switch (elements[v]) {
          case CGNS_ENUMV(TRI_3):   numCorners = 3;break;
          case CGNS_ENUMV(QUAD_4):  numCorners = 4;break;
          case CGNS_ENUMV(TETRA_4): numCorners = 4;break;
          case CGNS_ENUMV(HEXA_8):  numCorners = 8;break;
          default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) elements[v]);
          }
          ++v;
          for (v_loc = 0; v_loc < numCorners; ++v_loc, ++v) {
            cone[v_loc] = elements[v]+numCells-1;
          }
          /* Tetrahedra are inverted */
          if (elements[v] == CGNS_ENUMV(TETRA_4)) {
            PetscInt tmp = cone[0];
            cone[0] = cone[1];
            cone[1] = tmp;
          }
          /* Hexahedra are inverted */
          if (elements[v] == CGNS_ENUMV(HEXA_8)) {
            PetscInt tmp = cone[5];
            cone[5] = cone[7];
            cone[7] = tmp;
          }
          ierr = DMPlexSetCone(*dm, c, cone);CHKERRQ(ierr);
          ierr = DMSetLabelValue(*dm, "zone", c, z);CHKERRQ(ierr);
        }
      } else {
        switch (cellType) {
        case CGNS_ENUMV(TRI_3):   numCorners = 3;break;
        case CGNS_ENUMV(QUAD_4):  numCorners = 4;break;
        case CGNS_ENUMV(TETRA_4): numCorners = 4;break;
        case CGNS_ENUMV(HEXA_8):  numCorners = 8;break;
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) cellType);
        }

        /* CGNS uses Fortran-based indexing, DMPlex uses C-style and numbers cell first then vertices. */
        for (c_loc = 0, v = 0; c_loc <= numc; ++c_loc, ++c) {
          for (v_loc = 0; v_loc < numCorners; ++v_loc, ++v) {
            cone[v_loc] = elements[v]+numCells-1;
          }
          /* Tetrahedra are inverted */
          if (cellType == CGNS_ENUMV(TETRA_4)) {
            PetscInt tmp = cone[0];
            cone[0] = cone[1];
            cone[1] = tmp;
          }
          /* Hexahedra are inverted, and they give the top first */
          if (cellType == CGNS_ENUMV(HEXA_8)) {
            PetscInt tmp = cone[5];
            cone[5] = cone[7];
            cone[7] = tmp;
          }
          ierr = DMPlexSetCone(*dm, c, cone);CHKERRQ(ierr);
          ierr = DMSetLabelValue(*dm, "zone", c, z);CHKERRQ(ierr);
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
  ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, dim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numCells, numCells + numVertices);CHKERRQ(ierr);
  for (v = numCells; v < numCells+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
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

      ierr = cg_zone_read(cgid, 1, z, buffer, sizes);CHKERRQ(ierr);
      range_max[0] = sizes[0];
      ierr = cg_ngrids(cgid, 1, z, &ngrids);CHKERRQ(ierr);
      if (ngrids > 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a single grid, not %d\n",ngrids);
      ierr = cg_ncoords(cgid, 1, z, &ncoords);CHKERRQ(ierr);
      if (ncoords != dim) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a coordinate array for each dimension, not %d\n",ncoords);
      for (d = 0; d < dim; ++d) {
        ierr = cg_coord_info(cgid, 1, z, 1+d, &datatype, buffer);CHKERRQ(ierr);
        ierr = cg_coord_read(cgid, 1, z, buffer, CGNS_ENUMV(RealSingle), range_min, range_max, x[d]);CHKERRQ(ierr);
      }
      if (dim > 0) {
        for (v = 0; v < sizes[0]; ++v) coords[(v+off)*dim+0] = x[0][v];
      }
      if (dim > 1) {
        for (v = 0; v < sizes[0]; ++v) coords[(v+off)*dim+1] = x[1][v];
      }
      if (dim > 2) {
        for (v = 0; v < sizes[0]; ++v) coords[(v+off)*dim+2] = x[2][v];
      }
      off += sizes[0];
    }
    ierr = PetscFree3(x[0],x[1],x[2]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  /* Read boundary conditions */
  if (!rank) {
    DMLabel                     label;
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
      ierr = cg_nbocos(cgid, 1, z, &nbc);CHKERRQ(ierr);
      for (bc = 1; bc <= nbc; ++bc) {
        ierr = cg_boco_info(cgid, 1, z, bc, bcname, &bctype, &pointtype, &npoints, normal, &nnormals, &datatype, &ndatasets);CHKERRQ(ierr);
        ierr = DMCreateLabel(*dm, bcname);CHKERRQ(ierr);
        ierr = DMGetLabel(*dm, bcname, &label);CHKERRQ(ierr);
        ierr = PetscMalloc2(npoints, &points, nnormals, &normals);CHKERRQ(ierr);
        ierr = cg_boco_read(cgid, 1, z, bc, points, (void *) normals);CHKERRQ(ierr);
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
          ierr = cg_goto(cgid, 1, "Zone_t", z, "BC_t", bc, "end");CHKERRQ(ierr);
          ierr = cg_gridlocation_read(&gridloc);CHKERRQ(ierr);
          /* Range of points: assuming half-open interval since the documentation sucks */
          for (c = points[0]; c < points[1]; ++c) {
            if (gridloc == CGNS_ENUMV(Vertex)) {ierr = DMLabelSetValue(label, c - vertStart[z-1], 1);CHKERRQ(ierr);}
            else                               {ierr = DMLabelSetValue(label, c - cellStart[z-1], 1);CHKERRQ(ierr);}
          }
        } else if (pointtype == CGNS_ENUMV(PointList)) {
          CGNS_ENUMT(GridLocation_t) gridloc;

          /* List of points: Oh please, someone get the CGNS developers away from a computer. This is unconscionable. */
          ierr = cg_goto(cgid, 1, "Zone_t", z, "BC_t", bc, "end");
          ierr = cg_gridlocation_read(&gridloc);
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
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires CGNS support. Reconfigure using --with-cgns-dir");
#endif
}
