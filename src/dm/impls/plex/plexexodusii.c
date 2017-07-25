#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#if defined(PETSC_HAVE_EXODUSII)
#include <netcdf.h>
#include <exodusII.h>
#endif

/*@C
  DMPlexCreateExodus - Create a DMPlex mesh from an ExodusII file.

  Collective on comm

  Input Parameters:
+ comm  - The MPI communicator
. filename - The name of the ExodusII file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Level: beginner

.keywords: mesh,ExodusII
.seealso: DMPLEX, DMCreate(), DMPlexCreateExodus()
@*/
PetscErrorCode DMPlexCreateExodusFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscMPIInt    rank;
  PetscErrorCode ierr;
#if defined(PETSC_HAVE_EXODUSII)
  int   CPU_word_size = 0, IO_word_size = 0, exoid = -1;
  float version;
#endif

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 2);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
#if defined(PETSC_HAVE_EXODUSII)
  if (!rank) {
    exoid = ex_open(filename, EX_READ, &CPU_word_size, &IO_word_size, &version);
    if (exoid <= 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "ex_open(\"%s\",...) did not return a valid file ID", filename);
  }
  ierr = DMPlexCreateExodus(comm, exoid, interpolate, dm);CHKERRQ(ierr);
  if (!rank) {ierr = ex_close(exoid);CHKERRQ(ierr);}
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateExodus - Create a DMPlex mesh from an ExodusII file ID.

  Collective on comm

  Input Parameters:
+ comm  - The MPI communicator
. exoid - The ExodusII id associated with a exodus file and obtained using ex_open
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Level: beginner

.keywords: mesh,ExodusII
.seealso: DMPLEX, DMCreate()
@*/
PetscErrorCode DMPlexCreateExodus(MPI_Comm comm, PetscInt exoid, PetscBool interpolate, DM *dm)
{
#if defined(PETSC_HAVE_EXODUSII)
  PetscMPIInt    num_proc, rank;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar    *coords;
  PetscInt       coordSize, v;
  PetscErrorCode ierr;
  /* Read from ex_get_init() */
  char title[PETSC_MAX_PATH_LEN+1];
  int  dim    = 0, numVertices = 0, numCells = 0;
  int  num_cs = 0, num_vs = 0, num_fs = 0;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_EXODUSII)
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &num_proc);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  /* Open EXODUS II file and read basic informations on rank 0, then broadcast to all processors */
  if (!rank) {
    ierr = PetscMemzero(title,(PETSC_MAX_PATH_LEN+1)*sizeof(char));CHKERRQ(ierr);
    ierr = ex_get_init(exoid, title, &dim, &numVertices, &numCells, &num_cs, &num_vs, &num_fs);
    if (ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ExodusII ex_get_init() failed with error code %D\n",ierr);
    if (!num_cs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exodus file does not contain any cell set\n");
  }
  ierr = MPI_Bcast(title, PETSC_MAX_PATH_LEN+1, MPI_CHAR, 0, comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(&dim, 1, MPI_INT, 0, comm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, title);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMPlexSetChart(*dm, 0, numCells+numVertices);CHKERRQ(ierr);

  /* Read cell sets information */
  if (!rank) {
    PetscInt *cone;
    int      c, cs, c_loc, v, v_loc;
    /* Read from ex_get_elem_blk_ids() */
    int *cs_id;
    /* Read from ex_get_elem_block() */
    char buffer[PETSC_MAX_PATH_LEN+1];
    int  num_cell_in_set, num_vertex_per_cell, num_attr;
    /* Read from ex_get_elem_conn() */
    int *cs_connect;

    /* Get cell sets IDs */
    ierr = PetscMalloc1(num_cs, &cs_id);CHKERRQ(ierr);
    ierr = ex_get_ids (exoid, EX_ELEM_BLOCK, cs_id);CHKERRQ(ierr);
    /* Read the cell set connectivity table and build mesh topology
       EXO standard requires that cells in cell sets be numbered sequentially and be pairwise disjoint. */
    /* First set sizes */
    for (cs = 0, c = 0; cs < num_cs; cs++) {
      ierr = ex_get_block(exoid, EX_ELEM_BLOCK, cs_id[cs], buffer, &num_cell_in_set,&num_vertex_per_cell, 0, 0, &num_attr);CHKERRQ(ierr);
      for (c_loc = 0; c_loc < num_cell_in_set; ++c_loc, ++c) {
        ierr = DMPlexSetConeSize(*dm, c, num_vertex_per_cell);CHKERRQ(ierr);
      }
    }
    ierr = DMSetUp(*dm);CHKERRQ(ierr);
    for (cs = 0, c = 0; cs < num_cs; cs++) {
      ierr = ex_get_block(exoid, EX_ELEM_BLOCK, cs_id[cs], buffer, &num_cell_in_set, &num_vertex_per_cell, 0, 0, &num_attr);CHKERRQ(ierr);
      ierr = PetscMalloc2(num_vertex_per_cell*num_cell_in_set,&cs_connect,num_vertex_per_cell,&cone);CHKERRQ(ierr);
      ierr = ex_get_conn (exoid, EX_ELEM_BLOCK, cs_id[cs], cs_connect,NULL,NULL);CHKERRQ(ierr);
      /* EXO uses Fortran-based indexing, sieve uses C-style and numbers cell first then vertices. */
      for (c_loc = 0, v = 0; c_loc < num_cell_in_set; ++c_loc, ++c) {
        for (v_loc = 0; v_loc < num_vertex_per_cell; ++v_loc, ++v) {
          cone[v_loc] = cs_connect[v]+numCells-1;
        }
        if (dim == 3) {
          /* Tetrahedra are inverted */
          if (num_vertex_per_cell == 4) {
            PetscInt tmp = cone[0];
            cone[0] = cone[1];
            cone[1] = tmp;
          }
          /* Hexahedra are inverted */
          if (num_vertex_per_cell == 8) {
            PetscInt tmp = cone[1];
            cone[1] = cone[3];
            cone[3] = tmp;
          }
        }
        ierr = DMPlexSetCone(*dm, c, cone);CHKERRQ(ierr);
        ierr = DMSetLabelValue(*dm, "Cell Sets", c, cs_id[cs]);CHKERRQ(ierr);
      }
      ierr = PetscFree2(cs_connect,cone);CHKERRQ(ierr);
    }
    ierr = PetscFree(cs_id);CHKERRQ(ierr);
  }
  ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  if (interpolate) {
    DM idm = NULL;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    /* Maintain Cell Sets label */
    {
      DMLabel label;

      ierr = DMRemoveLabel(*dm, "Cell Sets", &label);CHKERRQ(ierr);
      if (label) {ierr = DMAddLabel(idm, label);CHKERRQ(ierr);}
    }
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }

  /* Create vertex set label */
  if (!rank && (num_vs > 0)) {
    int vs, v;
    /* Read from ex_get_node_set_ids() */
    int *vs_id;
    /* Read from ex_get_node_set_param() */
    int num_vertex_in_set;
    /* Read from ex_get_node_set() */
    int *vs_vertex_list;

    /* Get vertex set ids */
    ierr = PetscMalloc1(num_vs, &vs_id);CHKERRQ(ierr);
    ierr = ex_get_ids(exoid, EX_NODE_SET, vs_id);CHKERRQ(ierr);
    for (vs = 0; vs < num_vs; ++vs) {
      ierr = ex_get_set_param(exoid, EX_NODE_SET, vs_id[vs], &num_vertex_in_set, NULL);
      ierr = PetscMalloc1(num_vertex_in_set, &vs_vertex_list);CHKERRQ(ierr);
      ierr = ex_get_set(exoid, EX_NODE_SET, vs_id[vs], vs_vertex_list, NULL);CHKERRQ(ierr);
      for (v = 0; v < num_vertex_in_set; ++v) {
        ierr = DMSetLabelValue(*dm, "Vertex Sets", vs_vertex_list[v]+numCells-1, vs_id[vs]);CHKERRQ(ierr);
      }
      ierr = PetscFree(vs_vertex_list);CHKERRQ(ierr);
    }
    ierr = PetscFree(vs_id);CHKERRQ(ierr);
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
  ierr = VecSetBlockSize(coordinates, dim);CHKERRQ(ierr);
  ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  if (!rank) {
    float *x, *y, *z;

    ierr = PetscMalloc3(numVertices,&x,numVertices,&y,numVertices,&z);CHKERRQ(ierr);
    ierr = ex_get_coord(exoid, x, y, z);CHKERRQ(ierr);
    if (dim > 0) {
      for (v = 0; v < numVertices; ++v) coords[v*dim+0] = x[v];
    }
    if (dim > 1) {
      for (v = 0; v < numVertices; ++v) coords[v*dim+1] = y[v];
    }
    if (dim > 2) {
      for (v = 0; v < numVertices; ++v) coords[v*dim+2] = z[v];
    }
    ierr = PetscFree3(x,y,z);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);

  /* Create side set label */
  if (!rank && interpolate && (num_fs > 0)) {
    int fs, f, voff;
    /* Read from ex_get_side_set_ids() */
    int *fs_id;
    /* Read from ex_get_side_set_param() */
    int num_side_in_set;
    /* Read from ex_get_side_set_node_list() */
    int *fs_vertex_count_list, *fs_vertex_list;
    /* Read side set labels */
    char fs_name[MAX_STR_LENGTH+1];

    /* Get side set ids */
    ierr = PetscMalloc1(num_fs, &fs_id);CHKERRQ(ierr);
    ierr = ex_get_ids(exoid, EX_SIDE_SET, fs_id);CHKERRQ(ierr);
    for (fs = 0; fs < num_fs; ++fs) {
      ierr = ex_get_set_param(exoid, EX_SIDE_SET, fs_id[fs], &num_side_in_set, NULL);CHKERRQ(ierr);
      ierr = PetscMalloc2(num_side_in_set,&fs_vertex_count_list,num_side_in_set*4,&fs_vertex_list);CHKERRQ(ierr);
      ierr = ex_get_side_set_node_list(exoid, fs_id[fs], fs_vertex_count_list, fs_vertex_list);CHKERRQ(ierr);
      /* Get the specific name associated with this side set ID. */
      int fs_name_err = ex_get_name(exoid, EX_SIDE_SET, fs_id[fs], fs_name);
      for (f = 0, voff = 0; f < num_side_in_set; ++f) {
        const PetscInt *faces   = NULL;
        PetscInt       faceSize = fs_vertex_count_list[f], numFaces;
        PetscInt       faceVertices[4], v;

        if (faceSize > 4) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "ExodusII side cannot have %d > 4 vertices", faceSize);
        for (v = 0; v < faceSize; ++v, ++voff) {
          faceVertices[v] = fs_vertex_list[voff]+numCells-1;
        }
        ierr = DMPlexGetFullJoin(*dm, faceSize, faceVertices, &numFaces, &faces);CHKERRQ(ierr);
        if (numFaces != 1) SETERRQ3(comm, PETSC_ERR_ARG_WRONG, "Invalid ExodusII side %d in set %d maps to %d faces", f, fs, numFaces);
        ierr = DMSetLabelValue(*dm, "Face Sets", faces[0], fs_id[fs]);CHKERRQ(ierr);
        /* Only add the label if one has been detected for this side set. */
        if (!fs_name_err) {
          ierr = DMSetLabelValue(*dm, fs_name, faces[0], fs_id[fs]);CHKERRQ(ierr);
        }
        ierr = DMPlexRestoreJoin(*dm, faceSize, faceVertices, &numFaces, &faces);CHKERRQ(ierr);
      }
      ierr = PetscFree2(fs_vertex_count_list,fs_vertex_list);CHKERRQ(ierr);
    }
    ierr = PetscFree(fs_id);CHKERRQ(ierr);
  }
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}
