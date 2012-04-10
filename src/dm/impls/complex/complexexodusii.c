#define PETSCDM_DLL
#include <petsc-private/compleximpl.h>    /*I   "petscdmcomplex.h"   I*/

#ifdef PETSC_HAVE_EXODUSII
#include<netcdf.h>
#include<exodusII.h>
#endif

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateExodus"
/*@
  DMComplexCreateExodus - Create a DMComplex mesh from an ExodusII file.

  Collective on comm

  Input Parameters:
+ comm  - The MPI communicator
- exoid - The ExodusII id associated with a exodus file and obtained using ex_open

  Output Parameter:
. dm  - The DM object representing the mesh

  ExodusII side sets are ignored

  Interpolated meshes are not supported.

  Level: beginner

.keywords: mesh,ExodusII
.seealso: MeshCreate(), MeshCreateExodus()
@*/
PetscErrorCode DMComplexCreateExodus(MPI_Comm comm, PetscInt exoid, DM *dm)
{
#if defined(PETSC_HAVE_EXODUSII)
  DM_Complex    *mesh;
  PetscMPIInt    num_proc, rank;
  PetscScalar   *coords;
  PetscInt       coordSize, v;
  PetscErrorCode ierr;
  /* Read from ex_get_init() */
  char           title[PETSC_MAX_PATH_LEN+1];
  int            dim = 0, numVertices = 0, numCells = 0;
  int            num_cs = 0, num_vs = 0, num_fs = 0;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_EXODUSII)
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &num_proc);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMCOMPLEX);CHKERRQ(ierr);
  mesh = (DM_Complex *) (*dm)->data;
  /* Open EXODUS II file and read basic informations on rank 0, then broadcast to all processors */
  if (!rank) {
    ierr = ex_get_init(exoid, title, &dim, &numVertices, &numCells, &num_cs, &num_vs, &num_fs);CHKERRQ(ierr);
    if (!num_cs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exodus file does not contain any cell set\n");
  }
  ierr = MPI_Bcast(title, PETSC_MAX_PATH_LEN+1, MPI_CHAR, 0, comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(&dim, 1, MPI_INT, 0, comm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, title);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMComplexSetChart(*dm, 0, numCells+numVertices);CHKERRQ(ierr);

  /* Read cell sets information */
  if (!rank) {
    PetscInt *cone;
    int  c, cs, c_loc, v, v_loc;
    /* Read from ex_get_elem_blk_ids() */
    int *cs_id;
    /* Read from ex_get_elem_block() */
    char buffer[PETSC_MAX_PATH_LEN+1];
    int  num_cell_in_set, num_vertex_per_cell, num_attr;
    /* Read from ex_get_elem_conn() */
    int *cs_connect;

    /* Get cell sets IDs */
    ierr = PetscMalloc(num_cs * sizeof(int), &cs_id);CHKERRQ(ierr);
    ierr = ex_get_elem_blk_ids(exoid, cs_id);CHKERRQ(ierr);
    /* Read the cell set connectivity table and build mesh topology
       EXO standard requires that cells in cell sets be numbered sequentially and be pairwise disjoint. */
    /* First set sizes */
    for(cs = 0, c = 0; cs < num_cs; cs++) {
      ierr = ex_get_elem_block(exoid, cs_id[cs], buffer, &num_cell_in_set, &num_vertex_per_cell, &num_attr);CHKERRQ(ierr);
      for(c_loc = 0; c_loc < num_cell_in_set; ++c_loc, ++c) {
        ierr = DMComplexSetConeSize(*dm, c, num_vertex_per_cell);CHKERRQ(ierr);
      }
    }
    ierr = DMSetUp(*dm);CHKERRQ(ierr);
    for(cs = 0, c = 0; cs < num_cs; cs++) {
      ierr = ex_get_elem_block(exoid, cs_id[cs], buffer, &num_cell_in_set, &num_vertex_per_cell, &num_attr);CHKERRQ(ierr);
      ierr = PetscMalloc2(num_vertex_per_cell*num_cell_in_set,int,&cs_connect,num_vertex_per_cell,PetscInt,&cone);CHKERRQ(ierr);
      ierr = ex_get_elem_conn(exoid, cs_id[cs], cs_connect);CHKERRQ(ierr);
      /* EXO uses Fortran-based indexing, sieve uses C-style and numbers cell first then vertices. */
      for(c_loc = 0, v = 0; c_loc < num_cell_in_set; ++c_loc, ++c) {
        for(v_loc = 0; v_loc < num_vertex_per_cell; ++v_loc, ++v) {
          cone[v_loc] = cs_connect[v]+numCells-1;
        }
        ierr = DMComplexSetCone(*dm, c, cone);CHKERRQ(ierr);
        ierr = DMComplexSetLabelValue(*dm, "Cell Sets", c, cs_id[cs]);CHKERRQ(ierr);
      }
      ierr = PetscFree2(cs_connect,cone);CHKERRQ(ierr);
    }
    ierr = PetscFree(cs_id);CHKERRQ(ierr);
  }
  ierr = DMComplexSymmetrize(*dm);CHKERRQ(ierr);
  ierr = DMComplexStratify(*dm);CHKERRQ(ierr);

  /* Create vertex set label */
  if (!rank && (num_vs > 0)) {
    int  vs, v;
    /* Read from ex_get_node_set_ids() */
    int *vs_id;
    /* Read from ex_get_node_set_param() */
    int  num_vertex_in_set, num_attr;
    /* Read from ex_get_node_set() */
    int *vs_vertex_list;

    /* Get vertex set ids */
    ierr = PetscMalloc(num_vs * sizeof(int), &vs_id);CHKERRQ(ierr);
    ierr = ex_get_node_set_ids(exoid, vs_id);CHKERRQ(ierr);
    for(vs = 0; vs < num_vs; vs++) {
      ierr = ex_get_node_set_param(exoid, vs_id[vs], &num_vertex_in_set, &num_attr);CHKERRQ(ierr);
      ierr = PetscMalloc(num_vertex_in_set * sizeof(int), &vs_vertex_list);CHKERRQ(ierr);
      ierr = ex_get_node_set(exoid, vs_id[vs], vs_vertex_list);CHKERRQ(ierr);
      for(v = 0; v < num_vertex_in_set; v++) {
        ierr = DMComplexSetLabelValue(*dm, "Vertex Sets", vs_vertex_list[v]+numCells-1, vs_id[vs]);CHKERRQ(ierr);
      }
      ierr = PetscFree(vs_vertex_list);CHKERRQ(ierr);
    }
    ierr = PetscFree(vs_id);CHKERRQ(ierr);
  }
  /* Read coordinates */
  ierr = PetscSectionSetChart(mesh->coordSection, numCells, numCells + numVertices);CHKERRQ(ierr);
  for(v = numCells; v < numCells+numVertices; ++v) {
    ierr = PetscSectionSetDof(mesh->coordSection, v, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(mesh->coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(mesh->coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecSetSizes(mesh->coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(mesh->coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(mesh->coordinates, &coords);CHKERRQ(ierr);
  if (rank == 0) {
    PetscReal *x, *y, *z;

    ierr = PetscMalloc3(numVertices,PetscReal,&x,numVertices,PetscReal,&y,numVertices,PetscReal,&z);CHKERRQ(ierr);
    ierr = ex_get_coord(exoid, x, y, z);CHKERRQ(ierr);
    if (dim > 0) {for (v = 0; v < numVertices; ++v) {coords[v*dim+0] = x[v];}}
    if (dim > 1) {for (v = 0; v < numVertices; ++v) {coords[v*dim+1] = y[v];}}
    if (dim > 2) {for (v = 0; v < numVertices; ++v) {coords[v*dim+2] = z[v];}}
    ierr = PetscFree3(x,y,z);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(mesh->coordinates, &coords);CHKERRQ(ierr);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}
