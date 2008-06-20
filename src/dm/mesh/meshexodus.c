#include<petscmesh_formats.hh>   /*I      "petscmesh.h"   I*/

#ifdef PETSC_HAVE_EXODUS

#include<netcdf.h>
#include<exodusII.h>

#undef __FUNCT__
#define __FUNCT__ "PetscReadExodusII"
PetscErrorCode PetscReadExodusII(MPI_Comm comm, const char filename[], ALE::Obj<PETSC_MESH_TYPE>& mesh)
{
  int   exoid;
  int   CPU_word_size = 0, IO_word_size = 0;
  float version;
  char  title[MAX_LINE_LENGTH+1], elem_type[MAX_STR_LENGTH+1];
  int   num_dim, num_nodes, num_elem, num_elem_blk, num_node_sets, num_side_sets;
  int   ierr;

  PetscFunctionBegin;

  // Open EXODUS II file
  exoid = ex_open(filename, EX_READ, &CPU_word_size, &IO_word_size, &version);CHKERRQ(!exoid);
  // Read database parameters
  ierr = ex_get_init(exoid, title, &num_dim, &num_nodes, &num_elem, &num_elem_blk, &num_node_sets, &num_side_sets);CHKERRQ(ierr);
  printf ("database parameters:\n");
  printf ("title =  '%s'\n",title);
  printf ("num_dim = %3d\n",num_dim);
  printf ("num_nodes = %3d\n",num_nodes);
  printf ("num_elem = %3d\n",num_elem);
  printf ("num_elem_blk = %3d\n",num_elem_blk);
  printf ("num_node_sets = %3d\n",num_node_sets);
  printf ("num_side_sets = %3d\n",num_side_sets);

  // Read vertex coordinates
  float *x, *y, *z;
  ierr = PetscMalloc3(num_nodes,float,&x,num_nodes,float,&y,num_nodes,float,&z);CHKERRQ(ierr);
  ierr = ex_get_coord(exoid, x, y, z);CHKERRQ(ierr);
  printf ("x coords = \n"); for(int i=0; i<num_nodes; i++) {printf ("%5.1f\n", x[i]);}
  printf ("y coords = \n"); for(int i=0; i<num_nodes; i++) {printf ("%5.1f\n", y[i]);}
  if (num_dim >= 3) {
    printf ("z coords = \n"); for(int i=0; i<num_nodes; i++) {printf ("%5.1f\n", z[i]);}
  }

  int   *eb_ids, *num_elem_in_block, *num_nodes_per_elem, *num_attr;
  int  **connect;
  char **block_names;
  if (num_elem_blk > 0) {
    ierr = PetscMalloc5(num_elem_blk,int,&eb_ids,num_elem_blk,int,&num_elem_in_block,num_elem_blk,int,&num_nodes_per_elem,num_elem_blk,int,&num_attr,num_elem_blk,char*,&block_names);CHKERRQ(ierr);
    ierr = ex_get_elem_blk_ids(exoid, eb_ids);CHKERRQ(ierr);
    for(int eb = 0; eb < num_elem_blk; ++eb) {
      ierr = PetscMalloc((MAX_STR_LENGTH+1) * sizeof(char), &block_names[eb]);CHKERRQ(ierr);
    }
    ierr = ex_get_names(exoid, EX_ELEM_BLOCK, block_names);CHKERRQ(ierr);
    for(int eb = 0; eb < num_elem_blk; ++eb) {
      ierr = ex_get_elem_block(exoid, eb_ids[eb], elem_type, &num_elem_in_block[eb], &num_nodes_per_elem[eb], &num_attr[eb]);CHKERRQ(ierr);
      printf ("element block id = %2d\n",eb_ids[eb]);
      printf ("element type = '%s'\n", elem_type);
      printf ("num_elem_in_block = %2d\n",num_elem_in_block[eb]);
      printf ("num_nodes_per_elem = %2d\n",num_nodes_per_elem[eb]);
      printf ("num_attr = %2d\n",num_attr[eb]);
      printf ("name = '%s'\n",block_names[eb]);
      ierr = PetscFree(block_names[eb]);CHKERRQ(ierr);
    }

    // Read element connectivity
    ierr = PetscMalloc(num_elem_blk * sizeof(int*),&connect);CHKERRQ(ierr);
    for(int eb = 0; eb < num_elem_blk; ++eb) {
      if (num_elem_in_block[eb] > 0) {
        ierr = PetscMalloc(num_nodes_per_elem[eb]*num_elem_in_block[eb] * sizeof(int),&connect[eb]);CHKERRQ(ierr);
        ierr = ex_get_elem_conn(exoid, eb_ids[eb], connect[eb]);CHKERRQ(ierr);
        printf("connect array for elem block %2d\n", eb_ids[eb]);
        for(int j = 0; j < num_nodes_per_elem[eb]; ++j) {printf("%3d\n", connect[eb][j]);}
      }
    }
  }

  int *ns_ids;
  if (num_node_sets > 0) {
    ierr = PetscMalloc(num_node_sets * sizeof(int), &ns_ids);CHKERRQ(ierr);
    ierr = ex_get_node_set_ids(exoid, ns_ids);CHKERRQ(ierr);
    for(int ns = 0; ns < num_node_sets; ++ns) {
      int num_nodes_in_set, num_df_in_set;
      ierr = ex_get_node_set_param (exoid, ns_ids[ns], &num_nodes_in_set, &num_df_in_set);CHKERRQ(ierr);
      printf ("\nnode set %2d parameters: \n", ns_ids[ns]);
      printf ("num_nodes = %2d\n", num_nodes_in_set);
      int *node_list;
      ierr = PetscMalloc(num_nodes_in_set * sizeof(int), &node_list);CHKERRQ(ierr);
      ierr = ex_get_node_set(exoid, ns_ids[ns], node_list);
      printf ("\nnode list for node set %2d\n", ns_ids[ns]);
      for(int j = 0; j < num_nodes_in_set; ++j) {printf ("%3d\n", node_list[j]);}
      ierr = PetscFree(node_list);CHKERRQ(ierr);
	}
  }
  ierr = ex_close(exoid);CHKERRQ(ierr);

  int  numCorners = num_nodes_per_elem[0];
  int *cells;
  mesh->setDimension(num_dim);
  ierr = PetscMalloc(numCorners*num_elem * sizeof(int), &cells);CHKERRQ(ierr);
  for(int eb = 0, k = 0; eb < num_elem_blk; ++eb) {
    for(int e = 0; e < num_elem_in_block[eb]; ++e, ++k) {
      for(int c = 0; c < numCorners; ++c) {
        cells[k*numCorners+c] = connect[eb][e*numCorners+c];
      }
    }
    ierr = PetscFree(connect[eb]);CHKERRQ(ierr);
  }
  ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(mesh->comm(), mesh->debug());
  ALE::Obj<ALE::Mesh::sieve_type>       s     = new ALE::Mesh::sieve_type(mesh->comm(), mesh->debug());
  ALE::SieveBuilder<ALE::Mesh>::buildTopology(s, num_dim, num_elem, cells, num_nodes, false, numCorners);
  std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> renumbering;
  ALE::ISieveConverter::convertSieve(*s, *sieve, renumbering);
  mesh->setSieve(sieve);
  mesh->stratify();
  ierr = PetscFree(cells);CHKERRQ(ierr);
  if (num_elem_blk > 0) {
    ierr = PetscFree(connect);CHKERRQ(ierr);
    ierr = PetscFree5(eb_ids, num_elem_in_block, num_nodes_per_elem, num_attr, block_names);CHKERRQ(ierr);
  }
  double *coords;
  ierr = PetscMalloc(num_dim*num_nodes * sizeof(double), &coords);CHKERRQ(ierr);
  if (num_dim > 0) {for(int v = 0; v < num_nodes; ++v) {coords[v*num_dim+0] = x[v];}}
  if (num_dim > 1) {for(int v = 0; v < num_nodes; ++v) {coords[v*num_dim+1] = y[v];}}
  if (num_dim > 2) {for(int v = 0; v < num_nodes; ++v) {coords[v*num_dim+2] = z[v];}}
  ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(mesh, num_dim, coords);
  ierr = PetscFree(coords);CHKERRQ(ierr);
  ierr = PetscFree3(x, y, z);CHKERRQ(ierr);
  // Do vertex and cells sets
  if (num_node_sets > 0) {
    ierr = PetscFree(ns_ids);CHKERRQ(ierr);
  }
  mesh->view("Mesh");
  // Get coords and print in F90
  // Get connectivity and print in F90
  // Calculate cost function
  // Do in parallel
  //   Read in parallel
  //   Distribute
  //   Print out local meshes
  //   Do Blaise's assembly loop in parallel
  // Assemble function into Section
  // Assemble jacobian into Mat
  // Assemble in parallel
  // Convert Section to Vec
  PetscFunctionReturn(0);
}

#endif // PETSC_HAVE_EXODUS

#undef __FUNCT__
#define __FUNCT__ "MeshCreateExodus"
/*@C
  MeshCreateExodus - Create a Mesh from an ExodusII file.

  Not Collective

  Input Parameters:
+ comm - The MPI communicator
- filename - The ExodusII filename

  Output Parameter:
. mesh - The Mesh object

  Level: beginner

.keywords: mesh, ExodusII
.seealso: MeshCreate()
@*/
PetscErrorCode MeshCreateExodus(MPI_Comm comm, const char filename[], Mesh *mesh)
{
  PetscInt       debug = 0;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  ALE::Obj<PETSC_MESH_TYPE> m = new PETSC_MESH_TYPE(comm, -1, debug);
#ifdef PETSC_HAVE_EXODUS
  ierr = PetscReadExodusII(comm, filename, m);CHKERRQ(ierr);
#else
  SETERRQ(PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --with-exodus-dir=/path/to/exodus");
#endif
  if (debug) {m->view("Mesh");}
  ierr = MeshSetMesh(*mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshExodusGetInfo"
/*@
  MeshExodusGetInfo - Get information about an ExodusII Mesh.

  Not Collective

  Input Parameter:
. mesh - The Mesh object

  Output Parameters:
+ dim - The mesh dimension
. numVertices - The number of vertices in the mesh
. numCells - The number of cells in the mesh
. numCellBlocks - The number of cell blocks in the mesh
- numVertexSets - The number of vertex sets in the mesh

  Level: beginner

.keywords: mesh, ExodusII
.seealso: MeshCreateExodus()
@*/
PetscErrorCode MeshExodusGetInfo(Mesh mesh, PetscInt *dim, PetscInt *numVertices, PetscInt *numCells, PetscInt *numCellBlocks, PetscInt *numVertexSets)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  *dim           = m->getDimension();
  *numVertices   = m->depthStratum(0)->size();
  *numCells      = m->heightStratum(0)->size();
  *numCellBlocks = m->getLabel("CellBlocks")->getCapSize();
  *numVertexSets = m->getLabel("VertexSets")->getCapSize();
  PetscFunctionReturn(0);
}
