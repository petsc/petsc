#include<petscmesh_formats.hh>   /*I      "petscmesh.h"   I*/

#ifdef PETSC_HAVE_EXODUS

#include<netcdf.h>
#include<exodusII.h>

#undef __FUNCT__
#define __FUNCT__ "PetscReadExodusII"
PetscErrorCode PetscReadExodusII(MPI_Comm comm, const char filename[], PETSC_MESH_TYPE& mesh)
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

  mesh.setDimension(num_dim);
  // Write coord and connectivity code
  // Get Blaise repo
  // Replace info function
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
  ierr = PetscReadExodusII(comm, filename, *m);CHKERRQ(ierr);
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
  *numCellBlocks = 0;
  *numVertexSets = 0;
  PetscFunctionReturn(0);
}
