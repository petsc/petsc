#include<petscmesh_formats.hh>   /*I      "petscmesh.h"   I*/

#ifdef PETSC_HAVE_EXODUSII

// This is what I needed in my petscvariables:
//
// EXODUSII_INCLUDE = -I/PETSc3/mesh/exodusii-4.71/cbind/include
// EXODUSII_LIB = -L/PETSc3/mesh/exodusii-4.71/forbind/src -lexoIIv2for -L/PETSc3/mesh/exodusii-4.71/cbind/src -lexoIIv2c -lnetcdf

#include<netcdf.h>
#include<exodusII.h>

#undef __FUNCT__
#define __FUNCT__ "PetscReadExodusII"
PetscErrorCode PetscReadExodusII(MPI_Comm comm, const char filename[], ALE::Obj<PETSC_MESH_TYPE>& mesh)
{
  int   exoid;
  int   CPU_word_size = 0, IO_word_size = 0;
  const PetscMPIInt rank = mesh->commRank();
  float version;
  char  title[MAX_LINE_LENGTH+1], elem_type[MAX_STR_LENGTH+1];
  int   num_dim, num_nodes, num_elem, num_elem_blk, num_node_sets, num_side_sets;
  int   ierr;

  PetscFunctionBegin;
  // Open EXODUS II file
  exoid = ex_open(filename, EX_READ, &CPU_word_size, &IO_word_size, &version);CHKERRQ(!exoid);
  // Read database parameters
  ierr = ex_get_init(exoid, title, &num_dim, &num_nodes, &num_elem, &num_elem_blk, &num_node_sets, &num_side_sets);CHKERRQ(ierr);

  // Read vertex coordinates
  float *x, *y, *z;
  ierr = PetscMalloc3(num_nodes,float,&x,num_nodes,float,&y,num_nodes,float,&z);CHKERRQ(ierr);
  ierr = ex_get_coord(exoid, x, y, z);CHKERRQ(ierr);

  // Read element connectivity
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
      ierr = PetscFree(block_names[eb]);CHKERRQ(ierr);
    }
    ierr = PetscMalloc(num_elem_blk * sizeof(int*),&connect);CHKERRQ(ierr);
    for(int eb = 0; eb < num_elem_blk; ++eb) {
      if (num_elem_in_block[eb] > 0) {
        ierr = PetscMalloc(num_nodes_per_elem[eb]*num_elem_in_block[eb] * sizeof(int),&connect[eb]);CHKERRQ(ierr);
        ierr = ex_get_elem_conn(exoid, eb_ids[eb], connect[eb]);CHKERRQ(ierr);
      }
    }
  }

  // Read node sets
  int  *ns_ids, *num_nodes_in_set;
  int **node_list;
  if (num_node_sets > 0) {
    ierr = PetscMalloc3(num_node_sets,int,&ns_ids,num_node_sets,int,&num_nodes_in_set,num_node_sets,int*,&node_list);CHKERRQ(ierr);
    ierr = ex_get_node_set_ids(exoid, ns_ids);CHKERRQ(ierr);
    for(int ns = 0; ns < num_node_sets; ++ns) {
      int num_df_in_set;
      ierr = ex_get_node_set_param (exoid, ns_ids[ns], &num_nodes_in_set[ns], &num_df_in_set);CHKERRQ(ierr);
      ierr = PetscMalloc(num_nodes_in_set[ns] * sizeof(int), &node_list[ns]);CHKERRQ(ierr);
      ierr = ex_get_node_set(exoid, ns_ids[ns], node_list[ns]);
	}
  }
  ierr = ex_close(exoid);CHKERRQ(ierr);

  // Build mesh topology
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
  bool interpolate = false;

  try {
  mesh->setSieve(sieve);
  if (0 == rank) {
    if (!interpolate) {
      // Create the ISieve
      sieve->setChart(PETSC_MESH_TYPE::sieve_type::chart_type(0, num_elem+num_nodes));
      // Set cone and support sizes
      for (int c = 0; c < num_elem; ++c) {
	sieve->setConeSize(c, numCorners);
      }
      sieve->symmetrizeSizes(num_elem, numCorners, cells, num_elem - 1); /* Notice the -1 for 1-based indexing in cells[] */
      // Allocate point storage
      sieve->allocate();
      // Fill up cones
      int *cone  = new int[numCorners];
      int *coneO = new int[numCorners];
      for (int v = 0; v < numCorners; ++v) {
	coneO[v] = 1;
      }
      for (int c = 0; c < num_elem; ++c) {
        for (int v = 0; v < numCorners; ++v) {
	  cone[v] = cells[c*numCorners+v]+num_elem - 1;
	}
        sieve->setCone(cone, c);
        sieve->setConeOrientation(coneO, c);
      } // for
      delete[] cone; cone = 0;
      delete[] coneO; coneO = 0;
      // Symmetrize to fill up supports
      sieve->symmetrize();
    } else {
      // Same old thing
      ALE::Obj<ALE::Mesh::sieve_type> s = new ALE::Mesh::sieve_type(sieve->comm(), sieve->debug());

      ALE::SieveBuilder<ALE::Mesh>::buildTopology(s, num_dim, num_elem, cells, num_nodes, interpolate, numCorners);
      std::map<ALE::Mesh::point_type,ALE::Mesh::point_type> renumbering;
      ALE::ISieveConverter::convertSieve(*s, *sieve, renumbering);
    }
    if (!interpolate) {
      // Optimized stratification
      const ALE::Obj<PETSC_MESH_TYPE::label_type>& height = mesh->createLabel("height");
      const ALE::Obj<PETSC_MESH_TYPE::label_type>& depth  = mesh->createLabel("depth");

      for(int c = 0; c < num_elem; ++c) {
        height->setCone(0, c);
        depth->setCone(1, c);
      }
      for(int v = num_elem; v < num_elem+num_nodes; ++v) {
        height->setCone(1, v);
        depth->setCone(0, v);
      }
      mesh->setHeight(1);
      mesh->setDepth(1);
    } else {
      mesh->stratify();
    }
  } else {
    mesh->getSieve()->setChart(PETSC_MESH_TYPE::sieve_type::chart_type());
    mesh->getSieve()->allocate();
    mesh->stratify();
  }
  } catch (ALE::Exception e) {
    SETERRQ(PETSC_ERR_LIB, e.msg().c_str());
  }
  ierr = PetscFree(cells);CHKERRQ(ierr);

  // Build cell blocks
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& cellBlocks = mesh->createLabel("CellBlocks");
  if (rank == 0) {
    for(int eb = 0, k = 0; eb < num_elem_blk; ++eb) {
      for(int e = 0; e < num_elem_in_block[eb]; ++e, ++k) {
        mesh->setValue(cellBlocks, k, eb_ids[eb]);
      }
    }
  }
  if (num_elem_blk > 0) {
    ierr = PetscFree(connect);CHKERRQ(ierr);
    ierr = PetscFree5(eb_ids, num_elem_in_block, num_nodes_per_elem, num_attr, block_names);CHKERRQ(ierr);
  }

  // Build coordinates
  double *coords;
  ierr = PetscMalloc(num_dim*num_nodes * sizeof(double), &coords);CHKERRQ(ierr);
  if (num_dim > 0) {for(int v = 0; v < num_nodes; ++v) {coords[v*num_dim+0] = x[v];}}
  if (num_dim > 1) {for(int v = 0; v < num_nodes; ++v) {coords[v*num_dim+1] = y[v];}}
  if (num_dim > 2) {for(int v = 0; v < num_nodes; ++v) {coords[v*num_dim+2] = z[v];}}
  ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(mesh, num_dim, coords);
  ierr = PetscFree(coords);CHKERRQ(ierr);
  ierr = PetscFree3(x, y, z);CHKERRQ(ierr);

  // Build vertex sets
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& vertexSets = mesh->createLabel("VertexSets");
  if (rank == 0) {
    for(int ns = 0; ns < num_node_sets; ++ns) {
      for(int n = 0; n < num_nodes_in_set[ns]; ++n) {
        mesh->addValue(vertexSets, node_list[ns][n]+num_elem-1, ns_ids[ns]);
      }
      ierr = PetscFree(node_list[ns]);CHKERRQ(ierr);
    }
  }
  if (num_node_sets > 0) {
    ierr = PetscFree3(ns_ids,num_nodes_in_set,node_list);CHKERRQ(ierr);
  }

  //cellBlocks->view("Cell Blocks");
  //vertexSets->view("Vertex Sets");

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

#endif // PETSC_HAVE_EXODUSII

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
#ifdef PETSC_HAVE_EXODUSII
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
/*@C
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

#undef __FUNCT__
#define __FUNCT__ "MeshGetLabelSize"
/*@C
  MeshGetLabelSize - Get the number of different integer ids in a Label

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
- name - The label name

  Output Parameter:
. size - The label size (number of different integer ids)

  Level: beginner

.keywords: mesh, ExodusII
.seealso: MeshCreateExodus()
@*/
PetscErrorCode MeshGetLabelSize(Mesh mesh, const char name[], PetscInt *size)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  *size = m->getLabel(name)->getCapSize();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshGetLabelIds"
/*@C
  MeshGetLabelIds - Get the integer ids in a label

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
. name - The label name
- ids - The id storage array

  Output Parameter:
. ids - The integer ids

  Level: beginner

.keywords: mesh, ExodusII
.seealso: MeshCreateExodus()
@*/
PetscErrorCode MeshGetLabelIds(Mesh mesh, const char name[], PetscInt *ids)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_type::capSequence>&      labelIds = m->getLabel(name)->cap();
  const PETSC_MESH_TYPE::label_type::capSequence::const_iterator iEnd     = labelIds->end();
  PetscInt                                                       i        = 0;

  for(PETSC_MESH_TYPE::label_type::capSequence::const_iterator i_iter = labelIds->begin(); i_iter != iEnd; ++i_iter, ++i) {
    ids[i] = *i_iter;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshGetStratumSize"
/*@C
  MeshGetStratumSize - Get the number of points in a label stratum

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
. name - The label name
- value - The stratum value

  Output Parameter:
. size - The stratum size

  Level: beginner

.keywords: mesh, ExodusII
.seealso: MeshCreateExodus()
@*/
PetscErrorCode MeshGetStratumSize(Mesh mesh, const char name[], PetscInt value, PetscInt *size)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  *size = m->getLabelStratum(name, value)->size();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshGetStratum"
/*@C
  MeshGetStratum - Get the points in a label stratum

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
. name - The label name
. value - The stratum value
- points - The stratum points storage array

  Output Parameter:
. points - The stratum points

  Level: beginner

.keywords: mesh, ExodusII
.seealso: MeshCreateExodus()
@*/
PetscErrorCode MeshGetStratum(Mesh mesh, const char name[], PetscInt value, PetscInt *points)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& stratum = m->getLabelStratum(name, value);
  const PETSC_MESH_TYPE::label_sequence::iterator  sEnd    = stratum->end();
  PetscInt                                         s       = 0;

  for(PETSC_MESH_TYPE::label_sequence::iterator s_iter = stratum->begin(); s_iter != sEnd; ++s_iter, ++s) {
    points[s] = *s_iter;
  }
  PetscFunctionReturn(0);
}
