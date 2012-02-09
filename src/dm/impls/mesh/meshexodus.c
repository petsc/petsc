#include<petscdmmesh_formats.hh>   /*I "petscdmmesh.h" I*/

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
      typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
      ALE::Obj<FlexMesh::sieve_type> s = new FlexMesh::sieve_type(sieve->comm(), sieve->debug());

      ALE::SieveBuilder<FlexMesh>::buildTopology(s, num_dim, num_elem, cells, num_nodes, interpolate, numCorners);
      std::map<FlexMesh::point_type,FlexMesh::point_type> renumbering;
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
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB, e.msg().c_str());
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
#define __FUNCT__ "DMMeshCreateExodus"
/*@C
  DMMeshCreateExodus - Create a DMMesh from an ExodusII file.

  Not Collective

  Input Parameters:
+ comm - The MPI communicator
- filename - The ExodusII filename

  Output Parameter:
. dm - The DMMesh object

  Level: beginner

.keywords: mesh, ExodusII
.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshCreateExodus(MPI_Comm comm, const char filename[], DM *dm)
{
  PetscInt       debug = 0;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshCreate(comm, dm);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  ALE::Obj<PETSC_MESH_TYPE> m = new PETSC_MESH_TYPE(comm, -1, debug);
#ifdef PETSC_HAVE_EXODUSII
  ierr = PetscReadExodusII(comm, filename, m);CHKERRQ(ierr);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --with-exodus-dir=/path/to/exodus");
#endif
  if (debug) {m->view("Mesh");}
  ierr = DMMeshSetMesh(*dm, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshExodusGetInfo"
/*@
  DMMeshExodusGetInfo - Get information about an ExodusII Mesh.

  Not Collective

  Input Parameter:
. dm - The DMMesh object

  Output Parameters:
+ dim - The mesh dimension
. numVertices - The number of vertices in the mesh
. numCells - The number of cells in the mesh
. numCellBlocks - The number of cell blocks in the mesh
- numVertexSets - The number of vertex sets in the mesh

  Level: beginner

.keywords: mesh, ExodusII
.seealso: DMMeshCreateExodus()
@*/
PetscErrorCode DMMeshExodusGetInfo(DM dm, PetscInt *dim, PetscInt *numVertices, PetscInt *numCells, PetscInt *numCellBlocks, PetscInt *numVertexSets)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  *dim           = m->getDimension();
  *numVertices   = m->depthStratum(0)->size();
  *numCells      = m->heightStratum(0)->size();
  *numCellBlocks = m->getLabel("CellBlocks")->getCapSize();
  *numVertexSets = m->getLabel("VertexSets")->getCapSize();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateExodusNG"
/*@C
  DMMeshCreateExodusNG - Create a Mesh from an ExodusII file.

  Collective on comm

  Input Parameters:
+ comm - The MPI communicator
- filename - The ExodusII filename

  Output Parameter:
. dmBody  - The DM object representing the body
. dmFS    - The DM object representing the face sets or PETSC_NULL

  Face Sets (Side Sets in Exodus terminology) are ignored if dmFS is PETSC_NULL

  Interpolated meshes are not supported.

  Level: beginner

.keywords: mesh,ExodusII
.seealso: MeshCreate() MeshCreateExodus()
@*/
PetscErrorCode DMMeshCreateExodusNG(MPI_Comm comm,const char filename[],DM *dmBody,DM *dmFS)
{
  PetscBool               debug = PETSC_FALSE;
  PetscMPIInt             numproc,rank;
  PetscErrorCode          ierr;

  int                     exoid;
  int                     CPU_word_size = 0;
  int                     IO_word_size  = 0;
  int                     num_dim,num_vertices = 0,num_cell = 0;
  int                     num_cs = 0,num_vs = 0,num_fs = 0;
  float                   version;
  char                    title[MAX_LINE_LENGTH+1];
  char                    buffer[MAX_LINE_LENGTH+1];

  int                    *cs_id;
  int                     num_cell_in_set,num_vertex_per_cell,num_attr;
  int                    *cs_connect;

  int                    *vs_id;
  int                     num_vertex_in_set;
  int                    *vs_vertex_list;
  float                  *x,*y,*z;
  PetscReal              *coords;

  int                    *fs_id;
  int                     num_face_in_set;
  int                    *num_vertex_in_face,*fs_vertex_list,*fs_elem_list,*fs_face_list;

  PetscInt                my_num_cells,my_num_vertices;
  PetscInt                *local_points;
  ISLocalToGlobalMapping  point_mapping;


  DM                      dmBodySeq;
  PetscInt                f,v,c,f_loc,v_loc,c_loc,fs,vs,cs;

  typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
  typedef PETSC_MESH_TYPE::sieve_type     sieve_type;
  ALE::Obj<PETSC_MESH_TYPE>               meshBody,meshBodySeq,meshFS;
  ALE::Obj<FlexMesh::sieve_type>          sBody,sFS;
  ALE::Obj<FlexMesh>                      mBody,mFS;
  ALE::Obj<PETSC_MESH_TYPE::sieve_type>   sieveBody,sieveFS;
  std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> renumberingBody,renumberingFS;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_EXODUSII)
  ierr = MPI_Comm_rank(comm,&rank);
  ierr = MPI_Comm_size(comm,&numproc);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-debug",&debug,PETSC_NULL);CHKERRQ(ierr);

  ierr = DMMeshCreate(comm,&dmBodySeq);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dmBodySeq,&comm);CHKERRQ(ierr);
  if (numproc == 1) {
    *dmBody = dmBodySeq;
  } else {
    ierr = DMMeshCreate(comm,dmBody);CHKERRQ(ierr);
  }
  meshBodySeq = new PETSC_MESH_TYPE(comm,-1,debug);
  ierr = DMMeshSetMesh(dmBodySeq,meshBodySeq);CHKERRQ(ierr);

  sieveBody = new PETSC_MESH_TYPE::sieve_type(meshBodySeq->comm(),meshBodySeq->debug());
  sBody = new FlexMesh::sieve_type(meshBodySeq->comm(),meshBodySeq->debug());

  if (dmFS != PETSC_NULL) {
    ierr = DMMeshCreate(comm,dmFS);CHKERRQ(ierr);
    meshFS = new PETSC_MESH_TYPE(comm,-1,debug);
    ierr = DMMeshSetMesh(*dmFS,meshFS);CHKERRQ(ierr);
  }

  /*
    Open EXODUS II file and read basic informations on rank 0,
    then broadcast to all processors
  */
  if (rank == 0) {
    exoid = ex_open(filename,EX_READ,&CPU_word_size,&IO_word_size,&version);CHKERRQ(!exoid);
    ierr = ex_get_init(exoid,title,&num_dim,&num_vertices,&num_cell,&num_cs,&num_vs,&num_fs);CHKERRQ(ierr);
    if (num_cs == 0) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exodus file does not contain any cell set\n");
    }
  }

  ierr = MPI_Bcast(&num_dim,1,MPI_INT,0,comm);
  ierr = MPI_Bcast(&num_cs,1,MPI_INT,0,comm);
  ierr = MPI_Bcast(&num_vs,1,MPI_INT,0,comm);
  ierr = MPI_Bcast(&num_fs,1,MPI_INT,0,comm);

  meshBodySeq->setDimension(num_dim);
  meshFS->setDimension(num_dim-1);

  /*
    Read cell sets information then broadcast them
  */
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& cellSets = meshBodySeq->createLabel("Cell Sets");
  if (rank == 0) {
    /*
      Get cell sets IDs
    */
    ierr = PetscMalloc(num_cs*sizeof(int),&cs_id);CHKERRQ(ierr);
    ierr = ex_get_elem_blk_ids(exoid,cs_id);CHKERRQ(ierr);

    /*
      Read the cell set connectivity table and build mesh topology
      EXO standard requires that cells in cell sets be numbered sequentially
      and be pairwise disjoint.
    */
    for (c=0,cs = 0; cs < num_cs; cs++) {
      if (debug) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Building cell set %i\n",cs);CHKERRQ(ierr);
      }
      ierr = ex_get_elem_block(exoid,cs_id[cs],buffer,&num_cell_in_set,&num_vertex_per_cell,&num_attr);CHKERRQ(ierr);
      ierr = PetscMalloc(num_vertex_per_cell*num_cell_in_set*sizeof(int),&cs_connect);CHKERRQ(ierr);
      ierr = ex_get_elem_conn(exoid,cs_id[cs],cs_connect);CHKERRQ(ierr);
      /*
        EXO uses Fortran-based indexing, sieve uses C-style and numbers cell first then vertices.
      */
      for (v = 0,c_loc = 0; c_loc < num_cell_in_set; c_loc++,c++) {
        for (v_loc = 0; v_loc < num_vertex_per_cell; v_loc++,v++) {
          if (debug) {
            ierr = PetscPrintf(PETSC_COMM_SELF,"[%i]:\tinserting vertex \t%i in cell \t%i local vertex number \t%i\n",
                               rank,cs_connect[v]+num_cell-1,c,v_loc);CHKERRQ(ierr);
          }
          sBody->addArrow(sieve_type::point_type(cs_connect[v]+num_cell-1),
                          sieve_type::point_type(c),
                          v_loc);
        }
        meshBodySeq->setValue(cellSets,c,cs_id[cs]);
      }
      ierr = PetscFree(cs_connect);CHKERRQ(ierr);
    }
  }
  /*
    We do not interpolate and know that the numbering is compact (this is required in exo)
    so no need to renumber (renumber = false)
  */
  ALE::ISieveConverter::convertSieve(*sBody,*sieveBody,renumberingBody,false);
  meshBodySeq->setSieve(sieveBody);
  meshBodySeq->stratify();
  mBody = new FlexMesh(meshBodySeq->comm(),meshBodySeq->debug());
  ALE::ISieveConverter::convertOrientation(*sBody,*sieveBody,renumberingBody,mBody->getArrowSection("orientation").ptr());

  /*
    Create Vertex set label
  */
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& vertexSets = meshBodySeq->createLabel("Vertex Sets");
  if (num_vs >0) {
    if (rank == 0) {
      /*
        Get vertex set ids
      */
      ierr = PetscMalloc(num_vs*sizeof(int),&vs_id);
      ierr = ex_get_node_set_ids(exoid,vs_id);CHKERRQ(ierr);
      for (vs = 0; vs < num_vs; vs++) {
        ierr = ex_get_node_set_param(exoid,vs_id[vs],&num_vertex_in_set,&num_attr);CHKERRQ(ierr);
        ierr = PetscMalloc(num_vertex_in_set * sizeof(int),&vs_vertex_list);CHKERRQ(ierr);
        ierr = ex_get_node_set(exoid,vs_id[vs],vs_vertex_list);
        for (v = 0; v < num_vertex_in_set; v++) {
          meshBodySeq->addValue(vertexSets,vs_vertex_list[v]+num_cell-1,vs_id[vs]);
        }
        ierr = PetscFree(vs_vertex_list);
      }
      PetscFree(vs_id);
    }
  }
  /*
    Read coordinates
  */
    ierr = PetscMalloc4(num_vertices,float,&x,
                        num_vertices,float,&y,
                        num_vertices,float,&z,
                        num_dim*num_vertices,PetscReal,&coords);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = ex_get_coord(exoid,x,y,z);CHKERRQ(ierr);
    ierr = PetscMalloc(num_dim*num_vertices * sizeof(PetscReal), &coords);CHKERRQ(ierr);
    if (num_dim > 0) {for (v = 0; v < num_vertices; ++v) {coords[v*num_dim+0] = x[v];}}
    if (num_dim > 1) {for (v = 0; v < num_vertices; ++v) {coords[v*num_dim+1] = y[v];}}
    if (num_dim > 2) {for (v = 0; v < num_vertices; ++v) {coords[v*num_dim+2] = z[v];}}
  }
  ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(meshBodySeq,num_dim,coords);
  if (rank == 0) {
    ierr = PetscFree4(x,y,z,coords);CHKERRQ(ierr);
  }

  if (debug) {
    meshBodySeq->view("meshBodySeq");
    cellSets->view("Cell Sets");
    vertexSets->view("Vertex Sets");
  }

  /*
    Distribute
  */
  if (numproc>1) {
    ierr = DMMeshCreate(comm,dmBody);CHKERRQ(ierr);
    ierr = DMMeshDistribute(dmBodySeq,PETSC_NULL,dmBody);CHKERRQ(ierr);
    ierr = DMDestroy(&dmBodySeq);CHKERRQ(ierr);
  }
  ierr = DMMeshGetMesh(*dmBody,meshBody);CHKERRQ(ierr);
  my_num_cells     = meshBody->heightStratum(0)->size();
  my_num_vertices = meshBody->depthStratum(0)->size();
  if (debug) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%i]:\t my_num_cells: %i my_num_vertices: %i\n",rank,my_num_cells,my_num_vertices);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  }
  if(debug) {
    const ALE::Obj<PETSC_MESH_TYPE::label_type>& cellSets = meshBody->getLabel("Cell Sets");
    const ALE::Obj<PETSC_MESH_TYPE::label_type>& vertexSets = meshBody->getLabel("Vertex Sets");
    meshBody->view("meshBody");
    cellSets->view("Cell sets");
    vertexSets->view("Vertex sets");
  }

  /*
    Reading Face Sets (sidesets in exodusII linguo)
    - if numproc == 1: read face set vertex list and create face set mesh
    - if numproc > 1: create LocalToGlobalMapping using the renumbering iterator
                      read face set
                      broadcast face set
                      renumber face set, keeping only faces owned by a local element
                      create face set mesh
  */

  /*
    Build LocalToGlobalMapping from the sequential to the distributed mesh
  */
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& faceSets = meshFS->createLabel("Face Sets");
  sieveFS = new PETSC_MESH_TYPE::sieve_type(meshFS->comm(),meshFS->debug());
  sFS = new FlexMesh::sieve_type(meshFS->comm(),meshFS->debug());
  if (num_fs > 0) {

    FlexMesh::renumbering_type&            renumberingBody2 = meshBody->getRenumbering();
    if (numproc > 1) {
      ierr = PetscMalloc((my_num_cells + my_num_vertices) * sizeof(PetscInt),&local_points);CHKERRQ(ierr);
      for(FlexMesh::renumbering_type::const_iterator r_iter = renumberingBody2.begin(); r_iter != renumberingBody2.end(); ++r_iter) {
        /*
          r_iter->first  is a point number in the sequential mesh
          r_iter->second is the matching local point number in the distributed mesh
        */
        local_points[r_iter->second] = r_iter->first;
      }
      ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,my_num_cells + my_num_vertices,local_points,PETSC_OWN_POINTER,&point_mapping);CHKERRQ(ierr);
      if (debug) {
        ierr = ISLocalToGlobalMappingView(point_mapping,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      }
    }

    /*
      Build the face set mesh
    */
    /*
      Get face set ids
    */
    ierr = PetscMalloc(num_fs*sizeof(int),&fs_id);
    if (rank == 0) {
      ierr = ex_get_side_set_ids(exoid,fs_id);CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(fs_id,num_fs,MPI_INT,0,comm);
    int local_face = 0;
    for (f = 0,fs = 0; fs < num_fs; fs++) {
      if (debug) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Building face set %i\n",fs);CHKERRQ(ierr);
      }
      if (rank == 0) {
        ierr = ex_get_side_set_param(exoid,fs_id[fs],&num_face_in_set,&num_attr);CHKERRQ(ierr);
        ierr = ex_get_side_set_node_list_len(exoid,fs_id[fs],&num_vertex_in_set);CHKERRQ(ierr);
      }
      ierr = MPI_Bcast(&num_face_in_set,1,MPI_INT,0,comm);
      ierr = MPI_Bcast(&num_vertex_in_set,1,MPI_INT,0,comm);
      ierr = PetscMalloc4(num_face_in_set,int,&num_vertex_in_face,
                          num_vertex_in_set,int,&fs_vertex_list,
                          num_face_in_set,int,&fs_elem_list,
                          num_face_in_set,int,&fs_face_list);
      if (rank == 0) {
        ierr = ex_get_side_set_node_list(exoid,fs_id[fs],num_vertex_in_face,fs_vertex_list);
        /*
          num_vertex_in_face[f] is the number of vertices in face f
          fs_vertex_list is the list of all vertices in the current set (size = sum(num_vertex_in_face))
        */
        ierr = ex_get_side_set(exoid,fs_id[fs],fs_elem_list,fs_face_list);
        /*
          fs_elem_list[f] is the number of the element owning the face f (in exo numbering)
          fs_face_list[f] is the local face number of f in element fs_elem_list[f] (unused here)
        */

        for (v_loc = 0; v_loc < num_face_in_set; v_loc++) {
          fs_elem_list[v_loc]--;
        }
        for (f_loc=0; f_loc < num_vertex_in_set; f_loc++) {
          fs_vertex_list[f_loc] += num_cell-1;
        }
        if (debug) {
          PetscPrintf(PETSC_COMM_SELF,"fs_elem_list: ");
          PetscIntView(num_face_in_set,fs_elem_list,PETSC_VIEWER_STDOUT_SELF);
          PetscPrintf(PETSC_COMM_SELF,"fs_vertex_list: ");
          PetscIntView(num_vertex_in_set,fs_vertex_list,PETSC_VIEWER_STDOUT_SELF);
        }
      }
      ierr = MPI_Bcast(num_vertex_in_face,num_face_in_set,MPI_INT,0,comm);
      ierr = MPI_Bcast(fs_vertex_list,num_vertex_in_set,MPI_INT,0,comm);
      ierr = MPI_Bcast(fs_elem_list,num_face_in_set,MPI_INT,0,comm);
      ierr = MPI_Bcast(fs_face_list,num_face_in_set,MPI_INT,0,comm);

      /*
        Converting global cell and vertex numbers into local ones
      */

      if (numproc > 1) {
        ierr = ISGlobalToLocalMappingApply(point_mapping,IS_GTOLM_MASK,num_vertex_in_set,fs_vertex_list,PETSC_NULL,fs_vertex_list);
        ierr = ISGlobalToLocalMappingApply(point_mapping,IS_GTOLM_MASK,num_face_in_set,fs_elem_list,PETSC_NULL,fs_elem_list);
        if (debug) {
          PetscPrintf(PETSC_COMM_SELF,"[%i]:\trenumbered fs_elem_list: ",rank);
          PetscIntView(num_face_in_set,fs_elem_list,PETSC_VIEWER_STDOUT_SELF);
          PetscPrintf(PETSC_COMM_SELF,"[%i]:\trenumbered fs_vertex_list: ",rank);
          PetscIntView(num_vertex_in_set,fs_vertex_list,PETSC_VIEWER_STDOUT_SELF);
        }
      }

      for (v = 0,f_loc = 0; f_loc < num_face_in_set; f_loc++,f++) {
        if (fs_elem_list[f_loc] != -1) {
          for (v_loc = 0; v_loc <  num_vertex_in_face[f_loc]; v_loc++,v++) {
            if (debug) {
              ierr = PetscPrintf(PETSC_COMM_SELF,"[%i]:\tinserting vertex \t%i in face \t%i local vertex number \t%i\n",
                                 rank,fs_vertex_list[v],local_face,v_loc);CHKERRQ(ierr);
            }
            sFS->addArrow(sieve_type::point_type(fs_vertex_list[v]),
                          sieve_type::point_type(local_face),
                          v_loc);
          }
          meshFS->addValue(faceSets,local_face,fs_id[fs]);
          local_face++;
        } else {
          if (debug) {
            ierr = PetscPrintf(PETSC_COMM_SELF,"[%i]:\t... skipping non-local face %i\n",rank,f_loc);
          }
          v += num_vertex_in_face[f_loc];
        }
      }
      ierr = PetscFree4(num_vertex_in_face,fs_vertex_list,fs_elem_list,fs_face_list);
    }
    PetscFree(fs_id);

    if (numproc>1) {
      ierr = ISLocalToGlobalMappingDestroy(&point_mapping);CHKERRQ(ierr);
    }
    /*
      The numbering is NOT compact, but we want to preserve an ordering compatible with that
      of the body mesh, so we do not renumber (renumber = false)
    */
    ALE::ISieveConverter::convertSieve(*sFS,*sieveFS,renumberingFS,false);
    meshFS->setSieve(sieveFS);
    meshFS->stratify();
    mFS = new FlexMesh(meshFS->comm(),meshFS->debug());
    ALE::ISieveConverter::convertOrientation(*sFS,*sieveFS,renumberingFS,mFS->getArrowSection("orientation").ptr());
  } else {
    ALE::ISieveConverter::convertSieve(*sFS,*sieveFS,renumberingFS,false);
    meshFS->setSieve(sieveFS);
    meshFS->stratify();
  }
  if (debug) {
    meshFS->view("meshFS");
    faceSets->view("Face Sets");
  }

  if (rank == 0) {
    ierr = ex_close(exoid);CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This method requires ExodusII support. Reconfigure using --with-exodusii-dir=/path/to/exodus");
#endif
  PetscFunctionReturn(0);
}
