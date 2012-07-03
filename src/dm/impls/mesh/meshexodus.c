#include<petscdmmesh_formats.hh>   /*I "petscdmmesh.h" I*/

#ifdef PETSC_HAVE_EXODUSII
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
  char  title[PETSC_MAX_PATH_LEN+1], elem_type[PETSC_MAX_PATH_LEN+1];
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
      ierr = PetscMalloc((PETSC_MAX_PATH_LEN+1) * sizeof(char), &block_names[eb]);CHKERRQ(ierr);
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
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  if (debug) {m->view("Mesh");}
  ierr = DMMeshSetMesh(*dm, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateExodusNG"
/*@
  DMMeshCreateExodusNG - Create a Mesh from an ExodusII file.

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
.seealso: MeshCreate() MeshCreateExodus()
@*/
PetscErrorCode DMMeshCreateExodusNG(MPI_Comm comm, PetscInt exoid,DM *dm)
{
#if defined(PETSC_HAVE_EXODUSII)
  PetscBool               debug = PETSC_FALSE;
  PetscMPIInt             num_proc,rank;
  PetscErrorCode          ierr;

  int                     num_dim,num_vertices = 0,num_cell = 0;
  int                     num_cs = 0,num_vs = 0,num_fs = 0;
  char                    title[PETSC_MAX_PATH_LEN+1];
  char                    buffer[PETSC_MAX_PATH_LEN+1];

  int                    *cs_id;
  int                     num_cell_in_set,num_vertex_per_cell,num_attr;
  int                    *cs_connect;

  int                    *vs_id;
  int                     num_vertex_in_set;
  int                    *vs_vertex_list;
  PetscReal              *x,*y,*z;
  PetscReal              *coords;

  PetscInt                v,c,v_loc,c_loc,vs,cs;

  typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
  typedef PETSC_MESH_TYPE::sieve_type     sieve_type;
  ALE::Obj<PETSC_MESH_TYPE>               mesh;
  ALE::Obj<FlexMesh::sieve_type>          flexmesh_sieve;
  ALE::Obj<FlexMesh>                      flexmesh;
  ALE::Obj<PETSC_MESH_TYPE::sieve_type>   sieve;
  std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> renumbering;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_EXODUSII)
  ierr = MPI_Comm_rank(comm,&rank);
  ierr = MPI_Comm_size(comm,&num_proc);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-debug",&debug,PETSC_NULL);CHKERRQ(ierr);

  ierr = DMMeshCreate(comm,dm);CHKERRQ(ierr);
  /*
    I _really don't understand how this is needed
  */
  ierr = PetscObjectGetComm((PetscObject) *dm,&comm);CHKERRQ(ierr);


  mesh = new PETSC_MESH_TYPE(comm,-1,debug);
  ierr = DMMeshSetMesh(*dm,mesh);CHKERRQ(ierr);

  sieve = new PETSC_MESH_TYPE::sieve_type(mesh->comm(),mesh->debug());
  flexmesh_sieve = new FlexMesh::sieve_type(mesh->comm(),mesh->debug());

  /*
    Open EXODUS II file and read basic informations on rank 0,
    then broadcast to all processors
  */
  if (rank == 0) {
    ierr = ex_get_init(exoid,title,&num_dim,&num_vertices,&num_cell,&num_cs,&num_vs,&num_fs);CHKERRQ(ierr);
    if (num_cs == 0) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exodus file does not contain any cell set\n");
    }
  }

  ierr = MPI_Bcast(&num_dim,1,MPIU_INT,0,comm);
  ierr = MPI_Bcast(&num_cs,1,MPIU_INT,0,comm);
  ierr = MPI_Bcast(&num_vs,1,MPIU_INT,0,comm);
  ierr = MPI_Bcast(&num_fs,1,MPIU_INT,0,comm);
  ierr = MPI_Bcast(title,PETSC_MAX_PATH_LEN+1,MPI_CHAR,0,comm);

  ierr = PetscObjectSetName((PetscObject)*dm,title);CHKERRQ(ierr);
  mesh->setDimension(num_dim);

  /*
    Read cell sets information then broadcast them
  */
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& cellSets = mesh->createLabel("Cell Sets");
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
        ierr = PetscPrintf(comm,"Building cell set %i\n",cs);CHKERRQ(ierr);
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
          flexmesh_sieve->addArrow(sieve_type::point_type(cs_connect[v]+num_cell-1),
                          sieve_type::point_type(c),
                          v_loc);
        }
        mesh->setValue(cellSets,c,cs_id[cs]);
      }
      ierr = PetscFree(cs_connect);CHKERRQ(ierr);
    }
    ierr = PetscFree(cs_id);CHKERRQ(ierr);
  }
  /*
    We do not interpolate and know that the numbering is compact (this is required in exo)
    so no need to renumber (renumber = false)
  */
  ALE::ISieveConverter::convertSieve(*flexmesh_sieve,*sieve,renumbering,false);
  mesh->setSieve(sieve);
  mesh->stratify();
  flexmesh = new FlexMesh(mesh->comm(),mesh->debug());
  ALE::ISieveConverter::convertOrientation(*flexmesh_sieve,*sieve,renumbering,flexmesh->getArrowSection("orientation").ptr());

  /*
    Create Vertex set label
  */
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& vertexSets = mesh->createLabel("Vertex Sets");
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
          mesh->addValue(vertexSets,vs_vertex_list[v]+num_cell-1,vs_id[vs]);
        }
        ierr = PetscFree(vs_vertex_list);
      }
      PetscFree(vs_id);
    }
  }
  /*
    Read coordinates
  */
    ierr = PetscMalloc4(num_vertices,PetscReal,&x,
                        num_vertices,PetscReal,&y,
                        num_vertices,PetscReal,&z,
                        num_dim*num_vertices,PetscReal,&coords);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = ex_get_coord(exoid,x,y,z);CHKERRQ(ierr);
    ierr = PetscMalloc(num_dim*num_vertices * sizeof(PetscReal), &coords);CHKERRQ(ierr);
    if (num_dim > 0) {for (v = 0; v < num_vertices; ++v) {coords[v*num_dim+0] = x[v];}}
    if (num_dim > 1) {for (v = 0; v < num_vertices; ++v) {coords[v*num_dim+1] = y[v];}}
    if (num_dim > 2) {for (v = 0; v < num_vertices; ++v) {coords[v*num_dim+2] = z[v];}}
  }
  ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(mesh,num_dim,coords);
  if (rank == 0) {
    ierr = PetscFree4(x,y,z,coords);CHKERRQ(ierr);
  }

  /*
  if (rank == 0) {
    ierr = ex_close(exoid);CHKERRQ(ierr);
  }
  */
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshViewExodusSplit"
/*@
  DMMeshViewExodusSplit - Write a dmMesh geometry and topology into several ExodusII files.

  Collective on comm

  Input Parameters:
+ comm - The MPI communicator
- filename - The ExodusII filename. Must be different on each processor 
             (suggest using prefix-<rank>.gen or prefix-<rank>.exo)
. dm  - The DM object representing the body

  Face Sets (Side Sets in Exodus terminology) are ignored
  
  Interpolated meshes are not supported.
  
  Level: beginner

.keywords: mesh,ExodusII
.seealso: MeshCreate()
@*/
PetscErrorCode DMMeshViewExodusSplit(DM dm,PetscInt exoid)
{
#if defined(PETSC_HAVE_EXODUSII)
  PetscBool               debug = PETSC_FALSE;
  MPI_Comm                comm;
  PetscErrorCode          ierr;

  int                     num_dim,num_vertices = 0,num_cells = 0;
  int                     num_cs = 0,num_vs = 0;
  int                     num_cs_global = 0,num_vs_global = 0;
  const char             *title;
  IS                      csIS,vsIS;
  IS                      csIS_global,vsIS_global;
  const PetscInt         *csID,*vsID,*labels;

  PetscReal              *coord;

  PetscInt                c,v,c_offset;

  PetscInt                set,num_cell_in_set,num_vertex_per_cell;
  const PetscInt         *cells;
  IS                      cellsIS;
  const char             *cell_type;
  int                    *elem_map,*cs_connect,num_cs_attr=0;
  const PetscInt         *elem_connect;

  PetscInt                num_vertex_in_set,num_vs_attr=0;
  PetscInt               *vertices;
  IS                      verticesIS;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_EXODUSII)
  /*
    Extract mesh global properties from the DM
  */
  ierr = PetscObjectGetName((PetscObject)dm,&title);CHKERRQ(ierr);
  ierr = DMMeshGetDimension(dm,&num_dim);CHKERRQ(ierr);
  ierr = DMMeshGetStratumSize(dm,"height",0,&num_cells);CHKERRQ(ierr);
  ierr = DMMeshGetStratumSize(dm,"depth",0,&num_vertices);CHKERRQ(ierr);

  /*
    Get the local and  global number of sets
  */
  ierr = PetscObjectGetComm((PetscObject) dm,&comm);CHKERRQ(ierr);
  ierr = DMMeshGetLabelSize(dm,"Cell Sets",&num_cs);CHKERRQ(ierr);
  ierr = DMMeshGetLabelIdIS(dm,"Cell Sets",&csIS);CHKERRQ(ierr);
  ierr = ISGetTotalIndices(csIS,&labels);CHKERRQ(ierr);
  ierr = ISGetSize(csIS,&num_cs_global);CHKERRQ(ierr);
  ierr = PetscSortRemoveDupsInt(&num_cs_global,(PetscInt*)labels);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,num_cs_global,labels,PETSC_COPY_VALUES,&csIS_global);CHKERRQ(ierr);
  ierr = ISRestoreTotalIndices(csIS,&labels);CHKERRQ(ierr);

  ierr = DMMeshGetLabelSize(dm,"Vertex Sets",&num_vs);CHKERRQ(ierr);
  ierr = DMMeshGetLabelIdIS(dm,"Vertex Sets",&vsIS);CHKERRQ(ierr);
  ierr = ISGetSize(vsIS,&num_vs_global);CHKERRQ(ierr);
  if (num_vs_global > 0) {
    ierr = ISGetTotalIndices(vsIS,&labels);CHKERRQ(ierr);
    ierr = PetscSortRemoveDupsInt(&num_vs_global,(PetscInt*)labels);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,num_vs_global,labels,PETSC_COPY_VALUES,&vsIS_global);CHKERRQ(ierr);
    ierr = ISRestoreTotalIndices(vsIS,&labels);CHKERRQ(ierr);
  }
  ierr = ex_put_init(exoid,title,num_dim,num_vertices,num_cells,num_cs_global,num_vs_global,0);
  
  /*
    Write coordinates
  */
  ierr = DMMeshGetCoordinates(dm,PETSC_TRUE,&num_vertices,&num_dim,&coord);CHKERRQ(ierr);
  if (debug) {
    for (c = 0; c < num_dim; c++) {
      ierr = PetscPrintf(comm,"Coordinate %i:\n",c);CHKERRQ(ierr);
      ierr = PetscRealView(num_vertices,&coord[c*num_vertices],PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
  }
  ierr = ex_put_coord(exoid,coord,&coord[num_vertices],&coord[2*num_vertices]);
  
  /*
    Write cell set connectivity table and parameters
    Compute the element number map 
    The element number map is not needed as long as the cell sets are of the type
    0    .. n1
    n1+1 .. n2
    n2+1 ..n3 
    which is the case when a mesh is read from an exo file then distributed, but one never knows
  */
  
  /*
    The following loop should be based on csIS and not csIS_global, but EXO has no
    way to write the block id's other than ex_put_elem_block
    and ensight is bothered if all cell set ID's are not on all files...
    Not a huge deal
  */
    
  ierr = PetscMalloc(num_cells*sizeof(int),&elem_map);CHKERRQ(ierr);
  ierr = ISGetIndices(csIS_global,&csID);CHKERRQ(ierr);
  for (c_offset = 0,set = 0; set < num_cs_global; set++) {
    ierr = DMMeshGetStratumSize(dm,"Cell Sets",csID[set],&num_cell_in_set);CHKERRQ(ierr);
    ierr = DMMeshGetStratumIS(dm,"Cell Sets",csID[set],&cellsIS);CHKERRQ(ierr);
    ierr = ISGetIndices(cellsIS,&cells);CHKERRQ(ierr);
    if (debug) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Cell set %i: %i cells\n",csID[set],num_cell_in_set);CHKERRQ(ierr);
      ierr = PetscIntView(num_cell_in_set,cells,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
    /* 
      Add current block indices to elem_map. EXO uses fortran numbering
    */
    for (c = 0; c < num_cell_in_set; c++,c_offset++) {
      elem_map[c_offset] = cells[c]+1;
    }
    /*
      We make an educated guess as to the type of cell. This misses out quads in 
      a three-dimensional mesh.
      This most likely needs to be fixed by calling again ex_put_elem_block with
      the proper parameters
    */
    if (num_cell_in_set > 0) {
      ierr = DMMeshGetConeSize(dm,cells[0],&num_vertex_per_cell);CHKERRQ(ierr);
      if (num_vertex_per_cell == 2) {
        cell_type = "BAR";
      } else if (num_vertex_per_cell == 3) {
        cell_type = "TRI";
      } else if (num_vertex_per_cell == 8) {
        cell_type = "HEX";
      } else if (num_vertex_per_cell == 4 && num_dim == 2) {
          cell_type = "QUAD";
      } else if (num_vertex_per_cell == 4 && num_dim == 3) {
          cell_type = "TET";
      } else {
        cell_type = "UNKNOWN";
      }
    }  
    ierr = ex_put_elem_block (exoid,csID[set],cell_type,num_cell_in_set,num_vertex_per_cell,num_cs_attr);
    /* 
      Build connectivity table of the current block
    */
    ierr = PetscMalloc(num_cell_in_set*num_vertex_per_cell*sizeof(int),&cs_connect);CHKERRQ(ierr);
    for (c = 0; c < num_cell_in_set; c++) {
      ierr = DMMeshGetCone(dm,cells[c],&elem_connect);
      for (v = 0; v < num_vertex_per_cell; v++) {
        cs_connect[c*num_vertex_per_cell+v] = elem_connect[v]-num_cells+1;
      }
    }
    if (num_cell_in_set > 0) {
      ierr = ex_put_elem_conn(exoid,csID[set],cs_connect);
    }
    ierr = PetscFree(cs_connect);CHKERRQ(ierr);
    ierr = ISRestoreIndices(cellsIS,&cells);CHKERRQ(ierr);  
    ierr = ISDestroy(&cellsIS);CHKERRQ(ierr); 
  }
  ierr = ISRestoreIndices(csIS_global,&csID);CHKERRQ(ierr);
  ierr = ex_put_elem_num_map(exoid,elem_map);
  ierr = PetscFree(elem_map);CHKERRQ(ierr);
  
  /*
    Writing vertex sets 
  */  
  if (num_vs_global > 0) {
    ierr = ISGetIndices(vsIS_global,&vsID);CHKERRQ(ierr);
    for (set = 0; set < num_vs_global; set++) {
      ierr = DMMeshGetStratumSize(dm,"Vertex Sets",vsID[set],&num_vertex_in_set);CHKERRQ(ierr);
      ierr = ex_put_node_set_param(exoid,vsID[set],num_vertex_in_set,num_vs_attr);
      
      ierr = DMMeshGetStratumIS(dm,"Vertex Sets",vsID[set],&verticesIS);CHKERRQ(ierr);
      ierr = ISGetIndices(verticesIS,(const PetscInt**)&vertices);CHKERRQ(ierr);
      
      for (v = 0; v < num_vertex_in_set; v++) {
        vertices[v] -= num_cells-1;
      }
      
      if (num_vertex_in_set > 0) {
        ierr = ex_put_node_set(exoid,vsID[set],vertices);
      }
      ierr = ISRestoreIndices(verticesIS,(const PetscInt**)&vertices);CHKERRQ(ierr);
      ierr = ISDestroy(&verticesIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(vsIS_global,&vsID);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&csIS);CHKERRQ(ierr);
  ierr = ISDestroy(&vsIS);CHKERRQ(ierr);
  ierr = ISDestroy(&csIS_global);CHKERRQ(ierr);
#else
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateScatterToZeroVertex"
/*@
  DMMeshCreateScatterToZeroVertex - Creates the scatter required to scatter local (ghosted)
    Vecs associated with a field defined at the vertices of a mesh to a Vec on cpu 0.

  Input parameters:
. dm - the DMMesh representing the mesh

  Output parameters:
. scatter: the scatter

  Level: advanced

.keywords: mesh,ExodusII
.seealso DMMeshCreateScatterToZeroCell DMMeshCreateScatterToZeroVertexSet DMMeshCreateScatterToZeroCellSet
@*/
PetscErrorCode DMMeshCreateScatterToZeroVertex(DM dm,VecScatter *scatter)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscInt               *vertices;
  PetscInt                num_vertices,num_vertices_global,my_num_vertices_global;
  PetscInt                num_cells,num_cells_global,my_num_cells_global;
  Vec                     v_local,v_zero;
  MPI_Comm                comm;
  int                     rank;
  IS                      is_local;
  ALE::Obj<PETSC_MESH_TYPE>                                         mesh;
  typedef ALE::Mesh<PetscInt,PetscScalar>                           FlexMesh;
  std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> renumbering;


  PetscFunctionBegin;
  ierr = DMMeshGetStratumSize(dm,"depth",0,&num_vertices);CHKERRQ(ierr);
  ierr = DMMeshGetStratumSize(dm,"height",0,&num_cells);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);

  ierr = DMMeshGetMesh(dm,mesh);CHKERRQ(ierr);
  renumbering = mesh->getRenumbering();

  my_num_cells_global = 0;
  my_num_vertices_global = 0;
  ierr = PetscMalloc(num_vertices*sizeof(PetscInt),&vertices);CHKERRQ(ierr);
  /*
    Get the total number of cells and vertices from the mapping, and build array of global indices of the local vertices
    (TO array for the scatter) from the iterator
  */

  for (FlexMesh::renumbering_type::const_iterator r_iter = renumbering.begin(); r_iter != renumbering.end(); ++r_iter) {
    /*
       r_iter->first  is a point number in the sequential (global) mesh
       r_iter->second is the matching local point number in the distributed (local) mesh
    */
    if (r_iter->second > num_cells - 1 && r_iter->first > my_num_vertices_global) {
      my_num_vertices_global = r_iter->first;
    }
    if (r_iter->second < num_cells && r_iter->first > my_num_cells_global) {
      my_num_cells_global = r_iter->first;
    }
    if (r_iter->second > num_cells - 1) {
      vertices[r_iter->second - num_cells] = r_iter->first;
    }
  }
  my_num_cells_global++;
  ierr = MPI_Allreduce(&my_num_cells_global,&num_cells_global,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
  my_num_vertices_global -= num_cells_global-1;
  ierr = MPI_Allreduce(&my_num_vertices_global,&num_vertices_global,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);

  /*
    convert back vertices from point number to vertex numbers
  */
  for (i = 0; i < num_vertices; i++) {
    vertices[i] -= num_cells_global;
  }

  /*
    Build the IS and Vec required to create the VecScatter
    A MUCH better way would be to use VecScatterCreateLocal and not create then destroy
    Vecs, but I would need to understand better how it works
  */
  ierr = VecCreate(comm,&v_local);CHKERRQ(ierr);
  ierr = VecSetSizes(v_local,num_vertices,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v_local);CHKERRQ(ierr);
  ierr = VecCreate(comm,&v_zero);CHKERRQ(ierr);
  ierr = VecSetSizes(v_zero,num_vertices_global*(rank==0),PETSC_DECIDE);CHKERRQ(ierr);  
  ierr = VecSetFromOptions(v_zero);CHKERRQ(ierr);

  ierr = ISCreateGeneral(comm,num_vertices,vertices,PETSC_OWN_POINTER,&is_local);CHKERRQ(ierr);
  ierr = VecScatterCreate(v_local,PETSC_NULL,v_zero,is_local,scatter);CHKERRQ(ierr);

  ierr = ISDestroy(&is_local);CHKERRQ(ierr);
  ierr = VecDestroy(&v_local);CHKERRQ(ierr);
  ierr = VecDestroy(&v_zero);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateScatterToZeroVertexSet"
/*@
  DMMeshCreateScatterToZeroVertexSet - Creates the scatter required to scatter local (ghosted)
    Vecs associated with a field defined at the a vertex set of a mesh to a Vec on cpu 0.

  Input parameters:
. dm - the DMMesh representing the mesh

  Output parameters:
. scatter - the scatter

  Level: advanced

.keywords: mesh,ExodusII
.seealso DMMeshCreateScatterToZeroCell DMMeshCreateScatterToZeroVertex DMMeshCreateScatterToZeroCellSet
@*/
PetscErrorCode DMMeshCreateScatterToZeroVertexSet(DM dm,IS is_local,IS is_zero,VecScatter *scatter)
{
  PetscErrorCode          ierr;
  const PetscInt         *setvertices_local;
  PetscInt               *allvertices,*setvertices_zero,*setvertices_localtozero;
  PetscInt                num_vertices,num_vertices_global,my_num_vertices_global=0;
  PetscInt                num_cells,num_cells_global,my_num_cells_global=0;
  PetscInt                setsize_local,setsize_zero;
  Vec                     v_local,v_zero;
  MPI_Comm                comm;
  int                     rank,i,j,k,l,istart;
  IS                      is_localtozero;
  ALE::Obj<PETSC_MESH_TYPE>                                         mesh;
  typedef ALE::Mesh<PetscInt,PetscScalar>                           FlexMesh;
  std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> renumbering;


  PetscFunctionBegin;
  ierr = DMMeshGetStratumSize(dm,"height",0,&num_cells);CHKERRQ(ierr);
  ierr = DMMeshGetStratumSize(dm,"depth",0,&num_vertices);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);

  ierr = DMMeshGetMesh(dm,mesh);CHKERRQ(ierr);
  renumbering = mesh->getRenumbering();

  ierr = PetscMalloc(num_vertices*sizeof(PetscInt),&allvertices);
  /*
    Build array of global indices of the local vertices (TO array for the scatter)
    from the iterator
  */

  for (FlexMesh::renumbering_type::const_iterator r_iter = renumbering.begin(); r_iter != renumbering.end(); ++r_iter) {
    /*
       r_iter->first  is a point number in the sequential (global) mesh
       r_iter->second is the matching local point number in the distributed (local) mesh
    */
    if (r_iter->second > num_cells - 1 && r_iter->first > my_num_vertices_global) {
      my_num_vertices_global = r_iter->first;
    }
    if (r_iter->second < num_cells && r_iter->first > my_num_cells_global) {
      my_num_cells_global = r_iter->first;
    }
    if (r_iter->second > num_cells-1 ) {
      allvertices[r_iter->second - num_cells] = r_iter->first;
    }
  }
  my_num_cells_global++;
  ierr = MPI_Allreduce(&my_num_cells_global,&num_cells_global,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
  my_num_vertices_global -= num_cells_global-1;
  ierr = MPI_Allreduce(&my_num_vertices_global,&num_vertices_global,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);

  ierr = ISGetSize(is_local,&setsize_local);CHKERRQ(ierr);
  ierr = ISGetSize(is_zero,&setsize_zero);CHKERRQ(ierr);
  ierr = PetscMalloc(setsize_local*sizeof(PetscInt),&setvertices_localtozero);CHKERRQ(ierr);

  if (rank == 0)  {
    ierr = ISGetIndices(is_zero,(const PetscInt**)&setvertices_zero); 
  } else {
    ierr = PetscMalloc(setsize_zero*sizeof(PetscInt),&setvertices_zero);CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(setvertices_zero,setsize_zero,MPIU_INT,0,comm);
  ierr = ISGetIndices(is_local,&setvertices_local);    


  istart = 0;
  for (i = 0; i < setsize_local; i++) {
    j = allvertices[setvertices_local[i]-num_cells];
    /*
      j is the cell number in the seq mesh of the i-th vertex of the local vertex set.
      Search for the matching vertex in vertex set. Because vertex in the zero set are usually 
      numbered in ascending order, we start our search from the previous find.
`   */
        
    for (l = 0, k = istart; l < setsize_zero; l++,k = (l+istart)%setsize_zero) {
      if (setvertices_zero[k] == j) {
        break;
      }
    }
    setvertices_localtozero[i] = k;
    istart = (k+1)%setsize_zero;
  }
  ierr = ISRestoreIndices(is_local,&setvertices_local);
  if (rank == 0) {
    ierr = ISRestoreIndices(is_zero,(const PetscInt**)&setvertices_zero);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(setvertices_zero);CHKERRQ(ierr);   
  }                 
  /*
    Build the IS and Vec required to create the VecScatter
    A MUCH better way would be to use VecScatterCreateLocal and not create then destroy
    Vecs, but I would need to understand better how it works
  */
  ierr = VecCreate(comm,&v_local);CHKERRQ(ierr);
  ierr = VecSetSizes(v_local,setsize_local,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v_local);CHKERRQ(ierr);

  ierr = VecCreate(comm,&v_zero);CHKERRQ(ierr);
  ierr = VecSetSizes(v_zero,setsize_zero*(rank==0),PETSC_DECIDE);CHKERRQ(ierr);  
  ierr = VecSetFromOptions(v_zero);CHKERRQ(ierr);

  ierr = ISCreateGeneral(comm,setsize_local,setvertices_localtozero,PETSC_OWN_POINTER,&is_localtozero);CHKERRQ(ierr);
  ierr = VecScatterCreate(v_local,PETSC_NULL,v_zero,is_localtozero,scatter);CHKERRQ(ierr);

  ierr = ISDestroy(&is_localtozero);CHKERRQ(ierr);
  ierr = PetscFree(allvertices);CHKERRQ(ierr);
  ierr = VecDestroy(&v_local);CHKERRQ(ierr);
  ierr = VecDestroy(&v_zero);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateScatterToZeroCell"
/*@
  DMMeshCreateScatterToZeroCell - Creates the scatter required to scatter local (ghosted)
    Vecs associated with a field defined at the cells of a mesh to a Vec on cpu 0.

  Input parameters:
. dm - the DMMesh representing the mesh

  Output parameters:
. scatter - the scatter

  Level: advanced

.keywords: mesh,ExodusII
.seealso DMMeshCreateScatterToZeroVertex DMMeshCreateScatterToZeroVertexSet DMMeshCreateScatterToZeroCellSet
@*/
PetscErrorCode DMMeshCreateScatterToZeroCell(DM dm,VecScatter *scatter)
{
  PetscErrorCode          ierr;
  PetscInt               *cells;
  PetscInt                num_cells,num_cells_global,my_num_cells_global;
  Vec                     v_local,v_zero;
  MPI_Comm                comm;
  int                     rank;
  IS                      is_local;
  ALE::Obj<PETSC_MESH_TYPE>                                         mesh;
  typedef ALE::Mesh<PetscInt,PetscScalar>                           FlexMesh;
  std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> renumbering;


  PetscFunctionBegin;
  ierr = DMMeshGetStratumSize(dm,"height",0,&num_cells);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);

  ierr = DMMeshGetMesh(dm,mesh);CHKERRQ(ierr);
  renumbering = mesh->getRenumbering();

  my_num_cells_global = 0;
  ierr = PetscMalloc(num_cells*sizeof(PetscInt),&cells);
  /*
    Get the total number of cells from the mapping, and build array of global indices of the local cells (TO array for the scatter)
    from the iterator
  */

  for (FlexMesh::renumbering_type::const_iterator r_iter = renumbering.begin(); r_iter != renumbering.end(); ++r_iter) {
    /*
       r_iter->first  is a point number in the sequential (global) mesh
       r_iter->second is the matching local point number in the distributed (local) mesh
    */
    if (r_iter->second < num_cells && r_iter->first > my_num_cells_global) {
      my_num_cells_global = r_iter->first;
    }
    if (r_iter->second < num_cells ) {
      cells[r_iter->second] = r_iter->first;
    }
  }
  my_num_cells_global++;
  ierr = MPI_Allreduce(&my_num_cells_global,&num_cells_global,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);

  /*
    Build the IS and Vec required to create the VecScatter
    A MUCH better way would be to use VecScatterCreateLocal and not create then destroy
    Vecs, but I would need to understand better how it works
  */
  ierr = VecCreate(comm,&v_local);CHKERRQ(ierr);
  ierr = VecSetSizes(v_local,num_cells,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v_local);CHKERRQ(ierr);
  ierr = VecCreate(comm,&v_zero);CHKERRQ(ierr);
  ierr = VecSetSizes(v_zero,num_cells_global*(rank==0),PETSC_DECIDE);CHKERRQ(ierr);  
  ierr = VecSetFromOptions(v_zero);CHKERRQ(ierr);

  ierr = ISCreateGeneral(comm,num_cells,cells,PETSC_OWN_POINTER,&is_local);CHKERRQ(ierr);
  ierr = VecScatterCreate(v_local,PETSC_NULL,v_zero,is_local,scatter);CHKERRQ(ierr);

  ierr = ISDestroy(&is_local);CHKERRQ(ierr);
  ierr = VecDestroy(&v_local);CHKERRQ(ierr);
  ierr = VecDestroy(&v_zero);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateScatterToZeroCellSet"
/*@
  DMMeshCreateScatterToZeroCellSet - Creates the scatter required to scatter local (ghosted)
    Vecs associated with a field defined at a cell set of a mesh to a Vec on cpu 0.

  Input parameters:
. dm - the DMMesh representing the mesh

  Output parameters:
. scatter - the scatter

  Level: advanced

.keywords: mesh,ExodusII
.seealso DMMeshCreateScatterToZeroCell DMMeshCreateScatterToZeroVertexSet DMMeshCreateScatterToZeroVertex
@*/
PetscErrorCode DMMeshCreateScatterToZeroCellSet(DM dm,IS is_local,IS is_zero,VecScatter *scatter)
{
  PetscErrorCode          ierr;
  const PetscInt         *setcells_local;
  PetscInt               *allcells,*setcells_zero,*setcells_localtozero;
  PetscInt                num_cells;
  PetscInt                setsize_local,setsize_zero;
  Vec                     v_local,v_zero;
  MPI_Comm                comm;
  int                     rank,i,j,k,l,istart;
  IS                      is_localtozero;
  ALE::Obj<PETSC_MESH_TYPE>                                         mesh;
  typedef ALE::Mesh<PetscInt,PetscScalar>                           FlexMesh;
  std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> renumbering;


  PetscFunctionBegin;
  ierr = DMMeshGetStratumSize(dm,"height",0,&num_cells);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);

  ierr = DMMeshGetMesh(dm,mesh);CHKERRQ(ierr);
  renumbering = mesh->getRenumbering();

  ierr = PetscMalloc(num_cells*sizeof(PetscInt),&allcells);
  /*
    Build array of global indices of the local cells (TO array for the scatter)
    from the iterator
  */

  for (FlexMesh::renumbering_type::const_iterator r_iter = renumbering.begin(); r_iter != renumbering.end(); ++r_iter) {
    /*
       r_iter->first  is a point number in the sequential (global) mesh
       r_iter->second is the matching local point number in the distributed (local) mesh
    */
    if (r_iter->second < num_cells ) {
      allcells[r_iter->second] = r_iter->first;
    }
  }

  ierr = ISGetSize(is_local,&setsize_local);CHKERRQ(ierr);
  ierr = ISGetSize(is_zero,&setsize_zero);CHKERRQ(ierr);
  ierr = PetscMalloc(setsize_local*sizeof(PetscInt),&setcells_localtozero);CHKERRQ(ierr);

  if (rank == 0)  {
    ierr = ISGetIndices(is_zero,(const PetscInt**)&setcells_zero);    
  } else {
    ierr = PetscMalloc(setsize_zero*sizeof(PetscInt),&setcells_zero);CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(setcells_zero,setsize_zero,MPIU_INT,0,comm);
  ierr = ISGetIndices(is_local,&setcells_local);    

  istart = 0;
  for (i = 0; i < setsize_local; i++) {
    j = allcells[setcells_local[i]];
    /*
      j is the cell number in the seq mesh of the i-th cell of the local cell set.
      Search for the matching cell in cell set. Because cells in the zero set are usually 
      numbered sequentially, we start our search from the previous find.
`   */
        
    for (l = 0, k = istart; l < setsize_zero; l++,k = (l+istart)%setsize_zero) {
      if (setcells_zero[k] == j) {
        break;
      }
    }
    setcells_localtozero[i] = k;
    istart = (k+1)%setsize_zero;
  }
  ierr = ISRestoreIndices(is_local,&setcells_local);
  if (rank == 0) {
    ierr = ISRestoreIndices(is_zero,(const PetscInt**)&setcells_zero);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(setcells_zero);CHKERRQ(ierr);   
  }                 
  /*
    Build the IS and Vec required to create the VecScatter
    A MUCH better way would be to use VecScatterCreateLocal and not create then destroy
    Vecs, but I would need to understand better how it works
  */
  ierr = VecCreate(comm,&v_local);CHKERRQ(ierr);
  ierr = VecSetSizes(v_local,setsize_local,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v_local);CHKERRQ(ierr);

  ierr = VecCreate(comm,&v_zero);CHKERRQ(ierr);
  ierr = VecSetSizes(v_zero,setsize_zero*(rank==0),PETSC_DECIDE);CHKERRQ(ierr);  
  ierr = VecSetFromOptions(v_zero);CHKERRQ(ierr);

  ierr = ISCreateGeneral(comm,setsize_local,setcells_localtozero,PETSC_OWN_POINTER,&is_localtozero);CHKERRQ(ierr);
  ierr = VecScatterCreate(v_local,PETSC_NULL,v_zero,is_localtozero,scatter);CHKERRQ(ierr);

  ierr = PetscFree(setcells_localtozero);CHKERRQ(ierr);
  ierr = PetscFree(allcells);CHKERRQ(ierr);
  ierr = VecDestroy(&v_local);CHKERRQ(ierr);
  ierr = VecDestroy(&v_zero);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViewExodusVertex"
/*@

  VecViewExodusVertex - Write a Vec representing nodal values of some field in an exodusII file.

  Collective on comm

  The behavior differs depending on the size of comm:
    if size(comm) == 1 each processor writes its local vector into a separate file
    if size(comm) > 1, the values are sent to cpu0, and written in a single file


  Input Parameters:
+ dm   - the DMMesh representing the mesh
. v    - the LOCAL vector of values to be saved (i.e. with ghost values) obtained with SectionRealCreateLocalVector.
         if v represents a field with several components, the block size must be set accordingly using VecSetBlockSize()
. exoid - the id of the exodusII file (obtained with ex_open or ex_create)
. step  - the time step to write
- exofield - the position in the exodus field of the first component

  Interpolated meshes are not supported.

  Level: beginner

.keywords: mesh,ExodusII
.seealso: MeshCreate() MeshCreateExodus() SectionRealCreateLocalVector() VecLoadExodusVertex()
          VecViewExodusVertexSet() VecLoadExodusVertexSet() VecViewExodusCellSet() VecLoadExodusCellSet()
          VecViewExodusCell() VecLoadExodusCell()
@*/
PetscErrorCode VecViewExodusVertex(DM dm,Vec v,MPI_Comm comm,PetscInt exoid,PetscInt step,PetscInt exofield)
{
#ifdef PETSC_HAVE_EXODUSII
  PetscInt                rank,num_proc;
  PetscErrorCode          ierr;
  PetscInt                c,num_vertices,num_vertices_zero,num_dof;
  Vec                     vdof,vdof_zero;
  PetscReal               *vdof_array;
  VecScatter              scatter;
#endif

  PetscFunctionBegin;
#ifdef PETSC_HAVE_EXODUSII
  ierr = MPI_Comm_size(comm,&num_proc);
  ierr = MPI_Comm_rank(comm,&rank);
  ierr = VecGetBlockSize(v,&num_dof);

  ierr = DMMeshGetStratumSize(dm,"depth",0,&num_vertices);CHKERRQ(ierr);

  ierr = VecCreate(comm,&vdof);CHKERRQ(ierr);
  ierr = VecSetSizes(vdof,num_vertices,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(vdof);CHKERRQ(ierr);

  if (num_proc == 1) {
    ierr = PetscMalloc(num_vertices*sizeof(PetscReal),&vdof_array);CHKERRQ(ierr);
    for (c = 0; c < num_dof; c++) {
      ierr = VecStrideGather(v,c,vdof,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecGetArray(vdof,&vdof_array);CHKERRQ(ierr);
      ierr = ex_put_nodal_var(exoid,step,exofield+c,num_vertices,vdof_array); 
      ierr = VecRestoreArray(vdof,&vdof_array);CHKERRQ(ierr);
      if (ierr != 0) {
        SETERRQ2(comm,PETSC_ERR_FILE_WRITE,"Unable to write to file id %i. ex_put_nodal_var returned %i",exoid,ierr);
      }
    }
    ierr = PetscFree(vdof_array);CHKERRQ(ierr);
  } else {
    /*
      Build the scatter sending one dof towards cpu 0
    */
    ierr = DMMeshCreateScatterToZeroVertex(dm,&scatter);CHKERRQ(ierr);
    /*
      Another an ugly hack to get the total number of vertices in the mesh. This has got to stop...
    */
    num_vertices_zero = scatter->to_n;

    ierr = VecCreate(comm,&vdof_zero);CHKERRQ(ierr);
    ierr = VecSetSizes(vdof_zero,num_vertices_zero,PETSC_DECIDE);CHKERRQ(ierr);  
    ierr = VecSetFromOptions(vdof_zero);CHKERRQ(ierr);

    for (c = 0; c < num_dof; c++) {
      ierr = VecStrideGather(v,c,vdof,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecScatterBegin(scatter,vdof,vdof_zero,INSERT_VALUES,SCATTER_FORWARD);
      ierr = VecScatterEnd(scatter,vdof,vdof_zero,INSERT_VALUES,SCATTER_FORWARD);
      if (rank == 0) {
        ierr = VecGetArray(vdof_zero,&vdof_array);CHKERRQ(ierr);
        ierr = ex_put_nodal_var(exoid,step,exofield+c,num_vertices_zero,vdof_array);
        ierr = VecRestoreArray(vdof_zero,&vdof_array);CHKERRQ(ierr);
      }
    } 
    /*
      Clean up
    */
    ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
    ierr = VecDestroy(&vdof_zero);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&vdof);CHKERRQ(ierr);
#else
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecLoadExodusVertex"
/*@

  VecLoadExodusVertex - Loads a Vec representing nodal values of some field from an exodusII file.

  Collective on comm

  The behavior differs depending on the size of comm:
    if size(comm) == 1 each processor reads its local vector from a separate file
    if size(comm) > 1, the values are read by cpu 0 from a single file then scattered to each processor


  Input Parameters:
+ dm   - the DMMesh representing the mesh
. v    - the LOCAL vector of values to be read (i.e. with ghost values) obtained with SectionRealCreateLocalVector.
         if v represents a field with several components, the block size must be set accordingly using VecSetBlockSize()
. exoid - the id of the exodusII file (obtained with ex_open or ex_create)
. step  - the time step to read
- exofield - the position in the exodus field of the first component

  Interpolated meshes are not supported.

  Level: beginner

.keywords: mesh,ExodusII
.seealso: MeshCreate() MeshCreateExodus() SectionRealCreateLocalVector() VecViewExodusVertex()
          VecViewExodusVertexSet() VecLoadExodusVertexSet() VecViewExodusCellSet() VecLoadExodusCellSet()
          VecViewExodusCell() VecLoadExodusCell()

@*/
PetscErrorCode VecLoadExodusVertex(DM dm,Vec v,MPI_Comm comm,PetscInt exoid,PetscInt step,PetscInt exofield)
{
#ifdef PETSC_HAVE_EXODUSII
  PetscInt                rank,num_proc;
  PetscErrorCode          ierr;
  PetscInt                c,num_vertices,num_vertices_global,num_dof;
  Vec                     vdof,vzero;
  PetscReal               *vdof_array;
  VecScatter              scatter;
#endif

#ifdef PETSC_HAVE_EXODUSII
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&num_proc);
  ierr = MPI_Comm_rank(comm,&rank);
  ierr = VecGetBlockSize(v,&num_dof);

  ierr = DMMeshGetStratumSize(dm,"depth",0,&num_vertices);CHKERRQ(ierr);

  ierr = VecCreate(comm,&vdof);CHKERRQ(ierr);
  ierr = VecSetSizes(vdof,num_vertices,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(vdof);CHKERRQ(ierr);

  if (num_proc == 1) {
    for (c = 0; c < num_dof; c++) {
      ierr = VecGetArray(vdof,&vdof_array);CHKERRQ(ierr);
      ierr = ex_get_nodal_var(exoid,step,exofield+c,num_vertices,vdof_array); 
      ierr = VecRestoreArray(vdof,&vdof_array);CHKERRQ(ierr);
      if (ierr != 0) {
        SETERRQ2(comm,PETSC_ERR_FILE_READ,"Unable to read file id %i. ex_put_nodal_var returned %i",exoid,ierr);
      }
      ierr = VecStrideScatter(vdof,c,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  } else {
    /*
      Build the scatter sending one dof towards cpu 0
    */
    ierr = DMMeshCreateScatterToZeroVertex(dm,&scatter);CHKERRQ(ierr);
    /*
      Another an ugly hack to get the total number of vertices in the mesh. This has got to stop...
    */
    num_vertices_global = scatter->to_n;
    ierr = MPI_Bcast(&num_vertices_global,1,MPIU_INT,0,comm);CHKERRQ(ierr);

    ierr = VecCreate(comm,&vzero);CHKERRQ(ierr);
    ierr = VecSetSizes(vzero,num_vertices_global*(rank==0),PETSC_DECIDE);CHKERRQ(ierr);  
    ierr = VecSetFromOptions(vzero);CHKERRQ(ierr);

    for (c = 0; c < num_dof; c++) {
      if (rank == 0) {
        ierr = VecGetArray(vzero,&vdof_array);CHKERRQ(ierr);
        ierr = ex_get_nodal_var(exoid,step,exofield+c,num_vertices_global,vdof_array);
        ierr = VecRestoreArray(vzero,&vdof_array);CHKERRQ(ierr);
      }
      ierr = VecScatterBegin(scatter,vzero,vdof,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecScatterEnd(scatter,vzero,vdof,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecStrideScatter(vdof,c,v,INSERT_VALUES);CHKERRQ(ierr);
    } 
    /*
      Clean up
    */
    ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
    ierr = VecDestroy(&vzero);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&vdof);CHKERRQ(ierr);
#else
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViewExodusVertexSet"
/*@

  VecViewExodusVertexSet - Write a Vec representing nodal values of some field at a vertex set in an exodusII file.

  Collective on comm

  The behavior differs depending on the size of comm:
    if size(comm) == 1 each processor writes its local vector into a separate file
    if size(comm) > 1, the values are sent to cpu0, and written in a single file


  Input Parameters:
+ dm   - the DMMesh representing the mesh
. v    - the LOCAL vector of values to be saved (i.e. with ghost values) obtained with SectionRealCreateLocalVector.
         if v represents a field with several components, the block size must be set accordingly using VecSetBlockSize()
. vsID  - the vertex set ID
. exoid - the id of the exodusII file (obtained with ex_open or ex_create)
. step  - the time step to write
- exofield - the position in the exodus field of the first component

  Interpolated meshes are not supported.

  Level: beginner

.keywords: mesh,ExodusII
.seealso: MeshCreate() MeshCreateExodus() SectionRealCreateLocalVector() VecLoadExodusVertexSet()
          VecViewExodusVertex() VecLoadExodusVertex() VecViewExodusCellSet() VecLoadExodusCellSet()
          VecViewExodusCell() VecLoadExodusCell()

@*/
PetscErrorCode VecViewExodusVertexSet(DM dm,Vec v,PetscInt vsID,MPI_Comm comm,PetscInt exoid,PetscInt step,PetscInt exofield)
{
#ifdef PETSC_HAVE_EXODUSII
  PetscInt                rank,num_proc;
  PetscErrorCode          ierr;
  PetscInt                c,num_vertices_in_set=0,num_vertices_zero=0,num_cells_zero,num_dof;
  Vec                     vdof,vdof_zero;
  PetscReal               *vdof_array;
  VecScatter              scatter;
  PetscInt                i,junk1,junk2,junk3,junk4,junk5;
  char                    junk[MAX_LINE_LENGTH+1];
  IS                      vsIS,vsIS_zero;
  PetscInt               *vs_vertices_zero;
  MPI_Comm                dmcomm;
#endif

#ifdef PETSC_HAVE_EXODUSII
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&num_proc);
  ierr = MPI_Comm_rank(comm,&rank);
  ierr = VecGetBlockSize(v,&num_dof);
  ierr = PetscObjectGetComm((PetscObject) dm,&dmcomm);CHKERRQ(ierr);
  
  ierr = DMMeshGetStratumSize(dm,"Vertex Sets",vsID,&num_vertices_in_set);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,num_vertices_in_set,&vdof);CHKERRQ(ierr);

  if (num_proc == 1) {
    if (num_vertices_in_set > 0) {
      for (c = 0; c < num_dof; c++) {
        ierr = VecStrideGather(v,c,vdof,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecGetArray(vdof,&vdof_array);CHKERRQ(ierr);
        ierr = ex_put_nset_var(exoid,step,exofield+c,vsID,num_vertices_in_set,vdof_array); 
        if (ierr != 0) {
          SETERRQ2(comm,PETSC_ERR_FILE_WRITE,"Unable to write to file id %i. ex_put_nset_var returned %i",exoid,ierr);
        }
        ierr = VecRestoreArray(vdof,&vdof_array);CHKERRQ(ierr);
      }
    }
  } else {
    /*
      Build the scatter sending one dof towards cpu 0
    */
    ierr = DMMeshGetStratumIS(dm,"Vertex Sets",vsID,&vsIS);CHKERRQ(ierr);
    if (rank == 0) {
      ierr = ex_get_node_set_param(exoid,vsID,&num_vertices_zero,&junk1);
    }
    ierr = PetscMalloc(num_vertices_zero*sizeof(PetscInt),&vs_vertices_zero);CHKERRQ(ierr);
    if (rank == 0) {
      ierr = ex_get_node_set(exoid,vsID,vs_vertices_zero);
      ierr = ex_get_init(exoid,junk,&junk1,&junk2,&num_cells_zero,&junk3,&junk4,&junk5);CHKERRQ(ierr);
      for (i = 0; i < num_vertices_zero; i++) {
        vs_vertices_zero[i] += num_cells_zero-1;
      }
    }
    ierr = ISCreateGeneral(dmcomm,num_vertices_zero,vs_vertices_zero,PETSC_OWN_POINTER,&vsIS_zero);CHKERRQ(ierr);
    ierr = DMMeshCreateScatterToZeroVertexSet(dm,vsIS,vsIS_zero,&scatter);CHKERRQ(ierr);
    ierr = ISDestroy(&vsIS);CHKERRQ(ierr);
    ierr = ISDestroy(&vsIS_zero);CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF,num_vertices_zero*(rank == 0),&vdof_zero);CHKERRQ(ierr);

    for (c = 0; c < num_dof; c++) {
      ierr = VecStrideGather(v,c,vdof,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecScatterBegin(scatter,vdof,vdof_zero,INSERT_VALUES,SCATTER_FORWARD);
      ierr = VecScatterEnd(scatter,vdof,vdof_zero,INSERT_VALUES,SCATTER_FORWARD);

      if (rank == 0) {
        ierr = VecGetArray(vdof_zero,&vdof_array);CHKERRQ(ierr);
        ierr = ex_put_nset_var(exoid,step,exofield+c,vsID,num_vertices_zero,vdof_array);
        if (ierr != 0) {
          SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,"Unable to write to file id %i. ex_put_nset_var returned %i",exoid,ierr);
        }
        ierr = VecRestoreArray(vdof_zero,&vdof_array);CHKERRQ(ierr);
      }
    } 
    /*
      Clean up
    */
    ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
    ierr = VecDestroy(&vdof_zero);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&vdof);CHKERRQ(ierr);
#else
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecLoadExodusVertexSet"
/*@

  VecLoadExodusVertexSet - Write a Vec representing nodal values of some field at a vertex set in an exodusII file.

  Collective on comm

  The behavior differs depending on the size of comm:
    if size(comm) == 1 each processor writes its local vector into a separate file
    if size(comm) > 1, the values are sent to cpu0, and written in a single file


  Input Parameters:
+ dm   - the DMMesh representing the mesh
. v    - the LOCAL vector of values to be read (i.e. with ghost values) obtained with SectionRealCreateLocalVector.
         if v represents a field with several components, the block size must be set accordingly using VecSetBlockSize()
. vsID  - the vertex set ID
. exoid - the id of the exodusII file (obtained with ex_open or ex_create)
. step  - the time step to write
- exofield - the position in the exodus field of the first component

  Interpolated meshes are not supported.

  Level: beginner

.keywords: mesh,ExodusII
.seealso: MeshCreate() MeshCreateExodus() SectionRealCreateLocalVector() VecLoadExodusVertex()

@*/
PetscErrorCode VecLoadExodusVertexSet(DM dm,Vec v,PetscInt vsID,MPI_Comm comm,PetscInt exoid,PetscInt step,PetscInt exofield)
{
#ifdef PETSC_HAVE_EXODUSII
  PetscInt                rank,num_proc;
  PetscErrorCode          ierr;
  PetscInt                c,num_vertices_in_set=0,num_vertices_zero=0,num_cells_zero,num_dof;
  Vec                     vdof,vdof_zero;
  PetscReal               *vdof_array;
  VecScatter              scatter;
  PetscInt                i,junk1,junk2,junk3,junk4,junk5;
  char                    junk[MAX_LINE_LENGTH+1];
  IS                      vsIS,vsIS_zero;
  PetscInt               *vs_vertices_zero;
  MPI_Comm                dmcomm;
#endif

#ifdef PETSC_HAVE_EXODUSII
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&num_proc);
  ierr = MPI_Comm_rank(comm,&rank);
  ierr = VecGetBlockSize(v,&num_dof);
  ierr = PetscObjectGetComm((PetscObject) dm,&dmcomm);CHKERRQ(ierr);

  ierr = DMMeshGetStratumSize(dm,"Vertex Sets",vsID,&num_vertices_in_set);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,num_vertices_in_set,&vdof);CHKERRQ(ierr);

  if (num_proc == 1) {
    if (num_vertices_in_set > 0) {
      for (c = 0; c < num_dof; c++) {
        ierr = VecGetArray(vdof,&vdof_array);CHKERRQ(ierr);
        ierr = ex_get_nset_var(exoid,step,exofield+c,vsID,num_vertices_in_set,vdof_array); 
        if (ierr != 0) {
          SETERRQ2(comm,PETSC_ERR_FILE_READ,"Unable to read from file id %i. ex_put_nset_var returned %i",exoid,ierr);
        }
        ierr = VecRestoreArray(vdof,&vdof_array);CHKERRQ(ierr);
        ierr = VecStrideScatter(vdof,c,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  } else {
    /*
      Build the scatter sending one dof towards cpu 0
    */
    ierr = DMMeshGetStratumIS(dm,"Vertex Sets",vsID,&vsIS);CHKERRQ(ierr);
    if (rank == 0) {
      ierr = ex_get_node_set_param(exoid,vsID,&num_vertices_zero,&junk1);
    }
    ierr = PetscMalloc(num_vertices_zero*sizeof(PetscInt),&vs_vertices_zero);CHKERRQ(ierr);
    if (rank == 0) {
      ierr = ex_get_node_set(exoid,vsID,vs_vertices_zero);
      ierr = ex_get_init(exoid,junk,&junk1,&junk2,&num_cells_zero,&junk3,&junk4,&junk5);CHKERRQ(ierr);
      for (i = 0; i < num_vertices_zero; i++) {
        vs_vertices_zero[i] += num_cells_zero-1;
      }
    }
    ierr = ISCreateGeneral(dmcomm,num_vertices_zero,vs_vertices_zero,PETSC_OWN_POINTER,&vsIS_zero);CHKERRQ(ierr);
    ierr = DMMeshCreateScatterToZeroVertexSet(dm,vsIS,vsIS_zero,&scatter);CHKERRQ(ierr);
    ierr = ISDestroy(&vsIS);CHKERRQ(ierr);
    ierr = ISDestroy(&vsIS_zero);CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF,num_vertices_zero*(rank == 0),&vdof_zero);CHKERRQ(ierr);

    for (c = 0; c < num_dof; c++) {
      if (rank == 0) {
        ierr = VecGetArray(vdof_zero,&vdof_array);CHKERRQ(ierr);
        ierr = ex_get_nset_var(exoid,step,exofield+c,vsID,num_vertices_zero,vdof_array);
        if (ierr != 0) {
          SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read from file id %i. ex_get_nset_var returned %i",exoid,ierr);
        }
        ierr = VecRestoreArray(vdof_zero,&vdof_array);CHKERRQ(ierr);
      }
      ierr = VecScatterBegin(scatter,vdof_zero,vdof,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecScatterEnd(scatter,vdof_zero,vdof,INSERT_VALUES,SCATTER_REVERSE);

      ierr = VecStrideScatter(vdof,c,v,INSERT_VALUES);CHKERRQ(ierr);
    } 
    /*
      Clean up
    */
    ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
    ierr = VecDestroy(&vdof_zero);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&vdof);CHKERRQ(ierr);
#else
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViewExodusCell"
/*@

  VecViewExodusCell - Write a Vec representing the values of a field at all cells to an exodusII file.

  Input Parameters:
+ dm   - the DMMesh representing the mesh
. v    - the LOCAL vector of values to be saved (i.e. with ghost values) obtained with SectionRealCreateLocalVector.
         if v represents a field with several components, the block size must be set accordingly using VecSetBlockSize().
. comm - the communicator associated to the exo file
  + if size(comm) == 1 each processor writes its local vector into a separate file
  . if size(comm) > 1, the values are sent to cpu 0, and written in a single file
- exoid - the id of the exodusII file (obtained with ex_open or ex_create)
. step  - the time step to write
. exofield - the position in the exodus field of the first component

- Interpolated meshes are not supported.

  Level: beginner

.keywords: mesh,ExodusII
.seealso: MeshCreate() MeshCreateExodus() SectionRealCreateLocalVector() VecLoadExodusCell()
          VecViewExodusVertexSet() VecLoadExodusVertexSet() VecViewExodusCellSet() VecLoadExodusCellSet()
          VecViewExodusVertex() VecLoadExodusVertex()

@*/
PetscErrorCode VecViewExodusCell(DM dm,Vec v,MPI_Comm comm,PetscInt exoid,PetscInt step,PetscInt exofield)
{
#ifdef PETSC_HAVE_EXODUSII
  PetscErrorCode          ierr;
  PetscInt                rank,num_proc,num_dof,num_cells,num_cells_in_set,num_cells_zero=0;
  PetscInt                num_cs_zero,num_cs,set,c,istart;
  PetscInt               *setsID_zero;
  const PetscInt         *setsID;
  IS                      setIS,csIS,setIS_zero;
  VecScatter              setscatter,setscattertozero;
  Vec                     vdof,vdof_set,vdof_set_zero;
  PetscReal              *vdof_set_array;
  PetscInt                junk1,junk2,junk3,junk4,junk5;
  char                    junk[MAX_LINE_LENGTH+1];
  MPI_Comm                dmcomm;
#endif

#ifdef PETSC_HAVE_EXODUSII
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&num_proc);
  ierr = MPI_Comm_rank(comm,&rank);
  ierr = VecGetBlockSize(v,&num_dof);
  ierr = PetscObjectGetComm((PetscObject) dm,&dmcomm);CHKERRQ(ierr);
  ierr = DMMeshGetLabelSize(dm,"Cell Sets",&num_cs);CHKERRQ(ierr);
  ierr = DMMeshGetStratumSize(dm,"height",0,&num_cells);CHKERRQ(ierr);
  
  ierr = VecCreateSeq(PETSC_COMM_SELF,num_cells,&vdof);CHKERRQ(ierr);
  
  if (num_proc == 1) {
    ierr = DMMeshGetLabelSize(dm,"Cell Sets",&num_cs);CHKERRQ(ierr);
    ierr = DMMeshGetLabelIdIS(dm,"Cell Sets",&csIS);CHKERRQ(ierr);
    ierr = ISGetIndices(csIS,&setsID);CHKERRQ(ierr);

    for (set = 0; set < num_cs; set++) {
      /*
        Get the IS for the current set, the Vec containing a single dof and create
        the scatter for the restriction to the current cell set
      */
      ierr = DMMeshGetStratumIS(dm,"Cell Sets",setsID[set],&setIS);CHKERRQ(ierr);
      ierr = DMMeshGetStratumSize(dm,"Cell Sets",setsID[set],&num_cells_in_set);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,num_cells_in_set,&vdof_set);CHKERRQ(ierr);
      ierr = VecScatterCreate(vdof,setIS,vdof_set,PETSC_NULL,&setscatter);CHKERRQ(ierr);
      for (c = 0; c < num_dof; c++) {
        ierr = VecStrideGather(v,c,vdof,INSERT_VALUES);CHKERRQ(ierr);

        /*
          Get the restriction of vdof to the current set
        */
        ierr = VecScatterBegin(setscatter,vdof,vdof_set,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(setscatter,vdof,vdof_set,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        
        /*
          Write array to disk
        */
        ierr = VecGetArray(vdof_set,&vdof_set_array);CHKERRQ(ierr);
        ierr = ex_put_elem_var(exoid,step,exofield+c,setsID[set],num_cells_in_set,vdof_set_array);
        if (ierr != 0) {
          SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,"Unable to write to file id %i. ex_put_elem_var returned %i",exoid,ierr);
        }
        ierr = VecRestoreArray(vdof_set,&vdof_set_array);CHKERRQ(ierr);
      }
      ierr = VecDestroy(&vdof_set);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&setscatter);CHKERRQ(ierr);
      ierr = ISDestroy(&setIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(csIS,&setsID);CHKERRQ(ierr);
    ierr = ISDestroy(&csIS);CHKERRQ(ierr);
  } else {
    /*
      Get the number of blocks and the list of ID directly from  the file. This is easier
      than trying to reconstruct the global lists from all individual IS
    */
    if (rank == 0) {
      ierr = ex_get_init(exoid,junk,&junk1,&junk2,&junk3,&num_cs_zero,&junk4,&junk5);
    }
    ierr = MPI_Bcast(&num_cs_zero,1,MPIU_INT,0,dmcomm);CHKERRQ(ierr);
    ierr = PetscMalloc(num_cs_zero * sizeof(PetscInt),&setsID_zero);CHKERRQ(ierr);
    if (rank == 0) {
      ierr = ex_get_elem_blk_ids(exoid,setsID_zero);
    }
    ierr = MPI_Bcast(setsID_zero,num_cs_zero,MPIU_INT,0,dmcomm);
    
    istart = 0;
    for (set = 0; set < num_cs_zero; set++) {
      /*
        Get the size of the size of the set on cpu 0 and create a Vec to receive values
      */
      if (rank == 0) {
        ierr = ex_get_elem_block(exoid,setsID_zero[set],junk,&num_cells_zero,&junk1,&junk2);
      }
      ierr = VecCreateSeq(PETSC_COMM_SELF,num_cells_zero,&vdof_set_zero);CHKERRQ(ierr);

      /*
        Get the IS for the current set, the Vec containing a single dof and create
        the scatter for the restriction to the current cell set
      */
      ierr = DMMeshGetStratumIS(dm,"Cell Sets",setsID_zero[set],&setIS);CHKERRQ(ierr);
      ierr = DMMeshGetStratumSize(dm,"Cell Sets",setsID_zero[set],&num_cells_in_set);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,num_cells_in_set,&vdof_set);CHKERRQ(ierr);
      ierr = VecScatterCreate(vdof,setIS,vdof_set,PETSC_NULL,&setscatter);CHKERRQ(ierr);
      /*
        Get the scatter to send the values of a single dof at a single block to cpu 0
      */
      ierr = ISCreateStride(dmcomm,num_cells_zero,istart,1,&setIS_zero);
      ierr = DMMeshCreateScatterToZeroCellSet(dm,setIS,setIS_zero,&setscattertozero);
      ierr = ISDestroy(&setIS_zero);CHKERRQ(ierr);

      for (c = 0; c < num_dof; c++) {
        ierr = VecStrideGather(v,c,vdof,INSERT_VALUES);CHKERRQ(ierr);

        /*
          Get the restriction of vdof to the current set
        */
        ierr = VecScatterBegin(setscatter,vdof,vdof_set,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(setscatter,vdof,vdof_set,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        
        /*
          Scatter vdof_set to cpu 0
        */
        ierr = VecScatterBegin(setscattertozero,vdof_set,vdof_set_zero,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(setscattertozero,vdof_set,vdof_set_zero,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        
        /*
          Write array to disk
        */
        if (rank == 0) {
          ierr = VecGetArray(vdof_set_zero,&vdof_set_array);CHKERRQ(ierr);
          ierr = ex_put_elem_var(exoid,step,exofield+c,setsID_zero[set],num_cells_zero,vdof_set_array);
          ierr = VecRestoreArray(vdof_set_zero,&vdof_set_array);CHKERRQ(ierr);
        }
      }
      istart += num_cells_zero;
      ierr = VecDestroy(&vdof_set_zero);CHKERRQ(ierr);
      ierr = VecDestroy(&vdof_set);CHKERRQ(ierr);
      /*
        Does this really need to be protected with this test?
      */
      if (num_cells_in_set > 0) {
        /*
          Does this really need to be protected with this test?
        */
        ierr = VecScatterDestroy(&setscatter);CHKERRQ(ierr);
      }
      ierr = VecScatterDestroy(&setscattertozero);CHKERRQ(ierr);
      ierr = ISDestroy(&setIS);CHKERRQ(ierr);
    }
    ierr = PetscFree(setsID_zero);CHKERRQ(ierr);
  }  
  
  ierr = VecDestroy(&vdof);CHKERRQ(ierr);
#else
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecLoadExodusCell"
/*@

  VecLoadExodusCell - Read a Vec representing the values of a field at all cells from an exodusII file.

  Input Parameters:
+ dm   - the DMMesh representing the mesh
. v    - the LOCAL vector of values to be read (i.e. with ghost values) obtained with SectionRealCreateLocalVector.
         if v represents a field with several components, the block size must be set accordingly using VecSetBlockSize().
. comm - the communicator associated to the exo file
  + if size(comm) == 1 each processor writes its local vector into a separate file
  . if size(comm) > 1, the values are sent to cpu 0, and written in a single file
- exoid - the id of the exodusII file (obtained with ex_open or ex_create)
. step  - the time step to write
. exofield - the position in the exodus field of the first component

- Interpolated meshes are not supported.

  Level: beginner

.keywords: mesh,ExodusII
.seealso: MeshCreate() MeshCreateExodus() SectionRealCreateLocalVector() VecViewExodusCell()
          VecViewExodusVertexSet() VecLoadExodusVertexSet() VecViewExodusCellSet() VecLoadExodusCellSet()
          VecViewExodusVertex() VecLoadExodusVertex()

@*/
PetscErrorCode VecLoadExodusCell(DM dm,Vec v,MPI_Comm comm,PetscInt exoid,PetscInt step,PetscInt exofield)
{
#ifdef PETSC_HAVE_EXODUSII
  PetscErrorCode          ierr;
  PetscInt                rank,num_proc,num_dof,num_cells,num_cells_in_set,num_cells_zero=0;
  PetscInt                num_cs_zero,num_cs,set,c,istart;
  PetscInt               *setsID_zero;
  const PetscInt         *setsID;
  IS                      setIS,csIS,setIS_zero;
  VecScatter              setscatter,setscattertozero;
  Vec                     vdof,vdof_set,vdof_set_zero;
  PetscReal              *vdof_set_array;
  PetscInt                junk1,junk2,junk3,junk4,junk5;
  char                    junk[MAX_LINE_LENGTH+1];
  MPI_Comm                dmcomm;
#endif

#ifdef PETSC_HAVE_EXODUSII
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&num_proc);
  ierr = MPI_Comm_rank(comm,&rank);
  ierr = VecGetBlockSize(v,&num_dof);
  ierr = PetscObjectGetComm((PetscObject) dm,&dmcomm);CHKERRQ(ierr);
  ierr = DMMeshGetLabelSize(dm,"Cell Sets",&num_cs);CHKERRQ(ierr);
  ierr = DMMeshGetStratumSize(dm,"height",0,&num_cells);CHKERRQ(ierr);
  
  ierr = VecCreateSeq(PETSC_COMM_SELF,num_cells,&vdof);CHKERRQ(ierr);
  
  if (num_proc == 1) {
    ierr = DMMeshGetLabelSize(dm,"Cell Sets",&num_cs);CHKERRQ(ierr);
    ierr = DMMeshGetLabelIdIS(dm,"Cell Sets",&csIS);CHKERRQ(ierr);
    ierr = ISGetIndices(csIS,&setsID);CHKERRQ(ierr);

    for (c = 0; c < num_dof; c++) {        
      for (set = 0; set < num_cs; set++) {
        /*
          Get the IS for the current set, the Vec containing a single dof and create
          the scatter for the restriction to the current cell set
        */
        ierr = DMMeshGetStratumIS(dm,"Cell Sets",setsID[set],&setIS);CHKERRQ(ierr);
        ierr = DMMeshGetStratumSize(dm,"Cell Sets",setsID[set],&num_cells_in_set);CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF,num_cells_in_set,&vdof_set);CHKERRQ(ierr);
        ierr = VecScatterCreate(vdof,setIS,vdof_set,PETSC_NULL,&setscatter);CHKERRQ(ierr);
        /*
          Read array from disk
        */
        ierr = VecGetArray(vdof_set,&vdof_set_array);CHKERRQ(ierr);
        ierr = ex_get_elem_var(exoid,step,exofield+c,setsID[set],num_cells_in_set,vdof_set_array);
        ierr = VecRestoreArray(vdof_set,&vdof_set_array);CHKERRQ(ierr);
        /*
          Copy values to full dof vec
        */
        ierr = VecScatterBegin(setscatter,vdof_set,vdof,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(setscatter,vdof_set,vdof,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

      }
      ierr = VecStrideScatter(vdof,c,v,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecDestroy(&vdof_set);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&setscatter);CHKERRQ(ierr);
      ierr = ISDestroy(&setIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(csIS,&setsID);CHKERRQ(ierr);
    ierr = ISDestroy(&csIS);CHKERRQ(ierr);
  } else {
    /*
      Get the number of blocks and the list of ID directly from  the file. This is easier
      than trying to reconstruct the global lists from all individual IS
    */
    if (rank == 0) {
      ierr = ex_get_init(exoid,junk,&junk1,&junk2,&junk3,&num_cs_zero,&junk4,&junk5);
    }
    ierr = MPI_Bcast(&num_cs_zero,1,MPIU_INT,0,dmcomm);
    ierr = PetscMalloc(num_cs_zero * sizeof(PetscInt),&setsID_zero);CHKERRQ(ierr);
    if (rank == 0) {
      ierr = ex_get_elem_blk_ids(exoid,setsID_zero);
    }
    ierr = MPI_Bcast(setsID_zero,num_cs_zero,MPIU_INT,0,dmcomm);
    
    for (c = 0; c < num_dof; c++) {
      istart = 0;    
      for (set = 0; set < num_cs_zero; set++) {
        /*
          Get the size of the size of the set on cpu 0 and create a Vec to receive values
        */
        ierr = ex_get_elem_block(exoid,setsID_zero[set],junk,&num_cells_zero,&junk1,&junk2);
        ierr = VecCreateSeq(PETSC_COMM_SELF,num_cells_zero,&vdof_set_zero);CHKERRQ(ierr);
  
        /*
          Get the IS for the current set, the Vec containing a single dof and create
          the scatter for the restriction to the current cell set
        */
        ierr = DMMeshGetStratumIS(dm,"Cell Sets",setsID_zero[set],&setIS);CHKERRQ(ierr);
        ierr = DMMeshGetStratumSize(dm,"Cell Sets",setsID_zero[set],&num_cells_in_set);CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF,num_cells_in_set,&vdof_set);CHKERRQ(ierr);
        ierr = VecScatterCreate(vdof,setIS,vdof_set,PETSC_NULL,&setscatter);CHKERRQ(ierr);
        /*
          Get the scatter to send the values of a single dof at a single block to cpu 0
        */
        ierr = ISCreateStride(dmcomm,num_cells_zero,istart,1,&setIS_zero);
        ierr = DMMeshCreateScatterToZeroCellSet(dm,setIS,setIS_zero,&setscattertozero);
        ierr = ISDestroy(&setIS_zero);CHKERRQ(ierr);

        /*
          Read array from disk
        */
        if (rank == 0) {
          ierr = VecGetArray(vdof_set_zero,&vdof_set_array);CHKERRQ(ierr);
          ierr = ex_get_elem_var(exoid,step,exofield+c,setsID_zero[set],num_cells_zero,vdof_set_array);
          ierr = VecRestoreArray(vdof_set_zero,&vdof_set_array);CHKERRQ(ierr);
        }
        /*
          Send values back to their owning cpu
        */
        ierr = VecScatterBegin(setscattertozero,vdof_set_zero,vdof_set,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(setscattertozero,vdof_set_zero,vdof_set,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        /*
          Copy the values associated to the current set back into the component vec
        */
        ierr = VecScatterBegin(setscatter,vdof_set,vdof,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(setscatter,vdof_set,vdof,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        istart += num_cells_zero;
        ierr = MPI_Bcast(&istart,1,MPIU_INT,0,dmcomm);
      }
      /*
        Copy the component back into v
      */
      ierr = VecStrideScatter(vdof,c,v,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecDestroy(&vdof_set_zero);CHKERRQ(ierr);
      ierr = VecDestroy(&vdof_set);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&setscatter);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&setscattertozero);CHKERRQ(ierr);
      ierr = ISDestroy(&setIS);CHKERRQ(ierr);
    }
    ierr = PetscFree(setsID_zero);CHKERRQ(ierr);
    ierr = VecDestroy(&vdof);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&vdof);CHKERRQ(ierr);
#else
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViewExodusCellSet"
/*@

  VecViewExodusCellSet - Write a Vec representing the values of a field at a cell set to an exodusII file.

  Input Parameters:
+ dm   - the DMMesh representing the mesh
. v    - the LOCAL vector of values to be saved (i.e. with ghost values) obtained with SectionRealCreateLocalVector.
         if v represents a field with several components, the block size must be set accordingly using VecSetBlockSize().
. csID - the ID of the cell set
. comm - the communicator associated to the exo file
  + if size(comm) == 1 each processor writes its local vector into a separate file
  . if size(comm) > 1, the values are sent to cpu 0, and written in a single file
- exoid - the id of the exodusII file (obtained with ex_open or ex_create)
. step  - the time step to write
. exofield - the position in the exodus field of the first component

- Interpolated meshes are not supported.

  Level: beginner

.keywords: mesh,ExodusII
.seealso: MeshCreate() MeshCreateExodus() SectionRealCreateLocalVector() VecLoadExodusCellSet()
          VecViewExodusVertexSet() VecLoadExodusVertexSet() VecViewExodusCell() VecLoadExodusCell()
          VecViewExodusVertex() VecLoadExodusVertex()

@*/
PetscErrorCode VecViewExodusCellSet(DM dm,Vec v,PetscInt csID,MPI_Comm comm,PetscInt exoid,PetscInt step,PetscInt exofield)
{
#ifdef PETSC_HAVE_EXODUSII
  PetscErrorCode          ierr;
  PetscInt                rank,num_proc,num_dof,num_cells,num_cells_zero=0;
  PetscInt                i,c;
  PetscReal              *vdof_array;
  IS                      csIS,csIS_zero;
  Vec                     vdof,vdof_zero;
  VecScatter              scatter;
  PetscInt                istart = 0;

  PetscInt                junk1,junk2,junk3,junk4,junk5;
  char                    junk[MAX_LINE_LENGTH+1];
  PetscInt                num_cs_global;
  PetscInt               *csIDs;
  MPI_Comm                dmcomm;
#endif

#ifdef PETSC_HAVE_EXODUSII
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&num_proc);
  ierr = MPI_Comm_rank(comm,&rank);
  ierr = VecGetBlockSize(v,&num_dof);
  ierr = PetscObjectGetComm((PetscObject) dm,&dmcomm);CHKERRQ(ierr);
  ierr = DMMeshGetStratumIS(dm,"Cell Sets",csID,&csIS);CHKERRQ(ierr);
  ierr = ISGetSize(csIS,&num_cells);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,num_cells,&vdof);CHKERRQ(ierr);
  
  if (num_proc == 1) {
    for (c = 0; c < num_dof; c++) {
      ierr = VecStrideGather(v,c,vdof,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecGetArray(vdof,&vdof_array);CHKERRQ(ierr);
      ierr = ex_put_elem_var(exoid,step,exofield+c,csID,num_cells,vdof_array); 
      ierr = VecRestoreArray(vdof,&vdof_array);CHKERRQ(ierr);
      if (ierr != 0) {
        SETERRQ2(comm,PETSC_ERR_FILE_WRITE,"Unable to write to file id %i. ex_put_elem_var returned %i",exoid,ierr);
      }
    }
  } else {
    /*
      Get the global size of the cell set from the file because we can
      There is no direct way to get the index of the first cell in a cell set from exo.
      (which depends on the order on the file) so we need to seek the cell set in the file and 
      accumulate offsets... 
    */
    if (rank == 0) {
      istart = 0;
      ierr = ex_get_init(exoid,junk,&junk1,&junk2,&junk3,&num_cs_global,&junk4,&junk5);
      ierr = PetscMalloc(num_cs_global * sizeof(PetscInt),&csIDs);CHKERRQ(ierr);
      ierr = ex_get_elem_blk_ids(exoid,csIDs);
      for (i = 0; i < num_cs_global; i++) {
        ierr = ex_get_elem_block(exoid,csIDs[i],junk,&num_cells_zero,&junk1,&junk2);
        if (csIDs[i] == csID) {
          break;
        } else {
          istart += num_cells_zero;
        } 
      }
      ierr = PetscFree(csIDs);CHKERRQ(ierr);
      ierr = ex_get_elem_block(exoid,csID,junk,&num_cells_zero,&junk1,&junk2);
    }
    ierr = ISCreateStride(dmcomm,num_cells_zero,istart,1,&csIS_zero);
    ierr = DMMeshCreateScatterToZeroCellSet(dm,csIS,csIS_zero,&scatter);
    ierr = ISDestroy(&csIS_zero);CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF,num_cells_zero,&vdof_zero);CHKERRQ(ierr);

    for (c = 0; c < num_dof; c++) {
      ierr = VecStrideGather(v,c,vdof,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecScatterBegin(scatter,vdof,vdof_zero,INSERT_VALUES,SCATTER_FORWARD);
      ierr = VecScatterEnd(scatter,vdof,vdof_zero,INSERT_VALUES,SCATTER_FORWARD);
      if (rank == 0) {
        ierr = VecGetArray(vdof_zero,&vdof_array);CHKERRQ(ierr);
        ierr = ex_put_elem_var(exoid,step,exofield+c,csID,num_cells_zero,vdof_array);
        ierr = VecRestoreArray(vdof_zero,&vdof_array);CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy(&vdof_zero);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&csIS);CHKERRQ(ierr);
  ierr = VecDestroy(&vdof);CHKERRQ(ierr);
#else
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecLoadExodusCellSet"
/*@

  VecLoadExodusCellSet - Read a Vec representing the values of a field at a cell set from an exodusII file.

  Input Parameters:
+ dm   - the DMMesh representing the mesh
. v    - the LOCAL vector of values to be saved (i.e. with ghost values) obtained with SectionRealCreateLocalVector.
         if v represents a field with several components, the block size must be set accordingly using VecSetBlockSize().
. csID - the ID of the cell set
. comm - the communicator associated to the exo file
  + if size(comm) == 1 each processor writes its local vector into a separate file
  . if size(comm) > 1, the values are sent to cpu 0, and written in a single file
- exoid - the id of the exodusII file (obtained with ex_open or ex_create)
. step  - the time step to write
. exofield - the position in the exodus field of the first component

- Interpolated meshes are not supported.

  Level: beginner

.keywords: mesh,ExodusII
.seealso: MeshCreate() MeshCreateExodus() SectionRealCreateLocalVector() VecViewExodusCellSet()
          VecViewExodusVertexSet() VecLoadExodusVertexSet() VecViewExodusCell() VecLoadExodusCell()
          VecViewExodusVertex() VecLoadExodusVertex()

@*/
PetscErrorCode VecLoadExodusCellSet(DM dm,Vec v,PetscInt csID,MPI_Comm comm,PetscInt exoid,PetscInt step,PetscInt exofield)
{
#ifdef PETSC_HAVE_EXODUSII
  PetscErrorCode          ierr;
  PetscInt                rank,num_proc,num_dof,num_cells,num_cells_zero=0;
  PetscInt                i,c;
  PetscReal              *vdof_array;
  IS                      csIS,csIS_zero;
  Vec                     vdof,vdof_zero;
  VecScatter              scatter;
  PetscInt                istart = 0;

  PetscInt                junk1,junk2,junk3,junk4,junk5;
  PetscInt                num_cs_global;
  PetscInt               *csIDs;
  char                    junk[MAX_LINE_LENGTH+1];
  MPI_Comm                dmcomm;
#endif

#ifdef PETSC_HAVE_EXODUSII
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&num_proc);
  ierr = MPI_Comm_rank(comm,&rank);
  ierr = VecGetBlockSize(v,&num_dof);
  ierr = PetscObjectGetComm((PetscObject) dm,&dmcomm);CHKERRQ(ierr);
  
  ierr = DMMeshGetStratumIS(dm,"Cell Sets",csID,&csIS);CHKERRQ(ierr);
  ierr = ISGetSize(csIS,&num_cells);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,num_cells,&vdof);CHKERRQ(ierr);

  if (num_proc == 1) {
    for (c = 0; c < num_dof; c++) {
      ierr = VecGetArray(vdof,&vdof_array);CHKERRQ(ierr);
      ierr = ex_get_elem_var(exoid,step,exofield+c,csID,num_cells,vdof_array); 
      ierr = VecRestoreArray(vdof,&vdof_array);CHKERRQ(ierr);
      if (ierr != 0) {
        SETERRQ2(comm,PETSC_ERR_FILE_WRITE,"Unable to write to file id %i. ex_put_elem_var returned %i",exoid,ierr);
      }
      ierr = VecStrideScatter(vdof,c,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  } else {
    /*
      Get the global size of the cell set from the file because we can
      There is no direct way to get the index of the first cell in a cell set from exo.
      (which depends on the order on the file) so we need to seek the cell set in the file and 
      accumulate offsets... 
    */
    if (rank == 0) {
      istart = 0;
      ierr = ex_get_init(exoid,junk,&junk1,&junk2,&junk3,&num_cs_global,&junk4,&junk5);
      ierr = PetscMalloc(num_cs_global * sizeof(PetscInt),&csIDs);CHKERRQ(ierr);
      ierr = ex_get_elem_blk_ids(exoid,csIDs);
      for (i = 0; i < num_cs_global; i++) {
        ierr = ex_get_elem_block(exoid,csIDs[i],junk,&num_cells_zero,&junk1,&junk2);
        if (csIDs[i] == csID) {
          break;
        } else {
          istart += num_cells_zero;
        } 
      }
      ierr = PetscFree(csIDs);CHKERRQ(ierr);
      ierr = ex_get_elem_block(exoid,csID,junk,&num_cells_zero,&junk1,&junk2);
    }
    ierr = ISCreateStride(dmcomm,num_cells_zero,istart,1,&csIS_zero);
    ierr = DMMeshCreateScatterToZeroCellSet(dm,csIS,csIS_zero,&scatter);
    ierr = ISDestroy(&csIS_zero);CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF,num_cells_zero,&vdof_zero);CHKERRQ(ierr);
    
    for (c = 0; c < num_dof; c++) {
      if (rank == 0) {
        ierr = VecGetArray(vdof_zero,&vdof_array);CHKERRQ(ierr);
        ierr = ex_get_elem_var(exoid,step,exofield+c,csID,num_cells_zero,vdof_array);
        ierr = VecRestoreArray(vdof_zero,&vdof_array);CHKERRQ(ierr);
      }
      ierr = VecScatterBegin(scatter,vdof_zero,vdof,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecScatterEnd(scatter,vdof_zero,vdof,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecStrideScatter(vdof,c,v,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&vdof_zero);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&csIS);CHKERRQ(ierr);
  ierr = VecDestroy(&vdof);CHKERRQ(ierr);
#else
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  PetscFunctionReturn(0);
}
