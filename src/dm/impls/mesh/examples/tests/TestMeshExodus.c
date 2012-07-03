static char help[] = "Test a new generation ExodusII reader\n\n";

#include <petscsys.h>
#include <petscdmmesh.hh>
#include <sieve/Selection.hh>
#include <exodusII.h>


const char exo_KnownElements[] = "tri,tri3,triangle,triangle3,quad,quad4,tet,tet4,tetra,tetra4,hex,hex8";
/*
  The exodusII numbering scheme for faces
*/
const PetscInt exo_trifaces[3][2]  = {{0,1}, {1,2}, {2,0}};
const PetscInt exo_quadfaces[4][2] = {{0,1}, {1,2}, {2,3}, {3,0}};
const PetscInt exo_tetfaces[4][3]  = {{0,1,3}, {1,2,3}, {2,0,3}, {0,1,2}};
const PetscInt exo_hexfaces[6][4]  = {{0,1,5,4}, {1,2,6,5}, {2,3,7,6}, {0,4,7,3}, {0,3,2,1}, {4,5,6,7}};

typedef enum {
  EXO_TRI,
  EXO_QUAD,
  EXO_TET,
  EXO_HEX
} EXO_ELEM_TYPE;

PetscErrorCode EXOGetElemType(const char *name,EXO_ELEM_TYPE *elem_type);
PetscErrorCode MyDMMeshCreateExodus(MPI_Comm comm,const char filename[],DM *dmBody,DM *dmFS);
PetscErrorCode MyPetscReadExodusII(MPI_Comm comm,const char filename[],DM dmBody,DM dmFS);

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char ** argv) {
  DM             dmBody,dmFS;
  DM             dmBodyDist,dmFSDist;
  ALE::Obj<PETSC_MESH_TYPE> meshBody,meshBodyDist,meshFS,meshFSDist;
  PetscBool      inflag,outflag;
  char           infilename[PETSC_MAX_PATH_LEN+1],outfilename[PETSC_MAX_PATH_LEN+1];
  PetscViewer    viewer;
  int            rank,numproc;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-i",infilename,PETSC_MAX_PATH_LEN,&inflag);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&numproc);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (inflag) {
    ierr = MyDMMeshCreateExodus(PETSC_COMM_WORLD,infilename,&dmBody,&dmFS);CHKERRQ(ierr);
    ierr = DMMeshGetMesh(dmBody,meshBody);CHKERRQ(ierr);
    meshBody->view("meshBody");

    ierr = DMMeshGetMesh(dmFS,meshFS);CHKERRQ(ierr);
    meshFS->view("meshFS");

    if (numproc > 1) {
      ierr = DMMeshDistribute(dmBody,PETSC_NULL,&dmBodyDist);CHKERRQ(ierr);
      //ierr = DMMeshDistribute(dmFS,PETSC_NULL,&dmFSDist);CHKERRQ(ierr);
      ierr = DMMeshGetMesh(dmBodyDist,meshBodyDist);CHKERRQ(ierr);
      meshBodyDist->view("meshBodyDist");
      //ierr = DMMeshGetMesh(dmFSDist,meshFSDist);CHKERRQ(ierr);
      //meshFSDist->view("meshFS");

      //ierr = DMDestroy(&dmBodyDist);CHKERRQ(ierr);
      //ierr = DMDestroy(&dmFSDist);CHKERRQ(ierr);
    }

    //ierr = DMDestroy(&dmBody);CHKERRQ(ierr);
    //ierr = DMDestroy(&dmFS);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"No file name given\n");CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "MyDMMeshCreateExodus"
/*@C
  MyDMMeshCreateExodus - Create a Mesh from an ExodusII file.

  Not Collective

  Input Parameters:
+ comm - The MPI communicator
- filename - The ExodusII filename

  Output Parameter:
. dmBody  - The DM object representing the body
. dmFS - The DM object representing the face sets

  Level: beginner

.keywords: mesh,ExodusII
.seealso: MeshCreate()
@*/
PetscErrorCode MyDMMeshCreateExodus(MPI_Comm comm,const char filename[],DM *dmBody,DM *dmFS)
{
  PetscInt       debug = 0;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-debug",&debug,&flag);CHKERRQ(ierr);

  ierr = DMMeshCreate(comm,dmBody);CHKERRQ(ierr);
  ALE::Obj<PETSC_MESH_TYPE> meshBody = new PETSC_MESH_TYPE(comm,-1,debug);
  ierr = DMMeshSetMesh(*dmBody,meshBody);CHKERRQ(ierr);
  ierr = DMMeshCreate(comm,dmFS);CHKERRQ(ierr);
  ALE::Obj<PETSC_MESH_TYPE> meshFS = new PETSC_MESH_TYPE(comm,-1,debug);
  ierr = DMMeshSetMesh(*dmFS,meshFS);CHKERRQ(ierr);
#ifdef PETSC_HAVE_EXODUSII
  try {
    ierr = MyPetscReadExodusII(comm,filename,*dmBody,*dmFS);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cerr << "Error: " << e << std::endl;
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This method requires ExodusII support. Reconfigure using --with-exodusii-dir=/path/to/exodus");
#endif
  //if (debug) {mesh->view("Mesh");}
  //ierr = DMMeshSetMesh(*dm,mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MyPetscReadExodusII"
PetscErrorCode MyPetscReadExodusII(MPI_Comm comm,const char filename[],DM dmBody,DM dmFS)
{
  ALE::Obj<PETSC_MESH_TYPE>               meshBody,meshFS;
  typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
  //typedef std::set<FlexMesh::point_type>  PointSet;
  //ALE::Obj<FlexMesh>  boundarymesh;
  PetscMPIInt         rank;
  int                 CPU_word_size = 0;
  int                 IO_word_size  = 0;
  PetscBool           interpolate   = PETSC_FALSE;
  int               **connect = PETSC_NULL;
  int                 exoid;
  char                title[MAX_LINE_LENGTH+1];
  float               version;
  int                 num_dim,num_nodes = 0,num_elem = 0;
  int                 num_eb = 0,num_ns = 0,num_ss = 0;
  PetscErrorCode      ierr;
  //const char          known_elements[] = "tri,tri3,triangle,triangle3,quad,quad4,tet,tet4,tetra,tetra4,hex,hex8";
  float              *x = PETSC_NULL,*y = PETSC_NULL,*z = PETSC_NULL;
  PetscBool           debug = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-interpolate",&interpolate,PETSC_NULL);CHKERRQ(ierr);

  /*
    Get the sieve meshes from the dms
  */
  ierr = DMMeshGetMesh(dmBody,meshBody);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dmFS,meshFS);CHKERRQ(ierr);
  rank = meshBody->commRank();

  /*
    Open EXODUS II file and read basic informations on rank 0,
    then broadcast to all nodes
  */
  if (rank == 0) {
    exoid = ex_open(filename,EX_READ,&CPU_word_size,&IO_word_size,&version);CHKERRQ(!exoid);
    ierr = ex_get_init(exoid,title,&num_dim,&num_nodes,&num_elem,&num_eb,&num_ns,&num_ss);CHKERRQ(ierr);
    if (num_eb == 0) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exodus file does not contain any element block\n");
    }
    ierr = PetscMalloc3(num_nodes,float,&x,num_nodes,float,&y,num_nodes,float,&z);CHKERRQ(ierr);
    ierr = ex_get_coord(exoid,x,y,z);CHKERRQ(ierr);
  }

  ierr = MPI_Bcast(&num_dim,1,MPI_INT,0,comm);
  //ierr = MPI_Bcast(&num_eb,1,MPI_INT,0,comm);
  //ierr = MPI_Bcast(&num_ns,1,MPI_INT,0,comm);
  //ierr = MPI_Bcast(&num_ss,1,MPI_INT,0,comm);

  meshBody->setDimension(num_dim);
  meshFS->setDimension(num_dim-1);
  /*
    Read element connectivity
  */
  int      *eb_ids = PETSC_NULL,*num_elem_in_block = PETSC_NULL,*num_nodes_per_elem = PETSC_NULL,*num_attr = PETSC_NULL;
  char    **eb_name = PETSC_NULL,**eb_elemtype = PETSC_NULL;
  PetscBool eb_hasnoname;
  char     *elem_sig = PETSC_NULL;

  ierr = PetscMalloc6(num_eb,int,&eb_ids,
                      num_eb,int,&num_elem_in_block,
                      num_eb,int,&num_nodes_per_elem,
                      num_eb,int,&num_attr,
                      num_eb,char*,&eb_name,
                      num_eb,char*,&eb_elemtype);CHKERRQ(ierr);

  for (int eb = 0; eb < num_eb; eb++) {
    num_elem_in_block[eb] = 0;
    num_nodes_per_elem[eb] = 0;
    num_attr[eb] = 0;
    ierr = PetscMalloc2(MAX_STR_LENGTH+1,char,&eb_name[eb],MAX_STR_LENGTH+1,char,&eb_elemtype[eb]);CHKERRQ(ierr);
  }
  if (rank == 0) {
    /*
      Get EB names
    */
    ierr = ex_get_names(exoid,EX_ELEM_BLOCK,eb_name);CHKERRQ(ierr);
    for (int eb = 0; eb < num_eb; ++eb) {
      ierr = PetscStrcmp(eb_name[eb],"",&eb_hasnoname);CHKERRQ(ierr);
      if (eb_hasnoname) {
        ierr = PetscSNPrintf(eb_name[eb],MAX_STR_LENGTH,"CellBlock_%.4i",eb);CHKERRQ(ierr);
      }
    }
    ierr = ex_get_elem_blk_ids(exoid,eb_ids);CHKERRQ(ierr);

    /*
      Check that the element type in each block is known
    */
    for (int eb = 0; eb < num_eb; ++eb) {
      ierr = ex_get_elem_block(exoid,eb_ids[eb],eb_elemtype[eb],&num_elem_in_block[eb],&num_nodes_per_elem[eb],&num_attr[eb]);CHKERRQ(ierr);
      ierr = PetscStrtolower(eb_elemtype[eb]);CHKERRQ(ierr);
      ierr = PetscStrstr(exo_KnownElements,eb_elemtype[eb],&elem_sig);CHKERRQ(ierr);
      if (!elem_sig) {
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported element type: %s.\nSupported elements types are %s",eb_elemtype[eb],exo_KnownElements);
      }
    }

    /*
      Read Connectivity tables
    */
    ierr = PetscMalloc(num_eb * sizeof(int*),&connect);CHKERRQ(ierr);
    for (int eb = 0; eb < num_eb; ++eb) {
      ierr = PetscMalloc(num_nodes_per_elem[eb]*num_elem_in_block[eb] * sizeof(int),&connect[eb]);CHKERRQ(ierr);
      ierr = ex_get_elem_conn(exoid,eb_ids[eb],connect[eb]);CHKERRQ(ierr);
    }
  }
#if 0 // We do no currently make sections for the element blocks, and num_eb is not broadcast
  /* Broadcast element block names. This will be needed later when creating the matching sections */
  for(int eb = 0; eb < num_eb; ++eb) {
    ierr = MPI_Bcast(eb_name[eb],MAX_STR_LENGTH+1,MPI_CHAR,0,comm);
  }
#endif

  /*
    Read side sets
  */
  int      *ss_ids = PETSC_NULL,*num_sides_in_set = PETSC_NULL;
  int     **side_set_elem_list = PETSC_NULL,**side_set_side_list = PETSC_NULL;
  char    **ss_name = PETSC_NULL;
  PetscBool ss_hasnoname;
  int       num_df_in_sset;
  int       num_faces = 0;

  ierr = PetscMalloc5(num_ss,int,&ss_ids,
                      num_ss,int,&num_sides_in_set,
                      num_ss,int*,&side_set_elem_list,
                      num_ss,int*,&side_set_side_list,
                      num_ss,char*,&ss_name);CHKERRQ(ierr);
  for(int ss = 0; ss < num_ss; ++ss) {
    num_sides_in_set[ss] = 0;
    ierr = PetscMalloc((MAX_STR_LENGTH+1)*sizeof(char),&ss_name[ss]);CHKERRQ(ierr);
  }
  if (num_ss > 0) {
    ierr = ex_get_side_set_ids(exoid,ss_ids);CHKERRQ(ierr);
    ierr = ex_get_names(exoid,EX_SIDE_SET,ss_name);CHKERRQ(ierr);
  }
  for (int ss = 0; ss< num_ss; ++ss) {
    ierr = ex_get_side_set_param(exoid,ss_ids[ss],&num_sides_in_set[ss],&num_df_in_sset);
    ierr = PetscMalloc2(num_sides_in_set[ss],int,&side_set_elem_list[ss],
                        num_sides_in_set[ss],int,&side_set_side_list[ss]);CHKERRQ(ierr);
    ierr = ex_get_side_set(exoid,ss_ids[ss],side_set_elem_list[ss],side_set_side_list[ss]);
    ierr = PetscStrcmp(ss_name[ss],"",&ss_hasnoname);CHKERRQ(ierr);
    if (ss_hasnoname) {
      ierr = PetscSNPrintf(ss_name[ss],MAX_STR_LENGTH,"FaceSet_%.4i",ss_ids[ss]);CHKERRQ(ierr);
    }
    /*
     num_faces is an upper bound on the total number of faces unless
     the face sets are disjoints
    */
    num_faces += num_sides_in_set[ss];
  }
#if 0 // We do no currently make sections for the side sets, and num_ss is not broadcast
  /* Broadcast side sets names. This will be needed later when creating the matching sections */
  for(int ss = 0; ss < num_ss; ++ss) {
    ierr = MPI_Bcast(ss_name[ss],MAX_STR_LENGTH+1,MPI_CHAR,0,comm);
  }
#endif

  /*
    Read node sets
  */
  int  *ns_ids = PETSC_NULL,*num_nodes_in_set = PETSC_NULL;
  int **node_list = PETSC_NULL;
  char **ns_name = PETSC_NULL;
  PetscBool ns_hasnoname;
  ierr = PetscMalloc4(num_ns,int,&ns_ids,
                      num_ns,int,&num_nodes_in_set,
                      num_ns,int*,&node_list,
                      num_ns,char*,&ns_name);CHKERRQ(ierr);
  for(int ns = 0; ns < num_ns; ++ns) {
    num_nodes_in_set[ns] = 0;
    ierr = PetscMalloc((MAX_STR_LENGTH+1)*sizeof(char),&ns_name[ns]);CHKERRQ(ierr);
  }
  if (num_ns > 0) {
    ierr = ex_get_node_set_ids(exoid,ns_ids);CHKERRQ(ierr);
    ierr = ex_get_names(exoid,EX_NODE_SET,ns_name);CHKERRQ(ierr);
  }
  for (int ns = 0; ns < num_ns; ++ns) {
    int num_df_in_set;
    ierr = ex_get_node_set_param(exoid,ns_ids[ns],&num_nodes_in_set[ns],&num_df_in_set);CHKERRQ(ierr);
    ierr = PetscMalloc(num_nodes_in_set[ns] * sizeof(int),&node_list[ns]);CHKERRQ(ierr);
    ierr = ex_get_node_set(exoid,ns_ids[ns],node_list[ns]);
    ierr = PetscStrcmp(ns_name[ns],"",&ns_hasnoname);CHKERRQ(ierr);
    if (ns_hasnoname) {
      ierr = PetscSNPrintf(ns_name[ns],MAX_STR_LENGTH,"VertexSet_%.4i",ns_ids[ns]);CHKERRQ(ierr);
    }
  }
#if 0 // We do no currently make sections for the node sets, and num_ns is not broadcast
  /* Broadcast node sets names. This will be needed later when creating the matching sections */
  for(int ns = 0; ns < num_ns; ++ns) {
    ierr = MPI_Bcast(ns_name[ns],MAX_STR_LENGTH+1,MPI_CHAR,0,comm);
  }
#endif
  /* Done reading EXO,closing file */
  if (rank == 0) {
    ierr = ex_close(exoid);CHKERRQ(ierr);
  }

  /*
    Build mesh topology
  */
  int  *cells = PETSC_NULL;
  int **connectivity_table = PETSC_NULL;
  int   num_local_corners = 0;

  for(int eb = 0; eb < num_eb; ++eb) {
    num_local_corners += num_nodes_per_elem[eb] * num_elem_in_block[eb];
  }
  ierr = PetscMalloc2(num_local_corners,int,&cells,num_elem,int*,&connectivity_table);CHKERRQ(ierr);
  for (int eb = 0,k = 0; eb < num_eb; ++eb) {
    for (int e = 0; e < num_elem_in_block[eb]; ++e,++k) {
      for (int c = 0; c < num_nodes_per_elem[eb]; ++c) {
        cells[k*num_nodes_per_elem[eb]+c] = connect[eb][e*num_nodes_per_elem[eb]+c] - 1;
      }
      connectivity_table[k] = &cells[k*num_nodes_per_elem[eb]];
    }
    ierr = PetscFree(connect[eb]);CHKERRQ(ierr);
  }
  ierr = PetscFree(connect);CHKERRQ(ierr);

  ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(meshBody->comm(),meshBody->debug());
  ALE::Obj<FlexMesh>                    m     = new FlexMesh(meshBody->comm(),meshBody->debug());
  ALE::Obj<FlexMesh::sieve_type>        s     = new FlexMesh::sieve_type(meshBody->comm(),meshBody->debug());

  /*
    BUG!
    Here we assume that num_nodes_per_elem is constant accross blocks (i.e.)
    that all elements in the mesh are of the same type.
  */
  int numCorners = 0;
  if (rank == 0) {
    numCorners = num_nodes_per_elem[0];
  }

  ALE::SieveBuilder<FlexMesh>::buildTopology(s,num_dim,num_elem,cells,num_nodes,interpolate,numCorners,PETSC_DECIDE,m->getArrowSection("orientation"));

  std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> renumbering;
  ALE::ISieveConverter::convertSieve(*s,*sieve,renumbering,false);
  meshBody->setSieve(sieve);
  meshBody->stratify();
  ALE::ISieveConverter::convertOrientation(*s,*sieve,renumbering,m->getArrowSection("orientation").ptr());

  /*
    Build coordinates
  */
  double *coords = PETSC_NULL;
  ierr = PetscMalloc(num_dim*num_nodes * sizeof(double), &coords);CHKERRQ(ierr);
  if (num_dim > 0) {for (int v = 0; v < num_nodes; ++v) {coords[v*num_dim+0] = x[v];}}
  if (num_dim > 1) {for (int v = 0; v < num_nodes; ++v) {coords[v*num_dim+1] = y[v];}}
  if (num_dim > 2) {for (int v = 0; v < num_nodes; ++v) {coords[v*num_dim+2] = z[v];}}
  ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(meshBody,num_dim,coords);
  ierr = PetscFree(coords);CHKERRQ(ierr);

  /*
    Create label for element blocks
  */
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& cellBlocks = meshBody->createLabel("CellBlocks");
  for (int eb = 0,k = 0; eb < num_eb; ++eb) {
    for (int e = 0; e < num_elem_in_block[eb]; ++e,++k) {
      meshBody->setValue(cellBlocks,k,eb_ids[eb]);
    }
  }
  if (debug) {cellBlocks->view("Cell Blocks");}

  /*
    Initialize parent block mapping needed to build face sets
  */
  PetscInt *cellParentBlock = PETSC_NULL;

  ierr = PetscMalloc(num_elem * sizeof(PetscInt),&cellParentBlock);CHKERRQ(ierr);
  for (int eb = 0,k = 0; eb < num_eb; ++eb) {
    for (int e = 0; e < num_elem_in_block[eb]; ++e,++k) {
      cellParentBlock[k] = eb;
    }
  }

  /*
    Create Vertex set label
  */
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& vertexSets = meshBody->createLabel("VertexSets");
  for (int ns = 0; ns < num_ns; ++ns) {
    for (int n = 0; n < num_nodes_in_set[ns]; ++n) {
      meshBody->addValue(vertexSets,node_list[ns][n]+num_elem-1,ns_ids[ns]);
    }
  }
  if (debug) {vertexSets->view("Vertex Sets");}

  for (int ns = 0; ns < num_ns; ns++) {
    ierr = PetscFree(node_list[ns]);CHKERRQ(ierr);
    ierr = PetscFree(ns_name[ns]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(ns_ids,num_nodes_in_set,node_list,ns_name);CHKERRQ(ierr);

  /*
    Build face sets mesh
  */
  const PetscInt    *faceVertex = PETSC_NULL;
  PetscInt          *faces = PETSC_NULL,faceNumVertex,faceCell,faceNum,faceParentBlock;
  PetscInt           faceCount = 0,vertexCount = 0;
  EXO_ELEM_TYPE      cell_type;

  ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieveFS = new PETSC_MESH_TYPE::sieve_type(meshFS->comm(), meshFS->debug());
  ALE::Obj<FlexMesh>                    mFS     = new FlexMesh(meshBody->comm(),meshBody->debug());
  ALE::Obj<FlexMesh::sieve_type>        sFS     = new FlexMesh::sieve_type(meshBody->comm(),meshBody->debug());

  for(int k = 0,ss = 0; ss < num_ss; ++ss) {
    /*
     The type of parent cells for all faces in a face set has to be identical.
     We trust that this is the case and do not do any test.
    */
    faceParentBlock  = cellParentBlock[side_set_elem_list[ss][0]-1];
    ierr = EXOGetElemType(eb_elemtype[faceParentBlock],&cell_type);CHKERRQ(ierr);
    switch (cell_type) {
    case EXO_TRI:
      faceNumVertex = 2;
      break;
    case EXO_TET:
      faceNumVertex = 3;
      break;
    case EXO_QUAD:
      faceNumVertex = 2;
      break;
    case EXO_HEX:
      faceNumVertex = 4;
      break;
    }
    vertexCount += faceNumVertex * num_sides_in_set[ss];
  }
  /*
   Get the number of points in the face set
  */
  ierr = PetscMalloc(vertexCount * sizeof(PetscInt),&faces);CHKERRQ(ierr);
  for(int k = 0, ss = 0; ss < num_ss; ++ss) {
    /*
     The type of parent cells for all faces in a face set has to be identical.
     We trust that this is the case and do not do any test.
    */
    faceParentBlock  = cellParentBlock[side_set_elem_list[ss][0]-1];
    ierr = EXOGetElemType(eb_elemtype[faceParentBlock],&cell_type);CHKERRQ(ierr);
    for(int s = 0; s < num_sides_in_set[ss]; ++s) {
      /*
       Get the point number of the faces described by vertices side_set_side_list[ss][s]
       initialize side_set_point_list from side_set_elem_list and side_set_side_list
      */
      faceCell  = side_set_elem_list[ss][s]-1;
      faceNum   = side_set_side_list[ss][s]-1;
      switch (cell_type) {
      case EXO_TRI:
        faceNumVertex = 2;
        faceVertex    = exo_trifaces[faceNum];
        break;
      case EXO_TET:
        faceNumVertex = 3;
        faceVertex    = exo_tetfaces[faceNum];
        break;
      case EXO_QUAD:
        faceNumVertex = 2;
        faceVertex    = exo_quadfaces[faceNum];
        break;
      case EXO_HEX:
        faceNumVertex = 4;
        faceVertex    = exo_hexfaces[faceNum];
        break;
      }
      /*
       BUG:
       This is not going to work if the mesh contains several element types, or for elements whose faces
       have different number of vertices (prisms, for instance)
      */
      numCorners = faceNumVertex;
      for(int v = 0; v < faceNumVertex; v++,k++) {
        faces[k] = connectivity_table[faceCell][faceVertex[v]] + num_elem;
      }
      faceCount++;
    }
  }
  /*
   buildTopology default behavior is to offset the vertices with num_elem (well, really faceCount here).
   Here, we already computed the proper points numbers for the vertices, so we need to tell it to use our numbering.
   */
  int firstVertex = 0;
  ALE::SieveBuilder<FlexMesh>::buildTopology(sFS,num_dim-1,faceCount,faces,num_nodes,interpolate,numCorners,firstVertex,mFS->getArrowSection("orientation"));
  ierr = PetscFree(faces);CHKERRQ(ierr);

  std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> renumberingFS;
  ALE::ISieveConverter::convertSieve(*sFS,*sieveFS,renumbering,false);
  meshFS->setSieve(sieveFS);
  meshFS->stratify();
  ALE::ISieveConverter::convertOrientation(*sFS,*sieveFS,renumbering,mFS->getArrowSection("orientation").ptr());

  /*
   Build coordinates for meshFS
  */
  ierr = PetscMalloc((num_dim*num_nodes+num_elem) * sizeof(double), &coords);CHKERRQ(ierr);
  if (num_dim > 0) {for (int v = 0; v < num_nodes; ++v) {coords[v*num_dim+0+num_elem] = x[v];}}
  if (num_dim > 1) {for (int v = 0; v < num_nodes; ++v) {coords[v*num_dim+1+num_elem] = y[v];}}
  if (num_dim > 2) {for (int v = 0; v < num_nodes; ++v) {coords[v*num_dim+2+num_elem] = z[v];}}
  ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(meshFS,num_dim,coords);
  ierr = PetscFree(coords);CHKERRQ(ierr);

  /*
   Create Face set label
  */
  const ALE::Obj<PETSC_MESH_TYPE::label_type>& faceSets = meshFS->createLabel("FaceSets");
  for (int k=0,ss = 0; ss < num_ss; ++ss) {
    faceParentBlock  = cellParentBlock[side_set_elem_list[ss][0]-1];
    ierr = EXOGetElemType(eb_elemtype[faceParentBlock],&cell_type);CHKERRQ(ierr);
    for (int s = 0; s < num_sides_in_set[ss]; ++s,++k) {
      meshFS->addValue(faceSets,k,ss_ids[ss]);
    }
  }
  if (debug) {faceSets->view("Face Sets");}

  /*
    Free remaining element block temporary variables
  */
  for (int eb = 0; eb < num_eb; ++eb) {
    ierr = PetscFree2(eb_name[eb],eb_elemtype[eb]);CHKERRQ(ierr);
  }
  ierr = PetscFree6(eb_ids,num_elem_in_block,num_nodes_per_elem,num_attr,eb_name,eb_elemtype);CHKERRQ(ierr);
  ierr = PetscFree2(cells,connectivity_table);CHKERRQ(ierr);
  ierr = PetscFree(cellParentBlock);CHKERRQ(ierr);
  ierr = PetscFree3(x,y,z);CHKERRQ(ierr);

  //if (debug) {meshBody->view("Mesh");}
  if (debug) {meshFS->view("MeshFS");}
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "EXOGetElemType"
PetscErrorCode EXOGetElemType(const char *name,EXO_ELEM_TYPE *elem_type){
  PetscErrorCode ierr;
  char     *elem_sig;
  ierr = PetscStrstr("tri,tri3,triangle,triangle3",name,&elem_sig);CHKERRQ(ierr);
  if (elem_sig){
    *elem_type = EXO_TRI;
  }
  ierr = PetscStrstr("tet,tet4,tetra,tetra4",name,&elem_sig);CHKERRQ(ierr);
  if (elem_sig){
    *elem_type = EXO_TET;
  }
  ierr = PetscStrstr("quad,quad4",name,&elem_sig);CHKERRQ(ierr);
  if (elem_sig){
    *elem_type = EXO_QUAD;
  }
  ierr = PetscStrstr("hex,hex8",name,&elem_sig);CHKERRQ(ierr);
  if (elem_sig){
    *elem_type = EXO_HEX;
  }

  PetscFunctionReturn(0);
}
