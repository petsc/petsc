/*
  To do, bugs, questions:
    - Do I need to destroy the sections?
    - l. 378, shall I really free cellBlock?
    - There must be a million leaks
    - Test parallel version
*/
    
static char help[] = "Test distribution of properties using a mesh\n\n";

#include <petscsys.h>
#include <petscdmmesh.hh>
#include <sieve/Selection.hh>
#include <exodusII.h>

PetscErrorCode MyDMMeshCreateExodus(MPI_Comm comm, const char filename[], DM *dm);
PetscErrorCode MyPetscReadExodusII(MPI_Comm comm, const char filename[], DM dm);

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc, char ** argv) {
  DM             dm;
  PetscBool      inflag,outflag;
  char           infilename[PETSC_MAX_PATH_LEN+1],outfilename[PETSC_MAX_PATH_LEN+1];
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL, "-i", infilename, PETSC_MAX_PATH_LEN, &inflag);CHKERRQ(ierr);
  if (inflag) {
    ierr = MyDMMeshCreateExodus(PETSC_COMM_WORLD, infilename, &dm);CHKERRQ(ierr);
    //ierr = DMMeshCreateExodus(PETSC_COMM_WORLD, infilename, &dm);CHKERRQ(ierr);
    ierr = DMView(dm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetString(PETSC_NULL, "-o", outfilename, PETSC_MAX_PATH_LEN, &outflag);CHKERRQ(ierr);
  if (outflag) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outfilename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = DMView(dm,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dm);CHKERRQ(ierr);  
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
. dm - The DM object

  Level: beginner

.keywords: mesh, ExodusII
.seealso: MeshCreate()
@*/
PetscErrorCode MyDMMeshCreateExodus(MPI_Comm comm, const char filename[], DM *dm)
{
  PetscInt       debug = 1;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshCreate(comm, dm);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  ALE::Obj<PETSC_MESH_TYPE> mesh = new PETSC_MESH_TYPE(comm, -1, debug);
  ierr = DMMeshSetMesh(*dm, mesh);CHKERRQ(ierr);
#ifdef PETSC_HAVE_EXODUSII
  try {
    ierr = MyPetscReadExodusII(comm,filename,*dm);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cerr << "Error: " << e << std::endl;
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --with-exodusii-dir=/path/to/exodus");
#endif
  //if (debug) {mesh->view("Mesh");}
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MyPetscReadExodusII"
PetscErrorCode MyPetscReadExodusII(MPI_Comm comm, const char filename[],DM dm)
{
  ALE::Obj<PETSC_MESH_TYPE>               mesh;
  typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
  typedef std::set<FlexMesh::point_type>  PointSet;
  ALE::Obj<FlexMesh>  boundarymesh;
  PetscMPIInt         rank;
  int                 CPU_word_size = 0;
  int                 IO_word_size  = 0;
  PetscBool           interpolate   = PETSC_FALSE;
  PetscBool           addlabels     = PETSC_FALSE;
  int               **connect;
  int                 exoid;
  char                title[MAX_LINE_LENGTH+1];
  float               version;
  int                 num_dim,num_nodes = 0,num_elem = 0;
  int                 num_eb = 0,num_ns = 0,num_ss = 0;
  PetscErrorCode      ierr;
  const char          known_elements[] = "tri,tri3,triangle,triangle3,quad,quad4,tet,tet4,tetra,tetra4,hex,hex8";
  float              *x,*y,*z;
  PetscBool           debug = PETSC_TRUE;

  PetscFunctionBegin;
  /*
    Get the sieve mesh from the dm
  */
  ierr = DMMeshGetMesh(dm,mesh);CHKERRQ(ierr);
  rank = mesh->commRank();

  ierr = PetscOptionsGetBool(PETSC_NULL,"-interpolate",&interpolate,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-addlabels",&addlabels,PETSC_NULL);CHKERRQ(ierr);
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
  ierr = MPI_Bcast(&num_eb,1,MPI_INT,0,comm);
  ierr = MPI_Bcast(&num_ns,1,MPI_INT,0,comm);
  ierr = MPI_Bcast(&num_ss,1,MPI_INT,0,comm);

  mesh->setDimension(num_dim);

  /* 
    Read element connectivity
  */
  int      *eb_ids,*num_elem_in_block,*num_nodes_per_elem,*num_attr;
  char    **eb_name,**eb_elemtype;
  PetscBool eb_hasnoname;
  char     *elem_sig;

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
    ierr = PetscMalloc2(MAX_STR_LENGTH+1,char,&eb_name[eb], MAX_STR_LENGTH+1,char,&eb_elemtype[eb]);CHKERRQ(ierr);
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
    ierr = ex_get_elem_blk_ids(exoid, eb_ids);CHKERRQ(ierr);

    /*
      Check that the element type in each block is known
    */
    for (int eb = 0; eb < num_eb; ++eb) {
      ierr = ex_get_elem_block(exoid,eb_ids[eb],eb_elemtype[eb],&num_elem_in_block[eb],&num_nodes_per_elem[eb],&num_attr[eb]);CHKERRQ(ierr);
      ierr = PetscStrtolower(eb_elemtype[eb]);CHKERRQ(ierr);
      ierr = PetscStrstr(known_elements,eb_elemtype[eb],&elem_sig);CHKERRQ(ierr);
      if (!elem_sig) {
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported element type: %s.\nSupported elements types are %s",eb_elemtype[eb],known_elements);
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
  /*
    Broadcast element block names. This will be needed later when creating the matching sections
  */
  for (int eb = 0; eb < num_eb; ++eb) {
    ierr = MPI_Bcast(eb_name[eb],MAX_STR_LENGTH+1,MPI_CHAR,0,comm);
  }

  /* 
    Read side sets
  */
  int      *ss_ids, *num_sides_in_set;
  int     **side_set_elem_list, **side_set_side_list;
  char    **ss_name;
  PetscBool ss_hasnoname;
  int       num_df_in_sset;
  
  if (num_ss > 0) {
    ierr = PetscMalloc5(num_ss,int,&ss_ids,
                        num_ss,int,&num_sides_in_set,
                        num_ss,int*,&side_set_elem_list,
                        num_ss,int*,&side_set_side_list,
                        num_ss,char*,&ss_name);CHKERRQ(ierr);
    for (int ss=0; ss < num_ss; ss++) {
      num_sides_in_set[ss] = 0;
      ierr = PetscMalloc((MAX_STR_LENGTH+1)*sizeof(char),&ss_name[ss]);CHKERRQ(ierr);
    }
    if (rank == 0) {
      ierr = ex_get_side_set_ids(exoid, ss_ids);CHKERRQ(ierr);
      ierr = ex_get_names(exoid,EX_SIDE_SET,ss_name);CHKERRQ(ierr);
      for (int ss = 0; ss< num_ss; ++ss) {
        ierr = ex_get_side_set_param(exoid, ss_ids[ss], &num_sides_in_set[ss], &num_df_in_sset);
        ierr = PetscMalloc2(num_sides_in_set[ss],int,&side_set_elem_list[ss], 
                            num_sides_in_set[ss],int, &side_set_side_list[ss]);CHKERRQ(ierr);
        ierr = ex_get_side_set(exoid,ss_ids[ss],side_set_elem_list[ss],side_set_side_list[ss]);
        ierr = PetscStrcmp(ss_name[ss],"",&ss_hasnoname);CHKERRQ(ierr);
        if (ss_hasnoname) {
          ierr = PetscSNPrintf(ss_name[ss],MAX_STR_LENGTH,"FaceSet_%.4i",ss_ids[ss]);CHKERRQ(ierr);
        } 
      }
    }
    /*
      Broadcast side sets names. This will be needed later when creating the matching sections
    */
    for (int ss = 0; ss < num_ss; ++ss) {
      ierr = MPI_Bcast(ss_name[ss],MAX_STR_LENGTH+1,MPI_CHAR,0,comm);
    }
  }

  /* 
    Read node sets
  */
  int  *ns_ids, *num_nodes_in_set;
  int **node_list;
  char **ns_name;
  PetscBool ns_hasnoname;
  if (num_ns > 0) {
    ierr = PetscMalloc4(num_ns,int,&ns_ids,
                        num_ns,int,&num_nodes_in_set,
                        num_ns,int*,&node_list,
                        num_ns,char*,&ns_name);CHKERRQ(ierr);
    for (int ns=0; ns < num_ns; ns++) {
      num_nodes_in_set[ns] = 0;
      ierr = PetscMalloc((MAX_STR_LENGTH+1)*sizeof(char),&ns_name[ns]);CHKERRQ(ierr);
    }
    if (rank == 0) {
      ierr = ex_get_node_set_ids(exoid, ns_ids);CHKERRQ(ierr);
      ierr = ex_get_names(exoid,EX_NODE_SET,ns_name);CHKERRQ(ierr);

      for (int ns = 0; ns < num_ns; ++ns) {
        int num_df_in_set;
        ierr = ex_get_node_set_param(exoid, ns_ids[ns], &num_nodes_in_set[ns], &num_df_in_set);CHKERRQ(ierr);
        ierr = PetscMalloc(num_nodes_in_set[ns] * sizeof(int), &node_list[ns]);CHKERRQ(ierr);
        ierr = ex_get_node_set(exoid, ns_ids[ns], node_list[ns]);
        ierr = PetscStrcmp(ns_name[ns],"",&ns_hasnoname);CHKERRQ(ierr);
        if (ns_hasnoname) {
          ierr = PetscSNPrintf(ns_name[ns],MAX_STR_LENGTH,"VertexSet_%.4i",ns_ids[ns]);CHKERRQ(ierr);
        } 
      }
    }
    /*
      Broadcast node sets names. This will be needed later when creating the matching sections
    */
    for (int ns = 0; ns < num_ns; ++ns) {
      ierr = MPI_Bcast(ns_name[ns],MAX_STR_LENGTH+1,MPI_CHAR,0,comm);
    }
  }
  /*
    Done reading EXO, closing file
  */
  if (rank == 0) {
    ierr = ex_close(exoid);CHKERRQ(ierr);
  }


  /* 
    Build mesh topology
  */
  int  *cells;
  int **connectivity_table;
  int   num_local_corners = 0;
  if (rank == 0) {
    for (int eb=0; eb < num_eb; ++eb) {
      num_local_corners += num_nodes_per_elem[eb] * num_elem_in_block[eb];
    }
    ierr = PetscMalloc2(num_local_corners,int,&cells, num_elem,int*,&connectivity_table);CHKERRQ(ierr);
    for (int eb = 0, k = 0; eb < num_eb; ++eb) {
      for (int e = 0; e < num_elem_in_block[eb]; ++e, ++k) {
        for (int c = 0; c < num_nodes_per_elem[eb]; ++c) {
          cells[k*num_nodes_per_elem[eb]+c] = connect[eb][e*num_nodes_per_elem[eb]+c] - 1;
        }
        connectivity_table[k] = &cells[k*num_nodes_per_elem[eb]];
      }
      ierr = PetscFree(connect[eb]);CHKERRQ(ierr);
    }
    ierr = PetscFree(connect);CHKERRQ(ierr);
  }

  ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(mesh->comm(), mesh->debug());
  ALE::Obj<FlexMesh>                    m     = new FlexMesh(mesh->comm(), mesh->debug());
  ALE::Obj<FlexMesh::sieve_type>        s     = new FlexMesh::sieve_type(mesh->comm(), mesh->debug());

  /*
    BUG!
    Here we assume that num_nodes_per_elem is constant accross blocks (i.e.)
    that all elements in the mesh are of the same type.
  */
  int  numCorners=0;
  if (rank == 0) {
    numCorners = num_nodes_per_elem[0];
  }
  /*
    interpolating is required to build side sets
  */
  if (num_ss > 0) {
    interpolate = PETSC_TRUE;
  }

  ALE::SieveBuilder<FlexMesh>::buildTopology(s,num_dim,num_elem,cells,num_nodes,interpolate,numCorners,PETSC_DECIDE,m->getArrowSection("orientation"));

  std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> renumbering;
  ALE::ISieveConverter::convertSieve(*s, *sieve, renumbering, false);
  mesh->setSieve(sieve);
  mesh->stratify();
  ALE::ISieveConverter::convertOrientation(*s, *sieve, renumbering, m->getArrowSection("orientation").ptr());

  /* 
    Build coordinates
  */
  double *coords;
  if (rank == 0) {
    ierr = PetscMalloc(num_dim*num_nodes * sizeof(double), &coords);CHKERRQ(ierr);
    if (num_dim > 0) {for (int v = 0; v < num_nodes; ++v) {coords[v*num_dim+0] = x[v];}}
    if (num_dim > 1) {for (int v = 0; v < num_nodes; ++v) {coords[v*num_dim+1] = y[v];}}
    if (num_dim > 2) {for (int v = 0; v < num_nodes; ++v) {coords[v*num_dim+2] = z[v];}}
    ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(mesh,num_dim,coords);
    ierr = PetscFree(coords);CHKERRQ(ierr);
    ierr = PetscFree3(x, y, z);CHKERRQ(ierr);
  }

  /*
    Create a section for each Element Block.
    From the Exodus documentation:
    "The internal number of an element is defined implicitly by the order in which it appears
    in the file. Elements are numbered internally (beginning with 1) consecutively across all
    element blocks. See Node Number Map for a discussion of internal element numbering."
  */
  SectionInt *cellBlock,cellParentBlock;
  ierr = PetscMalloc(num_eb*sizeof(SectionInt),&cellBlock);CHKERRQ(ierr);
  ierr = DMMeshGetSectionInt(dm,"CellParentBlock",&cellParentBlock);CHKERRQ(ierr);

  for (int eb = 0,k = 0; eb < num_eb; eb++) {
    ierr = DMMeshGetSectionInt(dm,eb_name[eb],&cellBlock[eb]);CHKERRQ(ierr);
    for (int e = 0; e < num_elem_in_block[eb]; e++,k++) {
      ierr = SectionIntSetFiberDimension(cellBlock[eb],k,1);CHKERRQ(ierr);
      ierr = SectionIntSetFiberDimension(cellParentBlock,k,1);CHKERRQ(ierr);
    }
    ierr = SectionIntAllocate(cellBlock[eb]);CHKERRQ(ierr);
  }
  ierr = SectionIntAllocate(cellParentBlock);CHKERRQ(ierr);

  for (int eb = 0,k = 0; eb < num_eb; eb++) {
    for (int e = 0; e < num_elem_in_block[eb]; e++,k++) {
      ierr = SectionIntUpdate(cellParentBlock,k,&eb,INSERT_VALUES);CHKERRQ(ierr);
      ierr = SectionIntUpdate(cellBlock[eb],k,&eb,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /*
    Create label for element block , if needed
  */
  if (addlabels) {
    const ALE::Obj<PETSC_MESH_TYPE::label_type>& cellBlocks = mesh->createLabel("CellBlocks");
    if (rank == 0) {
      for (int eb = 0, k = 0; eb < num_eb; ++eb) {
        for (int e = 0; e < num_elem_in_block[eb]; ++e, ++k) {
          mesh->setValue(cellBlocks,k,eb_ids[eb]);
        }
      }
    }
    if (debug) {cellBlocks->view("Cell Blocks");}
  }
  ierr = PetscFree(cellBlock);CHKERRQ(ierr);  

  /*
    Build side sets
  */
  SectionInt        *faceSet,faceParentSet;
  PetscInt         **side_set_points;
  PetscInt           e,facepoint = 0;
  PetscInt          *parent_block;
  const PetscInt     triface[3][2]  = {{0,1}, {1,2}, {2,0}};
  const PetscInt     quadface[4][2] = {{0,1}, {1,2}, {2,3}, {3,0}};
  const PetscInt     tetface[4][3]  = {{0,1,3}, {1,2,3}, {2,0,3}, {0,1,2}};
  const PetscInt     hexface[6][4]  = {{0,1,5,4}, {1,2,6,5}, {2,3,7,6}, {0,4,7,3}, {0,3,2,1}, {4,5,6,7}};
  const PetscInt    *faceVertex;
  PetscInt           faceNumVertex,faceNum;
  ALE::Obj<PointSet> face = new PointSet();

  ierr = PetscMalloc2(num_ss,SectionInt,&faceSet,
                      num_ss,PetscInt*,&side_set_points);CHKERRQ(ierr);

  if (num_ss > 0) {
    ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> faceVisitor(1);
    /*
      Build the boundary mesh
    */
    boundarymesh = ALE::Selection<PETSC_MESH_TYPE>::boundaryV(mesh);
    /*
      This will fail on more than one processor with the error message:
      Error: ERROR: IFSieve points have not been allocated.
      Should boundaryV just return an empty boundary mesh when mesh is empty instead of an error?
    */      
    for (int ss = 0; ss < num_ss; ++ss) {
      ierr = DMMeshGetSectionInt(dm,ss_name[ss],&faceSet[ss]);CHKERRQ(ierr);
      ierr = PetscMalloc(num_sides_in_set[ss]*sizeof(PetscInt),&side_set_points[ss]);CHKERRQ(ierr);
      for (int s = 0; s < num_sides_in_set[ss]; ++s) {
        /*
          Get the point number of the faces described by vertices side_set_side_list[ss][s]
          initialize side_set_point_list from side_set_elem_list and side_set_side_list
        */
        e = side_set_elem_list[ss][s]-1;
        ierr = SectionIntRestrict(cellParentBlock,e,&parent_block);CHKERRQ(ierr);
        face->clear();
        faceNum = side_set_side_list[ss][s]-1;

        ierr = PetscStrstr("tri,tri3,triangle,triangle3",eb_elemtype[parent_block[0]],&elem_sig);CHKERRQ(ierr);
        if (elem_sig){
          faceNumVertex = 2;
          faceVertex = triface[faceNum];
        }
        ierr = PetscStrstr("tet,tet4,tetra,tetra4",eb_elemtype[parent_block[0]],&elem_sig);CHKERRQ(ierr);
        if (elem_sig){
          faceNumVertex = 3;
          faceVertex = tetface[faceNum];
        }
        ierr = PetscStrstr("quad,quad4",eb_elemtype[parent_block[0]],&elem_sig);CHKERRQ(ierr);
        if (elem_sig){
          faceNumVertex = 2;
          faceVertex = quadface[faceNum];
        }
        ierr = PetscStrstr("hex,hex8",eb_elemtype[parent_block[0]],&elem_sig);CHKERRQ(ierr);
        if (elem_sig){
          faceNumVertex = 4;
          faceVertex = hexface[faceNum];
        }
        for (int v = 0; v < faceNumVertex; v++) {
          face->insert(face->end(), connectivity_table[e][ faceVertex[v] ] + num_elem);
        }

        sieve->nJoin(face->begin(),face->end(),1,faceVisitor);

        int sz = faceVisitor.getSize();
        if (sz != 1) {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not find face with requested vertices.\n");
        }
        
        facepoint = faceVisitor.getPoints()[0];
        faceVisitor.clear();
        side_set_points[ss][s] = facepoint;
        /*
          Initialize fibration for the SectionInt faceSet[ss]
        */
        ierr = SectionIntSetFiberDimension(faceSet[ss],facepoint,1);CHKERRQ(ierr);
      }
      ierr = SectionIntAllocate(faceSet[ss]);CHKERRQ(ierr);
    }

    /*
      Initialize the faceSet sections
    */
    for (int ss = 0; ss < num_ss; ++ss) {
      for (int s = 0; s < num_sides_in_set[ss]; ++s) {
        ierr = SectionIntUpdate(faceSet[ss],side_set_points[ss][s],&ss,INSERT_VALUES);CHKERRQ(ierr);
      }
    }

    /*
      Inverse mapping: faceParentSet
    */
    PetscInt  *faceParentSetCount,*facecount;
    PetscInt **faceParentSetId;
    PetscInt   num_face = 0,faceid;
    
    ierr = DMMeshGetSectionInt(dm,"FaceParentSet",&faceParentSet);CHKERRQ(ierr);
    if (rank == 0) {
      num_face = mesh->depthStratum(1)->size();
      ierr = PetscMalloc3(num_face,PetscInt,&faceParentSetCount,
                          num_face,PetscInt*,&faceParentSetId,
                          num_face,PetscInt,&facecount);CHKERRQ(ierr);
      /* 
        Count parent set for each face
        We assume that the side sets are properly formed and that each face is listed exactly once in each set
      */
      for (int s = 0; s < num_face; s++) {
        facecount[s] = 0;
        faceParentSetCount[s] = 0;
      }
      for (int ss = 0; ss < num_ss; ss++) {
        for (int s = 0; s < num_sides_in_set[ss]; s++) {
          faceid = side_set_points[ss][s] - num_elem - num_nodes;
          faceParentSetCount[faceid]++;
        }
      }
      for (int ss = 0; ss < num_ss; ss++) {
        for (int s = 0; s < num_sides_in_set[ss]; s++) {
          facepoint = side_set_points[ss][s];
          faceid = side_set_points[ss][s] - num_elem - num_nodes;
          ierr = PetscMalloc(faceParentSetCount[faceid]*sizeof(PetscInt),&faceParentSetId[faceid]);CHKERRQ(ierr);
          ierr = SectionIntSetFiberDimension(faceParentSet,facepoint,faceParentSetCount[faceid]);CHKERRQ(ierr);
        }
      }
      ierr = SectionIntAllocate(faceParentSet);CHKERRQ(ierr);
      
      for (int ss = 0; ss < num_ss; ss++) {
        for (int s = 0; s < num_sides_in_set[ss]; s++) {
          faceid = side_set_points[ss][s] - num_elem - num_nodes;
          faceParentSetId[faceid][facecount[faceid]] = ss;
          facecount[faceid]++;
        }
      }          
      for (int ss = 0; ss < num_ss; ss++) {
        for (int s = 0; s < num_sides_in_set[ss]; s++) {
          facepoint = side_set_points[ss][s];
          faceid = side_set_points[ss][s] - num_elem - num_nodes;
          ierr = SectionIntUpdate(faceParentSet,facepoint,faceParentSetId[faceid],INSERT_VALUES);CHKERRQ(ierr);
          ierr = PetscFree(faceParentSetId[faceid]);CHKERRQ(ierr);
        }
      }
      ierr = PetscFree3(faceParentSetCount,faceParentSetId,facecount);CHKERRQ(ierr);
    } else {
      ierr = SectionIntAllocate(faceParentSet);CHKERRQ(ierr);
    }

    /*
      Create Face set label if needed
    */
    if (addlabels) {
      const ALE::Obj<PETSC_MESH_TYPE::label_type>& faceSets = mesh->createLabel("FaceSets");
      if (rank == 0) {
        for (int ss = 0; ss < num_ss; ++ss) {
          for (int s = 0; s < num_sides_in_set[ss]; s++) {
            mesh->addValue(faceSets,side_set_points[ss][s],ss_ids[ss]);
          }
        }
      }
      if (debug) {faceSets->view("Face Sets");}
    }

    /*
      Deallocate remaining side set arrays
    */
    for (int ss = 0; ss < num_ss; ++ss) {
      ierr = PetscFree(side_set_points[ss]);CHKERRQ(ierr);
      ierr = PetscFree2(side_set_elem_list[ss],side_set_side_list[ss]);CHKERRQ(ierr);
      ierr = PetscFree(ss_name[ss]);CHKERRQ(ierr);
    }
    

    ierr = PetscFree5(ss_ids,num_sides_in_set,side_set_elem_list,side_set_side_list,ss_name);CHKERRQ(ierr);
    ierr = PetscFree2(faceSet,side_set_points);CHKERRQ(ierr);
  }
  

  /*
    Build vertex sets
  */
  PetscInt   *vertexParentSetCount,**vertexParentSetId;
  SectionInt *vertexSet,vertexParentSet;
  PetscInt vertex_id, *vertexcount;
  ierr = PetscMalloc(num_ns*sizeof(SectionInt),&vertexSet);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = PetscMalloc2(num_nodes,PetscInt,&vertexParentSetCount,
                        num_nodes,PetscInt*,&vertexParentSetId);
    for (int s = 0; s < num_nodes; s++){
      vertexParentSetCount[s] = 0;
    }
  }
  for (int ns = 0; ns < num_ns; ns++) {
    ierr = DMMeshGetSectionInt(dm,ns_name[ns],&vertexSet[ns]);CHKERRQ(ierr);
    for (int v = 0; v < num_nodes_in_set[ns]; v++) {
      ierr = SectionIntSetFiberDimension(vertexSet[ns],node_list[ns][v]-1+num_elem,1);CHKERRQ(ierr);
      vertexParentSetCount[node_list[ns][v]-1]++;
    }
    ierr = SectionIntAllocate(vertexSet[ns]);CHKERRQ(ierr);
  }
  ierr = DMMeshGetSectionInt(dm,"VertexParentSet",&vertexParentSet);CHKERRQ(ierr);

  if (rank == 0) {
    ierr = PetscMalloc(num_nodes*sizeof(PetscInt),&vertexcount);CHKERRQ(ierr);
    for (int v = 0; v < num_nodes; v++) {
      if (vertexParentSetCount[v] > 0) {
        ierr = PetscMalloc(vertexParentSetCount[v]*sizeof(PetscInt),&vertexParentSetId[v]);
        ierr = SectionIntSetFiberDimension(vertexParentSet,v+num_elem,vertexParentSetCount[v]);CHKERRQ(ierr);
        vertexcount[v] = 0;
      }
    }
  }
  ierr = SectionIntAllocate(vertexParentSet);CHKERRQ(ierr);
  if (rank == 0) {
    for (int ns = 0; ns < num_ns; ns++) {
      for (int v = 0; v < num_nodes_in_set[ns]; v++) {
        vertex_id = node_list[ns][v]-1;
        ierr = SectionIntUpdate(vertexSet[ns],vertex_id+num_elem,&ns,INSERT_VALUES);CHKERRQ(ierr); 
      
        if (vertexParentSetCount[vertex_id] > 0) {
          vertexParentSetId[vertex_id][vertexcount[vertex_id]] = ns;
          vertexcount[vertex_id]++;
        }
      }
    }
    ierr = PetscFree(vertexcount);CHKERRQ(ierr);

    for (int v = 0; v < num_nodes; v++) {
      if (vertexParentSetCount[v] > 0) {
        ierr = SectionIntUpdate(vertexParentSet,v+num_elem,vertexParentSetId[v],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  if (rank == 0) {
    for (int v = 0; v < num_nodes; v++) {
      if (vertexParentSetCount[v] > 0) {
        ierr = PetscFree(vertexParentSetId[v]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree2(vertexParentSetCount,vertexParentSetId);CHKERRQ(ierr);
  }
  ierr = PetscFree(vertexSet);CHKERRQ(ierr);

  /*
    Create Node set label if needed
  */
  if (addlabels) {
    const ALE::Obj<PETSC_MESH_TYPE::label_type>& vertexSets = mesh->createLabel("VertexSets");
    if (rank == 0) {
      for (int ns = 0; ns < num_ns; ++ns) {
        for (int n = 0; n < num_nodes_in_set[ns]; ++n) {
          mesh->addValue(vertexSets, node_list[ns][n]+num_elem-1, ns_ids[ns]);
        }
      }
    }
    if (debug) {vertexSets->view("Vertex Sets");}
  }
  if (num_ns > 0) {
    for (int ns = 0; ns < num_ns; ns++) {
      ierr = PetscFree(node_list[ns]);CHKERRQ(ierr);
      ierr = PetscFree(ns_name[ns]);CHKERRQ(ierr);
    }
    ierr = PetscFree4(ns_ids,num_nodes_in_set,node_list,ns_name);CHKERRQ(ierr);
  }

  /* 
    Free remaining element block temporary variables
  */
  for (int eb = 0; eb < num_eb; ++eb) {
    ierr = PetscFree2(eb_name[eb],eb_elemtype[eb]);CHKERRQ(ierr);
  }
  ierr = PetscFree6(eb_ids,num_elem_in_block,num_nodes_per_elem,num_attr,eb_name,eb_elemtype);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = PetscFree2(cells,connectivity_table);CHKERRQ(ierr);
  }

  if (debug && num_ss > 0) {boundarymesh->view("\n\nBoundary Mesh");}
  if (debug) {mesh->view("\n\nMesh");}
  PetscFunctionReturn(0);
}
