/*
  To do, bugs, questions:
    - Do I need to destroy the sections?
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
    ierr = DMView(dm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetString(PETSC_NULL, "-o", outfilename, PETSC_MAX_PATH_LEN, &outflag);CHKERRQ(ierr);
  if (outflag) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outfilename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = DMView(dm,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
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
  //ierr = DMMeshSetMesh(*dm, mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MyPetscReadExodusII"
PetscErrorCode MyPetscReadExodusII(MPI_Comm comm, const char filename[],DM dm)
{
// ALE::Obj<PETSC_MESH_TYPE>& mesh
  ALE::Obj<PETSC_MESH_TYPE>               mesh;
  typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
  typedef std::set<FlexMesh::point_type>  PointSet;
  ALE::Obj<FlexMesh>  boundarymesh;
  PetscMPIInt         rank;
  int                 CPU_word_size = 0;
  int                 IO_word_size  = 0;
  PetscBool           interpolate   = PETSC_FALSE;
  PetscBool           addlabels     = PETSC_FALSE;
  int                 **connect;
  int                 exoid;
  char                title[MAX_LINE_LENGTH+1];
  float               version;
  int                 num_dim, num_nodes=0, num_elem=0;
  int                 num_elem_blk=0, num_node_sets=0, num_side_sets=0;
  PetscErrorCode      ierr;
  const char          known_elements[] = "tri,tri3,quad,quad4,tet,tet4,hex,hex8";
  float               *x, *y, *z;
  PetscBool           debug = PETSC_TRUE;

  PetscFunctionBegin;
  /*
    Get the sieve mesh from the dm
  */
  ierr = DMMeshGetMesh(dm,mesh);CHKERRQ(ierr);
  rank = mesh->commRank();

  ierr = PetscOptionsGetBool(PETSC_NULL, "-interpolate", &interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL, "-addlabels", &addlabels, PETSC_NULL);CHKERRQ(ierr);
  /*
    Open EXODUS II file and read basic informations on rank 0,
    then broadcast to all nodes
  */
  if (rank == 0) {
    exoid = ex_open(filename,EX_READ,&CPU_word_size,&IO_word_size,&version);CHKERRQ(!exoid);
    // Read database parameters
    ierr = ex_get_init(exoid, title,&num_dim,&num_nodes,&num_elem,&num_elem_blk,&num_node_sets,&num_side_sets);CHKERRQ(ierr);
    if (num_elem_blk == 0) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exodus file does not contain any element block\n");
    }
    ierr = PetscMalloc3(num_nodes,float,&x,num_nodes,float,&y,num_nodes,float,&z);CHKERRQ(ierr);
    ierr = ex_get_coord(exoid,x,y,z);CHKERRQ(ierr);
  }

  ierr = MPI_Bcast(&num_dim,1,MPI_INT,0,comm);
  //ierr = MPI_Bcast(&num_nodes,1,MPI_INT,0,comm);
  //ierr = MPI_Bcast(&num_elem,1,MPI_INT,0,comm);
  ierr = MPI_Bcast(&num_elem_blk,1,MPI_INT,0,comm);
  ierr = MPI_Bcast(&num_node_sets,1,MPI_INT,0,comm);
  ierr = MPI_Bcast(&num_side_sets,1,MPI_INT,0,comm);

  mesh->setDimension(num_dim);

  // Read element connectivity
  int   *eb_ids,*num_elem_in_block,*num_nodes_per_elem,*num_attr;
  char **block_name,**block_elemtype;
  char* elem_sig;
  int  *ss_ids, *num_sides_in_set;
  int **side_set_elem_list, **side_set_side_list;
  int  *ns_ids, *num_nodes_in_set;
  int **node_list;

  ierr = PetscMalloc6(num_elem_blk,int,&eb_ids,
                      num_elem_blk,int,&num_elem_in_block,
                      num_elem_blk,int,&num_nodes_per_elem,
                      num_elem_blk,int,&num_attr,
                      num_elem_blk,char*,&block_name,
                      num_elem_blk,char*,&block_elemtype);CHKERRQ(ierr);

  for (int eb = 0; eb < num_elem_blk; eb++) {
    num_elem_in_block[eb] = 0;
    num_nodes_per_elem[eb] = 0;
    num_attr[eb] = 0;
  }
  if (rank == 0) {
    for (int eb = 0; eb < num_elem_blk; ++eb) {
      ierr = PetscMalloc2(MAX_STR_LENGTH+1,char,&block_name[eb], MAX_STR_LENGTH+1,char,&block_elemtype[eb]);CHKERRQ(ierr);
    }
    ierr = ex_get_names(exoid,EX_ELEM_BLOCK,block_name);CHKERRQ(ierr);
    ierr = ex_get_elem_blk_ids(exoid, eb_ids);CHKERRQ(ierr);

    /*
      Check that the element type in each block is known
    */
    for (int eb = 0; eb < num_elem_blk; ++eb) {
      ierr = ex_get_elem_block(exoid,eb_ids[eb],block_elemtype[eb],&num_elem_in_block[eb],&num_nodes_per_elem[eb],&num_attr[eb]);CHKERRQ(ierr);
      ierr = PetscStrtolower(block_elemtype[eb]);CHKERRQ(ierr);
      ierr = PetscStrstr(known_elements,block_elemtype[eb],&elem_sig);CHKERRQ(ierr);
      if (!elem_sig) {
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported element type: %s.\nSupported elements types are %s",block_elemtype[eb],known_elements);
      }
      ierr = PetscFree(block_name[eb]);CHKERRQ(ierr);
    }

    /*
      Read Connectivity tables
    */
    ierr = PetscMalloc(num_elem_blk * sizeof(int*),&connect);CHKERRQ(ierr);
    for (int eb = 0; eb < num_elem_blk; ++eb) {
      ierr = PetscMalloc(num_nodes_per_elem[eb]*num_elem_in_block[eb] * sizeof(int),&connect[eb]);CHKERRQ(ierr);
      ierr = ex_get_elem_conn(exoid,eb_ids[eb],connect[eb]);CHKERRQ(ierr);
    }
  }

  // Read side sets
  if (num_side_sets > 0) {
    ierr = PetscMalloc4(num_side_sets,int,&ss_ids,
                        num_side_sets,int,&num_sides_in_set,
                        num_side_sets,int*,&side_set_elem_list,
                        num_side_sets,int*,&side_set_side_list);CHKERRQ(ierr);
    for (int ss=0; ss < num_side_sets; ss++) {
      num_sides_in_set[ss] = 0;
    }
    if (rank == 0) {
      ierr = ex_get_side_set_ids(exoid, ss_ids);CHKERRQ(ierr);
      for (int ss = 0; ss< num_side_sets; ++ss) {
        int num_df_in_sset;
        ierr = ex_get_side_set_param(exoid, ss_ids[ss], &num_sides_in_set[ss], &num_df_in_sset);
        ierr = PetscMalloc2(num_sides_in_set[ss],int,&side_set_elem_list[ss], num_sides_in_set[ss],int, &side_set_side_list[ss]);CHKERRQ(ierr);
        ierr = ex_get_side_set(exoid, ss_ids[ss], side_set_elem_list[ss], side_set_side_list[ss]);
      }
    }
  }

  // Read node sets
  if (num_node_sets > 0) {
    ierr = PetscMalloc3(num_node_sets,int,&ns_ids,
                        num_node_sets,int,&num_nodes_in_set,
                        num_node_sets,int*,&node_list);CHKERRQ(ierr);
    for (int ns=0; ns < num_node_sets; ns++) {
      num_nodes_in_set[ns] = 0;
    }
    if (rank == 0) {
      ierr = ex_get_node_set_ids(exoid, ns_ids);CHKERRQ(ierr);
      for (int ns = 0; ns < num_node_sets; ++ns) {
        int num_df_in_set;
        ierr = ex_get_node_set_param (exoid, ns_ids[ns], &num_nodes_in_set[ns], &num_df_in_set);CHKERRQ(ierr);
        ierr = PetscMalloc(num_nodes_in_set[ns] * sizeof(int), &node_list[ns]);CHKERRQ(ierr);
        ierr = ex_get_node_set(exoid, ns_ids[ns], node_list[ns]);
      }
    }
  }
  if (rank == 0) {
    ierr = ex_close(exoid);CHKERRQ(ierr);
  }

  // Build mesh topology
  int *cells;
  int **connectivity_table;
  int num_local_corners = 0;
  if (rank == 0) {
    for (int eb=0; eb < num_elem_blk; ++eb) {
      num_local_corners += num_nodes_per_elem[eb] * num_elem_in_block[eb];
    }
    ierr = PetscMalloc2(num_local_corners,int,&cells, num_elem,int*,&connectivity_table);CHKERRQ(ierr);
    for (int eb = 0, k = 0; eb < num_elem_blk; ++eb) {
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
  if (num_side_sets > 0) {
    interpolate = PETSC_TRUE;
  }

  ALE::SieveBuilder<FlexMesh>::buildTopology(s,num_dim,num_elem,cells,num_nodes,interpolate,numCorners,PETSC_DECIDE,m->getArrowSection("orientation"));

  std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> renumbering;
  ALE::ISieveConverter::convertSieve(*s, *sieve, renumbering, false);
  mesh->setSieve(sieve);
  mesh->stratify();
  ALE::ISieveConverter::convertOrientation(*s, *sieve, renumbering, m->getArrowSection("orientation").ptr());

  // Build coordinates
  double *coords;
  if (rank == 0) {
    ierr = PetscMalloc(num_dim*num_nodes * sizeof(double), &coords);CHKERRQ(ierr);
    if (num_dim > 0) {for(int v = 0; v < num_nodes; ++v) {coords[v*num_dim+0] = x[v];}}
    if (num_dim > 1) {for(int v = 0; v < num_nodes; ++v) {coords[v*num_dim+1] = y[v];}}
    if (num_dim > 2) {for(int v = 0; v < num_nodes; ++v) {coords[v*num_dim+2] = z[v];}}
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
  char EBSectionName[256];
  SectionInt *cellBlock,cellParentBlock;
  ierr = PetscMalloc(num_elem_blk*sizeof(SectionInt),&cellBlock);CHKERRQ(ierr);
  ierr = DMMeshGetSectionInt(dm,"CellParentBlock",&cellParentBlock);CHKERRQ(ierr);

  for (int eb = 0,k = 0; eb < num_elem_blk; eb++) {
    ierr = PetscSNPrintf(EBSectionName,sizeof(EBSectionName),"CellBlock_%.4i",eb);CHKERRQ(ierr);
    ierr = DMMeshGetSectionInt(dm,EBSectionName,&cellBlock[eb]);CHKERRQ(ierr);
    for (int e = 0; e < num_elem_in_block[eb]; e++,k++) {
      ierr = SectionIntSetFiberDimension(cellBlock[eb],k,1);CHKERRQ(ierr);
      ierr = SectionIntSetFiberDimension(cellParentBlock,k,1);CHKERRQ(ierr);
    }
    ierr = SectionIntAllocate(cellBlock[eb]);CHKERRQ(ierr);
  }
  ierr = SectionIntAllocate(cellParentBlock);CHKERRQ(ierr);

  for (int eb = 0,k = 0; eb < num_elem_blk; eb++) {
    for (int e = 0; e < num_elem_in_block[eb]; e++,k++) {
      ierr = SectionIntUpdate(cellParentBlock,k,&eb,INSERT_VALUES);CHKERRQ(ierr);
      ierr = SectionIntUpdate(cellBlock[eb],k,&eb,INSERT_VALUES);CHKERRQ(ierr);
    }
    //ierr = SectionIntComplete(cellBlock[eb]);CHKERRQ(ierr);
    if (debug) {ierr = SectionIntView(cellBlock[eb],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  }
  //ierr = SectionIntComplete(cellParentBlock);CHKERRQ(ierr);
  if (debug) {ierr = SectionIntView(cellParentBlock,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}

  if (addlabels) {
    const ALE::Obj<PETSC_MESH_TYPE::label_type>& cellBlocks = mesh->createLabel("CellBlocks");
    if (rank == 0) {
      for(int eb = 0, k = 0; eb < num_elem_blk; ++eb) {
        for(int e = 0; e < num_elem_in_block[eb]; ++e, ++k) {
          mesh->setValue(cellBlocks, k, eb_ids[eb]);
        }
      }
    }
  }
  ierr = PetscFree(cellBlock);CHKERRQ(ierr);


  /*
    Build vertex sets
  */
  char       NSSectionName[256];
  PetscInt   *vertexParentSetCount,**vertexParentSetId;
  SectionInt *vertexSet,vertexParentSet;
  ierr = PetscMalloc(num_node_sets*sizeof(SectionInt),&vertexSet);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = PetscMalloc2(num_nodes,PetscInt,&vertexParentSetCount,
                        num_nodes,PetscInt*,&vertexParentSetId);
    for (int s = 0; s < num_nodes; s++){
      vertexParentSetCount[s] = 0;
    }
  }
  for (int ns = 0; ns < num_node_sets; ns++) {
    ierr = PetscSNPrintf(NSSectionName,sizeof(NSSectionName),"VertexSet_%.4i",ns);CHKERRQ(ierr);
    ierr = DMMeshGetSectionInt(dm,NSSectionName,&vertexSet[ns]);CHKERRQ(ierr);
    for (int v = 0; v < num_nodes_in_set[ns]; v++) {
      ierr = SectionIntSetFiberDimension(vertexSet[ns],node_list[ns][v]-1+num_elem,1);CHKERRQ(ierr);
      vertexParentSetCount[node_list[ns][v]-1]++;
    }
    ierr = SectionIntAllocate(vertexSet[ns]);CHKERRQ(ierr);
  }
  ierr = DMMeshGetSectionInt(dm,"ParentVertexSet",&vertexParentSet);CHKERRQ(ierr);

  PetscInt vertex_id, *count;
  if (rank == 0) {
    ierr = PetscMalloc(num_nodes*sizeof(PetscInt),&count);CHKERRQ(ierr);
    for (int v = 0; v < num_nodes; v++) {
      if (vertexParentSetCount[v] > 0) {
        ierr = PetscMalloc(vertexParentSetCount[v]*sizeof(PetscInt),&vertexParentSetId[v]);
        ierr = SectionIntSetFiberDimension(vertexParentSet,v+num_elem,vertexParentSetCount[v]);CHKERRQ(ierr);
        count[v] = 0;
      }
    }
  }
  ierr = SectionIntAllocate(vertexParentSet);CHKERRQ(ierr);
  if (rank == 0) {
    for (int ns = 0; ns < num_node_sets; ns++) {
      for (int v = 0; v < num_nodes_in_set[ns]; v++) {
        ierr = SectionIntUpdate(vertexSet[ns],node_list[ns][v]-1+num_elem,&ns,INSERT_VALUES);CHKERRQ(ierr); 
        vertex_id = node_list[ns][v]-1;
        if (vertexParentSetCount[vertex_id] > 0) {
          vertexParentSetId[vertex_id][count[vertex_id]] = ns;
          count[vertex_id]++;
        }
      }
      //ierr = SectionIntComplete(vertexSet[ns]);CHKERRQ(ierr);
      if (debug) {ierr = SectionIntView(vertexSet[ns],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
    }
    ierr = PetscFree(count);CHKERRQ(ierr);

    for (int v = 0; v < num_nodes; v++) {
      if (vertexParentSetCount[v] > 0) {
        ierr = SectionIntUpdate(vertexParentSet,v+num_elem,vertexParentSetId[v],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  //ierr = SectionIntComplete(vertexParentSet);CHKERRQ(ierr);
  if (debug) {ierr = SectionIntView(vertexParentSet,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  if (rank == 0) {
    for (int v = 0; v < num_nodes; v++) {
      ierr = PetscFree(vertexParentSetId[v]);CHKERRQ(ierr);
    }
    ierr = PetscFree2(vertexParentSetCount,vertexParentSetId);CHKERRQ(ierr);
  }
  ierr = PetscFree(vertexSet);CHKERRQ(ierr);

  if (addlabels) {
    const ALE::Obj<PETSC_MESH_TYPE::label_type>& vertexSets = mesh->createLabel("VertexSets");
    if (rank == 0) {
      for (int ns = 0; ns < num_node_sets; ++ns) {
        for (int n = 0; n < num_nodes_in_set[ns]; ++n) {
          mesh->addValue(vertexSets, node_list[ns][n]+num_elem-1, ns_ids[ns]);
        }
        ierr = PetscFree(node_list[ns]);CHKERRQ(ierr);
      }
    }
  }
  if (num_node_sets > 0) {
    ierr = PetscFree3(ns_ids,num_nodes_in_set,node_list);CHKERRQ(ierr);
  }

  /*
    Build side sets
  */
  char               SSSectionName[256];
  SectionInt         *faceSet,faceParentSet;
  PetscInt           **side_set_points;
  PetscBool          flag1;
  PetscInt           e,facepoint=0;
  PetscInt          *parent_block;
  ALE::Obj<PointSet> face = new PointSet();

  ierr = PetscMalloc2(num_side_sets,SectionInt,&faceSet,
                      num_side_sets,PetscInt*,&side_set_points);CHKERRQ(ierr);
  ierr = DMMeshGetSectionInt(dm,"FaceParentSet",&faceParentSet);CHKERRQ(ierr);

  if (num_side_sets > 0) {
    /*
      Build the boundary mesh
    */
    boundarymesh = ALE::Selection<PETSC_MESH_TYPE>::boundaryV(mesh);

    for (int ss = 0; ss < num_side_sets; ++ss) {
      ierr = PetscSNPrintf(SSSectionName,sizeof(SSSectionName),"FaceSet_%.4i",ss);CHKERRQ(ierr);
      ierr = DMMeshGetSectionInt(dm,SSSectionName,&faceSet[ss]);CHKERRQ(ierr);
      ierr = PetscMalloc(num_sides_in_set[ss]*sizeof(PetscInt),&side_set_points[ss]);CHKERRQ(ierr);
      for (int s = 0; s < num_sides_in_set[ss]; ++s) {
        /*
          initialize side_set_point_list from side_set_elem_list and side_set_side_list
        */
        e = side_set_elem_list[ss][s]-1;
        ierr = SectionIntRestrict(cellParentBlock,e,&parent_block);CHKERRQ(ierr);
        face->clear();
        // TRI
        ierr = PetscStrstr("tri,tri3",block_elemtype[parent_block[0]],&elem_sig);CHKERRQ(ierr);
        if (elem_sig){
          switch (side_set_side_list[ss][s]){
            case 1:
              face->insert(face->end(), connectivity_table[e][0]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][1]+num_elem-1);
              break;
            case 2:
              face->insert(face->end(), connectivity_table[e][1]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][2]+num_elem-1);
              break;
            case 3:
              face->insert(face->end(), connectivity_table[e][2]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][0]+num_elem-1);
              break;
          }
        }

        // TET
        ierr = PetscStrstr("tet,tet4",block_elemtype[parent_block[0]],&elem_sig);CHKERRQ(ierr);
        if (elem_sig){
          switch (side_set_side_list[ss][s]){
            case 1:
              face->insert(face->end(), connectivity_table[e][0]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][1]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][3]+num_elem-1);
              break;
            case 2:
              face->insert(face->end(), connectivity_table[e][1]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][2]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][3]+num_elem-1);
              break;
            case 3:
              face->insert(face->end(), connectivity_table[e][2]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][0]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][3]+num_elem-1);
              break;
            case 4:
              face->insert(face->end(), connectivity_table[e][0]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][2]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][1]+num_elem-1);
              break;
          }
        }

        // QUAD
        ierr = PetscStrstr("quad,quad4",block_elemtype[parent_block[0]],&elem_sig);CHKERRQ(ierr);
        if (elem_sig){
          switch (side_set_side_list[ss][s]){
            case 1:
              face->insert(face->end(), connectivity_table[e][0]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][1]+num_elem-1);
              break;
            case 2:
              face->insert(face->end(), connectivity_table[e][1]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][2]+num_elem-1);
              break;
            case 3:
              face->insert(face->end(), connectivity_table[e][2]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][3]+num_elem-1);
              break;
            case 4:
              face->insert(face->end(), connectivity_table[e][3]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][0]+num_elem-1);
              break;
          }
        }

        //HEX
        ierr = PetscStrstr("hex,hex8",block_elemtype[parent_block[0]],&elem_sig);CHKERRQ(ierr);
        if (elem_sig){
          switch (side_set_side_list[ss][s]){
            case 1:
              face->insert(face->end(), connectivity_table[e][0]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][1]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][5]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][4]+num_elem-1);
              break;
            case 2:
              face->insert(face->end(), connectivity_table[e][1]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][2]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][6]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][5]+num_elem-1);
              break;
            case 3:
              face->insert(face->end(), connectivity_table[e][2]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][3]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][7]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][6]+num_elem-1);
              break;
            case 4:
              face->insert(face->end(), connectivity_table[e][3]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][0]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][4]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][7]+num_elem-1);
              break;
            case 5:
              face->insert(face->end(), connectivity_table[e][3]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][2]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][1]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][0]+num_elem-1);
              break;
            case 6:
              face->insert(face->end(), connectivity_table[e][4]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][5]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][6]+num_elem-1);
              face->insert(face->end(), connectivity_table[e][7]+num_elem-1);
              break;
          }
        }

        /*
          Matt, can you figure out how to get the facepoint from here?
          facepoint = mesh->nJoin1(face);
        */
        printf("************* sideset: %i\n",ss);
        printf("              side:    %i\n",s);
        printf("              element: %i\n",e);
        printf("              edge:    %i\n",side_set_side_list[ss][s]);
        printf("              point:   %i\n",facepoint);
        side_set_points[ss][s] = facepoint;

        /*
          Initialize fibration for the SectionInt FaceSet_%.4i
        */
        ierr = SectionIntSetFiberDimension(faceSet[ss],facepoint,1);CHKERRQ(ierr);
      }
      ierr = SectionIntAllocate(faceSet[ss]);CHKERRQ(ierr);
    }

    /*
      To do:
        compute side sets inverse mapping with overlap
    */
    ierr = SectionIntAllocate(faceParentSet);CHKERRQ(ierr);

    /*
      Initialize the faceSet sections
    */
    for (int ss = 0; ss < num_side_sets; ++ss) {
      for (int s = 0; s < num_sides_in_set[ss]; ++s) {
        ierr = SectionIntUpdate(faceSet[ss],side_set_points[ss][s],&ss,INSERT_VALUES);CHKERRQ(ierr);
      }
      //ierr = SectionIntComplete(faceSet[ss]);CHKERRQ(ierr);
      if (debug) {ierr = SectionIntView(faceSet[ss],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
      ierr = PetscFree(side_set_points[ss]);CHKERRQ(ierr);
    }
    //ierr = SectionIntComplete(faceParentSet);CHKERRQ(ierr);
    if (debug) {ierr = SectionIntView(faceParentSet,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}

    for(int ss = 0; ss < num_side_sets; ++ss) {
      ierr = PetscFree2(side_set_elem_list[ss],side_set_side_list[ss]);CHKERRQ(ierr);
    }
    ierr = PetscFree4(ss_ids,num_sides_in_set,side_set_elem_list,side_set_side_list);CHKERRQ(ierr);

    ierr = PetscFree2(faceSet,side_set_points);CHKERRQ(ierr);
  }

  // Free element block temporary variables
  if (num_elem_blk > 0) {
    for (int eb = 0; eb < num_elem_blk; ++eb) {
      ierr = PetscFree2(block_name[eb],block_elemtype[eb]);CHKERRQ(ierr);
    }
    ierr = PetscFree6(eb_ids, num_elem_in_block, num_nodes_per_elem, num_attr, block_name, block_elemtype);CHKERRQ(ierr);
  }
  ierr = PetscFree2(cells,connectivity_table);CHKERRQ(ierr);

  //if (debug) {cellBlocks->view("Cell Blocks");}
  //if (debug) {vertexSets->view("Vertex Sets");}
  if (debug) {mesh->view("Mesh");}
  if (debug) {boundarymesh->view("Boundary Mesh");}
  PetscFunctionReturn(0);
}
