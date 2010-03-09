static char help[] = "Test distribution of properties using a mesh\n\n";

#include <stdio.h>
#include <stdlib.h>
#include <petscsys.h>
#include <petscmesh.hh>
#include <sieve/Selection.hh>
#include <exodusII.h>

PetscErrorCode MyPetscReadExodusII(MPI_Comm comm, const char filename[], ALE::Obj<PETSC_MESH_TYPE>& mesh);
PetscErrorCode MeshCreateExodus(MPI_Comm comm, const char filename[], Mesh *mesh);

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc, char ** argv) {
  Mesh           mesh;
  PetscTruth     flag;
  char           filename[PETSC_MAX_PATH_LEN+1];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL, "-f", filename, PETSC_MAX_PATH_LEN, &flag);CHKERRQ(ierr);
  if (flag) {
    try {
      ierr = MeshCreateExodus(PETSC_COMM_WORLD, filename, &mesh);CHKERRQ(ierr);
      ierr = MeshView(mesh, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    } catch(ALE::Exception e) {
      std::cerr << "Error: " << e << std::endl;
    }
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;

}

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
  PetscInt       debug = 1;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  ALE::Obj<PETSC_MESH_TYPE> m = new PETSC_MESH_TYPE(comm, -1, debug);
#ifdef PETSC_HAVE_EXODUSII
  ierr = MyPetscReadExodusII(comm, filename, m);CHKERRQ(ierr);
#else
  SETERRQ(PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --with-exodusii-dir=/path/to/exodus");
#endif
  if (debug) {m->view("Mesh");}
  ierr = MeshSetMesh(*mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MyPetscReadExodusII"
PetscErrorCode MyPetscReadExodusII(MPI_Comm comm, const char filename[], ALE::Obj<PETSC_MESH_TYPE>& mesh)
{
  typedef std::set<ALE::Mesh::point_type> PointSet;
  ALE::Obj<ALE::Mesh> boundarymesh; 
  const PetscMPIInt   rank          = mesh->commRank();
  int                 CPU_word_size = 0;
  int                 IO_word_size  = 0;
  PetscTruth          interpolate   = PETSC_FALSE;
  int                 exoid;
  char                title[MAX_LINE_LENGTH+1], elem_type[MAX_STR_LENGTH+1];
  float               version;
  int                 num_dim, num_nodes, num_elem, num_elem_blk, num_node_sets, num_side_sets;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetTruth(PETSC_NULL, "-interpolate", &interpolate, PETSC_NULL);CHKERRQ(ierr);
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
  char **block_names, **block_elem_sig, elem_type_sig[4];
  PetscTruth is_known_elem_type=PETSC_FALSE;
  if (num_elem_blk > 0) {
    ierr = PetscMalloc6(num_elem_blk,int,&eb_ids, num_elem_blk,int,&num_elem_in_block, num_elem_blk,int,&num_nodes_per_elem, num_elem_blk,int,&num_attr, num_elem_blk,char*,&block_names, num_elem_blk,char*,&block_elem_sig);CHKERRQ(ierr);
    ierr = ex_get_elem_blk_ids(exoid, eb_ids);CHKERRQ(ierr);
    for(int eb = 0; eb < num_elem_blk; ++eb) {
//      ierr = PetscMalloc((MAX_STR_LENGTH+1) * sizeof(char), &block_names[eb]);CHKERRQ(ierr);
//      ierr = PetscMalloc(4 * sizeof(char), &block_elem_sig[eb]);CHKERRQ(ierr);
      ierr = PetscMalloc2(MAX_STR_LENGTH+1,char,&block_names[eb], 4, char, &block_elem_sig[eb]);CHKERRQ(ierr);
    }
    ierr = ex_get_names(exoid, EX_ELEM_BLOCK, block_names);CHKERRQ(ierr);
    for(int eb = 0; eb < num_elem_blk; ++eb) {
      ierr = ex_get_elem_block(exoid, eb_ids[eb], elem_type, &num_elem_in_block[eb], &num_nodes_per_elem[eb], &num_attr[eb]);CHKERRQ(ierr);
      ierr = PetscFree(block_names[eb]);CHKERRQ(ierr);
      // Do some validation
      ierr = PetscStrncpy(block_elem_sig[eb],elem_type,3);CHKERRQ(ierr);
      ierr = PetscStrtolower(block_elem_sig[eb]);CHKERRQ(ierr);
      ierr = PetscStrcmp(block_elem_sig[eb], "tri", &is_known_elem_type);CHKERRQ(ierr);
      if (!is_known_elem_type){
        ierr = PetscStrcasecmp(block_elem_sig[eb], "tet", &is_known_elem_type);CHKERRQ(ierr);
      }
      if (!is_known_elem_type){
        ierr = PetscStrcasecmp(block_elem_sig[eb], "qua", &is_known_elem_type);CHKERRQ(ierr);
      }
      if (!is_known_elem_type){
        ierr = PetscStrcasecmp(block_elem_sig[eb], "she", &is_known_elem_type);CHKERRQ(ierr);
      }
      if (!is_known_elem_type){
        ierr = PetscStrcasecmp(block_elem_sig[eb], "hex", &is_known_elem_type);CHKERRQ(ierr);
      }
      if (!is_known_elem_type){
        SETERRQ3(PETSC_ERR_SUP, "Unsupported element type: %s / %s in block %i\n", block_elem_sig[eb], elem_type, eb);
      }
      // I could simply search block_elem_sig[eb] in "tri|tet|qua|she|hex"
    }
    ierr = PetscMalloc(num_elem_blk * sizeof(int*),&connect);CHKERRQ(ierr);
    for(int eb = 0; eb < num_elem_blk; ++eb) {
      if (num_elem_in_block[eb] > 0) {
        ierr = PetscMalloc(num_nodes_per_elem[eb]*num_elem_in_block[eb] * sizeof(int),&connect[eb]);CHKERRQ(ierr);
        ierr = ex_get_elem_conn(exoid, eb_ids[eb], connect[eb]);CHKERRQ(ierr);
      }
    }
  }
  
  // Read side sets
  int  *ss_ids, *num_sides_in_set;
  int **side_set_elem_list, **side_set_side_list;
  if (num_side_sets > 0) {
    ierr = PetscMalloc4(num_side_sets,int,&ss_ids, num_side_sets,int,&num_sides_in_set, num_side_sets,int*,&side_set_elem_list, num_side_sets,int*,&side_set_side_list);CHKERRQ(ierr);
    ierr = ex_get_side_set_ids(exoid, ss_ids);CHKERRQ(ierr);
    for(int ss = 0; ss< num_side_sets; ++ss) {
      int num_df_in_sset;
      ierr = ex_get_side_set_param(exoid, ss_ids[ss], &num_sides_in_set[ss], &num_df_in_sset);
      ierr = PetscMalloc2(num_sides_in_set[ss],int,&side_set_elem_list[ss], num_sides_in_set[ss],int, &side_set_side_list[ss]);CHKERRQ(ierr);
      ierr = ex_get_side_set(exoid, ss_ids[ss], side_set_elem_list[ss], side_set_side_list[ss]);
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
  int *cells;
  int **connectivity_table;
  int num_local_corners = 0;
  for(int eb=0; eb < num_elem_blk; ++eb) {
    num_local_corners += num_nodes_per_elem[eb] * num_elem_in_block[eb];
  }
  mesh->setDimension(num_dim);
  ierr = PetscMalloc2(num_local_corners,int,&cells, num_elem, int*, &connectivity_table);CHKERRQ(ierr); 
  for(int eb = 0, k = 0; eb < num_elem_blk; ++eb) {
    for(int e = 0; e < num_elem_in_block[eb]; ++e, ++k) {
      for(int c = 0; c < num_nodes_per_elem[eb]; ++c) {
        cells[k*num_nodes_per_elem[eb]+c] = connect[eb][e*num_nodes_per_elem[eb]+c] - 1;
      }
      connectivity_table[k] = &cells[k*num_nodes_per_elem[eb]];
    }
  ierr = PetscFree(connect[eb]);CHKERRQ(ierr);
  }


  ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(mesh->comm(), mesh->debug());
  ALE::Obj<ALE::Mesh>                   m     = new ALE::Mesh(mesh->comm(), mesh->debug());
  ALE::Obj<ALE::Mesh::sieve_type>       s     = new ALE::Mesh::sieve_type(mesh->comm(), mesh->debug());

  // Here we assume that num_nodes_per_elem is constant accross blocks (i.e.) 
  // that all elements in the mesh are of the same type
  int  numCorners = num_nodes_per_elem[0];
  ALE::SieveBuilder<ALE::Mesh>::buildTopology(s, num_dim, num_elem, cells, num_nodes, interpolate, numCorners, PETSC_DECIDE, m->getArrowSection("orientation"));
  
  std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> renumbering;
  ALE::ISieveConverter::convertSieve(*s, *sieve, renumbering, false);
  mesh->setSieve(sieve);
  mesh->stratify();
  ALE::ISieveConverter::convertOrientation(*s, *sieve, renumbering, m->getArrowSection("orientation").ptr());

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
  }

  /*
  Do not free yet, I need num_elem_in_block to initialize the side sets
  if (num_elem_blk > 0) {
    ierr = PetscFree5(eb_ids, num_elem_in_block, num_nodes_per_elem, num_attr, block_names);CHKERRQ(ierr);
  }
  */

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

  // Build side sets
  if (num_side_sets > 0){
    ierr = PetscPrintf(mesh->comm(), "Building boundary\n");CHKERRQ(ierr);
    // Build the boundary mesh
    boundarymesh = ALE::Selection<PETSC_MESH_TYPE>::boundaryV(mesh);
//    const ALE::Obj<PETSC_MESH_TYPE::label_type>& sideSets = boundarymesh->createLabel("SideSets");

    /* 
      Find the parent block of each element 
    */
    if (rank == 0) {
      int *parent_block;
      PetscTruth flag1, flag2;
      ierr = PetscMalloc(num_elem*sizeof(int), &parent_block);CHKERRQ(ierr);
      for(int eb = 0, k = 0; eb < num_elem_blk; ++eb) {
        for(int e = 0; e < num_elem_in_block[eb]; ++e, ++k) {
          parent_block[k] = eb;
        }
      }

      ALE::Obj<PointSet> face = new PointSet();
      int facepoint;
      int num_side_corners, *side_corners;
      for (int ss=0; ss < num_side_sets; ++ss){
        for (int s = 0; s < num_sides_in_set[ss]; ++s){
          int e = side_set_elem_list[ss][s];
          face->clear();
          /* 
            There is currently no easy way to recover the point number of a vertex if there are more 
            than one type of elements present in the mesh.
            The following will need to be adapted when we can deal with that 
            hint: the connectivity table "cells" will have to be a 2 dimensional array, 
                  or we will need to play with pointer arithmetic
          */
          /* 
<<<<<<< local
            the numberic scheme for vertices, faces and edges in EXO has been designed by a maniac
=======
            the numbering scheme for vertices, faces and edges in EXO has been designed by a maniac
>>>>>>> other
            cf. Figure 5 in exodus documentation or 
            https://redmine.schur.math.lsu.edu/attachments/36/Exodus_Sides_Ordering.png
          */
          // TRI
          ierr = PetscStrcasecmp(block_elem_sig[parent_block[e-1]], "tri", &flag1);CHKERRQ(ierr);
          if (flag1){
            switch (side_set_side_list[ss][s]){
              case 1:
                face->insert(face->end(), connectivity_table[e-1][0]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][1]+num_elem-1); 
                break;
              case 2:
                face->insert(face->end(), connectivity_table[e-1][1]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][2]+num_elem-1); 
                break;
              case 3:
                face->insert(face->end(), connectivity_table[e-1][2]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][0]+num_elem-1); 
                break;
            }
          }

          // TET
          ierr = PetscStrcasecmp(block_elem_sig[parent_block[side_set_elem_list[ss][s]-1]], "tet", &flag1);CHKERRQ(ierr);
          if (flag1){
            switch (side_set_side_list[ss][s]){
              case 1:
                face->insert(face->end(), connectivity_table[e-1][0]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][1]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][3]+num_elem-1);
                break;
              case 2:
                face->insert(face->end(), connectivity_table[e-1][1]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][2]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][3]+num_elem-1); 
                break;
              case 3:
                face->insert(face->end(), connectivity_table[e-1][2]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][0]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][3]+num_elem-1); 
                break;
              case 4:
                face->insert(face->end(), connectivity_table[e-1][0]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][2]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][1]+num_elem-1); 
                break;
            }
          }
          
          // QUAD
          ierr = PetscStrcasecmp(block_elem_sig[parent_block[side_set_elem_list[ss][s]-1]], "qua", &flag1);CHKERRQ(ierr);
          ierr = PetscStrcasecmp(block_elem_sig[parent_block[side_set_elem_list[ss][s]-1]], "she", &flag2);CHKERRQ(ierr);
          if (flag1 || flag2){
            switch (side_set_side_list[ss][s]){
              case 1:
                face->insert(face->end(), connectivity_table[e-1][0]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][1]+num_elem-1);
                break;
              case 2:
                face->insert(face->end(), connectivity_table[e-1][1]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][2]+num_elem-1);
                break;
              case 3:
                face->insert(face->end(), connectivity_table[e-1][2]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][3]+num_elem-1);
                break;
              case 4:
                face->insert(face->end(), connectivity_table[e-1][3]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][0]+num_elem-1);
                break;
            }
          }
          
          //HEX
          ierr = PetscStrcasecmp(block_elem_sig[parent_block[side_set_elem_list[ss][s]-1]], "hex", &flag1);CHKERRQ(ierr);
          if (flag1){
            switch (side_set_side_list[ss][s]){
              case 1:
                face->insert(face->end(), connectivity_table[e-1][0]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][1]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][5]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][4]+num_elem-1);
                break;
              case 2:
                face->insert(face->end(), connectivity_table[e-1][1]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][2]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][6]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][5]+num_elem-1);
                break;
              case 3:
                face->insert(face->end(), connectivity_table[e-1][2]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][3]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][7]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][6]+num_elem-1);
                break;
              case 4:
                face->insert(face->end(), connectivity_table[e-1][3]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][0]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][4]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][7]+num_elem-1);
                break;
              case 5:
                face->insert(face->end(), connectivity_table[e-1][3]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][2]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][1]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][0]+num_elem-1);
                break;
              case 6:
                face->insert(face->end(), connectivity_table[e-1][4]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][5]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][6]+num_elem-1); 
                face->insert(face->end(), connectivity_table[e-1][7]+num_elem-1);
                break;
            }
          }
//          facepoint = boundarymesh->nJoin1(face);
          // Do a join to get the point number of the edge with vertices side_corners
          // Add the edge
        }
      }
      ierr = PetscFree(parent_block);CHKERRQ(ierr);
    }
  }
  boundarymesh->view("Boundary Mesh");

  // Free element block temporary variables
  if (num_elem_blk > 0) {
    ierr = PetscFree6(eb_ids, num_elem_in_block, num_nodes_per_elem, num_attr, block_names, block_elem_sig);CHKERRQ(ierr);
  }
  ierr = PetscFree(connectivity_table);CHKERRQ(ierr);
  ierr = PetscFree(cells);CHKERRQ(ierr);
  
  //cellBlocks->view("Cell Blocks");
  //vertexSets->view("Vertex Sets");
  PetscFunctionReturn(0);
}
