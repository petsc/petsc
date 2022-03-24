static char help[] = "This example is based on ex1 using MATNEST. \n";

/* T
  Concepts: DMNetwork
  Concepts: KSP
*/

#include <petscdmnetwork.h>
#include <petscksp.h>

/* The topology looks like:

            (1)
            /|\
           / | \
          /  |  \
         R   R   V
        /    |b4  \
    b1 /    (4)    \ b2
      /    /   \    R
     /   R       R   \
    /  /           \  \
   / / b5        b6  \ \
  //                   \\
(2)--------- R -------- (3)
             b3

  Nodes:          (1), ... (4)
  Branches:       b1, ... b6
  Resistances:    R
  Voltage source: V

  Additionally, there is a current source from (2) to (1).
*/

/*
  Structures containing physical data of circuit.
  Note that no topology is defined
*/

typedef struct {
  PetscInt    id;  /* node id */
  PetscScalar inj; /* current injection (A) */
  PetscBool   gr;  /* grounded ? */
} Node;

typedef struct {
  PetscInt    id;  /* branch id */
  PetscScalar r;   /* resistance (ohms) */
  PetscScalar bat; /* battery (V) */
} Branch;

/*
  read_data: this routine fills data structures with problem data.
  This can be substituted by an external parser.
*/

PetscErrorCode read_data(PetscInt *pnnode,PetscInt *pnbranch,Node **pnode,Branch **pbranch,PetscInt **pedgelist)
{
  PetscInt          nnode, nbranch, i;
  Branch            *branch;
  Node              *node;
  PetscInt          *edgelist;

  PetscFunctionBeginUser;
  nnode   = 4;
  nbranch = 6;

  CHKERRQ(PetscCalloc1(nnode,&node));
  CHKERRQ(PetscCalloc1(nbranch,&branch));

  for (i = 0; i < nnode; i++) {
    node[i].id  = i;
    node[i].inj = 0;
    node[i].gr = PETSC_FALSE;
  }

  for (i = 0; i < nbranch; i++) {
    branch[i].id  = i;
    branch[i].r   = 1.0;
    branch[i].bat = 0;
  }

  /*
    Branch 1 contains a voltage source of 12.0 V
    From node 0 to 1 there exists a current source of 4.0 A
    Node 3 is grounded, hence the voltage is 0.
  */
  branch[1].bat = 12.0;
  node[0].inj   = -4.0;
  node[1].inj   =  4.0;
  node[3].gr    =  PETSC_TRUE;

  /*
    edgelist is an array of len = 2*nbranch. that defines the
    topology of the network. For branch i we would have that:
      edgelist[2*i]     = from node
      edgelist[2*i + 1] = to node
  */

  CHKERRQ(PetscCalloc1(2*nbranch, &edgelist));

  for (i = 0; i < nbranch; i++) {
    switch (i) {
      case 0:
        edgelist[2*i]     = 0;
        edgelist[2*i + 1] = 1;
        break;
      case 1:
        edgelist[2*i]     = 0;
        edgelist[2*i + 1] = 2;
        break;
      case 2:
        edgelist[2*i]     = 1;
        edgelist[2*i + 1] = 2;
        break;
      case 3:
        edgelist[2*i]     = 0;
        edgelist[2*i + 1] = 3;
        break;
      case 4:
        edgelist[2*i]     = 1;
        edgelist[2*i + 1] = 3;
        break;
      case 5:
        edgelist[2*i]     = 2;
        edgelist[2*i + 1] = 3;
        break;
      default:
        break;
    }
  }

  /* assign pointers */
  *pnnode    = nnode;
  *pnbranch  = nbranch;
  *pedgelist = edgelist;
  *pbranch   = branch;
  *pnode     = node;
  PetscFunctionReturn(0);
}

PetscErrorCode FormOperator(DM networkdm,Mat A,Vec b)
{
  Vec               localb;
  Branch            *branch;
  Node              *node;
  PetscInt          e;
  PetscInt          v,vStart,vEnd;
  PetscInt          eStart, eEnd;
  PetscBool         ghost;
  const PetscInt    *cone;
  PetscScalar       *barr;
  PetscInt          lofst, lofst_to, lofst_fr;
  PetscInt          key;
  PetscInt          row[2],col[6];
  PetscScalar       val[6];
  Mat               e11, c12, c21, v22;
  Mat               **mats;

  PetscFunctionBegin;
  CHKERRQ(DMGetLocalVector(networkdm,&localb));
  CHKERRQ(VecSet(b,0.0));
  CHKERRQ(VecSet(localb,0.0));

  CHKERRQ(VecGetArray(localb,&barr));

  /* Get nested submatrices */
  CHKERRQ(MatNestGetSubMats(A,NULL,NULL,&mats));
  e11 = mats[0][0];  /* edges */
  c12 = mats[0][1];  /* couplings */
  c21 = mats[1][0];  /* couplings */
  v22 = mats[1][1];  /* vertices */

  /* Get vertices and edge range */
  CHKERRQ(DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd));
  CHKERRQ(DMNetworkGetVertexRange(networkdm,&vStart,&vEnd));

  for (e = 0; e < eEnd; e++) {
    CHKERRQ(DMNetworkGetComponent(networkdm,e,0,&key,(void**)&branch,NULL));
    CHKERRQ(DMNetworkGetEdgeOffset(networkdm,e,&lofst));

    CHKERRQ(DMNetworkGetConnectedVertices(networkdm,e,&cone));
    CHKERRQ(DMNetworkGetVertexOffset(networkdm,cone[0],&lofst_fr));
    CHKERRQ(DMNetworkGetVertexOffset(networkdm,cone[1],&lofst_to));

    /* These are edge-edge and go to e11 */
    row[0] = lofst;
    col[0] = lofst;     val[0] =  1;
    CHKERRQ(MatSetValuesLocal(e11,1,row,1,col,val,INSERT_VALUES));

    /* These are edge-vertex and go to c12 */
    col[0] = lofst_to;  val[0] =  1;
    col[1] = lofst_fr;  val[1] = -1;
    CHKERRQ(MatSetValuesLocal(c12,1,row,2,col,val,INSERT_VALUES));

    /* These are edge-vertex and go to c12 */
    /* from node */
    CHKERRQ(DMNetworkGetComponent(networkdm,cone[0],0,&key,(void**)&node,NULL));

    if (!node->gr) {
      row[0] = lofst_fr;
      col[0] = lofst;   val[0] =  1;
      CHKERRQ(MatSetValuesLocal(c21,1,row,1,col,val,INSERT_VALUES));
    }

    /* to node */
    CHKERRQ(DMNetworkGetComponent(networkdm,cone[1],0,&key,(void**)&node,NULL));

    if (!node->gr) {
      row[0] = lofst_to;
      col[0] = lofst;   val[0] =  -1;
      CHKERRQ(MatSetValuesLocal(c21,1,row,1,col,val,INSERT_VALUES));
    }

    /* TODO: this is not a nested vector. Need to implement nested vector */
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,e,ALL_COMPONENTS,&lofst));
    barr[lofst] = branch->bat;
  }

  for (v = vStart; v < vEnd; v++) {
    CHKERRQ(DMNetworkIsGhostVertex(networkdm,v,&ghost));
    if (!ghost) {
      CHKERRQ(DMNetworkGetComponent(networkdm,v,0,&key,(void**)&node,NULL));
      CHKERRQ(DMNetworkGetVertexOffset(networkdm,v,&lofst));

      if (node->gr) {
        row[0] = lofst;
        col[0] = lofst;   val[0] =  1;
        CHKERRQ(MatSetValuesLocal(v22,1,row,1,col,val,INSERT_VALUES));
      } else {
        /* TODO: this is not a nested vector. Need to implement nested vector */
        CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,v,ALL_COMPONENTS,&lofst));
        barr[lofst] -= node->inj;
      }
    }
  }

  CHKERRQ(VecRestoreArray(localb,&barr));

  CHKERRQ(DMLocalToGlobalBegin(networkdm,localb,ADD_VALUES,b));
  CHKERRQ(DMLocalToGlobalEnd(networkdm,localb,ADD_VALUES,b));
  CHKERRQ(DMRestoreLocalVector(networkdm,&localb));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char ** argv)
{
  PetscInt          i, nnode = 0, nbranch = 0;
  PetscInt          eStart, eEnd, vStart, vEnd;
  PetscMPIInt       size, rank;
  DM                networkdm;
  Vec               x, b;
  Mat               A;
  KSP               ksp;
  PetscInt          *edgelist = NULL;
  PetscInt          componentkey[2];
  Node              *node;
  Branch            *branch;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* "read" data only for processor 0 */
  if (rank == 0) {
    CHKERRQ(read_data(&nnode, &nbranch, &node, &branch, &edgelist));
  }

  CHKERRQ(DMNetworkCreate(PETSC_COMM_WORLD,&networkdm));
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"nstr",sizeof(Node),&componentkey[0]));
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"bsrt",sizeof(Branch),&componentkey[1]));

  /* Set number of nodes/edges, add edge connectivity */
  CHKERRQ(DMNetworkSetNumSubNetworks(networkdm,PETSC_DECIDE,1));
  CHKERRQ(DMNetworkAddSubnetwork(networkdm,"",nbranch,edgelist,NULL));

  /* Set up the network layout */
  CHKERRQ(DMNetworkLayoutSetUp(networkdm));

  /* Add network components (physical parameters of nodes and branches) and num of variables */
  if (rank == 0) {
    CHKERRQ(DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd));
    for (i = eStart; i < eEnd; i++) {
      CHKERRQ(DMNetworkAddComponent(networkdm,i,componentkey[1],&branch[i-eStart],1));
    }

    CHKERRQ(DMNetworkGetVertexRange(networkdm,&vStart,&vEnd));
    for (i = vStart; i < vEnd; i++) {
      CHKERRQ(DMNetworkAddComponent(networkdm,i,componentkey[0],&node[i-vStart],1));
    }
  }

  /* Network partitioning and distribution of data */
  CHKERRQ(DMSetUp(networkdm));
  CHKERRQ(DMNetworkDistribute(&networkdm,0));

  CHKERRQ(DMNetworkAssembleGraphStructures(networkdm));

  /* We don't use these data structures anymore since they have been copied to networkdm */
  if (rank == 0) {
    CHKERRQ(PetscFree(edgelist));
    CHKERRQ(PetscFree(node));
    CHKERRQ(PetscFree(branch));
  }

  CHKERRQ(DMCreateGlobalVector(networkdm,&x));
  CHKERRQ(VecDuplicate(x,&b));

  CHKERRQ(DMSetMatType(networkdm, MATNEST));
  CHKERRQ(DMCreateMatrix(networkdm,&A));

  /* Assembly system of equations */
  CHKERRQ(FormOperator(networkdm,A,b));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ksp));
  CHKERRQ(KSPSetOperators(ksp, A, A));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp, b, x));
  CHKERRQ(VecView(x, 0));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(DMDestroy(&networkdm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex double defined(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      args: -ksp_converged_reason

   test:
      suffix: 2
      nsize: 2
      args: -petscpartitioner_type simple -ksp_converged_reason

TEST*/
