static char help[] = "This example demonstrates the use of DMNetwork interface for solving a simple electric circuit. \n\
                      The example can be found in p.150 of 'Strang, Gilbert. Computational Science and Engineering. Wellesley, MA'.\n\n";

/* T
  Concepts: DMNetwork
  Concepts: KSP
*/

#include <petscdmnetwork.h>
#include <petscksp.h>

/* The topology looks like:

            (0)
            /|\
           / | \
          /  |  \
         R   R   V
        /    |b3  \
    b0 /    (3)    \ b1
      /    /   \    R
     /   R       R   \
    /  /           \  \
   / / b4        b5  \ \
  //                   \\
(1)--------- R -------- (2)
             b2

  Nodes:          (0), ... (3)
  Branches:       b0, ... b5
  Resistances:    R
  Voltage source: V

  Additionally, there is a current source from (1) to (0).
*/

/*
  Structures containing physical data of circuit.
  Note that no topology is defined
*/

typedef struct {
  PetscInt    id;  /* node id */
  PetscScalar inj; /* current injection (A) */
  PetscBool   gr;  /* boundary node */
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

  CHKERRQ(PetscCalloc2(nnode,&node,nbranch,&branch));

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
    edgelist is an array of len = 2*nbranch that defines the
    topology of the network. For branch i we have:
      edgelist[2*i]     = from node
      edgelist[2*i + 1] = to node.
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

PetscErrorCode FormOperator(DM dmnetwork,Mat A,Vec b)
{
  Branch            *branch;
  Node              *node;
  PetscInt          e,v,vStart,vEnd,eStart, eEnd;
  PetscInt          lofst,lofst_to,lofst_fr,row[2],col[6];
  PetscBool         ghost;
  const PetscInt    *cone;
  PetscScalar       *barr,val[6];

  PetscFunctionBegin;
  CHKERRQ(MatZeroEntries(A));

  CHKERRQ(VecSet(b,0.0));
  CHKERRQ(VecGetArray(b,&barr));

  /*
    We define the current i as an "edge characteristic" and the voltage v as a "vertex characteristic".
    We iterate the list of edges and vertices, query the associated voltages and currents
    and use them to write the Kirchoff equations:

    Branch equations: i/r + v_to - v_from     = v_source (battery)
    Node equations:   sum(i_to) - sum(i_from) = i_source
   */
  CHKERRQ(DMNetworkGetEdgeRange(dmnetwork,&eStart,&eEnd));
  for (e = 0; e < eEnd; e++) {
    CHKERRQ(DMNetworkGetComponent(dmnetwork,e,0,NULL,(void**)&branch,NULL));
    CHKERRQ(DMNetworkGetLocalVecOffset(dmnetwork,e,ALL_COMPONENTS,&lofst));

    CHKERRQ(DMNetworkGetConnectedVertices(dmnetwork,e,&cone));
    CHKERRQ(DMNetworkGetLocalVecOffset(dmnetwork,cone[0],ALL_COMPONENTS,&lofst_fr));
    CHKERRQ(DMNetworkGetLocalVecOffset(dmnetwork,cone[1],ALL_COMPONENTS,&lofst_to));

    /* set rhs b for Branch equation */
    barr[lofst] = branch->bat;

    /* set Branch equation */
    row[0] = lofst;
    col[0] = lofst;     val[0] =  1./branch->r;
    col[1] = lofst_to;  val[1] =  1;
    col[2] = lofst_fr;  val[2] = -1;
    CHKERRQ(MatSetValuesLocal(A,1,row,3,col,val,ADD_VALUES));

    /* set Node equation */
    CHKERRQ(DMNetworkGetComponent(dmnetwork,cone[0],0,NULL,(void**)&node,NULL));

    /* from node */
    if (!node->gr) { /* not a boundary node */
      row[0] = lofst_fr;
      col[0] = lofst;   val[0] = -1;
      CHKERRQ(MatSetValuesLocal(A,1,row,1,col,val,ADD_VALUES));
    }

    /* to node */
    CHKERRQ(DMNetworkGetComponent(dmnetwork,cone[1],0,NULL,(void**)&node,NULL));

    if (!node->gr) { /* not a boundary node */
      row[0] = lofst_to;
      col[0] = lofst;   val[0] = 1;
      CHKERRQ(MatSetValuesLocal(A,1,row,1,col,val,ADD_VALUES));
    }
  }

  /* set rhs b for Node equation */
  CHKERRQ(DMNetworkGetVertexRange(dmnetwork,&vStart,&vEnd));
  for (v = vStart; v < vEnd; v++) {
    CHKERRQ(DMNetworkIsGhostVertex(dmnetwork,v,&ghost));
    if (!ghost) {
      CHKERRQ(DMNetworkGetComponent(dmnetwork,v,0,NULL,(void**)&node,NULL));
      CHKERRQ(DMNetworkGetLocalVecOffset(dmnetwork,v,ALL_COMPONENTS,&lofst));

      if (node->gr) { /* a boundary node */
        row[0] = lofst;
        col[0] = lofst;   val[0] = 1;
        CHKERRQ(MatSetValuesLocal(A,1,row,1,col,val,ADD_VALUES));
      } else {       /* not a boundary node */
        barr[lofst] += node->inj;
      }
    }
  }

  CHKERRQ(VecRestoreArray(b,&barr));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char ** argv)
{
  PetscInt          i, nnode = 0, nbranch = 0, eStart, eEnd, vStart, vEnd;
  PetscMPIInt       size, rank;
  DM                dmnetwork;
  Vec               x, b;
  Mat               A;
  KSP               ksp;
  PetscInt          *edgelist = NULL;
  PetscInt          componentkey[2];
  Node              *node;
  Branch            *branch;
  PetscInt          nE[1];

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* "Read" data only for processor 0 */
  if (rank == 0) {
    CHKERRQ(read_data(&nnode, &nbranch, &node, &branch, &edgelist));
  }

  CHKERRQ(DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork));
  CHKERRQ(DMNetworkRegisterComponent(dmnetwork,"nstr",sizeof(Node),&componentkey[0]));
  CHKERRQ(DMNetworkRegisterComponent(dmnetwork,"bsrt",sizeof(Branch),&componentkey[1]));

  /* Set local number of nodes/edges, add edge connectivity */
  nE[0] = nbranch;
  CHKERRQ(DMNetworkSetNumSubNetworks(dmnetwork,PETSC_DECIDE,1));
  CHKERRQ(DMNetworkAddSubnetwork(dmnetwork,"",nE[0],edgelist,NULL));

  /* Set up the network layout */
  CHKERRQ(DMNetworkLayoutSetUp(dmnetwork));

  /* Add network components (physical parameters of nodes and branches) and num of variables */
  if (rank == 0) {
    CHKERRQ(DMNetworkGetEdgeRange(dmnetwork,&eStart,&eEnd));
    for (i = eStart; i < eEnd; i++) {
      CHKERRQ(DMNetworkAddComponent(dmnetwork,i,componentkey[1],&branch[i-eStart],1));
    }

    CHKERRQ(DMNetworkGetVertexRange(dmnetwork,&vStart,&vEnd));
    for (i = vStart; i < vEnd; i++) {
      CHKERRQ(DMNetworkAddComponent(dmnetwork,i,componentkey[0],&node[i-vStart],1));
    }
  }

  /* Network partitioning and distribution of data */
  CHKERRQ(DMSetUp(dmnetwork));
  CHKERRQ(DMNetworkDistribute(&dmnetwork,0));

  /* We do not use these data structures anymore since they have been copied to dmnetwork */
  if (rank == 0) {
    CHKERRQ(PetscFree(edgelist));
    CHKERRQ(PetscFree2(node,branch));
  }

  /* Create vectors and matrix */
  CHKERRQ(DMCreateGlobalVector(dmnetwork,&x));
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(DMCreateMatrix(dmnetwork,&A));

  /* Assembly system of equations */
  CHKERRQ(FormOperator(dmnetwork,A,b));

  /* Solve linear system: A x = b */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ksp));
  CHKERRQ(KSPSetOperators(ksp, A, A));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp, b, x));
  CHKERRQ(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  /* Free work space */
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(DMDestroy(&dmnetwork));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex double defined(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      args: -ksp_monitor_short

   test:
      suffix: 2
      nsize: 2
      args: -petscpartitioner_type simple -ksp_converged_reason

TEST*/
