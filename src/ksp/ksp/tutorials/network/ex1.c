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
  PetscErrorCode    ierr;
  PetscInt          nnode, nbranch, i;
  Branch            *branch;
  Node              *node;
  PetscInt          *edgelist;

  PetscFunctionBeginUser;
  nnode   = 4;
  nbranch = 6;

  ierr = PetscCalloc2(nnode,&node,nbranch,&branch);CHKERRQ(ierr);

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
  ierr = PetscCalloc1(2*nbranch, &edgelist);CHKERRQ(ierr);

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
  PetscErrorCode    ierr;
  Branch            *branch;
  Node              *node;
  PetscInt          e,v,vStart,vEnd,eStart, eEnd;
  PetscInt          lofst,lofst_to,lofst_fr,row[2],col[6];
  PetscBool         ghost;
  const PetscInt    *cone;
  PetscScalar       *barr,val[6];

  PetscFunctionBegin;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);

  ierr = VecSet(b,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(b,&barr);CHKERRQ(ierr);

  /*
    We define the current i as an "edge characteristic" and the voltage v as a "vertex characteristic".
    We iterate the list of edges and vertices, query the associated voltages and currents
    and use them to write the Kirchoff equations:

    Branch equations: i/r + v_to - v_from     = v_source (battery)
    Node equations:   sum(i_to) - sum(i_from) = i_source
   */
  ierr = DMNetworkGetEdgeRange(dmnetwork,&eStart,&eEnd);CHKERRQ(ierr);
  for (e = 0; e < eEnd; e++) {
    ierr = DMNetworkGetComponent(dmnetwork,e,0,NULL,(void**)&branch,NULL);CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(dmnetwork,e,ALL_COMPONENTS,&lofst);CHKERRQ(ierr);

    ierr = DMNetworkGetConnectedVertices(dmnetwork,e,&cone);CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(dmnetwork,cone[0],ALL_COMPONENTS,&lofst_fr);CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(dmnetwork,cone[1],ALL_COMPONENTS,&lofst_to);CHKERRQ(ierr);

    /* set rhs b for Branch equation */
    barr[lofst] = branch->bat;

    /* set Branch equation */
    row[0] = lofst;
    col[0] = lofst;     val[0] =  1./branch->r;
    col[1] = lofst_to;  val[1] =  1;
    col[2] = lofst_fr;  val[2] = -1;
    ierr = MatSetValuesLocal(A,1,row,3,col,val,ADD_VALUES);CHKERRQ(ierr);

    /* set Node equation */
    ierr = DMNetworkGetComponent(dmnetwork,cone[0],0,NULL,(void**)&node,NULL);CHKERRQ(ierr);

    /* from node */
    if (!node->gr) { /* not a boundary node */
      row[0] = lofst_fr;
      col[0] = lofst;   val[0] = -1;
      ierr = MatSetValuesLocal(A,1,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
    }

    /* to node */
    ierr = DMNetworkGetComponent(dmnetwork,cone[1],0,NULL,(void**)&node,NULL);CHKERRQ(ierr);

    if (!node->gr) { /* not a boundary node */
      row[0] = lofst_to;
      col[0] = lofst;   val[0] = 1;
      ierr = MatSetValuesLocal(A,1,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  /* set rhs b for Node equation */
  ierr = DMNetworkGetVertexRange(dmnetwork,&vStart,&vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; v++) {
    ierr = DMNetworkIsGhostVertex(dmnetwork,v,&ghost);CHKERRQ(ierr);
    if (!ghost) {
      ierr = DMNetworkGetComponent(dmnetwork,v,0,NULL,(void**)&node,NULL);CHKERRQ(ierr);
      ierr = DMNetworkGetLocalVecOffset(dmnetwork,v,ALL_COMPONENTS,&lofst);CHKERRQ(ierr);

      if (node->gr) { /* a boundary node */
        row[0] = lofst;
        col[0] = lofst;   val[0] = 1;
        ierr = MatSetValuesLocal(A,1,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
      } else {       /* not a boundary node */
        barr[lofst] += node->inj;
      }
    }
  }

  ierr = VecRestoreArray(b,&barr);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char ** argv)
{
  PetscErrorCode    ierr;
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
  PetscInt          nV[1],nE[1];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* "Read" data only for processor 0 */
  if (!rank) {
    ierr = read_data(&nnode, &nbranch, &node, &branch, &edgelist);CHKERRQ(ierr);
  }

  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(dmnetwork,"nstr",sizeof(Node),&componentkey[0]);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(dmnetwork,"bsrt",sizeof(Branch),&componentkey[1]);CHKERRQ(ierr);

  /* Set local number of nodes/edges, add edge connectivity */
  nV[0] = nnode; nE[0] = nbranch;
  ierr = DMNetworkSetNumSubNetworks(dmnetwork,PETSC_DECIDE,1);CHKERRQ(ierr);
  ierr = DMNetworkAddSubnetwork(dmnetwork,"",nV[0],nE[0],edgelist,NULL);CHKERRQ(ierr);

  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUp(dmnetwork);CHKERRQ(ierr);

  /* Add network components (physical parameters of nodes and branches) and num of variables */
  if (!rank) {
    ierr = DMNetworkGetEdgeRange(dmnetwork,&eStart,&eEnd);CHKERRQ(ierr);
    for (i = eStart; i < eEnd; i++) {
      ierr = DMNetworkAddComponent(dmnetwork,i,componentkey[1],&branch[i-eStart],1);CHKERRQ(ierr);
    }

    ierr = DMNetworkGetVertexRange(dmnetwork,&vStart,&vEnd);CHKERRQ(ierr);
    for (i = vStart; i < vEnd; i++) {
      ierr = DMNetworkAddComponent(dmnetwork,i,componentkey[0],&node[i-vStart],1);CHKERRQ(ierr);
    }
  }

  /* Network partitioning and distribution of data */
  ierr = DMSetUp(dmnetwork);CHKERRQ(ierr);
  ierr = DMNetworkDistribute(&dmnetwork,0);CHKERRQ(ierr);

  /* We do not use these data structures anymore since they have been copied to dmnetwork */
  if (!rank) {
    ierr = PetscFree(edgelist);CHKERRQ(ierr);
    ierr = PetscFree2(node,branch);CHKERRQ(ierr);
  }

  /* Create vectors and matrix */
  ierr = DMCreateGlobalVector(dmnetwork,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dmnetwork,&A);CHKERRQ(ierr);

  /* Assembly system of equations */
  ierr = FormOperator(dmnetwork,A,b);CHKERRQ(ierr);

  /* Solve linear system: A x = b */
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);
  ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free work space */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = DMDestroy(&dmnetwork);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   build:
      requires: !complex double define(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      args: -ksp_monitor_short

   test:
      suffix: 2
      nsize: 2
      args: -petscpartitioner_type simple -ksp_converged_reason

TEST*/
