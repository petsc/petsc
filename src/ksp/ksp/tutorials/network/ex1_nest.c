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
  PetscErrorCode    ierr;
  PetscInt          nnode, nbranch, i;
  Branch            *branch;
  Node              *node;
  PetscInt          *edgelist;

  nnode   = 4;
  nbranch = 6;

  ierr = PetscCalloc1(nnode,&node);CHKERRQ(ierr);
  ierr = PetscCalloc1(nbranch,&branch);CHKERRQ(ierr);

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

PetscErrorCode FormOperator(DM networkdm,Mat A,Vec b)
{
  PetscErrorCode    ierr;
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
  ierr = DMGetLocalVector(networkdm,&localb);CHKERRQ(ierr);
  ierr = VecSet(b,0.0);CHKERRQ(ierr);
  ierr = VecSet(localb,0.0);CHKERRQ(ierr);

  ierr = VecGetArray(localb,&barr);CHKERRQ(ierr);

  /* Get nested submatrices */
  ierr = MatNestGetSubMats(A,NULL,NULL,&mats);CHKERRQ(ierr);
  e11 = mats[0][0];  /* edges */
  c12 = mats[0][1];  /* couplings */
  c21 = mats[1][0];  /* couplings */
  v22 = mats[1][1];  /* vertices */

  /* Get vertices and edge range */
  ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);

  for (e = 0; e < eEnd; e++) {
    ierr = DMNetworkGetComponent(networkdm,e,0,&key,(void**)&branch);CHKERRQ(ierr);
    ierr = DMNetworkGetEdgeOffset(networkdm,e,&lofst);CHKERRQ(ierr);

    ierr = DMNetworkGetConnectedVertices(networkdm,e,&cone);CHKERRQ(ierr);
    ierr = DMNetworkGetVertexOffset(networkdm,cone[0],&lofst_fr);CHKERRQ(ierr);
    ierr = DMNetworkGetVertexOffset(networkdm,cone[1],&lofst_to);CHKERRQ(ierr);

    /* These are edge-edge and go to e11 */
    row[0] = lofst;
    col[0] = lofst;     val[0] =  1;
    ierr = MatSetValuesLocal(e11,1,row,1,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /* These are edge-vertex and go to c12 */
    col[0] = lofst_to;  val[0] =  1;
    col[1] = lofst_fr;  val[1] = -1;
    ierr = MatSetValuesLocal(c12,1,row,2,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /* These are edge-vertex and go to c12 */
    /* from node */
    ierr = DMNetworkGetComponent(networkdm,cone[0],0,&key,(void**)&node);CHKERRQ(ierr);

    if (!node->gr) {
      row[0] = lofst_fr;
      col[0] = lofst;   val[0] =  1;
      ierr = MatSetValuesLocal(c21,1,row,1,col,val,INSERT_VALUES);CHKERRQ(ierr);
    }

    /* to node */
    ierr = DMNetworkGetComponent(networkdm,cone[1],0,&key,(void**)&node);CHKERRQ(ierr);

    if (!node->gr) {
      row[0] = lofst_to;
      col[0] = lofst;   val[0] =  -1;
      ierr = MatSetValuesLocal(c21,1,row,1,col,val,INSERT_VALUES);CHKERRQ(ierr);
    }

    /* TODO: this is not a nested vector. Need to implement nested vector */
    ierr = DMNetworkGetVariableOffset(networkdm,e,&lofst);CHKERRQ(ierr);
    barr[lofst] = branch->bat;
  }

  for (v = vStart; v < vEnd; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm,v,&ghost);CHKERRQ(ierr);
    if (!ghost) {
      ierr = DMNetworkGetComponent(networkdm,v,0,&key,(void**)&node);CHKERRQ(ierr);
      ierr = DMNetworkGetVertexOffset(networkdm,v,&lofst);CHKERRQ(ierr);

      if (node->gr) {
        row[0] = lofst;
        col[0] = lofst;   val[0] =  1;
        ierr = MatSetValuesLocal(v22,1,row,1,col,val,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        /* TODO: this is not a nested vector. Need to implement nested vector */
        ierr = DMNetworkGetVariableOffset(networkdm,v,&lofst);CHKERRQ(ierr);
        barr[lofst] -= node->inj;
      }
    }
  }

  ierr = VecRestoreArray(localb,&barr);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm,localb,ADD_VALUES,b);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localb,ADD_VALUES,b);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localb);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char ** argv)
{
  PetscErrorCode    ierr;
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
  PetscInt          nV[1],nE[1],*edgelists[1];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* "read" data only for processor 0 */
  if (!rank) {
    ierr = read_data(&nnode, &nbranch, &node, &branch, &edgelist);CHKERRQ(ierr);
  }

  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&networkdm);CHKERRQ(ierr);

  ierr = DMNetworkRegisterComponent(networkdm,"nstr",sizeof(Node),&componentkey[0]);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"bsrt",sizeof(Branch),&componentkey[1]);CHKERRQ(ierr);


  /* Set number of nodes/edges */
  nV[0] = nnode; nE[0] = nbranch;
  ierr = DMNetworkSetSizes(networkdm,1,nV,nE,0,NULL);CHKERRQ(ierr);
  /* Add edge connectivity */
  edgelists[0] = edgelist;
  ierr = DMNetworkSetEdgeList(networkdm,edgelists,NULL);CHKERRQ(ierr);
  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

  /* Add network components: physical parameters of nodes and branches*/
  if (!rank) {
    ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
    for (i = eStart; i < eEnd; i++) {
      ierr = DMNetworkAddComponent(networkdm,i,componentkey[1],&branch[i-eStart]);CHKERRQ(ierr);
      ierr = DMNetworkAddNumVariables(networkdm,i,1);CHKERRQ(ierr);
    }

    ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);
    for (i = vStart; i < vEnd; i++) {
      ierr = DMNetworkAddComponent(networkdm,i,componentkey[0],&node[i-vStart]);CHKERRQ(ierr);
      /* Add number of variables */
      ierr = DMNetworkAddNumVariables(networkdm,i,1);CHKERRQ(ierr);
    }
  }

  /* Network partitioning and distribution of data */
  ierr = DMSetUp(networkdm);CHKERRQ(ierr);
  ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);

  ierr = DMNetworkAssembleGraphStructures(networkdm);CHKERRQ(ierr);

  /* Print some info */
#if 0
  PetscInt offset, goffset;
  ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);

  for (i = eStart; i < eEnd; i++) {
    ierr = DMNetworkGetVariableOffset(networkdm,i,&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableGlobalOffset(networkdm,i,&goffset);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"rank[%d] edge %d - loff: %d, goff: %d .\n",rank,i,offset,goffset);CHKERRQ(ierr);
  }
  for (i = vStart; i < vEnd; i++) {
    ierr = DMNetworkGetVariableOffset(networkdm,i,&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableGlobalOffset(networkdm,i,&goffset);CHKERRQ(ierr);
    ierr = PetscPrintf("rank[%d] vertex %d - loff: %d, goff: %d .\n",rank,i,offset,goffset);CHKERRQ(ierr);
  }
#endif

  /* We don't use these data structures anymore since they have been copied to networkdm */
  if (!rank) {
    ierr = PetscFree(edgelist);CHKERRQ(ierr);
    ierr = PetscFree(node);CHKERRQ(ierr);
    ierr = PetscFree(branch);CHKERRQ(ierr);
  }

  ierr = DMCreateGlobalVector(networkdm,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);

  ierr = DMSetMatType(networkdm, MATNEST);CHKERRQ(ierr);
  ierr = DMCreateMatrix(networkdm,&A);CHKERRQ(ierr);

  /* Assembly system of equations */
  ierr = FormOperator(networkdm,A,b);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);
  ierr = VecView(x, 0);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = DMDestroy(&networkdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   build:
      requires: !complex double define(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      args: -ksp_converged_reason

   test:
      suffix: 2
      nsize: 2
      args: -petscpartitioner_type simple -ksp_converged_reason

TEST*/

