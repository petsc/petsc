static char help[] = "This example is based on ex1.c, but generates a random network of chosen sizes and parameters. \n\
  Usage: -n determines number of nodes. The nonnegative seed can be specified with the flag -seed, otherwise the program generates a random seed.\n\n";

/* T
  Concepts: DMNetwork
  Concepts: KSP
*/

#include <petscdmnetwork.h>
#include <petscksp.h>
#include <time.h>

typedef struct {
  PetscInt    id; /* node id */
  PetscScalar inj; /* current injection (A) */
  PetscBool   gr; /* grounded ? */
} Node;

typedef struct {
  PetscInt    id;  /* branch id */
  PetscScalar r;   /* resistance (ohms) */
  PetscScalar bat; /* battery (V) */
} Branch;

typedef struct Edge {
  PetscInt      n;
  PetscInt      i;
  PetscInt      j;
  struct Edge   *next;
} Edge;

PetscReal findDistance(PetscReal x1, PetscReal x2, PetscReal y1, PetscReal y2)
{
  return PetscSqrtReal(PetscPowReal(x2-x1,2.0) + PetscPowReal(y2-y1,2.0));
}

/*
  The algorithm for network formation is based on the paper:
  Routing of Multipoint Connections, Bernard M. Waxman. 1988
*/

PetscErrorCode random_network(PetscInt nvertex,PetscInt *pnbranch,Node **pnode,Branch **pbranch,PetscInt **pedgelist,PetscInt seed)
{
  PetscInt       i, j, nedges = 0;
  PetscInt       *edgelist;
  PetscInt       nbat, ncurr, fr, to;
  PetscReal      *x, *y, value, xmax = 10.0; /* generate points in square */
  PetscReal      maxdist = 0.0, dist, alpha, beta, prob;
  PetscRandom    rnd;
  Branch         *branch;
  Node           *node;
  Edge           *head = NULL, *nnew= NULL, *aux= NULL;

  PetscFunctionBeginUser;
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rnd));
  CHKERRQ(PetscRandomSetFromOptions(rnd));

  CHKERRQ(PetscRandomSetSeed(rnd, seed));
  CHKERRQ(PetscRandomSeed(rnd));

  /* These parameters might be modified for experimentation */
  nbat  = (PetscInt)(0.1*nvertex);
  ncurr = (PetscInt)(0.1*nvertex);
  alpha = 0.6;
  beta  = 0.2;

  CHKERRQ(PetscMalloc2(nvertex,&x,nvertex,&y));

  CHKERRQ(PetscRandomSetInterval(rnd,0.0,xmax));
  for (i=0; i<nvertex; i++) {
    CHKERRQ(PetscRandomGetValueReal(rnd,&x[i]));
    CHKERRQ(PetscRandomGetValueReal(rnd,&y[i]));
  }

  /* find maximum distance */
  for (i=0; i<nvertex; i++) {
    for (j=0; j<nvertex; j++) {
      dist = findDistance(x[i],x[j],y[i],y[j]);
      if (dist >= maxdist) maxdist = dist;
    }
  }

  CHKERRQ(PetscRandomSetInterval(rnd,0.0,1.0));
  for (i=0; i<nvertex; i++) {
    for (j=0; j<nvertex; j++) {
      if (j != i) {
        dist = findDistance(x[i],x[j],y[i],y[j]);
        prob = beta*PetscExpScalar(-dist/(maxdist*alpha));
        CHKERRQ(PetscRandomGetValueReal(rnd,&value));
        if (value <= prob) {
          CHKERRQ(PetscMalloc1(1,&nnew));
          if (head == NULL) {
            head       = nnew;
            head->next = NULL;
            head->n    = nedges;
            head->i    = i;
            head->j    = j;
          } else {
            aux = head;
            head = nnew;
            head->n    = nedges;
            head->next = aux;
            head->i    = i;
            head->j    = j;
          }
          nedges += 1;
        }
      }
    }
  }

  CHKERRQ(PetscMalloc1(2*nedges,&edgelist));

  for (aux = head; aux; aux = aux->next) {
    edgelist[(aux->n)*2]     = aux->i;
    edgelist[(aux->n)*2 + 1] = aux->j;
  }

  aux = head;
  while (aux != NULL) {
    nnew = aux;
    aux = aux->next;
    CHKERRQ(PetscFree(nnew));
  }

  CHKERRQ(PetscCalloc2(nvertex,&node,nedges,&branch));

  for (i = 0; i < nvertex; i++) {
    node[i].id  = i;
    node[i].inj = 0;
    node[i].gr = PETSC_FALSE;
  }

  for (i = 0; i < nedges; i++) {
    branch[i].id  = i;
    branch[i].r   = 1.0;
    branch[i].bat = 0;
  }

  /* Chose random node as ground voltage */
  CHKERRQ(PetscRandomSetInterval(rnd,0.0,nvertex));
  CHKERRQ(PetscRandomGetValueReal(rnd,&value));
  node[(int)value].gr = PETSC_TRUE;

  /* Create random current and battery injectionsa */
  for (i=0; i<ncurr; i++) {
    CHKERRQ(PetscRandomSetInterval(rnd,0.0,nvertex));
    CHKERRQ(PetscRandomGetValueReal(rnd,&value));
    fr   = edgelist[(int)value*2];
    to   = edgelist[(int)value*2 + 1];
    node[fr].inj += 1.0;
    node[to].inj -= 1.0;
  }

  for (i=0; i<nbat; i++) {
    CHKERRQ(PetscRandomSetInterval(rnd,0.0,nedges));
    CHKERRQ(PetscRandomGetValueReal(rnd,&value));
    branch[(int)value].bat += 1.0;
  }

  CHKERRQ(PetscFree2(x,y));
  CHKERRQ(PetscRandomDestroy(&rnd));

  /* assign pointers */
  *pnbranch  = nedges;
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
  PetscInt          e,v,vStart,vEnd,eStart, eEnd;
  PetscInt          lofst,lofst_to,lofst_fr,row[2],col[6];
  PetscBool         ghost;
  const PetscInt    *cone;
  PetscScalar       *barr,val[6];

  PetscFunctionBegin;
  CHKERRQ(DMGetLocalVector(networkdm,&localb));
  CHKERRQ(VecSet(b,0.0));
  CHKERRQ(VecSet(localb,0.0));
  CHKERRQ(MatZeroEntries(A));

  CHKERRQ(VecGetArray(localb,&barr));

  /*
    We can define the current as a "edge characteristic" and the voltage
    and the voltage as a "vertex characteristic". With that, we can iterate
    the list of edges and vertices, query the associated voltages and currents
    and use them to write the Kirchoff equations.
  */

  /* Branch equations: i/r + uj - ui = battery */
  CHKERRQ(DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd));
  for (e = 0; e < eEnd; e++) {
    CHKERRQ(DMNetworkGetComponent(networkdm,e,0,NULL,(void**)&branch,NULL));
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,e,ALL_COMPONENTS,&lofst));

    CHKERRQ(DMNetworkGetConnectedVertices(networkdm,e,&cone));
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,cone[0],ALL_COMPONENTS,&lofst_fr));
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,cone[1],ALL_COMPONENTS,&lofst_to));

    barr[lofst] = branch->bat;

    row[0] = lofst;
    col[0] = lofst;     val[0] =  1;
    col[1] = lofst_to;  val[1] =  1;
    col[2] = lofst_fr;  val[2] = -1;
    CHKERRQ(MatSetValuesLocal(A,1,row,3,col,val,ADD_VALUES));

    /* from node */
    CHKERRQ(DMNetworkGetComponent(networkdm,cone[0],0,NULL,(void**)&node,NULL));

    if (!node->gr) {
      row[0] = lofst_fr;
      col[0] = lofst;   val[0] =  1;
      CHKERRQ(MatSetValuesLocal(A,1,row,1,col,val,ADD_VALUES));
    }

    /* to node */
    CHKERRQ(DMNetworkGetComponent(networkdm,cone[1],0,NULL,(void**)&node,NULL));

    if (!node->gr) {
      row[0] = lofst_to;
      col[0] = lofst;   val[0] =  -1;
      CHKERRQ(MatSetValuesLocal(A,1,row,1,col,val,ADD_VALUES));
    }
  }

  CHKERRQ(DMNetworkGetVertexRange(networkdm,&vStart,&vEnd));
  for (v = vStart; v < vEnd; v++) {
    CHKERRQ(DMNetworkIsGhostVertex(networkdm,v,&ghost));
    if (!ghost) {
      CHKERRQ(DMNetworkGetComponent(networkdm,v,0,NULL,(void**)&node,NULL));
      CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,v,ALL_COMPONENTS,&lofst));

      if (node->gr) {
        row[0] = lofst;
        col[0] = lofst;   val[0] =  1;
        CHKERRQ(MatSetValuesLocal(A,1,row,1,col,val,ADD_VALUES));
      } else {
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
  PetscInt          i, nbranch = 0, eStart, eEnd, vStart, vEnd;
  PetscInt          seed = 0, nnode = 0;
  PetscMPIInt       size, rank;
  DM                networkdm;
  Vec               x, b;
  Mat               A;
  KSP               ksp;
  PetscInt          *edgelist = NULL;
  PetscInt          componentkey[2];
  Node              *node;
  Branch            *branch;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage[3];
#endif

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-seed",&seed,NULL));

  CHKERRQ(PetscLogStageRegister("Network Creation", &stage[0]));
  CHKERRQ(PetscLogStageRegister("DMNetwork data structures", &stage[1]));
  CHKERRQ(PetscLogStageRegister("KSP", &stage[2]));

  CHKERRQ(PetscLogStagePush(stage[0]));
  /* "read" data only for processor 0 */
  if (rank == 0) {
    nnode = 100;
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&nnode,NULL));
    CHKERRQ(random_network(nnode, &nbranch, &node, &branch, &edgelist, seed));
  }
  CHKERRQ(PetscLogStagePop());

  CHKERRQ(PetscLogStagePush(stage[1]));
  CHKERRQ(DMNetworkCreate(PETSC_COMM_WORLD,&networkdm));
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"nstr",sizeof(Node),&componentkey[0]));
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"bsrt",sizeof(Branch),&componentkey[1]));

  /* Set number of nodes/edges and edge connectivity */
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
    CHKERRQ(PetscFree2(node,branch));
  }

  /* Create vectors and matrix */
  CHKERRQ(DMCreateGlobalVector(networkdm,&x));
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(DMCreateMatrix(networkdm,&A));

  CHKERRQ(PetscLogStagePop());

  CHKERRQ(PetscLogStagePush(stage[2]));
  /* Assembly system of equations */
  CHKERRQ(FormOperator(networkdm,A,b));

  /* Solve linear system: A x = b */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ksp));
  CHKERRQ(KSPSetOperators(ksp, A, A));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp, b, x));

  CHKERRQ(PetscLogStagePop());

  /* Free work space */
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
      requires: !single double defined(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      args: -ksp_converged_reason

   test:
      suffix: 2
      nsize: 2
      args: -petscpartitioner_type simple -pc_type asm -sub_pc_type ilu -ksp_converged_reason

   test:
      suffix: 3
      nsize: 4
      args: -petscpartitioner_type simple -pc_type asm -sub_pc_type lu -sub_pc_factor_shift_type nonzero -ksp_converged_reason

   test:
      suffix: graphindex
      args: -n 20 -vertex_global_section_view -edge_global_section_view

   test:
      suffix: graphindex_2
      nsize: 2
      args: -petscpartitioner_type simple -n 20 -vertex_global_section_view -edge_global_section_view

TEST*/
