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
  PetscErrorCode ierr;
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
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);

  ierr = PetscRandomSetSeed(rnd, seed);CHKERRQ(ierr);
  ierr = PetscRandomSeed(rnd);CHKERRQ(ierr);

  /* These parameters might be modified for experimentation */
  nbat  = (PetscInt)(0.1*nvertex);
  ncurr = (PetscInt)(0.1*nvertex);
  alpha = 0.6;
  beta  = 0.2;

  ierr = PetscMalloc2(nvertex,&x,nvertex,&y);CHKERRQ(ierr);

  ierr = PetscRandomSetInterval(rnd,0.0,xmax);CHKERRQ(ierr);
  for (i=0; i<nvertex; i++) {
    ierr = PetscRandomGetValueReal(rnd,&x[i]);CHKERRQ(ierr);
    ierr = PetscRandomGetValueReal(rnd,&y[i]);CHKERRQ(ierr);
  }

  /* find maximum distance */
  for (i=0; i<nvertex; i++) {
    for (j=0; j<nvertex; j++) {
      dist = findDistance(x[i],x[j],y[i],y[j]);
      if (dist >= maxdist) maxdist = dist;
    }
  }

  ierr = PetscRandomSetInterval(rnd,0.0,1.0);CHKERRQ(ierr);
  for (i=0; i<nvertex; i++) {
    for (j=0; j<nvertex; j++) {
      if (j != i) {
        dist = findDistance(x[i],x[j],y[i],y[j]);
        prob = beta*PetscExpScalar(-dist/(maxdist*alpha));
        ierr = PetscRandomGetValueReal(rnd,&value);CHKERRQ(ierr);
        if (value <= prob) {
          ierr = PetscMalloc1(1,&nnew);CHKERRQ(ierr);
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

  ierr = PetscMalloc1(2*nedges,&edgelist);CHKERRQ(ierr);

  for (aux = head; aux; aux = aux->next) {
    edgelist[(aux->n)*2]     = aux->i;
    edgelist[(aux->n)*2 + 1] = aux->j;
  }

  aux = head;
  while (aux != NULL) {
    nnew = aux;
    aux = aux->next;
    ierr = PetscFree(nnew);CHKERRQ(ierr);
  }

  ierr = PetscCalloc2(nvertex,&node,nedges,&branch);CHKERRQ(ierr);

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
  ierr = PetscRandomSetInterval(rnd,0.0,nvertex);CHKERRQ(ierr);
  ierr = PetscRandomGetValueReal(rnd,&value);CHKERRQ(ierr);
  node[(int)value].gr = PETSC_TRUE;

  /* Create random current and battery injectionsa */
  for (i=0; i<ncurr; i++) {
    ierr = PetscRandomSetInterval(rnd,0.0,nvertex);CHKERRQ(ierr);
    ierr = PetscRandomGetValueReal(rnd,&value);CHKERRQ(ierr);
    fr   = edgelist[(int)value*2];
    to   = edgelist[(int)value*2 + 1];
    node[fr].inj += 1.0;
    node[to].inj -= 1.0;
  }

  for (i=0; i<nbat; i++) {
    ierr = PetscRandomSetInterval(rnd,0.0,nedges);CHKERRQ(ierr);
    ierr = PetscRandomGetValueReal(rnd,&value);CHKERRQ(ierr);
    branch[(int)value].bat += 1.0;
  }

  ierr = PetscFree2(x,y);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);

  /* assign pointers */
  *pnbranch  = nedges;
  *pedgelist = edgelist;
  *pbranch   = branch;
  *pnode     = node;
  PetscFunctionReturn(ierr);
}

PetscErrorCode FormOperator(DM networkdm,Mat A,Vec b)
{
  PetscErrorCode    ierr;
  Vec               localb;
  Branch            *branch;
  Node              *node;
  PetscInt          e,v,vStart,vEnd,eStart, eEnd;
  PetscInt          lofst,lofst_to,lofst_fr,row[2],col[6];
  PetscBool         ghost;
  const PetscInt    *cone;
  PetscScalar       *barr,val[6];

  PetscFunctionBegin;
  ierr = DMGetLocalVector(networkdm,&localb);CHKERRQ(ierr);
  ierr = VecSet(b,0.0);CHKERRQ(ierr);
  ierr = VecSet(localb,0.0);CHKERRQ(ierr);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);

  ierr = VecGetArray(localb,&barr);CHKERRQ(ierr);

  /*
    We can define the current as a "edge characteristic" and the voltage
    and the voltage as a "vertex characteristic". With that, we can iterate
    the list of edges and vertices, query the associated voltages and currents
    and use them to write the Kirchoff equations.
  */

  /* Branch equations: i/r + uj - ui = battery */
  ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
  for (e = 0; e < eEnd; e++) {
    ierr = DMNetworkGetComponent(networkdm,e,0,NULL,(void**)&branch,NULL);CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(networkdm,e,ALL_COMPONENTS,&lofst);CHKERRQ(ierr);

    ierr = DMNetworkGetConnectedVertices(networkdm,e,&cone);CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(networkdm,cone[0],ALL_COMPONENTS,&lofst_fr);CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(networkdm,cone[1],ALL_COMPONENTS,&lofst_to);CHKERRQ(ierr);

    barr[lofst] = branch->bat;

    row[0] = lofst;
    col[0] = lofst;     val[0] =  1;
    col[1] = lofst_to;  val[1] =  1;
    col[2] = lofst_fr;  val[2] = -1;
    ierr = MatSetValuesLocal(A,1,row,3,col,val,ADD_VALUES);CHKERRQ(ierr);

    /* from node */
    ierr = DMNetworkGetComponent(networkdm,cone[0],0,NULL,(void**)&node,NULL);CHKERRQ(ierr);

    if (!node->gr) {
      row[0] = lofst_fr;
      col[0] = lofst;   val[0] =  1;
      ierr = MatSetValuesLocal(A,1,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
    }

    /* to node */
    ierr = DMNetworkGetComponent(networkdm,cone[1],0,NULL,(void**)&node,NULL);CHKERRQ(ierr);

    if (!node->gr) {
      row[0] = lofst_to;
      col[0] = lofst;   val[0] =  -1;
      ierr = MatSetValuesLocal(A,1,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm,v,&ghost);CHKERRQ(ierr);
    if (!ghost) {
      ierr = DMNetworkGetComponent(networkdm,v,0,NULL,(void**)&node,NULL);CHKERRQ(ierr);
      ierr = DMNetworkGetLocalVecOffset(networkdm,v,ALL_COMPONENTS,&lofst);CHKERRQ(ierr);

      if (node->gr) {
        row[0] = lofst;
        col[0] = lofst;   val[0] =  1;
        ierr = MatSetValuesLocal(A,1,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
      } else {
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

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-seed",&seed,NULL);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Network Creation", &stage[0]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("DMNetwork data structures", &stage[1]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("KSP", &stage[2]);CHKERRQ(ierr);

  ierr = PetscLogStagePush(stage[0]);CHKERRQ(ierr);
  /* "read" data only for processor 0 */
  if (!rank) {
    nnode = 100;
    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&nnode,NULL);CHKERRQ(ierr);
    ierr = random_network(nnode, &nbranch, &node, &branch, &edgelist, seed);CHKERRQ(ierr);
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscLogStagePush(stage[1]);CHKERRQ(ierr);
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&networkdm);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"nstr",sizeof(Node),&componentkey[0]);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"bsrt",sizeof(Branch),&componentkey[1]);CHKERRQ(ierr);

  /* Set number of nodes/edges and edge connectivity */
  ierr = DMNetworkSetNumSubNetworks(networkdm,PETSC_DECIDE,1);CHKERRQ(ierr);
  ierr = DMNetworkAddSubnetwork(networkdm,"",nnode,nbranch,edgelist,NULL);CHKERRQ(ierr);

  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

  /* Add network components (physical parameters of nodes and branches) and num of variables */
  if (!rank) {
    ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
    for (i = eStart; i < eEnd; i++) {
      ierr = DMNetworkAddComponent(networkdm,i,componentkey[1],&branch[i-eStart],1);CHKERRQ(ierr);
    }

    ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);
    for (i = vStart; i < vEnd; i++) {
      ierr = DMNetworkAddComponent(networkdm,i,componentkey[0],&node[i-vStart],1);CHKERRQ(ierr);
    }
  }

  /* Network partitioning and distribution of data */
  ierr = DMSetUp(networkdm);CHKERRQ(ierr);
  ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);
  ierr = DMNetworkAssembleGraphStructures(networkdm);CHKERRQ(ierr);

  /* We don't use these data structures anymore since they have been copied to networkdm */
  if (!rank) {
    ierr = PetscFree(edgelist);CHKERRQ(ierr);
    ierr = PetscFree2(node,branch);CHKERRQ(ierr);
  }

  /* Create vectors and matrix */
  ierr = DMCreateGlobalVector(networkdm,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = DMCreateMatrix(networkdm,&A);CHKERRQ(ierr);

  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscLogStagePush(stage[2]);CHKERRQ(ierr);
  /* Assembly system of equations */
  ierr = FormOperator(networkdm,A,b);CHKERRQ(ierr);

  /* Solve linear system: A x = b */
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);

  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /* Free work space */
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
