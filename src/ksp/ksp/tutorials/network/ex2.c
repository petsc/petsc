static char help[] = "This example is based on ex1.c, but generates a random network of chosen sizes and parameters. \n\
  Usage: -n determines number of nodes. The nonnegative seed can be specified with the flag -seed, otherwise the program generates a random seed.\n\n";

#include <petscdmnetwork.h>
#include <petscksp.h>
#include <time.h>

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

typedef struct Edge {
  PetscInt     n;
  PetscInt     i;
  PetscInt     j;
  struct Edge *next;
} Edge;

PetscReal findDistance(PetscReal x1, PetscReal x2, PetscReal y1, PetscReal y2)
{
  return PetscSqrtReal(PetscPowReal(x2 - x1, 2.0) + PetscPowReal(y2 - y1, 2.0));
}

/*
  The algorithm for network formation is based on the paper:
  Routing of Multipoint Connections, Bernard M. Waxman. 1988
*/

PetscErrorCode random_network(PetscInt nvertex, PetscInt *pnbranch, Node **pnode, Branch **pbranch, PetscInt **pedgelist, PetscInt seed)
{
  PetscInt    i, j, nedges = 0;
  PetscInt   *edgelist;
  PetscInt    nbat, ncurr, fr, to;
  PetscReal  *x, *y, value, xmax = 10.0; /* generate points in square */
  PetscReal   maxdist = 0.0, dist, alpha, beta, prob;
  PetscRandom rnd;
  Branch     *branch;
  Node       *node;
  Edge       *head = NULL, *nnew = NULL, *aux = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rnd));
  PetscCall(PetscRandomSetFromOptions(rnd));

  PetscCall(PetscRandomSetSeed(rnd, seed));
  PetscCall(PetscRandomSeed(rnd));

  /* These parameters might be modified for experimentation */
  nbat  = (PetscInt)(0.1 * nvertex);
  ncurr = (PetscInt)(0.1 * nvertex);
  alpha = 0.6;
  beta  = 0.2;

  PetscCall(PetscMalloc2(nvertex, &x, nvertex, &y));

  PetscCall(PetscRandomSetInterval(rnd, 0.0, xmax));
  for (i = 0; i < nvertex; i++) {
    PetscCall(PetscRandomGetValueReal(rnd, &x[i]));
    PetscCall(PetscRandomGetValueReal(rnd, &y[i]));
  }

  /* find maximum distance */
  for (i = 0; i < nvertex; i++) {
    for (j = 0; j < nvertex; j++) {
      dist = findDistance(x[i], x[j], y[i], y[j]);
      if (dist >= maxdist) maxdist = dist;
    }
  }

  PetscCall(PetscRandomSetInterval(rnd, 0.0, 1.0));
  for (i = 0; i < nvertex; i++) {
    for (j = 0; j < nvertex; j++) {
      if (j != i) {
        dist = findDistance(x[i], x[j], y[i], y[j]);
        prob = beta * PetscExpScalar(-dist / (maxdist * alpha));
        PetscCall(PetscRandomGetValueReal(rnd, &value));
        if (value <= prob) {
          PetscCall(PetscMalloc1(1, &nnew));
          if (head == NULL) {
            head       = nnew;
            head->next = NULL;
            head->n    = nedges;
            head->i    = i;
            head->j    = j;
          } else {
            aux        = head;
            head       = nnew;
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

  PetscCall(PetscMalloc1(2 * nedges, &edgelist));

  for (aux = head; aux; aux = aux->next) {
    edgelist[(aux->n) * 2]     = aux->i;
    edgelist[(aux->n) * 2 + 1] = aux->j;
  }

  aux = head;
  while (aux != NULL) {
    nnew = aux;
    aux  = aux->next;
    PetscCall(PetscFree(nnew));
  }

  PetscCall(PetscCalloc2(nvertex, &node, nedges, &branch));

  for (i = 0; i < nvertex; i++) {
    node[i].id  = i;
    node[i].inj = 0;
    node[i].gr  = PETSC_FALSE;
  }

  for (i = 0; i < nedges; i++) {
    branch[i].id  = i;
    branch[i].r   = 1.0;
    branch[i].bat = 0;
  }

  /* Chose random node as ground voltage */
  PetscCall(PetscRandomSetInterval(rnd, 0.0, nvertex));
  PetscCall(PetscRandomGetValueReal(rnd, &value));
  node[(int)value].gr = PETSC_TRUE;

  /* Create random current and battery injectionsa */
  for (i = 0; i < ncurr; i++) {
    PetscCall(PetscRandomSetInterval(rnd, 0.0, nvertex));
    PetscCall(PetscRandomGetValueReal(rnd, &value));
    fr = edgelist[(int)value * 2];
    to = edgelist[(int)value * 2 + 1];
    node[fr].inj += 1.0;
    node[to].inj -= 1.0;
  }

  for (i = 0; i < nbat; i++) {
    PetscCall(PetscRandomSetInterval(rnd, 0.0, nedges));
    PetscCall(PetscRandomGetValueReal(rnd, &value));
    branch[(int)value].bat += 1.0;
  }

  PetscCall(PetscFree2(x, y));
  PetscCall(PetscRandomDestroy(&rnd));

  /* assign pointers */
  *pnbranch  = nedges;
  *pedgelist = edgelist;
  *pbranch   = branch;
  *pnode     = node;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormOperator(DM networkdm, Mat A, Vec b)
{
  Vec             localb;
  Branch         *branch;
  Node           *node;
  PetscInt        e, v, vStart, vEnd, eStart, eEnd;
  PetscInt        lofst, lofst_to, lofst_fr, row[2], col[6];
  PetscBool       ghost;
  const PetscInt *cone;
  PetscScalar    *barr, val[6];

  PetscFunctionBegin;
  PetscCall(DMGetLocalVector(networkdm, &localb));
  PetscCall(VecSet(b, 0.0));
  PetscCall(VecSet(localb, 0.0));
  PetscCall(MatZeroEntries(A));

  PetscCall(VecGetArray(localb, &barr));

  /*
    We can define the current as a "edge characteristic" and the voltage
    and the voltage as a "vertex characteristic". With that, we can iterate
    the list of edges and vertices, query the associated voltages and currents
    and use them to write the Kirchoff equations.
  */

  /* Branch equations: i/r + uj - ui = battery */
  PetscCall(DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd));
  for (e = 0; e < eEnd; e++) {
    PetscCall(DMNetworkGetComponent(networkdm, e, 0, NULL, (void **)&branch, NULL));
    PetscCall(DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &lofst));

    PetscCall(DMNetworkGetConnectedVertices(networkdm, e, &cone));
    PetscCall(DMNetworkGetLocalVecOffset(networkdm, cone[0], ALL_COMPONENTS, &lofst_fr));
    PetscCall(DMNetworkGetLocalVecOffset(networkdm, cone[1], ALL_COMPONENTS, &lofst_to));

    barr[lofst] = branch->bat;

    row[0] = lofst;
    col[0] = lofst;
    val[0] = 1;
    col[1] = lofst_to;
    val[1] = 1;
    col[2] = lofst_fr;
    val[2] = -1;
    PetscCall(MatSetValuesLocal(A, 1, row, 3, col, val, ADD_VALUES));

    /* from node */
    PetscCall(DMNetworkGetComponent(networkdm, cone[0], 0, NULL, (void **)&node, NULL));

    if (!node->gr) {
      row[0] = lofst_fr;
      col[0] = lofst;
      val[0] = 1;
      PetscCall(MatSetValuesLocal(A, 1, row, 1, col, val, ADD_VALUES));
    }

    /* to node */
    PetscCall(DMNetworkGetComponent(networkdm, cone[1], 0, NULL, (void **)&node, NULL));

    if (!node->gr) {
      row[0] = lofst_to;
      col[0] = lofst;
      val[0] = -1;
      PetscCall(MatSetValuesLocal(A, 1, row, 1, col, val, ADD_VALUES));
    }
  }

  PetscCall(DMNetworkGetVertexRange(networkdm, &vStart, &vEnd));
  for (v = vStart; v < vEnd; v++) {
    PetscCall(DMNetworkIsGhostVertex(networkdm, v, &ghost));
    if (!ghost) {
      PetscCall(DMNetworkGetComponent(networkdm, v, 0, NULL, (void **)&node, NULL));
      PetscCall(DMNetworkGetLocalVecOffset(networkdm, v, ALL_COMPONENTS, &lofst));

      if (node->gr) {
        row[0] = lofst;
        col[0] = lofst;
        val[0] = 1;
        PetscCall(MatSetValuesLocal(A, 1, row, 1, col, val, ADD_VALUES));
      } else {
        barr[lofst] -= node->inj;
      }
    }
  }

  PetscCall(VecRestoreArray(localb, &barr));

  PetscCall(DMLocalToGlobalBegin(networkdm, localb, ADD_VALUES, b));
  PetscCall(DMLocalToGlobalEnd(networkdm, localb, ADD_VALUES, b));
  PetscCall(DMRestoreLocalVector(networkdm, &localb));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt    i, nbranch = 0, eStart, eEnd, vStart, vEnd;
  PetscInt    seed = 0, nnode = 0;
  PetscMPIInt size, rank;
  DM          networkdm;
  Vec         x, b;
  Mat         A;
  KSP         ksp;
  PetscInt   *edgelist = NULL;
  PetscInt    componentkey[2];
  Node       *node;
  Branch     *branch;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage[3];
#endif

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-seed", &seed, NULL));

  PetscCall(PetscLogStageRegister("Network Creation", &stage[0]));
  PetscCall(PetscLogStageRegister("DMNetwork data structures", &stage[1]));
  PetscCall(PetscLogStageRegister("KSP", &stage[2]));

  PetscCall(PetscLogStagePush(stage[0]));
  /* "read" data only for processor 0 */
  if (rank == 0) {
    nnode = 100;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &nnode, NULL));
    PetscCall(random_network(nnode, &nbranch, &node, &branch, &edgelist, seed));
  }
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(stage[1]));
  PetscCall(DMNetworkCreate(PETSC_COMM_WORLD, &networkdm));
  PetscCall(DMNetworkRegisterComponent(networkdm, "nstr", sizeof(Node), &componentkey[0]));
  PetscCall(DMNetworkRegisterComponent(networkdm, "bsrt", sizeof(Branch), &componentkey[1]));

  /* Set number of nodes/edges and edge connectivity */
  PetscCall(DMNetworkSetNumSubNetworks(networkdm, PETSC_DECIDE, 1));
  PetscCall(DMNetworkAddSubnetwork(networkdm, "", nbranch, edgelist, NULL));

  /* Set up the network layout */
  PetscCall(DMNetworkLayoutSetUp(networkdm));

  /* Add network components (physical parameters of nodes and branches) and num of variables */
  if (rank == 0) {
    PetscCall(DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd));
    for (i = eStart; i < eEnd; i++) PetscCall(DMNetworkAddComponent(networkdm, i, componentkey[1], &branch[i - eStart], 1));

    PetscCall(DMNetworkGetVertexRange(networkdm, &vStart, &vEnd));
    for (i = vStart; i < vEnd; i++) PetscCall(DMNetworkAddComponent(networkdm, i, componentkey[0], &node[i - vStart], 1));
  }

  /* Network partitioning and distribution of data */
  PetscCall(DMSetUp(networkdm));
  PetscCall(DMNetworkDistribute(&networkdm, 0));
  PetscCall(DMNetworkAssembleGraphStructures(networkdm));

  /* We don't use these data structures anymore since they have been copied to networkdm */
  if (rank == 0) {
    PetscCall(PetscFree(edgelist));
    PetscCall(PetscFree2(node, branch));
  }

  /* Create vectors and matrix */
  PetscCall(DMCreateGlobalVector(networkdm, &x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(DMCreateMatrix(networkdm, &A));

  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(stage[2]));
  /* Assembly system of equations */
  PetscCall(FormOperator(networkdm, A, b));

  /* Solve linear system: A x = b */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(PetscLogStagePop());

  /* Free work space */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&networkdm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !single double defined(PETSC_HAVE_ATTRIBUTEALIGNED) 64bitptr

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
