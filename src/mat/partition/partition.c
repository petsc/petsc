
#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/

/* Logging support */
PetscClassId MAT_PARTITIONING_CLASSID;

/*
   Simplest partitioning, keeps the current partitioning.
*/
static PetscErrorCode MatPartitioningApply_Current(MatPartitioning part, IS *partitioning)
{
  PetscInt    m;
  PetscMPIInt rank, size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)part), &size));
  if (part->n != size) {
    const char *prefix;
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)part, &prefix));
    SETERRQ(PetscObjectComm((PetscObject)part), PETSC_ERR_SUP, "This is the DEFAULT NO-OP partitioner, it currently only supports one domain per processor\nuse -%smat_partitioning_type parmetis or chaco or ptscotch for more than one subdomain per processor", prefix ? prefix : "");
  }
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)part), &rank));

  PetscCall(MatGetLocalSize(part->adj, &m, NULL));
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)part), m, rank, 0, partitioning));
  PetscFunctionReturn(0);
}

/*
   partition an index to rebalance the computation
*/
static PetscErrorCode MatPartitioningApply_Average(MatPartitioning part, IS *partitioning)
{
  PetscInt m, M, nparts, *indices, r, d, *parts, i, start, end, loc;

  PetscFunctionBegin;
  PetscCall(MatGetSize(part->adj, &M, NULL));
  PetscCall(MatGetLocalSize(part->adj, &m, NULL));
  nparts = part->n;
  PetscCall(PetscMalloc1(nparts, &parts));
  d = M / nparts;
  for (i = 0; i < nparts; i++) parts[i] = d;
  r = M % nparts;
  for (i = 0; i < r; i++) parts[i] += 1;
  for (i = 1; i < nparts; i++) parts[i] += parts[i - 1];
  PetscCall(PetscMalloc1(m, &indices));
  PetscCall(MatGetOwnershipRange(part->adj, &start, &end));
  for (i = start; i < end; i++) {
    PetscCall(PetscFindInt(i, nparts, parts, &loc));
    if (loc < 0) loc = -(loc + 1);
    else loc = loc + 1;
    indices[i - start] = loc;
  }
  PetscCall(PetscFree(parts));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)part), m, indices, PETSC_OWN_POINTER, partitioning));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPartitioningApply_Square(MatPartitioning part, IS *partitioning)
{
  PetscInt    cell, n, N, p, rstart, rend, *color;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)part), &size));
  PetscCheck(part->n == size, PetscObjectComm((PetscObject)part), PETSC_ERR_SUP, "Currently only supports one domain per processor");
  p = (PetscInt)PetscSqrtReal((PetscReal)part->n);
  PetscCheck(p * p == part->n, PetscObjectComm((PetscObject)part), PETSC_ERR_SUP, "Square partitioning requires \"perfect square\" number of domains");

  PetscCall(MatGetSize(part->adj, &N, NULL));
  n = (PetscInt)PetscSqrtReal((PetscReal)N);
  PetscCheck(n * n == N, PetscObjectComm((PetscObject)part), PETSC_ERR_SUP, "Square partitioning requires square domain");
  PetscCheck(n % p == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "Square partitioning requires p to divide n");
  PetscCall(MatGetOwnershipRange(part->adj, &rstart, &rend));
  PetscCall(PetscMalloc1(rend - rstart, &color));
  /* for (int cell=rstart; cell<rend; cell++) color[cell-rstart] = ((cell%n) < (n/2)) + 2 * ((cell/n) < (n/2)); */
  for (cell = rstart; cell < rend; cell++) color[cell - rstart] = ((cell % n) / (n / p)) + p * ((cell / n) / (n / p));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)part), rend - rstart, color, PETSC_OWN_POINTER, partitioning));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Current(MatPartitioning part)
{
  PetscFunctionBegin;
  part->ops->apply   = MatPartitioningApply_Current;
  part->ops->view    = NULL;
  part->ops->destroy = NULL;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Average(MatPartitioning part)
{
  PetscFunctionBegin;
  part->ops->apply   = MatPartitioningApply_Average;
  part->ops->view    = NULL;
  part->ops->destroy = NULL;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Square(MatPartitioning part)
{
  PetscFunctionBegin;
  part->ops->apply   = MatPartitioningApply_Square;
  part->ops->view    = NULL;
  part->ops->destroy = NULL;
  PetscFunctionReturn(0);
}

/* gets as input the "sizes" array computed by ParMetis_*_NodeND and returns
       seps[  0 :         2*p) : the start and end node of each subdomain
       seps[2*p : 2*p+2*(p-1)) : the start and end node of each separator
     levels[  0 :         p-1) : level in the tree for each separator (-1 root, -2 and -3 first level and so on)
   The arrays must be large enough
*/
PETSC_INTERN PetscErrorCode MatPartitioningSizesToSep_Private(PetscInt p, PetscInt sizes[], PetscInt seps[], PetscInt level[])
{
  PetscInt l2p, i, pTree, pStartTree;

  PetscFunctionBegin;
  l2p = PetscLog2Real(p);
  PetscCheck(!(l2p - (PetscInt)PetscLog2Real(p)), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "%" PetscInt_FMT " is not a power of 2", p);
  if (!p) PetscFunctionReturn(0);
  PetscCall(PetscArrayzero(seps, 2 * p - 2));
  PetscCall(PetscArrayzero(level, p - 1));
  seps[2 * p - 2] = sizes[2 * p - 2];
  pTree           = p;
  pStartTree      = 0;
  while (pTree != 1) {
    for (i = pStartTree; i < pStartTree + pTree; i++) {
      seps[i] += sizes[i];
      seps[pStartTree + pTree + (i - pStartTree) / 2] += seps[i];
    }
    pStartTree += pTree;
    pTree = pTree / 2;
  }
  seps[2 * p - 2] -= sizes[2 * p - 2];

  pStartTree = 2 * p - 2;
  pTree      = 1;
  while (pStartTree > 0) {
    for (i = pStartTree; i < pStartTree + pTree; i++) {
      PetscInt k = 2 * i - (pStartTree + 2 * pTree);
      PetscInt n = seps[k + 1];

      seps[k + 1]  = seps[i] - sizes[k + 1];
      seps[k]      = seps[k + 1] + sizes[k + 1] - n - sizes[k];
      level[i - p] = -pTree - i + pStartTree;
    }
    pTree *= 2;
    pStartTree -= pTree;
  }
  /* I know there should be a formula */
  PetscCall(PetscSortIntWithArrayPair(p - 1, seps + p, sizes + p, level));
  for (i = 2 * p - 2; i >= 0; i--) {
    seps[2 * i]     = seps[i];
    seps[2 * i + 1] = seps[i] + PetscMax(sizes[i] - 1, 0);
  }
  PetscFunctionReturn(0);
}

/* ===========================================================================================*/

PetscFunctionList MatPartitioningList              = NULL;
PetscBool         MatPartitioningRegisterAllCalled = PETSC_FALSE;

/*@C
   MatPartitioningRegister - Adds a new sparse matrix partitioning to the  matrix package.

   Not Collective

   Input Parameters:
+  sname - name of partitioning (for example `MATPARTITIONINGCURRENT`) or `MATPARTITIONINGPARMETIS`
-  function - function pointer that creates the partitioning type

   Level: developer

   Sample usage:
.vb
   MatPartitioningRegister("my_part",MyPartCreate);
.ve

   Then, your partitioner can be chosen with the procedural interface via
$     MatPartitioningSetType(part,"my_part")
   or at runtime via the option
$     -mat_partitioning_type my_part

.seealso: `MatPartitioning`, `MatPartitioningType`, `MatPartitioningCreate()`, `MatPartitioningRegisterDestroy()`, `MatPartitioningRegisterAll()`
@*/
PetscErrorCode MatPartitioningRegister(const char sname[], PetscErrorCode (*function)(MatPartitioning))
{
  PetscFunctionBegin;
  PetscCall(MatInitializePackage());
  PetscCall(PetscFunctionListAdd(&MatPartitioningList, sname, function));
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningGetType - Gets the Partitioning method type and name (as a string)
        from the partitioning context.

   Not collective

   Input Parameter:
.  partitioning - the partitioning context

   Output Parameter:
.  type - partitioner type

   Level: intermediate

   Not Collective

.seealso: `MatPartitioning`, `MatPartitioningType`, `MatPartitioningCreate()`, `MatPartitioningRegisterDestroy()`, `MatPartitioningRegisterAll()`
@*/
PetscErrorCode MatPartitioningGetType(MatPartitioning partitioning, MatPartitioningType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(partitioning, MAT_PARTITIONING_CLASSID, 1);
  PetscValidPointer(type, 2);
  *type = ((PetscObject)partitioning)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningSetNParts - Set how many partitions need to be created;
        by default this is one per processor. Certain partitioning schemes may
        in fact only support that option.

   Collective on part

   Input Parameters:
+  partitioning - the partitioning context
-  n - the number of partitions

   Level: intermediate

.seealso: `MatPartitioning`, `MatPartitioningCreate()`, `MatPartitioningApply()`
@*/
PetscErrorCode MatPartitioningSetNParts(MatPartitioning part, PetscInt n)
{
  PetscFunctionBegin;
  part->n = n;
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningApplyND - Gets a nested dissection partitioning for a matrix.

   Collective on Mat

   Input Parameters:
.  matp - the matrix partitioning object

   Output Parameters:
.   partitioning - the partitioning. For each local node, a positive value indicates the processor
                   number the node has been assigned to. Negative x values indicate the separator level -(x+1).

   Level: intermediate

   Note:
   The user can define additional partitionings; see `MatPartitioningRegister()`.

.seealso: `MatPartitioningApplyND()`, `MatPartitioningRegister()`, `MatPartitioningCreate()`,
          `MatPartitioningDestroy()`, `MatPartitioningSetAdjacency()`, `ISPartitioningToNumbering()`,
          `ISPartitioningCount()`
@*/
PetscErrorCode MatPartitioningApplyND(MatPartitioning matp, IS *partitioning)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matp, MAT_PARTITIONING_CLASSID, 1);
  PetscValidPointer(partitioning, 2);
  PetscCheck(matp->adj->assembled, PetscObjectComm((PetscObject)matp), PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled matrix");
  PetscCheck(!matp->adj->factortype, PetscObjectComm((PetscObject)matp), PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  PetscCall(PetscLogEventBegin(MAT_PartitioningND, matp, 0, 0, 0));
  PetscUseTypeMethod(matp, applynd, partitioning);
  PetscCall(PetscLogEventEnd(MAT_PartitioningND, matp, 0, 0, 0));

  PetscCall(MatPartitioningViewFromOptions(matp, NULL, "-mat_partitioning_view"));
  PetscCall(ISViewFromOptions(*partitioning, NULL, "-mat_partitioning_view"));
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningApply - Gets a partitioning for the graph represented by a sparse matrix.

   Collective on matp

   Input Parameters:
.  matp - the matrix partitioning object

   Output Parameters:
.   partitioning - the partitioning. For each local node this tells the processor
                   number that that node is assigned to.

   Options Database Keys:
   To specify the partitioning through the options database, use one of
   the following
$    -mat_partitioning_type parmetis, -mat_partitioning current
   To see the partitioning result
$    -mat_partitioning_view

   Level: beginner

   The user can define additional partitionings; see `MatPartitioningRegister()`.

.seealso: `MatPartitioning`, `MatPartitioningType`, `MatPartitioningRegister()`, `MatPartitioningCreate()`,
          `MatPartitioningDestroy()`, `MatPartitioningSetAdjacency()`, `ISPartitioningToNumbering()`,
          `ISPartitioningCount()`
@*/
PetscErrorCode MatPartitioningApply(MatPartitioning matp, IS *partitioning)
{
  PetscBool viewbalance, improve;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(matp, MAT_PARTITIONING_CLASSID, 1);
  PetscValidPointer(partitioning, 2);
  PetscCheck(matp->adj->assembled, PetscObjectComm((PetscObject)matp), PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled matrix");
  PetscCheck(!matp->adj->factortype, PetscObjectComm((PetscObject)matp), PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  PetscCall(PetscLogEventBegin(MAT_Partitioning, matp, 0, 0, 0));
  PetscUseTypeMethod(matp, apply, partitioning);
  PetscCall(PetscLogEventEnd(MAT_Partitioning, matp, 0, 0, 0));

  PetscCall(MatPartitioningViewFromOptions(matp, NULL, "-mat_partitioning_view"));
  PetscCall(ISViewFromOptions(*partitioning, NULL, "-mat_partitioning_view"));

  PetscObjectOptionsBegin((PetscObject)matp);
  viewbalance = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-mat_partitioning_view_imbalance", "Display imbalance information of a partition", NULL, PETSC_FALSE, &viewbalance, NULL));
  improve = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-mat_partitioning_improve", "Improve the quality of a partition", NULL, PETSC_FALSE, &improve, NULL));
  PetscOptionsEnd();

  if (improve) PetscCall(MatPartitioningImprove(matp, partitioning));

  if (viewbalance) PetscCall(MatPartitioningViewImbalance(matp, *partitioning));
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningImprove - Improves the quality of a given partition.

   Collective on matp

   Input Parameters:
+  matp - the matrix partitioning object
-  partitioning - the partitioning. For each local node this tells the processor
                   number that that node is assigned to.

   Output Parameters:
.   partitioning - the partitioning. For each local node this tells the processor
                   number that that node is assigned to.

   Options Database Keys:
   To improve the quality of the partition
$    -mat_partitioning_improve

   Level: beginner

.seealso: `MatPartitioning`, `MatPartitioningType`, `MatPartitioningApply()`, `MatPartitioningCreate()`,
          `MatPartitioningDestroy()`, `MatPartitioningSetAdjacency()`, `ISPartitioningToNumbering()`,
          `ISPartitioningCount()`
@*/
PetscErrorCode MatPartitioningImprove(MatPartitioning matp, IS *partitioning)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matp, MAT_PARTITIONING_CLASSID, 1);
  PetscValidPointer(partitioning, 2);
  PetscCheck(matp->adj->assembled, PetscObjectComm((PetscObject)matp), PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled matrix");
  PetscCheck(!matp->adj->factortype, PetscObjectComm((PetscObject)matp), PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  PetscCall(PetscLogEventBegin(MAT_Partitioning, matp, 0, 0, 0));
  PetscTryTypeMethod(matp, improve, partitioning);
  PetscCall(PetscLogEventEnd(MAT_Partitioning, matp, 0, 0, 0));
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningViewImbalance - Display partitioning imbalance information.

   Collective on matp

   Input Parameters:
+  matp - the matrix partitioning object
-  partitioning - the partitioning. For each local node this tells the processor
                   number that that node is assigned to.

   Options Database Keys:
   To see the partitioning imbalance information
$    -mat_partitioning_view_balance

   Level: beginner

.seealso: `MatPartitioning`, `MatPartitioningType`, `MatPartitioningApply()`, `MatPartitioningView()`
@*/
PetscErrorCode MatPartitioningViewImbalance(MatPartitioning matp, IS partitioning)
{
  PetscInt        nparts, *subdomainsizes, *subdomainsizes_tmp, nlocal, i, maxsub, minsub, avgsub;
  const PetscInt *indices;
  PetscViewer     viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(matp, MAT_PARTITIONING_CLASSID, 1);
  PetscValidHeaderSpecific(partitioning, IS_CLASSID, 2);
  nparts = matp->n;
  PetscCall(PetscCalloc2(nparts, &subdomainsizes, nparts, &subdomainsizes_tmp));
  PetscCall(ISGetLocalSize(partitioning, &nlocal));
  PetscCall(ISGetIndices(partitioning, &indices));
  for (i = 0; i < nlocal; i++) subdomainsizes_tmp[indices[i]] += matp->vertex_weights ? matp->vertex_weights[i] : 1;
  PetscCallMPI(MPI_Allreduce(subdomainsizes_tmp, subdomainsizes, nparts, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)matp)));
  PetscCall(ISRestoreIndices(partitioning, &indices));
  minsub = PETSC_MAX_INT, maxsub = PETSC_MIN_INT, avgsub = 0;
  for (i = 0; i < nparts; i++) {
    minsub = PetscMin(minsub, subdomainsizes[i]);
    maxsub = PetscMax(maxsub, subdomainsizes[i]);
    avgsub += subdomainsizes[i];
  }
  avgsub /= nparts;
  PetscCall(PetscFree2(subdomainsizes, subdomainsizes_tmp));
  PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)matp), &viewer));
  PetscCall(MatPartitioningView(matp, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Partitioning Imbalance Info: Max %" PetscInt_FMT ", Min %" PetscInt_FMT ", Avg %" PetscInt_FMT ", R %g\n", maxsub, minsub, avgsub, (double)(maxsub / (PetscReal)minsub)));
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningSetAdjacency - Sets the adjacency graph (matrix) of the thing to be
      partitioned.

   Collective on part

   Input Parameters:
+  part - the partitioning context
-  adj - the adjacency matrix, this can be any `MatType` but the natural representation is `MATMPIADJ`

   Level: beginner

.seealso: `MatPartitioning`, `MatPartitioningType`, `MatPartitioningCreate()`
@*/
PetscErrorCode MatPartitioningSetAdjacency(MatPartitioning part, Mat adj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, MAT_PARTITIONING_CLASSID, 1);
  PetscValidHeaderSpecific(adj, MAT_CLASSID, 2);
  part->adj = adj;
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningDestroy - Destroys the partitioning context.

   Collective on part

   Input Parameters:
.  part - the partitioning context

   Level: beginner

.seealso: `MatPartitioning`, `MatPartitioningType`, `MatPartitioningCreate()`
@*/
PetscErrorCode MatPartitioningDestroy(MatPartitioning *part)
{
  PetscFunctionBegin;
  if (!*part) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*part), MAT_PARTITIONING_CLASSID, 1);
  if (--((PetscObject)(*part))->refct > 0) {
    *part = NULL;
    PetscFunctionReturn(0);
  }

  if ((*part)->ops->destroy) PetscCall((*(*part)->ops->destroy)((*part)));
  PetscCall(PetscFree((*part)->vertex_weights));
  PetscCall(PetscFree((*part)->part_weights));
  PetscCall(PetscHeaderDestroy(part));
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningSetVertexWeights - Sets the weights for vertices for a partitioning.

   Logically Collective on part

   Input Parameters:
+  part - the partitioning context
-  weights - the weights, on each process this array must have the same size as the number of local rows times the value passed with `MatPartitioningSetNumberVertexWeights()` or
             1 if that is not provided

   Level: beginner

   Notes:
      The array weights is freed by PETSc so the user should not free the array. In C/C++
   the array must be obtained with a call to `PetscMalloc()`, not malloc().

   The weights may not be used by some partitioners

.seealso: `MatPartitioning`, `MatPartitioningCreate()`, `MatPartitioningSetType()`, `MatPartitioningSetPartitionWeights()`, `MatPartitioningSetNumberVertexWeights()`
@*/
PetscErrorCode MatPartitioningSetVertexWeights(MatPartitioning part, const PetscInt weights[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, MAT_PARTITIONING_CLASSID, 1);
  PetscCall(PetscFree(part->vertex_weights));
  part->vertex_weights = (PetscInt *)weights;
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningSetPartitionWeights - Sets the weights for each partition.

   Logically Collective on part

   Input Parameters:
+  part - the partitioning context
-  weights - An array of size nparts that is used to specify the fraction of
             vertex weight that should be distributed to each sub-domain for
             the balance constraint. If all of the sub-domains are to be of
             the same size, then each of the nparts elements should be set
             to a value of 1/nparts. Note that the sum of all of the weights
             should be one.

   Level: beginner

   Note:
      The array weights is freed by PETSc so the user should not free the array. In C/C++
   the array must be obtained with a call to `PetscMalloc()`, not malloc().

.seealso:  `MatPartitioning`, `MatPartitioningSetVertexWeights()`, `MatPartitioningCreate()`, `MatPartitioningSetType()`, `MatPartitioningSetVertexWeights()`
@*/
PetscErrorCode MatPartitioningSetPartitionWeights(MatPartitioning part, const PetscReal weights[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, MAT_PARTITIONING_CLASSID, 1);
  PetscCall(PetscFree(part->part_weights));
  part->part_weights = (PetscReal *)weights;
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningSetUseEdgeWeights - Set a flag to indicate whether or not to use edge weights.

   Logically Collective on part

   Input Parameters:
+  part - the partitioning context
-  use_edge_weights - the flag indicateing whether or not to use edge weights. By default no edge weights will be used,
                      that is, use_edge_weights is set to FALSE. If set use_edge_weights to TRUE, users need to make sure legal
                      edge weights are stored in an ADJ matrix.
   Level: beginner

   Options Database Keys:
.  -mat_partitioning_use_edge_weights - (true or false)

.seealso: `MatPartitioning`, `MatPartitioningCreate()`, `MatPartitioningSetType()`, `MatPartitioningSetVertexWeights()`, `MatPartitioningSetPartitionWeights()`
@*/
PetscErrorCode MatPartitioningSetUseEdgeWeights(MatPartitioning part, PetscBool use_edge_weights)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, MAT_PARTITIONING_CLASSID, 1);
  part->use_edge_weights = use_edge_weights;
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningGetUseEdgeWeights - Get a flag that indicates whether or not to edge weights are used.

   Logically Collective on part

   Input Parameters:
.  part - the partitioning context

   Output Parameters:
.  use_edge_weights - the flag indicateing whether or not to edge weights are used.

   Level: beginner

.seealso: `MatPartitioning`, `MatPartitioningCreate()`, `MatPartitioningSetType()`, `MatPartitioningSetVertexWeights()`, `MatPartitioningSetPartitionWeights()`,
          `MatPartitioningSetUseEdgeWeights`
@*/
PetscErrorCode MatPartitioningGetUseEdgeWeights(MatPartitioning part, PetscBool *use_edge_weights)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, MAT_PARTITIONING_CLASSID, 1);
  PetscValidBoolPointer(use_edge_weights, 2);
  *use_edge_weights = part->use_edge_weights;
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningCreate - Creates a partitioning context.

   Collective

   Input Parameter:
.   comm - MPI communicator

   Output Parameter:
.  newp - location to put the context

   Level: beginner

.seealso: `MatPartitioning`, `MatPartitioningSetType()`, `MatPartitioningApply()`, `MatPartitioningDestroy()`,
          `MatPartitioningSetAdjacency()`
@*/
PetscErrorCode MatPartitioningCreate(MPI_Comm comm, MatPartitioning *newp)
{
  MatPartitioning part;
  PetscMPIInt     size;

  PetscFunctionBegin;
  *newp = NULL;

  PetscCall(MatInitializePackage());
  PetscCall(PetscHeaderCreate(part, MAT_PARTITIONING_CLASSID, "MatPartitioning", "Matrix/graph partitioning", "MatOrderings", comm, MatPartitioningDestroy, MatPartitioningView));
  part->vertex_weights   = NULL;
  part->part_weights     = NULL;
  part->use_edge_weights = PETSC_FALSE; /* By default we don't use edge weights */

  PetscCallMPI(MPI_Comm_size(comm, &size));
  part->n    = (PetscInt)size;
  part->ncon = 1;

  *newp = part;
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningViewFromOptions - View a partitioning context from the options database

   Collective on A

   Input Parameters:
+  A - the partitioning context
.  obj - Optional object
-  name - command line option

   Level: intermediate

  Options Database:
.  -mat_partitioning_view [viewertype]:... - the viewer and its options

  Note:
.vb
    If no value is provided ascii:stdout is used
       ascii[:[filename][:[format][:append]]]    defaults to stdout - format can be one of ascii_info, ascii_info_detail, or ascii_matlab,
                                                  for example ascii::ascii_info prints just the information about the object not all details
                                                  unless :append is given filename opens in write mode, overwriting what was already there
       binary[:[filename][:[format][:append]]]   defaults to the file binaryoutput
       draw[:drawtype[:filename]]                for example, draw:tikz, draw:tikz:figure.tex  or draw:x
       socket[:port]                             defaults to the standard output port
       saws[:communicatorname]                    publishes object to the Scientific Application Webserver (SAWs)
.ve

.seealso: `MatPartitioning`, `MatPartitioningView()`, `PetscObjectViewFromOptions()`, `MatPartitioningCreate()`
@*/
PetscErrorCode MatPartitioningViewFromOptions(MatPartitioning A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_PARTITIONING_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningView - Prints the partitioning data structure.

   Collective on part

   Input Parameters:
+  part - the partitioning context
-  viewer - optional visualization context

   Level: intermediate

   Note:
   The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The user can open alternative visualization contexts with
.     `PetscViewerASCIIOpen()` - output to a specified file

.seealso: `MatPartitioning`, `PetscViewer`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode MatPartitioningView(MatPartitioning part, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, MAT_PARTITIONING_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)part), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(part, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)part, viewer));
    if (part->vertex_weights) PetscCall(PetscViewerASCIIPrintf(viewer, "  Using vertex weights\n"));
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscTryTypeMethod(part, view, viewer);
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningSetType - Sets the type of partitioner to use

   Collective on part

   Input Parameters:
+  part - the partitioning context.
-  type - a known method

   Options Database Key:
.  -mat_partitioning_type  <type> - (for instance, parmetis), use -help for a list of available methods

   Level: intermediate

.seealso: `MatPartitioning`, `MatPartitioningCreate()`, `MatPartitioningApply()`, `MatPartitioningType`
@*/
PetscErrorCode MatPartitioningSetType(MatPartitioning part, MatPartitioningType type)
{
  PetscBool match;
  PetscErrorCode (*r)(MatPartitioning);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, MAT_PARTITIONING_CLASSID, 1);
  PetscValidCharPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)part, type, &match));
  if (match) PetscFunctionReturn(0);

  PetscTryTypeMethod(part, destroy);
  part->ops->destroy = NULL;

  part->setupcalled = 0;
  part->data        = NULL;
  PetscCall(PetscMemzero(part->ops, sizeof(struct _MatPartitioningOps)));

  PetscCall(PetscFunctionListFind(MatPartitioningList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)part), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown partitioning type %s", type);

  PetscCall((*r)(part));

  PetscCall(PetscFree(((PetscObject)part)->type_name));
  PetscCall(PetscStrallocpy(type, &((PetscObject)part)->type_name));
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningSetFromOptions - Sets various partitioning options from the
        options database for the partitioning object

   Collective on part

   Input Parameter:
.  part - the partitioning context.

   Options Database Keys:
+  -mat_partitioning_type  <type> - (for instance, parmetis), use -help for a list of available methods
-  -mat_partitioning_nparts - number of subgraphs

   Level: beginner

   Note:
    If the partitioner has not been set by the user it uses one of the installed partitioner such as ParMetis. If there are
   no installed partitioners it uses current which means no repartioning.

.seealso: `MatPartitioning`
@*/
PetscErrorCode MatPartitioningSetFromOptions(MatPartitioning part)
{
  PetscBool   flag;
  char        type[256];
  const char *def;

  PetscFunctionBegin;
  PetscObjectOptionsBegin((PetscObject)part);
  if (!((PetscObject)part)->type_name) {
#if defined(PETSC_HAVE_PARMETIS)
    def = MATPARTITIONINGPARMETIS;
#elif defined(PETSC_HAVE_CHACO)
    def = MATPARTITIONINGCHACO;
#elif defined(PETSC_HAVE_PARTY)
    def = MATPARTITIONINGPARTY;
#elif defined(PETSC_HAVE_PTSCOTCH)
    def = MATPARTITIONINGPTSCOTCH;
#else
    def = MATPARTITIONINGCURRENT;
#endif
  } else {
    def = ((PetscObject)part)->type_name;
  }
  PetscCall(PetscOptionsFList("-mat_partitioning_type", "Type of partitioner", "MatPartitioningSetType", MatPartitioningList, def, type, 256, &flag));
  if (flag) PetscCall(MatPartitioningSetType(part, type));

  PetscCall(PetscOptionsInt("-mat_partitioning_nparts", "number of fine parts", NULL, part->n, &part->n, &flag));

  PetscCall(PetscOptionsBool("-mat_partitioning_use_edge_weights", "whether or not to use edge weights", NULL, part->use_edge_weights, &part->use_edge_weights, &flag));

  /*
    Set the type if it was never set.
  */
  if (!((PetscObject)part)->type_name) PetscCall(MatPartitioningSetType(part, def));

  PetscTryTypeMethod(part, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningSetNumberVertexWeights - Sets the number of weights per vertex

   Not collective

   Input Parameters:
+  partitioning - the partitioning context
-  ncon - the number of weights

   Level: intermediate

.seealso: `MatPartitioning`, `MatPartitioningSetVertexWeights()`
@*/
PetscErrorCode MatPartitioningSetNumberVertexWeights(MatPartitioning partitioning, PetscInt ncon)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(partitioning, MAT_PARTITIONING_CLASSID, 1);
  partitioning->ncon = ncon;
  PetscFunctionReturn(0);
}
