
#include <petsc/private/matimpl.h>               /*I "petscmat.h" I*/

/* Logging support */
PetscClassId MAT_PARTITIONING_CLASSID;

/*
   Simplest partitioning, keeps the current partitioning.
*/
static PetscErrorCode MatPartitioningApply_Current(MatPartitioning part,IS *partitioning)
{
  PetscErrorCode ierr;
  PetscInt       m;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)part),&size);CHKERRQ(ierr);
  if (part->n != size) {
    const char *prefix;
    ierr = PetscObjectGetOptionsPrefix((PetscObject)part,&prefix);CHKERRQ(ierr);
    SETERRQ1(PetscObjectComm((PetscObject)part),PETSC_ERR_SUP,"This is the DEFAULT NO-OP partitioner, it currently only supports one domain per processor\nuse -%smat_partitioning_type parmetis or chaco or ptscotch for more than one subdomain per processor",prefix ? prefix : "");
  }
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)part),&rank);CHKERRQ(ierr);

  ierr = MatGetLocalSize(part->adj,&m,NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(PetscObjectComm((PetscObject)part),m,rank,0,partitioning);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   partition an index to rebalance the computation
*/
static PetscErrorCode MatPartitioningApply_Average(MatPartitioning part,IS *partitioning)
{
  PetscErrorCode ierr;
  PetscInt       m,M,nparts,*indices,r,d,*parts,i,start,end,loc;

  PetscFunctionBegin;
  ierr   = MatGetSize(part->adj,&M,NULL);CHKERRQ(ierr);
  ierr   = MatGetLocalSize(part->adj,&m,NULL);CHKERRQ(ierr);
  nparts = part->n;
  ierr   = PetscMalloc1(nparts,&parts);CHKERRQ(ierr);
  d      = M/nparts;
  for (i=0; i<nparts; i++) parts[i] = d;
  r = M%nparts;
  for (i=0; i<r; i++) parts[i] += 1;
  for (i=1; i<nparts; i++) parts[i] += parts[i-1];
  ierr = PetscMalloc1(m,&indices);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(part->adj,&start,&end);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    ierr = PetscFindInt(i,nparts,parts,&loc);CHKERRQ(ierr);
    if (loc<0) loc = -(loc+1);
    else loc = loc+1;
    indices[i-start] = loc;
  }
  ierr = PetscFree(parts);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),m,indices,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPartitioningApply_Square(MatPartitioning part,IS *partitioning)
{
  PetscErrorCode ierr;
  PetscInt       cell,n,N,p,rstart,rend,*color;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)part),&size);CHKERRQ(ierr);
  if (part->n != size) SETERRQ(PetscObjectComm((PetscObject)part),PETSC_ERR_SUP,"Currently only supports one domain per processor");
  p = (PetscInt)PetscSqrtReal((PetscReal)part->n);
  if (p*p != part->n) SETERRQ(PetscObjectComm((PetscObject)part),PETSC_ERR_SUP,"Square partitioning requires \"perfect square\" number of domains");

  ierr = MatGetSize(part->adj,&N,NULL);CHKERRQ(ierr);
  n    = (PetscInt)PetscSqrtReal((PetscReal)N);
  if (n*n != N) SETERRQ(PetscObjectComm((PetscObject)part),PETSC_ERR_SUP,"Square partitioning requires square domain");
  if (n%p != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Square partitioning requires p to divide n");
  ierr = MatGetOwnershipRange(part->adj,&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscMalloc1(rend-rstart,&color);CHKERRQ(ierr);
  /* for (int cell=rstart; cell<rend; cell++) { color[cell-rstart] = ((cell%n) < (n/2)) + 2 * ((cell/n) < (n/2)); } */
  for (cell=rstart; cell<rend; cell++) {
    color[cell-rstart] = ((cell%n) / (n/p)) + p * ((cell/n) / (n/p));
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),rend-rstart,color,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Current(MatPartitioning part)
{
  PetscFunctionBegin;
  part->ops->apply   = MatPartitioningApply_Current;
  part->ops->view    = 0;
  part->ops->destroy = 0;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Average(MatPartitioning part)
{
  PetscFunctionBegin;
  part->ops->apply   = MatPartitioningApply_Average;
  part->ops->view    = 0;
  part->ops->destroy = 0;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Square(MatPartitioning part)
{
  PetscFunctionBegin;
  part->ops->apply   = MatPartitioningApply_Square;
  part->ops->view    = 0;
  part->ops->destroy = 0;
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
  PetscInt       l2p,i,pTree,pStartTree;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  l2p = PetscLog2Real(p);
  if (l2p - (PetscInt)PetscLog2Real(p)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"%D is not a power of 2",p);
  if (!p) PetscFunctionReturn(0);
  ierr = PetscArrayzero(seps,2*p-2);CHKERRQ(ierr);
  ierr = PetscArrayzero(level,p-1);CHKERRQ(ierr);
  seps[2*p-2] = sizes[2*p-2];
  pTree = p;
  pStartTree = 0;
  while (pTree != 1) {
    for (i = pStartTree; i < pStartTree + pTree; i++) {
      seps[i] += sizes[i];
      seps[pStartTree + pTree + (i-pStartTree)/2] += seps[i];
    }
    pStartTree += pTree;
    pTree = pTree/2;
  }
  seps[2*p-2] -= sizes[2*p-2];

  pStartTree = 2*p-2;
  pTree      = 1;
  while (pStartTree > 0) {
    for (i = pStartTree; i < pStartTree + pTree; i++) {
      PetscInt k = 2*i - (pStartTree +2*pTree);
      PetscInt n = seps[k+1];

      seps[k+1]  = seps[i]   - sizes[k+1];
      seps[k]    = seps[k+1] + sizes[k+1] - n - sizes[k];
      level[i-p] = -pTree - i + pStartTree;
    }
    pTree *= 2;
    pStartTree -= pTree;
  }
  /* I know there should be a formula */
  ierr = PetscSortIntWithArrayPair(p-1,seps+p,sizes+p,level);CHKERRQ(ierr);
  for (i=2*p-2;i>=0;i--) { seps[2*i] = seps[i]; seps[2*i+1] = seps[i] + PetscMax(sizes[i] - 1,0); }
  PetscFunctionReturn(0);
}

/* ===========================================================================================*/

PetscFunctionList MatPartitioningList              = 0;
PetscBool         MatPartitioningRegisterAllCalled = PETSC_FALSE;


/*@C
   MatPartitioningRegister - Adds a new sparse matrix partitioning to the  matrix package.

   Not Collective

   Input Parameters:
+  sname - name of partitioning (for example MATPARTITIONINGCURRENT) or parmetis
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

.seealso: MatPartitioningRegisterDestroy(), MatPartitioningRegisterAll()
@*/
PetscErrorCode  MatPartitioningRegister(const char sname[],PetscErrorCode (*function)(MatPartitioning))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&MatPartitioningList,sname,function);CHKERRQ(ierr);
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

@*/
PetscErrorCode  MatPartitioningGetType(MatPartitioning partitioning,MatPartitioningType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(partitioning,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)partitioning)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningSetNParts - Set how many partitions need to be created;
        by default this is one per processor. Certain partitioning schemes may
        in fact only support that option.

   Not collective

   Input Parameter:
+  partitioning - the partitioning context
-  n - the number of partitions

   Level: intermediate

   Not Collective

.seealso: MatPartitioningCreate(), MatPartitioningApply()
@*/
PetscErrorCode  MatPartitioningSetNParts(MatPartitioning part,PetscInt n)
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

   Level: beginner

   The user can define additional partitionings; see MatPartitioningRegister().

.seealso:  MatPartitioningRegister(), MatPartitioningCreate(),
           MatPartitioningDestroy(), MatPartitioningSetAdjacency(), ISPartitioningToNumbering(),
           ISPartitioningCount()
@*/
PetscErrorCode  MatPartitioningApplyND(MatPartitioning matp,IS *partitioning)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(matp,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(partitioning,2);
  if (!matp->adj->assembled) SETERRQ(PetscObjectComm((PetscObject)matp),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (matp->adj->factortype) SETERRQ(PetscObjectComm((PetscObject)matp),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  if (!matp->ops->applynd) SETERRQ1(PetscObjectComm((PetscObject)matp),PETSC_ERR_SUP,"Nested dissection not provided by MatPartitioningType %s",((PetscObject)matp)->type_name);
  ierr = PetscLogEventBegin(MAT_PartitioningND,matp,0,0,0);CHKERRQ(ierr);
  ierr = (*matp->ops->applynd)(matp,partitioning);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PartitioningND,matp,0,0,0);CHKERRQ(ierr);

  ierr = MatPartitioningViewFromOptions(matp,NULL,"-mat_partitioning_view");CHKERRQ(ierr);
  ierr = ISViewFromOptions(*partitioning,NULL,"-mat_partitioning_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningApply - Gets a partitioning for a matrix.

   Collective on Mat

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

   The user can define additional partitionings; see MatPartitioningRegister().

.seealso:  MatPartitioningRegister(), MatPartitioningCreate(),
           MatPartitioningDestroy(), MatPartitioningSetAdjacency(), ISPartitioningToNumbering(),
           ISPartitioningCount()
@*/
PetscErrorCode  MatPartitioningApply(MatPartitioning matp,IS *partitioning)
{
  PetscErrorCode ierr;
  PetscBool      viewbalance,improve;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(matp,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(partitioning,2);
  if (!matp->adj->assembled) SETERRQ(PetscObjectComm((PetscObject)matp),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (matp->adj->factortype) SETERRQ(PetscObjectComm((PetscObject)matp),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  if (!matp->ops->apply) SETERRQ(PetscObjectComm((PetscObject)matp),PETSC_ERR_ARG_WRONGSTATE,"Must set type with MatPartitioningSetFromOptions() or MatPartitioningSetType()");
  ierr = PetscLogEventBegin(MAT_Partitioning,matp,0,0,0);CHKERRQ(ierr);
  ierr = (*matp->ops->apply)(matp,partitioning);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Partitioning,matp,0,0,0);CHKERRQ(ierr);

  ierr = MatPartitioningViewFromOptions(matp,NULL,"-mat_partitioning_view");CHKERRQ(ierr);
  ierr = ISViewFromOptions(*partitioning,NULL,"-mat_partitioning_view");CHKERRQ(ierr);

  ierr = PetscObjectOptionsBegin((PetscObject)matp);CHKERRQ(ierr);
  viewbalance = PETSC_FALSE;
  ierr = PetscOptionsBool("-mat_partitioning_view_imbalance","Display imbalance information of a partition",NULL,PETSC_FALSE,&viewbalance,NULL);CHKERRQ(ierr);
  improve = PETSC_FALSE;
  ierr = PetscOptionsBool("-mat_partitioning_improve","Improve the quality of a partition",NULL,PETSC_FALSE,&improve,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (improve) {
    ierr = MatPartitioningImprove(matp,partitioning);CHKERRQ(ierr);
  }

  if (viewbalance) {
    ierr = MatPartitioningViewImbalance(matp,*partitioning);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*@
   MatPartitioningImprove - Improves the quality of a given partition.

   Collective on Mat

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


.seealso:  MatPartitioningApply(), MatPartitioningCreate(),
           MatPartitioningDestroy(), MatPartitioningSetAdjacency(), ISPartitioningToNumbering(),
           ISPartitioningCount()
@*/
PetscErrorCode  MatPartitioningImprove(MatPartitioning matp,IS *partitioning)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(matp,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(partitioning,2);
  if (!matp->adj->assembled) SETERRQ(PetscObjectComm((PetscObject)matp),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (matp->adj->factortype) SETERRQ(PetscObjectComm((PetscObject)matp),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  ierr = PetscLogEventBegin(MAT_Partitioning,matp,0,0,0);CHKERRQ(ierr);
  if (matp->ops->improve) {
    ierr = (*matp->ops->improve)(matp,partitioning);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_Partitioning,matp,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningViewImbalance - Display partitioning imbalance information.

   Collective on MatPartitioning

   Input Parameters:
+  matp - the matrix partitioning object
-  partitioning - the partitioning. For each local node this tells the processor
                   number that that node is assigned to.

   Options Database Keys:
   To see the partitioning imbalance information
$    -mat_partitioning_view_balance

   Level: beginner

.seealso:  MatPartitioningApply(), MatPartitioningView()
@*/
PetscErrorCode  MatPartitioningViewImbalance(MatPartitioning matp, IS partitioning)
{
  PetscErrorCode  ierr;
  PetscInt        nparts,*subdomainsizes,*subdomainsizes_tmp,nlocal,i,maxsub,minsub,avgsub;
  const PetscInt  *indices;
  PetscViewer     viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(matp,MAT_PARTITIONING_CLASSID,1);
  PetscValidHeaderSpecific(partitioning,IS_CLASSID,2);
  nparts = matp->n;
  ierr = PetscCalloc2(nparts,&subdomainsizes,nparts,&subdomainsizes_tmp);CHKERRQ(ierr);
  ierr = ISGetLocalSize(partitioning,&nlocal);CHKERRQ(ierr);
  ierr = ISGetIndices(partitioning,&indices);CHKERRQ(ierr);
  for (i=0;i<nlocal;i++) {
    subdomainsizes_tmp[indices[i]] += matp->vertex_weights? matp->vertex_weights[i]:1;
  }
  ierr = MPI_Allreduce(subdomainsizes_tmp,subdomainsizes,nparts,MPIU_INT,MPI_SUM, PetscObjectComm((PetscObject)matp));CHKERRQ(ierr);
  ierr = ISRestoreIndices(partitioning,&indices);CHKERRQ(ierr);
  minsub = PETSC_MAX_INT, maxsub = PETSC_MIN_INT, avgsub=0;
  for (i=0; i<nparts; i++) {
    minsub = PetscMin(minsub,subdomainsizes[i]);
    maxsub = PetscMax(maxsub,subdomainsizes[i]);
    avgsub += subdomainsizes[i];
  }
  avgsub /=nparts;
  ierr = PetscFree2(subdomainsizes,subdomainsizes_tmp);CHKERRQ(ierr);
  ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)matp),&viewer);CHKERRQ(ierr);
  ierr = MatPartitioningView(matp,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Partitioning Imbalance Info: Max %D, Min %D, Avg %D, R %g\n",maxsub, minsub, avgsub, (double)(maxsub/(PetscReal)minsub));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningSetAdjacency - Sets the adjacency graph (matrix) of the thing to be
      partitioned.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  adj - the adjacency matrix

   Level: beginner

.seealso: MatPartitioningCreate()
@*/
PetscErrorCode  MatPartitioningSetAdjacency(MatPartitioning part,Mat adj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidHeaderSpecific(adj,MAT_CLASSID,2);
  part->adj = adj;
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningDestroy - Destroys the partitioning context.

   Collective on Partitioning

   Input Parameters:
.  part - the partitioning context

   Level: beginner

.seealso: MatPartitioningCreate()
@*/
PetscErrorCode  MatPartitioningDestroy(MatPartitioning *part)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*part) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*part),MAT_PARTITIONING_CLASSID,1);
  if (--((PetscObject)(*part))->refct > 0) {*part = 0; PetscFunctionReturn(0);}

  if ((*part)->ops->destroy) {
    ierr = (*(*part)->ops->destroy)((*part));CHKERRQ(ierr);
  }
  ierr = PetscFree((*part)->vertex_weights);CHKERRQ(ierr);
  ierr = PetscFree((*part)->part_weights);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningSetVertexWeights - Sets the weights for vertices for a partitioning.

   Logically Collective on Partitioning

   Input Parameters:
+  part - the partitioning context
-  weights - the weights, on each process this array must have the same size as the number of local rows

   Level: beginner

   Notes:
      The array weights is freed by PETSc so the user should not free the array. In C/C++
   the array must be obtained with a call to PetscMalloc(), not malloc().

.seealso: MatPartitioningCreate(), MatPartitioningSetType(), MatPartitioningSetPartitionWeights()
@*/
PetscErrorCode  MatPartitioningSetVertexWeights(MatPartitioning part,const PetscInt weights[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  ierr = PetscFree(part->vertex_weights);CHKERRQ(ierr);
  part->vertex_weights = (PetscInt*)weights;
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningSetPartitionWeights - Sets the weights for each partition.

   Logically Collective on Partitioning

   Input Parameters:
+  part - the partitioning context
-  weights - An array of size nparts that is used to specify the fraction of
             vertex weight that should be distributed to each sub-domain for
             the balance constraint. If all of the sub-domains are to be of
             the same size, then each of the nparts elements should be set
             to a value of 1/nparts. Note that the sum of all of the weights
             should be one.

   Level: beginner

   Notes:
      The array weights is freed by PETSc so the user should not free the array. In C/C++
   the array must be obtained with a call to PetscMalloc(), not malloc().

.seealso: MatPartitioningCreate(), MatPartitioningSetType(), MatPartitioningSetVertexWeights()
@*/
PetscErrorCode  MatPartitioningSetPartitionWeights(MatPartitioning part,const PetscReal weights[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  ierr = PetscFree(part->part_weights);CHKERRQ(ierr);
  part->part_weights = (PetscReal*)weights;
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

.seealso: MatPartitioningSetType(), MatPartitioningApply(), MatPartitioningDestroy(),
          MatPartitioningSetAdjacency()

@*/
PetscErrorCode  MatPartitioningCreate(MPI_Comm comm,MatPartitioning *newp)
{
  MatPartitioning part;
  PetscErrorCode  ierr;
  PetscMPIInt     size;

  PetscFunctionBegin;
  *newp = 0;

  ierr = MatInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(part,MAT_PARTITIONING_CLASSID,"MatPartitioning","Matrix/graph partitioning","MatOrderings",comm,MatPartitioningDestroy,MatPartitioningView);CHKERRQ(ierr);
  part->vertex_weights = NULL;
  part->part_weights   = NULL;

  ierr    = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  part->n = (PetscInt)size;

  *newp = part;
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningViewFromOptions - View from Options

   Collective on MatPartitioning

   Input Parameters:
+  A - the partitioning context
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  MatPartitioning, MatPartitioningView, PetscObjectViewFromOptions(), MatPartitioningCreate()
@*/
PetscErrorCode  MatPartitioningViewFromOptions(MatPartitioning A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_PARTITIONING_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningView - Prints the partitioning data structure.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  viewer - optional visualization context

   Level: intermediate

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The user can open alternative visualization contexts with
.     PetscViewerASCIIOpen() - output to a specified file

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  MatPartitioningView(MatPartitioning part,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)part),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(part,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)part,viewer);CHKERRQ(ierr);
    if (part->vertex_weights) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using vertex weights\n");CHKERRQ(ierr);
    }
  }
  if (part->ops->view) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = (*part->ops->view)(part,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   MatPartitioningSetType - Sets the type of partitioner to use

   Collective on MatPartitioning

   Input Parameter:
+  part - the partitioning context.
-  type - a known method

   Options Database Command:
$  -mat_partitioning_type  <type>
$      Use -help for a list of available methods
$      (for instance, parmetis)

   Level: intermediate

.seealso: MatPartitioningCreate(), MatPartitioningApply(), MatPartitioningType

@*/
PetscErrorCode  MatPartitioningSetType(MatPartitioning part,MatPartitioningType type)
{
  PetscErrorCode ierr,(*r)(MatPartitioning);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)part,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (part->ops->destroy) {
    ierr = (*part->ops->destroy)(part);CHKERRQ(ierr);
    part->ops->destroy = NULL;
  }
  part->setupcalled = 0;
  part->data        = 0;
  ierr = PetscMemzero(part->ops,sizeof(struct _MatPartitioningOps));CHKERRQ(ierr);

  ierr = PetscFunctionListFind(MatPartitioningList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)part),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown partitioning type %s",type);

  ierr = (*r)(part);CHKERRQ(ierr);

  ierr = PetscFree(((PetscObject)part)->type_name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type,&((PetscObject)part)->type_name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningSetFromOptions - Sets various partitioning options from the
        options database.

   Collective on MatPartitioning

   Input Parameter:
.  part - the partitioning context.

   Options Database Command:
$  -mat_partitioning_type  <type>
$      Use -help for a list of available methods
$      (for instance, parmetis)
$  -mat_partitioning_nparts - number of subgraphs


   Notes:
    If the partitioner has not been set by the user it uses one of the installed partitioner such as ParMetis. If there are
   no installed partitioners it uses current which means no repartioning.

   Level: beginner

@*/
PetscErrorCode  MatPartitioningSetFromOptions(MatPartitioning part)
{
  PetscErrorCode ierr;
  PetscBool      flag;
  char           type[256];
  const char     *def;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject)part);CHKERRQ(ierr);
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
  ierr = PetscOptionsFList("-mat_partitioning_type","Type of partitioner","MatPartitioningSetType",MatPartitioningList,def,type,256,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatPartitioningSetType(part,type);CHKERRQ(ierr);
  }

  ierr = PetscOptionsInt("-mat_partitioning_nparts","number of fine parts",NULL,part->n,& part->n,&flag);CHKERRQ(ierr);

  /*
    Set the type if it was never set.
  */
  if (!((PetscObject)part)->type_name) {
    ierr = MatPartitioningSetType(part,def);CHKERRQ(ierr);
  }

  if (part->ops->setfromoptions) {
    ierr = (*part->ops->setfromoptions)(PetscOptionsObject,part);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
