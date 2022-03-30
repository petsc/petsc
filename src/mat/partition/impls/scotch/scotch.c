
#include <../src/mat/impls/adj/mpi/mpiadj.h>       /*I "petscmat.h" I*/

EXTERN_C_BEGIN
#include <ptscotch.h>
#if defined(PETSC_HAVE_SCOTCH_PARMETIS_V3_NODEND)
/* we define the prototype instead of include SCOTCH's parmetis.h */
void SCOTCH_ParMETIS_V3_NodeND(const SCOTCH_Num * const,SCOTCH_Num * const, SCOTCH_Num * const,const SCOTCH_Num * const,const SCOTCH_Num * const,SCOTCH_Num * const,SCOTCH_Num * const,MPI_Comm *);
#endif
EXTERN_C_END

typedef struct {
  double     imbalance;
  SCOTCH_Num strategy;
} MatPartitioning_PTScotch;

/*@
   MatPartitioningPTScotchSetImbalance - Sets the value of the load imbalance
   ratio to be used during strategy selection.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  imb  - the load imbalance ratio

   Options Database:
.  -mat_partitioning_ptscotch_imbalance <imb> - set load imbalance ratio

   Note:
   Must be in the range [0,1]. The default value is 0.01.

   Level: advanced

.seealso: MatPartitioningPTScotchSetStrategy(), MatPartitioningPTScotchGetImbalance()
@*/
PetscErrorCode MatPartitioningPTScotchSetImbalance(MatPartitioning part,PetscReal imb)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveReal(part,imb,2);
  PetscTryMethod(part,"MatPartitioningPTScotchSetImbalance_C",(MatPartitioning,PetscReal),(part,imb));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningPTScotchSetImbalance_PTScotch(MatPartitioning part,PetscReal imb)
{
  MatPartitioning_PTScotch *scotch = (MatPartitioning_PTScotch*)part->data;

  PetscFunctionBegin;
  if (imb==PETSC_DEFAULT) scotch->imbalance = 0.01;
  else {
    PetscCheck(imb>=0.0 && imb<=1.0,PetscObjectComm((PetscObject)part),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of imb. Must be in range [0,1]");
    scotch->imbalance = (double)imb;
  }
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningPTScotchGetImbalance - Gets the value of the load imbalance
   ratio used during strategy selection.

   Not Collective

   Input Parameter:
.  part - the partitioning context

   Output Parameter:
.  imb  - the load imbalance ratio

   Level: advanced

.seealso: MatPartitioningPTScotchSetImbalance()
@*/
PetscErrorCode MatPartitioningPTScotchGetImbalance(MatPartitioning part,PetscReal *imb)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(imb,2);
  PetscUseMethod(part,"MatPartitioningPTScotchGetImbalance_C",(MatPartitioning,PetscReal*),(part,imb));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningPTScotchGetImbalance_PTScotch(MatPartitioning part,PetscReal *imb)
{
  MatPartitioning_PTScotch *scotch = (MatPartitioning_PTScotch*)part->data;

  PetscFunctionBegin;
  *imb = scotch->imbalance;
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningPTScotchSetStrategy - Sets the strategy to be used in PTScotch.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  strategy - the strategy, one of
.vb
     MP_PTSCOTCH_DEFAULT     - Default behavior
     MP_PTSCOTCH_QUALITY     - Prioritize quality over speed
     MP_PTSCOTCH_SPEED       - Prioritize speed over quality
     MP_PTSCOTCH_BALANCE     - Enforce load balance
     MP_PTSCOTCH_SAFETY      - Avoid methods that may fail
     MP_PTSCOTCH_SCALABILITY - Favor scalability as much as possible
.ve

   Options Database:
.  -mat_partitioning_ptscotch_strategy [quality,speed,balance,safety,scalability] - strategy

   Level: advanced

   Notes:
   The default is MP_SCOTCH_QUALITY. See the PTScotch documentation for more information.

.seealso: MatPartitioningPTScotchSetImbalance(), MatPartitioningPTScotchGetStrategy()
@*/
PetscErrorCode MatPartitioningPTScotchSetStrategy(MatPartitioning part,MPPTScotchStrategyType strategy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveEnum(part,strategy,2);
  PetscTryMethod(part,"MatPartitioningPTScotchSetStrategy_C",(MatPartitioning,MPPTScotchStrategyType),(part,strategy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningPTScotchSetStrategy_PTScotch(MatPartitioning part,MPPTScotchStrategyType strategy)
{
  MatPartitioning_PTScotch *scotch = (MatPartitioning_PTScotch*)part->data;

  PetscFunctionBegin;
  switch (strategy) {
  case MP_PTSCOTCH_QUALITY:     scotch->strategy = SCOTCH_STRATQUALITY; break;
  case MP_PTSCOTCH_SPEED:       scotch->strategy = SCOTCH_STRATSPEED; break;
  case MP_PTSCOTCH_BALANCE:     scotch->strategy = SCOTCH_STRATBALANCE; break;
  case MP_PTSCOTCH_SAFETY:      scotch->strategy = SCOTCH_STRATSAFETY; break;
  case MP_PTSCOTCH_SCALABILITY: scotch->strategy = SCOTCH_STRATSCALABILITY; break;
  default:                      scotch->strategy = SCOTCH_STRATDEFAULT; break;
  }
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningPTScotchGetStrategy - Gets the strategy used in PTScotch.

   Not Collective

   Input Parameter:
.  part - the partitioning context

   Output Parameter:
.  strategy - the strategy

   Level: advanced

.seealso: MatPartitioningPTScotchSetStrategy()
@*/
PetscErrorCode MatPartitioningPTScotchGetStrategy(MatPartitioning part,MPPTScotchStrategyType *strategy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(strategy,2);
  PetscUseMethod(part,"MatPartitioningPTScotchGetStrategy_C",(MatPartitioning,MPPTScotchStrategyType*),(part,strategy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningPTScotchGetStrategy_PTScotch(MatPartitioning part,MPPTScotchStrategyType *strategy)
{
  MatPartitioning_PTScotch *scotch = (MatPartitioning_PTScotch*)part->data;

  PetscFunctionBegin;
  switch (scotch->strategy) {
  case SCOTCH_STRATQUALITY:     *strategy = MP_PTSCOTCH_QUALITY; break;
  case SCOTCH_STRATSPEED:       *strategy = MP_PTSCOTCH_SPEED; break;
  case SCOTCH_STRATBALANCE:     *strategy = MP_PTSCOTCH_BALANCE; break;
  case SCOTCH_STRATSAFETY:      *strategy = MP_PTSCOTCH_SAFETY; break;
  case SCOTCH_STRATSCALABILITY: *strategy = MP_PTSCOTCH_SCALABILITY; break;
  default:                      *strategy = MP_PTSCOTCH_DEFAULT; break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningView_PTScotch(MatPartitioning part, PetscViewer viewer)
{
  MatPartitioning_PTScotch *scotch = (MatPartitioning_PTScotch*)part->data;
  PetscBool                isascii;
  const char               *str=NULL;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    switch (scotch->strategy) {
    case SCOTCH_STRATQUALITY:     str = "Prioritize quality over speed"; break;
    case SCOTCH_STRATSPEED:       str = "Prioritize speed over quality"; break;
    case SCOTCH_STRATBALANCE:     str = "Enforce load balance"; break;
    case SCOTCH_STRATSAFETY:      str = "Avoid methods that may fail"; break;
    case SCOTCH_STRATSCALABILITY: str = "Favor scalability as much as possible"; break;
    default:                      str = "Default behavior"; break;
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Strategy=%s\n",str));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Load imbalance ratio=%g\n",scotch->imbalance));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningSetFromOptions_PTScotch(PetscOptionItems *PetscOptionsObject,MatPartitioning part)
{
  PetscBool                flag;
  PetscReal                r;
  MatPartitioning_PTScotch *scotch = (MatPartitioning_PTScotch*)part->data;
  MPPTScotchStrategyType   strat;

  PetscFunctionBegin;
  PetscCall(MatPartitioningPTScotchGetStrategy(part,&strat));
  PetscCall(PetscOptionsHead(PetscOptionsObject,"PTScotch partitioning options"));
  PetscCall(PetscOptionsEnum("-mat_partitioning_ptscotch_strategy","Strategy","MatPartitioningPTScotchSetStrategy",MPPTScotchStrategyTypes,(PetscEnum)strat,(PetscEnum*)&strat,&flag));
  if (flag) PetscCall(MatPartitioningPTScotchSetStrategy(part,strat));
  PetscCall(PetscOptionsReal("-mat_partitioning_ptscotch_imbalance","Load imbalance ratio","MatPartitioningPTScotchSetImbalance",scotch->imbalance,&r,&flag));
  if (flag) PetscCall(MatPartitioningPTScotchSetImbalance(part,r));
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPartitioningApply_PTScotch_Private(MatPartitioning part, PetscBool useND, IS *partitioning)
{
  MPI_Comm                 pcomm,comm;
  MatPartitioning_PTScotch *scotch = (MatPartitioning_PTScotch*)part->data;
  PetscMPIInt              rank;
  Mat                      mat  = part->adj;
  Mat_MPIAdj               *adj = (Mat_MPIAdj*)mat->data;
  PetscBool                flg,distributed;
  PetscBool                proc_weight_flg;
  PetscInt                 i,j,p,bs=1,nold;
  PetscInt                 *NDorder = NULL;
  PetscReal                *vwgttab,deltval;
  SCOTCH_Num               *locals,*velotab,*veloloctab,*edloloctab,vertlocnbr,edgelocnbr,nparts=part->n;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)part,&pcomm));
  /* Duplicate the communicator to be sure that PTSCOTCH attribute caching does not interfere with PETSc. */
  PetscCallMPI(MPI_Comm_dup(pcomm,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATMPIADJ,&flg));
  if (!flg) {
    /* bs indicates if the converted matrix is "reduced" from the original and hence the
       resulting partition results need to be stretched to match the original matrix */
    nold = mat->rmap->n;
    PetscCall(MatConvert(mat,MATMPIADJ,MAT_INITIAL_MATRIX,&mat));
    if (mat->rmap->n > 0) bs = nold/mat->rmap->n;
    adj  = (Mat_MPIAdj*)mat->data;
  }

  proc_weight_flg = part->part_weights ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mat_partitioning_ptscotch_proc_weight", &proc_weight_flg, NULL));

  PetscCall(PetscMalloc1(mat->rmap->n+1,&locals));

  if (useND) {
#if defined(PETSC_HAVE_SCOTCH_PARMETIS_V3_NODEND)
    PetscInt    *sizes, *seps, log2size, subd, *level, base = 0;
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(comm,&size));
    log2size = PetscLog2Real(size);
    subd = PetscPowInt(2,log2size);
    PetscCheck(subd == size,comm,PETSC_ERR_SUP,"Only power of 2 communicator sizes");
    PetscCall(PetscMalloc1(mat->rmap->n,&NDorder));
    PetscCall(PetscMalloc3(2*size,&sizes,4*size,&seps,size,&level));
    SCOTCH_ParMETIS_V3_NodeND(mat->rmap->range,adj->i,adj->j,&base,NULL,NDorder,sizes,&comm);
    PetscCall(MatPartitioningSizesToSep_Private(subd,sizes,seps,level));
    for (i=0;i<mat->rmap->n;i++) {
      PetscInt loc;

      PetscCall(PetscFindInt(NDorder[i],2*subd,seps,&loc));
      if (loc < 0) {
        loc = -(loc+1);
        if (loc%2) { /* part of subdomain */
          locals[i] = loc/2;
        } else {
          PetscCall(PetscFindInt(NDorder[i],2*(subd-1),seps+2*subd,&loc));
          loc = loc < 0 ? -(loc+1)/2 : loc/2;
          locals[i] = level[loc];
        }
      } else locals[i] = loc/2;
    }
    PetscCall(PetscFree3(sizes,seps,level));
#else
    SETERRQ(pcomm,PETSC_ERR_SUP,"Need libptscotchparmetis.a compiled with -DSCOTCH_METIS_PREFIX");
#endif
  } else {
    velotab = NULL;
    if (proc_weight_flg) {
      PetscCall(PetscMalloc1(nparts,&vwgttab));
      PetscCall(PetscMalloc1(nparts,&velotab));
      for (j=0; j<nparts; j++) {
        if (part->part_weights) vwgttab[j] = part->part_weights[j]*nparts;
        else vwgttab[j] = 1.0;
      }
      for (i=0; i<nparts; i++) {
        deltval = PetscAbsReal(vwgttab[i]-PetscFloorReal(vwgttab[i]+0.5));
        if (deltval>0.01) {
          for (j=0; j<nparts; j++) vwgttab[j] /= deltval;
        }
      }
      for (i=0; i<nparts; i++) velotab[i] = (SCOTCH_Num)(vwgttab[i] + 0.5);
      PetscCall(PetscFree(vwgttab));
    }

    vertlocnbr = mat->rmap->range[rank+1] - mat->rmap->range[rank];
    edgelocnbr = adj->i[vertlocnbr];
    veloloctab = part->vertex_weights;
    edloloctab = part->use_edge_weights? adj->values:NULL;

    /* detect whether all vertices are located at the same process in original graph */
    for (p = 0; !mat->rmap->range[p+1] && p < nparts; ++p);
    distributed = (mat->rmap->range[p+1] == mat->rmap->N) ? PETSC_FALSE : PETSC_TRUE;
    if (distributed) {
      SCOTCH_Arch     archdat;
      SCOTCH_Dgraph   grafdat;
      SCOTCH_Dmapping mappdat;
      SCOTCH_Strat    stradat;

      PetscCall(SCOTCH_dgraphInit(&grafdat,comm));
      PetscCall(SCOTCH_dgraphBuild(&grafdat,0,vertlocnbr,vertlocnbr,adj->i,adj->i+1,veloloctab,
                                 NULL,edgelocnbr,edgelocnbr,adj->j,NULL,edloloctab));

      if (PetscDefined(USE_DEBUG)) PetscCall(SCOTCH_dgraphCheck(&grafdat));

      PetscCall(SCOTCH_archInit(&archdat));
      PetscCall(SCOTCH_stratInit(&stradat));
      PetscCall(SCOTCH_stratDgraphMapBuild(&stradat,scotch->strategy,nparts,nparts,scotch->imbalance));

      if (velotab) {
        PetscCall(SCOTCH_archCmpltw(&archdat,nparts,velotab));
      } else {
        PetscCall(SCOTCH_archCmplt(&archdat,nparts));
      }
      PetscCall(SCOTCH_dgraphMapInit(&grafdat,&mappdat,&archdat,locals));
      PetscCall(SCOTCH_dgraphMapCompute(&grafdat,&mappdat,&stradat));

      SCOTCH_dgraphMapExit(&grafdat,&mappdat);
      SCOTCH_archExit(&archdat);
      SCOTCH_stratExit(&stradat);
      SCOTCH_dgraphExit(&grafdat);

    } else if (rank == p) {
      SCOTCH_Graph grafdat;
      SCOTCH_Strat stradat;

      PetscCall(SCOTCH_graphInit(&grafdat));
      PetscCall(SCOTCH_graphBuild(&grafdat,0,vertlocnbr,adj->i,adj->i+1,veloloctab,NULL,edgelocnbr,adj->j,edloloctab));
      if (PetscDefined(USE_DEBUG)) PetscCall(SCOTCH_graphCheck(&grafdat));
      PetscCall(SCOTCH_stratInit(&stradat));
      PetscCall(SCOTCH_stratGraphMapBuild(&stradat,scotch->strategy,nparts,scotch->imbalance));
      if (velotab) {
        SCOTCH_Arch archdat;
        PetscCall(SCOTCH_archInit(&archdat));
        PetscCall(SCOTCH_archCmpltw(&archdat,nparts,velotab));
        PetscCall(SCOTCH_graphMap(&grafdat,&archdat,&stradat,locals));
        SCOTCH_archExit(&archdat);
      } else {
        PetscCall(SCOTCH_graphPart(&grafdat,nparts,&stradat,locals));
      }
      SCOTCH_stratExit(&stradat);
      SCOTCH_graphExit(&grafdat);
    }

    PetscCall(PetscFree(velotab));
  }
  PetscCallMPI(MPI_Comm_free(&comm));

  if (bs > 1) {
    PetscInt *newlocals;
    PetscCall(PetscMalloc1(bs*mat->rmap->n,&newlocals));
    for (i=0;i<mat->rmap->n;i++) {
      for (j=0;j<bs;j++) {
        newlocals[bs*i+j] = locals[i];
      }
    }
    PetscCall(PetscFree(locals));
    PetscCall(ISCreateGeneral(pcomm,bs*mat->rmap->n,newlocals,PETSC_OWN_POINTER,partitioning));
  } else {
    PetscCall(ISCreateGeneral(pcomm,mat->rmap->n,locals,PETSC_OWN_POINTER,partitioning));
  }
  if (useND) {
    IS ndis;

    if (bs > 1) {
      PetscCall(ISCreateBlock(pcomm,bs,mat->rmap->n,NDorder,PETSC_OWN_POINTER,&ndis));
    } else {
      PetscCall(ISCreateGeneral(pcomm,mat->rmap->n,NDorder,PETSC_OWN_POINTER,&ndis));
    }
    PetscCall(ISSetPermutation(ndis));
    PetscCall(PetscObjectCompose((PetscObject)(*partitioning),"_petsc_matpartitioning_ndorder",(PetscObject)ndis));
    PetscCall(ISDestroy(&ndis));
  }

  if (!flg) {
    PetscCall(MatDestroy(&mat));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningApply_PTScotch(MatPartitioning part,IS *partitioning)
{
  PetscFunctionBegin;
  PetscCall(MatPartitioningApply_PTScotch_Private(part,PETSC_FALSE,partitioning));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningApplyND_PTScotch(MatPartitioning part,IS *partitioning)
{
  PetscFunctionBegin;
  PetscCall(MatPartitioningApply_PTScotch_Private(part,PETSC_TRUE,partitioning));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningDestroy_PTScotch(MatPartitioning part)
{
  MatPartitioning_PTScotch *scotch = (MatPartitioning_PTScotch*)part->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(scotch));
  /* clear composed functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningPTScotchSetImbalance_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningPTScotchGetImbalance_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningPTScotchSetStrategy_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningPTScotchGetStrategy_C",NULL));
  PetscFunctionReturn(0);
}

/*MC
   MATPARTITIONINGPTSCOTCH - Creates a partitioning context via the external package SCOTCH.

   Level: beginner

   Notes:
    See http://www.labri.fr/perso/pelegrin/scotch/

.seealso: MatPartitioningSetType(), MatPartitioningType
M*/

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_PTScotch(MatPartitioning part)
{
  MatPartitioning_PTScotch *scotch;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(part,&scotch));
  part->data = (void*)scotch;

  scotch->imbalance = 0.01;
  scotch->strategy  = SCOTCH_STRATDEFAULT;

  part->ops->apply          = MatPartitioningApply_PTScotch;
  part->ops->applynd        = MatPartitioningApplyND_PTScotch;
  part->ops->view           = MatPartitioningView_PTScotch;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_PTScotch;
  part->ops->destroy        = MatPartitioningDestroy_PTScotch;

  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningPTScotchSetImbalance_C",MatPartitioningPTScotchSetImbalance_PTScotch));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningPTScotchGetImbalance_C",MatPartitioningPTScotchGetImbalance_PTScotch));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningPTScotchSetStrategy_C",MatPartitioningPTScotchSetStrategy_PTScotch));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningPTScotchGetStrategy_C",MatPartitioningPTScotchGetStrategy_PTScotch));
  PetscFunctionReturn(0);
}
