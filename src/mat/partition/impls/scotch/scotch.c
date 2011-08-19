
#include <../src/mat/impls/adj/mpi/mpiadj.h>       /*I "petscmat.h" I*/

EXTERN_C_BEGIN
#include <ptscotch.h>
EXTERN_C_END

typedef struct {
  double     imbalance;
  SCOTCH_Num strategy;
} MatPartitioning_Scotch;

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningScotchSetImbalance"
/*@
   MatPartitioningScotchSetImbalance - Sets the value of the load imbalance
   ratio to be used during strategy selection.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  imb  - the load imbalance ratio

   Options Database:
.  -mat_partitioning_scotch_imbalance <imb>

   Note:
   Must be in the range [0,1]. The default value is 0.01.

   Level: advanced

.seealso: MatPartitioningScotchSetStrategy(), MatPartitioningScotchGetImbalance()
@*/
PetscErrorCode MatPartitioningScotchSetImbalance(MatPartitioning part,PetscReal imb)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveReal(part,imb,2);
  ierr = PetscTryMethod(part,"MatPartitioningScotchSetImbalance_C",(MatPartitioning,PetscReal),(part,imb));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningScotchSetImbalance_Scotch" 
PetscErrorCode MatPartitioningScotchSetImbalance_Scotch(MatPartitioning part,PetscReal imb)
{
  MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch*)part->data;

  PetscFunctionBegin;
  if (imb==PETSC_DEFAULT) scotch->imbalance = 0.01;
  else {
    if (imb<0.0 || imb>1.0) SETERRQ(((PetscObject)part)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of imb. Must be in range [0,1]");
    scotch->imbalance = (double)imb;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningScotchGetImbalance"
/*@
   MatPartitioningScotchGetImbalance - Gets the value of the load imbalance
   ratio used during strategy selection.

   Input Parameter:
.  part - the partitioning context

   Output Parameter:
.  imb  - the load imbalance ratio

   Level: advanced

.seealso: MatPartitioningScotchSetImbalance()
@*/
PetscErrorCode MatPartitioningScotchGetImbalance(MatPartitioning part,PetscReal *imb)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(imb,2);
  ierr = PetscTryMethod(part,"MatPartitioningScotchGetImbalance_C",(MatPartitioning,PetscReal*),(part,imb));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningScotchGetImbalance_Scotch" 
PetscErrorCode MatPartitioningScotchGetImbalance_Scotch(MatPartitioning part,PetscReal *imb)
{
  MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch*)part->data;

  PetscFunctionBegin;
  *imb = scotch->imbalance;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningScotchSetStrategy"
/*@
   MatPartitioningScotchSetStrategy - Sets the strategy to be used in Scotch.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  strategy - the strategy, one of
.vb
     MP_SCOTCH_QUALITY     - Prioritize quality over speed
     MP_SCOTCH_SPEED       - Prioritize speed over quality
     MP_SCOTCH_BALANCE     - Enforce load balance
     MP_SCOTCH_SAFETY      - Avoid methods that may fail
     MP_SCOTCH_SCALABILITY - Favor scalability as much as possible
.ve

   Options Database:
.  -mat_partitioning_scotch_strategy [quality,speed,balance,safety,scalability] - strategy 
 
   Level: advanced

   Notes:
   The default is MP_SCOTCH_QUALITY. See the Scotch documentation for more information.

.seealso: MatPartitioningScotchSetImbalance(), MatPartitioningScotchGetStrategy()
@*/
PetscErrorCode MatPartitioningScotchSetStrategy(MatPartitioning part,MPScotchStrategyType strategy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveEnum(part,strategy,2);
  ierr = PetscTryMethod(part,"MatPartitioningScotchSetStrategy_C",(MatPartitioning,MPScotchStrategyType),(part,strategy));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningScotchSetStrategy_Scotch"
PetscErrorCode MatPartitioningScotchSetStrategy_Scotch(MatPartitioning part,MPScotchStrategyType strategy)
{
  MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch*)part->data;

  PetscFunctionBegin;
  switch (strategy) {
    case MP_SCOTCH_QUALITY:     scotch->strategy = SCOTCH_STRATQUALITY; break;
    case MP_SCOTCH_SPEED:       scotch->strategy = SCOTCH_STRATSPEED; break;
    case MP_SCOTCH_BALANCE:     scotch->strategy = SCOTCH_STRATBALANCE; break;
    case MP_SCOTCH_SAFETY:      scotch->strategy = SCOTCH_STRATSAFETY; break;
    case MP_SCOTCH_SCALABILITY: scotch->strategy = SCOTCH_STRATSCALABILITY; break;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningScotchGetStrategy"
/*@
   MatPartitioningScotchGetStrategy - Gets the strategy used in Scotch.

   Not Collective

   Input Parameter:
.  part - the partitioning context

   Output Parameter:
.  strategy - the strategy
 
   Level: advanced

.seealso: MatPartitioningScotchSetStrategy()
@*/
PetscErrorCode MatPartitioningScotchGetStrategy(MatPartitioning part,MPScotchStrategyType *strategy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(strategy,2);
  ierr = PetscTryMethod(part,"MatPartitioningScotchGetStrategy_C",(MatPartitioning,MPScotchStrategyType*),(part,strategy));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningScotchGetStrategy_Scotch"
PetscErrorCode MatPartitioningScotchGetStrategy_Scotch(MatPartitioning part,MPScotchStrategyType *strategy)
{
  MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch*)part->data;

  PetscFunctionBegin;
  switch (scotch->strategy) {
    case SCOTCH_STRATQUALITY:     *strategy = MP_SCOTCH_QUALITY; break;
    case SCOTCH_STRATSPEED:       *strategy = MP_SCOTCH_SPEED; break;
    case SCOTCH_STRATBALANCE:     *strategy = MP_SCOTCH_BALANCE; break;
    case SCOTCH_STRATSAFETY:      *strategy = MP_SCOTCH_SAFETY; break;
    case SCOTCH_STRATSCALABILITY: *strategy = MP_SCOTCH_SCALABILITY; break;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningView_Scotch"
PetscErrorCode MatPartitioningView_Scotch(MatPartitioning part, PetscViewer viewer)
{
  MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch*)part->data;
  PetscErrorCode         ierr;
  PetscBool              isascii;
  const char             *str;
  
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    switch (scotch->strategy) {
      case SCOTCH_STRATQUALITY:     str = "Prioritize quality over speed"; break;
      case SCOTCH_STRATSPEED:       str = "Prioritize speed over quality"; break;
      case SCOTCH_STRATBALANCE:     str = "Enforce load balance"; break;
      case SCOTCH_STRATSAFETY:      str = "Avoid methods that may fail"; break;
      case SCOTCH_STRATSCALABILITY: str = "Favor scalability as much as possible"; break;
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  Strategy=%s\n",str);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Load imbalance ratio=%g\n",scotch->imbalance);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for Scotch partitioner",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningSetFromOptions_Scotch"
PetscErrorCode MatPartitioningSetFromOptions_Scotch(MatPartitioning part)
{
  PetscErrorCode         ierr;
  PetscBool              flag;
  PetscReal              r;
  MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch*)part->data;
  MPScotchStrategyType   strat;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Scotch partitioning options");CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-mat_partitioning_scotch_strategy","Strategy","MatPartitioningScotchSetStrategy",MPScotchStrategyTypes,MP_SCOTCH_QUALITY,(PetscEnum*)&strat,&flag);CHKERRQ(ierr);
    if (flag) { ierr = MatPartitioningScotchSetStrategy(part,strat);CHKERRQ(ierr); }
    ierr = PetscOptionsReal("-mat_partitioning_scotch_imbalance","Load imbalance ratio","MatPartitioningScotchSetImbalance",scotch->imbalance,&r,&flag);CHKERRQ(ierr);
    if (flag) { ierr = MatPartitioningScotchSetImbalance(part,r);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningApply_Scotch"
PetscErrorCode MatPartitioningApply_Scotch(MatPartitioning part,IS *partitioning)
{
  MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch*)part->data;
  PetscErrorCode         ierr;
  PetscMPIInt            rank;
  Mat                    mat = part->adj;
  Mat_MPIAdj             *adj = (Mat_MPIAdj*)mat->data;
  PetscBool              flg;
  PetscInt               i,j,wgtflag=0,bs=1,nold;
  PetscReal              *vwgttab,deltval;
  SCOTCH_Num             *locals,*velotab,*veloloctab,*edloloctab,vertlocnbr,edgelocnbr,nparts=part->n;
  SCOTCH_Arch            archdat;
  SCOTCH_Dgraph          grafdat;
  SCOTCH_Dmapping        mappdat;
  SCOTCH_Strat           stradat;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)part)->comm,&rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)mat,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (!flg) {
    /* bs indicates if the converted matrix is "reduced" from the original and hence the 
       resulting partition results need to be stretched to match the original matrix */
    nold = mat->rmap->n;
    ierr = MatConvert(mat,MATMPIADJ,MAT_INITIAL_MATRIX,&mat);CHKERRQ(ierr);
    bs   = nold/mat->rmap->n;
    adj  = (Mat_MPIAdj*)mat->data;
  }

  ierr = PetscMalloc((mat->rmap->n+1)*sizeof(SCOTCH_Num),&locals);CHKERRQ(ierr);
  ierr = PetscMalloc(nparts*sizeof(PetscReal),&vwgttab);CHKERRQ(ierr);
  ierr = PetscMalloc(nparts*sizeof(SCOTCH_Num),&velotab);CHKERRQ(ierr);
  for (j=0;j<nparts;j++) {
    if (part->part_weights) vwgttab[j] = part->part_weights[j]*nparts;
    else vwgttab[j] = 1.0;
  }
  for (i=0;i<nparts;i++) {
    deltval = PetscAbsReal(vwgttab[i]-floor(vwgttab[i]+0.5));
    if (deltval>0.01) {
      for (j=0;j<nparts;j++) vwgttab[j] /= deltval;
    }
  }
  for (i=0;i<nparts;i++)
    velotab[i] = (SCOTCH_Num) (vwgttab[i] + 0.5);
  ierr = PetscFree(vwgttab);CHKERRQ(ierr);

  ierr = SCOTCH_dgraphInit(&grafdat,((PetscObject)mat)->comm);CHKERRQ(ierr);

  vertlocnbr = mat->rmap->range[rank+1] - mat->rmap->range[rank];
  edgelocnbr = adj->i[vertlocnbr];
  veloloctab = (!part->vertex_weights && !(wgtflag & 2)) ? part->vertex_weights: PETSC_NULL;
  edloloctab = (!adj->values && !(wgtflag & 1)) ? adj->values: PETSC_NULL;

  ierr = SCOTCH_dgraphBuild(&grafdat,0,vertlocnbr,vertlocnbr,adj->i,adj->i+1,veloloctab,
                    PETSC_NULL,edgelocnbr,edgelocnbr,adj->j,PETSC_NULL,edloloctab);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG)
  ierr = SCOTCH_dgraphCheck(&grafdat);CHKERRQ(ierr);
#endif

  ierr = SCOTCH_archInit(&archdat);CHKERRQ(ierr);
  ierr = SCOTCH_stratInit(&stradat);CHKERRQ(ierr);
  ierr = SCOTCH_stratDgraphMapBuild(&stradat,scotch->strategy,nparts,nparts,scotch->imbalance);CHKERRQ(ierr);

  ierr = SCOTCH_archCmpltw(&archdat,nparts,velotab);CHKERRQ(ierr);
  ierr = SCOTCH_dgraphMapInit(&grafdat,&mappdat,&archdat,locals);CHKERRQ(ierr);
  ierr = SCOTCH_dgraphMapCompute(&grafdat,&mappdat,&stradat);CHKERRQ(ierr);

  SCOTCH_dgraphMapExit (&grafdat,&mappdat);
  SCOTCH_archExit(&archdat);
  SCOTCH_stratExit(&stradat);
  SCOTCH_dgraphExit(&grafdat);
  ierr = PetscFree(velotab);CHKERRQ(ierr);

  if (bs > 1) {
    PetscInt *newlocals;
    ierr = PetscMalloc(bs*mat->rmap->n*sizeof(PetscInt),&newlocals);CHKERRQ(ierr);
    for (i=0;i<mat->rmap->n;i++) {
      for (j=0;j<bs;j++) {
        newlocals[bs*i+j] = locals[i];
      }
    }
    ierr = PetscFree(locals);CHKERRQ(ierr);
    ierr = ISCreateGeneral(((PetscObject)part)->comm,bs*mat->rmap->n,newlocals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
  } else {
    ierr = ISCreateGeneral(((PetscObject)part)->comm,mat->rmap->n,locals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
  }

  if (!flg) {
    ierr = MatDestroy(&mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningDestroy_Scotch"
PetscErrorCode MatPartitioningDestroy_Scotch(MatPartitioning part)
{
  MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch*)part->data;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscFree(scotch);CHKERRQ(ierr);
  /* clear composed functions */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningScotchSetImbalance_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningScotchGetImbalance_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningScotchSetStrategy_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningScotchGetStrategy_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATPARTITIONINGSCOTCH - Creates a partitioning context via the external package SCOTCH.

   Level: beginner

   Notes: See http://www.labri.fr/perso/pelegrin/scotch/

.keywords: Partitioning, create, context

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatPartitioningCreate_Scotch"
PetscErrorCode  MatPartitioningCreate_Scotch(MatPartitioning part)
{
  PetscErrorCode         ierr;
  MatPartitioning_Scotch *scotch;

  PetscFunctionBegin;
  ierr = PetscNewLog(part,MatPartitioning_Scotch,&scotch);CHKERRQ(ierr);
  part->data = (void*)scotch;

  scotch->imbalance = 0.01;
  scotch->strategy  = SCOTCH_STRATQUALITY;

  part->ops->apply          = MatPartitioningApply_Scotch;
  part->ops->view           = MatPartitioningView_Scotch;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_Scotch;
  part->ops->destroy        = MatPartitioningDestroy_Scotch;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningScotchSetImbalance_C","MatPartitioningScotchSetImbalance_Scotch",MatPartitioningScotchSetImbalance_Scotch);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningScotchGetImbalance_C","MatPartitioningScotchGetImbalance_Scotch",MatPartitioningScotchGetImbalance_Scotch);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningScotchSetStrategy_C","MatPartitioningScotchSetStrategy_Scotch",MatPartitioningScotchSetStrategy_Scotch);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningScotchGetStrategy_C","MatPartitioningScotchGetStrategy_Scotch",MatPartitioningScotchGetStrategy_Scotch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
