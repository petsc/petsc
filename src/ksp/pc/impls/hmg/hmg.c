#include <petscdm.h>
#include <petscctable.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/pcmgimpl.h>
#include <petsc/private/pcimpl.h>      /*I "petscpc.h" I*/

typedef struct {
  PC               innerpc;            /* A MG inner PC (Hypre or PCGAMG) to setup interpolations and coarse operators  */
  char*            innerpctype;        /* PCGAMG or PCHYPRE */
  PetscBool        reuseinterp;        /* A flag indicates if or not to reuse the interpolations */
  PetscBool        subcoarsening;      /* If or not to use a subspace-based coarsening algorithm */
  PetscBool        usematmaij;         /* If or not to use MatMAIJ for saving memory */
  PetscInt         component;          /* Which subspace is used for the subspace-based coarsening algorithm? */
} PC_HMG;

PetscErrorCode PCSetFromOptions_HMG(PetscOptionItems*,PC);
PetscErrorCode PCReset_MG(PC);

static PetscErrorCode PCHMGExtractSubMatrix_Private(Mat pmat,Mat *submat,MatReuse reuse,PetscInt component,PetscInt blocksize)
{
  IS             isrow;
  PetscInt       rstart,rend;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)pmat,&comm));
  PetscCheckFalse(component>=blocksize,comm,PETSC_ERR_ARG_INCOMP,"Component %D should be less than block size %D ",component,blocksize);
  CHKERRQ(MatGetOwnershipRange(pmat,&rstart,&rend));
  PetscCheckFalse((rend-rstart)%blocksize != 0,comm,PETSC_ERR_ARG_INCOMP,"Block size %D is inconsistent for [%D, %D) ",blocksize,rstart,rend);
  CHKERRQ(ISCreateStride(comm,(rend-rstart)/blocksize,rstart+component,blocksize,&isrow));
  CHKERRQ(MatCreateSubMatrix(pmat,isrow,isrow,reuse,submat));
  CHKERRQ(ISDestroy(&isrow));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHMGExpandInterpolation_Private(Mat subinterp, Mat *interp, PetscInt blocksize)
{
  PetscInt              subrstart,subrend,subrowsize,subcolsize,subcstart,subcend,rowsize,colsize;
  PetscInt              subrow,row,nz,*d_nnz,*o_nnz,i,j,dnz,onz,max_nz,*indices;
  const PetscInt        *idx;
  const PetscScalar     *values;
  MPI_Comm              comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)subinterp,&comm));
  CHKERRQ(MatGetOwnershipRange(subinterp,&subrstart,&subrend));
  subrowsize = subrend-subrstart;
  rowsize = subrowsize*blocksize;
  CHKERRQ(PetscCalloc2(rowsize,&d_nnz,rowsize,&o_nnz));
  CHKERRQ(MatGetOwnershipRangeColumn(subinterp,&subcstart,&subcend));
  subcolsize = subcend - subcstart;
  colsize    = subcolsize*blocksize;
  max_nz = 0;
  for (subrow=subrstart;subrow<subrend;subrow++) {
    CHKERRQ(MatGetRow(subinterp,subrow,&nz,&idx,NULL));
    if (max_nz<nz) max_nz = nz;
    dnz = 0; onz = 0;
    for (i=0;i<nz;i++) {
      if (idx[i]>=subcstart && idx[i]<subcend) dnz++;
      else onz++;
    }
    for (i=0;i<blocksize;i++) {
      d_nnz[(subrow-subrstart)*blocksize+i] = dnz;
      o_nnz[(subrow-subrstart)*blocksize+i] = onz;
    }
    CHKERRQ(MatRestoreRow(subinterp,subrow,&nz,&idx,NULL));
  }
  CHKERRQ(MatCreateAIJ(comm,rowsize,colsize,PETSC_DETERMINE,PETSC_DETERMINE,0,d_nnz,0,o_nnz,interp));
  CHKERRQ(MatSetOption(*interp,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE));
  CHKERRQ(MatSetOption(*interp,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));
  CHKERRQ(MatSetOption(*interp,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  CHKERRQ(MatSetFromOptions(*interp));

  CHKERRQ(MatSetUp(*interp));
  CHKERRQ(PetscFree2(d_nnz,o_nnz));
  CHKERRQ(PetscMalloc1(max_nz,&indices));
  for (subrow=subrstart; subrow<subrend; subrow++) {
    CHKERRQ(MatGetRow(subinterp,subrow,&nz,&idx,&values));
    for (i=0;i<blocksize;i++) {
      row = subrow*blocksize+i;
      for (j=0;j<nz;j++) {
        indices[j] = idx[j]*blocksize+i;
      }
      CHKERRQ(MatSetValues(*interp,1,&row,nz,indices,values,INSERT_VALUES));
    }
    CHKERRQ(MatRestoreRow(subinterp,subrow,&nz,&idx,&values));
  }
  CHKERRQ(PetscFree(indices));
  CHKERRQ(MatAssemblyBegin(*interp,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*interp,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetUp_HMG(PC pc)
{
  PetscErrorCode     ierr;
  Mat                PA, submat;
  PC_MG              *mg   = (PC_MG*)pc->data;
  PC_HMG             *hmg   = (PC_HMG*) mg->innerctx;
  MPI_Comm           comm;
  PetscInt           level;
  PetscInt           num_levels;
  Mat                *operators,*interpolations;
  PetscInt           blocksize;
  const char         *prefix;
  PCMGGalerkinType   galerkin;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  if (pc->setupcalled) {
   if (hmg->reuseinterp) {
     /* If we did not use Galerkin in the last call or we have a different sparsity pattern now,
      * we have to build from scratch
      * */
     CHKERRQ(PCMGGetGalerkin(pc,&galerkin));
     if (galerkin == PC_MG_GALERKIN_NONE || pc->flag != SAME_NONZERO_PATTERN) pc->setupcalled = PETSC_FALSE;
     CHKERRQ(PCMGSetGalerkin(pc,PC_MG_GALERKIN_PMAT));
     CHKERRQ(PCSetUp_MG(pc));
     PetscFunctionReturn(0);
    } else {
     CHKERRQ(PCReset_MG(pc));
     pc->setupcalled = PETSC_FALSE;
    }
  }

  /* Create an inner PC (GAMG or HYPRE) */
  if (!hmg->innerpc) {
    CHKERRQ(PCCreate(comm,&hmg->innerpc));
    /* If users do not set an inner pc type, we need to set a default value */
    if (!hmg->innerpctype) {
    /* If hypre is available, use hypre, otherwise, use gamg */
#if PETSC_HAVE_HYPRE
      CHKERRQ(PetscStrallocpy(PCHYPRE,&(hmg->innerpctype)));
#else
      CHKERRQ(PetscStrallocpy(PCGAMG,&(hmg->innerpctype)));
#endif
    }
    CHKERRQ(PCSetType(hmg->innerpc,hmg->innerpctype));
  }
  CHKERRQ(PCGetOperators(pc,NULL,&PA));
  /* Users need to correctly set a block size of matrix in order to use subspace coarsening */
  CHKERRQ(MatGetBlockSize(PA,&blocksize));
  if (blocksize<=1) hmg->subcoarsening = PETSC_FALSE;
  /* Extract a submatrix for constructing subinterpolations */
  if (hmg->subcoarsening) {
    CHKERRQ(PCHMGExtractSubMatrix_Private(PA,&submat,MAT_INITIAL_MATRIX,hmg->component,blocksize));
    PA = submat;
  }
  CHKERRQ(PCSetOperators(hmg->innerpc,PA,PA));
  if (hmg->subcoarsening) {
   CHKERRQ(MatDestroy(&PA));
  }
  /* Setup inner PC correctly. During this step, matrix will be coarsened */
  CHKERRQ(PCSetUseAmat(hmg->innerpc,PETSC_FALSE));
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)pc,&prefix));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)hmg->innerpc,prefix));
  CHKERRQ(PetscObjectAppendOptionsPrefix((PetscObject)hmg->innerpc,"hmg_inner_"));
  CHKERRQ(PCSetFromOptions(hmg->innerpc));
  CHKERRQ(PCSetUp(hmg->innerpc));

  /* Obtain interpolations IN PLACE. For BoomerAMG, (I,J,data) is reused to avoid memory overhead */
  CHKERRQ(PCGetInterpolations(hmg->innerpc,&num_levels,&interpolations));
  /* We can reuse the coarse operators when we do the full space coarsening */
  if (!hmg->subcoarsening) {
    CHKERRQ(PCGetCoarseOperators(hmg->innerpc,&num_levels,&operators));
  }

  CHKERRQ(PCDestroy(&hmg->innerpc));
  hmg->innerpc = NULL;
  CHKERRQ(PCMGSetLevels_MG(pc,num_levels,NULL));
  /* Set coarse matrices and interpolations to PCMG */
  for (level=num_levels-1; level>0; level--) {
    Mat P=NULL, pmat=NULL;
    Vec b, x,r;
    if (hmg->subcoarsening) {
      if (hmg->usematmaij) {
        CHKERRQ(MatCreateMAIJ(interpolations[level-1],blocksize,&P));
        CHKERRQ(MatDestroy(&interpolations[level-1]));
      } else {
        /* Grow interpolation. In the future, we should use MAIJ */
        CHKERRQ(PCHMGExpandInterpolation_Private(interpolations[level-1],&P,blocksize));
        CHKERRQ(MatDestroy(&interpolations[level-1]));
      }
    } else {
      P = interpolations[level-1];
    }
    CHKERRQ(MatCreateVecs(P,&b,&r));
    CHKERRQ(PCMGSetInterpolation(pc,level,P));
    CHKERRQ(PCMGSetRestriction(pc,level,P));
    CHKERRQ(MatDestroy(&P));
    /* We reuse the matrices when we do not do subspace coarsening */
    if ((level-1)>=0 && !hmg->subcoarsening) {
      pmat = operators[level-1];
      CHKERRQ(PCMGSetOperators(pc,level-1,pmat,pmat));
      CHKERRQ(MatDestroy(&pmat));
    }
    CHKERRQ(PCMGSetRhs(pc,level-1,b));

    CHKERRQ(PCMGSetR(pc,level,r));
    CHKERRQ(VecDestroy(&r));

    CHKERRQ(VecDuplicate(b,&x));
    CHKERRQ(PCMGSetX(pc,level-1,x));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&b));
  }
  CHKERRQ(PetscFree(interpolations));
  if (!hmg->subcoarsening) {
    CHKERRQ(PetscFree(operators));
  }
  /* Turn Galerkin off when we already have coarse operators */
  CHKERRQ(PCMGSetGalerkin(pc,hmg->subcoarsening ? PC_MG_GALERKIN_PMAT:PC_MG_GALERKIN_NONE));
  CHKERRQ(PCSetDM(pc,NULL));
  CHKERRQ(PCSetUseAmat(pc,PETSC_FALSE));
  ierr = PetscObjectOptionsBegin((PetscObject)pc);CHKERRQ(ierr);
  CHKERRQ(PCSetFromOptions_MG(PetscOptionsObject,pc)); /* should be called in PCSetFromOptions_HMG(), but cannot be called prior to PCMGSetLevels() */
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CHKERRQ(PCSetUp_MG(pc));
  PetscFunctionReturn(0);
}

PetscErrorCode PCDestroy_HMG(PC pc)
{
  PC_MG          *mg  = (PC_MG*)pc->data;
  PC_HMG         *hmg = (PC_HMG*) mg->innerctx;

  PetscFunctionBegin;
  CHKERRQ(PCDestroy(&hmg->innerpc));
  CHKERRQ(PetscFree(hmg->innerpctype));
  CHKERRQ(PetscFree(hmg));
  CHKERRQ(PCDestroy_MG(pc));

  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetReuseInterpolation_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetUseSubspaceCoarsening_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetInnerPCType_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetCoarseningComponent_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode PCView_HMG(PC pc,PetscViewer viewer)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_HMG         *hmg = (PC_HMG*) mg->innerctx;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," Reuse interpolation: %s\n",hmg->reuseinterp? "true":"false"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer," Use subspace coarsening: %s\n",hmg->subcoarsening? "true":"false"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer," Coarsening component: %D \n",hmg->component));
    CHKERRQ(PetscViewerASCIIPrintf(viewer," Use MatMAIJ: %s \n",hmg->usematmaij? "true":"false"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer," Inner PC type: %s \n",hmg->innerpctype));
  }
  CHKERRQ(PCView_MG(pc,viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_HMG(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_HMG         *hmg = (PC_HMG*) mg->innerctx;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"HMG"));
  CHKERRQ(PetscOptionsBool("-pc_hmg_reuse_interpolation","Reuse the interpolation operators when possible (cheaper, weaker when matrix entries change a lot)","PCHMGSetReuseInterpolation",hmg->reuseinterp,&hmg->reuseinterp,NULL));
  CHKERRQ(PetscOptionsBool("-pc_hmg_use_subspace_coarsening","Use the subspace coarsening to compute the interpolations","PCHMGSetUseSubspaceCoarsening",hmg->subcoarsening,&hmg->subcoarsening,NULL));
  CHKERRQ(PetscOptionsBool("-pc_hmg_use_matmaij","Use MatMAIJ store interpolation for saving memory","PCHMGSetInnerPCType",hmg->usematmaij,&hmg->usematmaij,NULL));
  CHKERRQ(PetscOptionsInt("-pc_hmg_coarsening_component","Which component is chosen for the subspace-based coarsening algorithm","PCHMGSetCoarseningComponent",hmg->component,&hmg->component,NULL));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHMGSetReuseInterpolation_HMG(PC pc, PetscBool reuse)
{
  PC_MG          *mg  = (PC_MG*)pc->data;
  PC_HMG         *hmg = (PC_HMG*) mg->innerctx;

  PetscFunctionBegin;
  hmg->reuseinterp = reuse;
  PetscFunctionReturn(0);
}

/*@
   PCHMGSetReuseInterpolation - Reuse interpolation matrices in HMG

   Logically Collective on PC

   Input Parameters:
+  pc - the HMG context
-  reuse - True indicates that HMG will reuse the interpolations

   Options Database Keys:
.  -pc_hmg_reuse_interpolation <true | false> - Whether or not to reuse the interpolations. If true, it potentially save the compute time.

   Level: beginner

.keywords: HMG, multigrid, interpolation, reuse, set

.seealso: PCHMG
@*/
PetscErrorCode PCHMGSetReuseInterpolation(PC pc, PetscBool reuse)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCHMGSetReuseInterpolation_C",(PC,PetscBool),(pc,reuse)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHMGSetUseSubspaceCoarsening_HMG(PC pc, PetscBool subspace)
{
  PC_MG          *mg  = (PC_MG*)pc->data;
  PC_HMG         *hmg = (PC_HMG*) mg->innerctx;

  PetscFunctionBegin;
  hmg->subcoarsening = subspace;
  PetscFunctionReturn(0);
}

/*@
   PCHMGSetUseSubspaceCoarsening - Use subspace coarsening in HMG

   Logically Collective on PC

   Input Parameters:
+  pc - the HMG context
-  reuse - True indicates that HMG will use the subspace coarsening

   Options Database Keys:
.  -pc_hmg_use_subspace_coarsening  <true | false> - Whether or not to use subspace coarsening (that is, coarsen a submatrix).

   Level: beginner

.keywords: HMG, multigrid, interpolation, subspace, coarsening

.seealso: PCHMG
@*/
PetscErrorCode PCHMGSetUseSubspaceCoarsening(PC pc, PetscBool subspace)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCHMGSetUseSubspaceCoarsening_C",(PC,PetscBool),(pc,subspace)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHMGSetInnerPCType_HMG(PC pc, PCType type)
{
  PC_MG           *mg  = (PC_MG*)pc->data;
  PC_HMG          *hmg = (PC_HMG*) mg->innerctx;

  PetscFunctionBegin;
  CHKERRQ(PetscStrallocpy(type,&(hmg->innerpctype)));
  PetscFunctionReturn(0);
}

/*@C
   PCHMGSetInnerPCType - Set an inner PC type

   Logically Collective on PC

   Input Parameters:
+  pc - the HMG context
-  type - <hypre, gamg> coarsening algorithm

   Options Database Keys:
.  -hmg_inner_pc_type <hypre, gamg> - What method is used to coarsen matrix

   Level: beginner

.keywords: HMG, multigrid, interpolation, coarsening

.seealso: PCHMG, PCType
@*/
PetscErrorCode PCHMGSetInnerPCType(PC pc, PCType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCHMGSetInnerPCType_C",(PC,PCType),(pc,type)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHMGSetCoarseningComponent_HMG(PC pc, PetscInt component)
{
  PC_MG           *mg  = (PC_MG*)pc->data;
  PC_HMG          *hmg = (PC_HMG*) mg->innerctx;

  PetscFunctionBegin;
  hmg->component = component;
  PetscFunctionReturn(0);
}

/*@
   PCHMGSetCoarseningComponent - Set which component is used for the subspace-based coarsening algorithm

   Logically Collective on PC

   Input Parameters:
+  pc - the HMG context
-  component - which component PC will coarsen

   Options Database Keys:
.  -pc_hmg_coarsening_component - Which component is chosen for the subspace-based coarsening algorithm

   Level: beginner

.keywords: HMG, multigrid, interpolation, coarsening, component

.seealso: PCHMG, PCType
@*/
PetscErrorCode PCHMGSetCoarseningComponent(PC pc, PetscInt component)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCHMGSetCoarseningComponent_C",(PC,PetscInt),(pc,component)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHMGUseMatMAIJ_HMG(PC pc, PetscBool usematmaij)
{
  PC_MG           *mg  = (PC_MG*)pc->data;
  PC_HMG          *hmg = (PC_HMG*) mg->innerctx;

  PetscFunctionBegin;
  hmg->usematmaij = usematmaij;
  PetscFunctionReturn(0);
}

/*@
   PCHMGUseMatMAIJ - Set a flag that indicates if or not to use MatMAIJ for interpolations for saving memory

   Logically Collective on PC

   Input Parameters:
+  pc - the HMG context
-  usematmaij - if or not to use MatMAIJ for interpolations. By default, it is true for saving memory

   Options Database Keys:
.  -pc_hmg_use_matmaij - <true | false >

   Level: beginner

.keywords: HMG, multigrid, interpolation, coarsening, MatMAIJ

.seealso: PCHMG, PCType
@*/
PetscErrorCode PCHMGUseMatMAIJ(PC pc, PetscBool usematmaij)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCHMGUseMatMAIJ_C",(PC,PetscBool),(pc,usematmaij)));
  PetscFunctionReturn(0);
}

/*MC
   PCHMG - Hybrid of PETSc preconditioners (such as ASM, BJacobi, SOR, etc.) and Hypre BoomerAMG, GAMG or other multilevel methods. BoomerAMG, GAMG
           or other multilevel methods is used to coarsen matrix and generate a sequence of coarse matrices and interpolations. The matrices and
           interpolations are employed to construct PCMG, and then any available PETSc preconditioners can be chosen as smoothers and the coarse solver.

   Options Database Keys:
+  -pc_hmg_reuse_interpolation <true | false> - Whether or not to reuse the interpolations. If true, it potentially save the compute time.
.  -pc_hmg_use_subspace_coarsening  <true | false> - Whether or not to use subspace coarsening (that is, coarsen a submatrix).
.  -hmg_inner_pc_type <hypre, gamg, ...> - What method is used to coarsen matrix
-  -pc_hmg_use_matmaij <true | false> - Whether or not to use MatMAIJ for multicomponent problems for saving memory

   Notes:
    For multicomponent problems, we can just coarsen one submatrix associated with one particular component. In this way, the preconditioner setup
    time is significantly reduced. One typical use case is neutron transport equations. There are many variables on each mesh vertex due to the
    of angle and energy. Each variable, in fact, corresponds to the same PDEs but with different material properties.

   Level: beginner

   Concepts: Hybrid of ASM and MG, Subspace Coarsening

    References:
.   * - Fande Kong, Yaqi Wang, Derek R Gaston, Cody J Permann, Andrew E Slaughter, Alexander D Lindsay, Richard C Martineau, A highly parallel multilevel
    Newton-Krylov-Schwarz method with subspace-based coarsening and partition-based balancing for the multigroup neutron transport equations on
    3D unstructured meshes, arXiv preprint arXiv:1903.03659, 2019

.seealso:  PCCreate(), PCSetType(), PCType, PC, PCMG, PCHYPRE, PCHMG, PCGetCoarseOperators(), PCGetInterpolations(), PCHMGSetReuseInterpolation(), PCHMGSetUseSubspaceCoarsening(),
           PCHMGSetInnerPCType()

M*/
PETSC_EXTERN PetscErrorCode PCCreate_HMG(PC pc)
{
  PC_HMG         *hmg;
  PC_MG          *mg;

  PetscFunctionBegin;
  /* if type was previously mg; must manually destroy it because call to PCSetType(pc,PCMG) will not destroy it */
  if (pc->ops->destroy) {
    CHKERRQ((*pc->ops->destroy)(pc));
    pc->data = NULL;
  }
  CHKERRQ(PetscFree(((PetscObject)pc)->type_name));

  CHKERRQ(PCSetType(pc,PCMG));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)pc, PCHMG));
  CHKERRQ(PetscNew(&hmg));

  mg                      = (PC_MG*) pc->data;
  mg->innerctx            = hmg;
  hmg->reuseinterp        = PETSC_FALSE;
  hmg->subcoarsening      = PETSC_FALSE;
  hmg->usematmaij         = PETSC_TRUE;
  hmg->component          = 0;
  hmg->innerpc            = NULL;

  pc->ops->setfromoptions = PCSetFromOptions_HMG;
  pc->ops->view           = PCView_HMG;
  pc->ops->destroy        = PCDestroy_HMG;
  pc->ops->setup          = PCSetUp_HMG;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetReuseInterpolation_C",PCHMGSetReuseInterpolation_HMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetUseSubspaceCoarsening_C",PCHMGSetUseSubspaceCoarsening_HMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetInnerPCType_C",PCHMGSetInnerPCType_HMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetCoarseningComponent_C",PCHMGSetCoarseningComponent_HMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCHMGUseMatMAIJ_C",PCHMGUseMatMAIJ_HMG));
  PetscFunctionReturn(0);
}
