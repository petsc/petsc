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
  PetscErrorCode ierr;
  PetscInt       rstart,rend;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pmat,&comm);CHKERRQ(ierr);
  if (component>=blocksize) SETERRQ2(comm,PETSC_ERR_ARG_INCOMP,"Component %D should be less than block size %D \n",component,blocksize);
  ierr = MatGetOwnershipRange(pmat,&rstart,&rend);CHKERRQ(ierr);
  if ((rend-rstart)%blocksize != 0) SETERRQ3(comm,PETSC_ERR_ARG_INCOMP,"Block size %D is inconsistent for [%D, %D) \n",blocksize,rstart,rend);
  ierr = ISCreateStride(comm,(rend-rstart)/blocksize,rstart+component,blocksize,&isrow);
  ierr = MatCreateSubMatrix(pmat,isrow,isrow,reuse,submat);CHKERRQ(ierr);
  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHMGExpandInterpolation_Private(Mat subinterp, Mat *interp, PetscInt blocksize)
{
  PetscInt              subrstart,subrend,subrowsize,subcolsize,subcstart,subcend,rowsize,colsize;
  PetscInt              subrow,row,nz,*d_nnz,*o_nnz,i,j,dnz,onz,max_nz,*indices;
  const PetscInt        *idx;
  const PetscScalar     *values;
  PetscErrorCode        ierr;
  MPI_Comm              comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)subinterp,&comm);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(subinterp,&subrstart,&subrend);CHKERRQ(ierr);
  subrowsize = subrend-subrstart;
  rowsize = subrowsize*blocksize;
  ierr = PetscCalloc2(rowsize,&d_nnz,rowsize,&o_nnz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(subinterp,&subcstart,&subcend);CHKERRQ(ierr);
  subcolsize = subcend - subcstart;
  colsize    = subcolsize*blocksize;
  max_nz = 0;
  for (subrow=subrstart;subrow<subrend;subrow++) {
    ierr = MatGetRow(subinterp,subrow,&nz,&idx,NULL);CHKERRQ(ierr);
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
    ierr = MatRestoreRow(subinterp,subrow,&nz,&idx,NULL);CHKERRQ(ierr);
  }
  ierr = MatCreateAIJ(comm,rowsize,colsize,PETSC_DETERMINE,PETSC_DETERMINE,0,d_nnz,0,o_nnz,interp);CHKERRQ(ierr);
  ierr = MatSetOption(*interp,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(*interp,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(*interp,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*interp);CHKERRQ(ierr);

  ierr = MatSetUp(*interp);CHKERRQ(ierr);
  ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);
  ierr = PetscMalloc1(max_nz,&indices);CHKERRQ(ierr);
  for (subrow=subrstart; subrow<subrend; subrow++) {
    ierr = MatGetRow(subinterp,subrow,&nz,&idx,&values);CHKERRQ(ierr);
    for (i=0;i<blocksize;i++) {
      row = subrow*blocksize+i;
      for (j=0;j<nz;j++) {
        indices[j] = idx[j]*blocksize+i;
      }
      ierr = MatSetValues(*interp,1,&row,nz,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(subinterp,subrow,&nz,&idx,&values);CHKERRQ(ierr);
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*interp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*interp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  if (pc->setupcalled) {
   if (hmg->reuseinterp) {
     /* If we did not use Galerkin in the last call or we have a different sparsity pattern now,
      * we have to build from scratch
      * */
     ierr = PCMGGetGalerkin(pc,&galerkin);CHKERRQ(ierr);
     if (galerkin == PC_MG_GALERKIN_NONE || pc->flag != SAME_NONZERO_PATTERN) pc->setupcalled = PETSC_FALSE;
     ierr = PCMGSetGalerkin(pc,PC_MG_GALERKIN_PMAT);CHKERRQ(ierr);
     ierr = PCSetUp_MG(pc);CHKERRQ(ierr);
     PetscFunctionReturn(0);
    } else {
     ierr = PCReset_MG(pc);CHKERRQ(ierr);
     pc->setupcalled = PETSC_FALSE;
    }
  }

  /* Create an inner PC (GAMG or HYPRE) */
  if (!hmg->innerpc) {
    ierr = PCCreate(comm,&hmg->innerpc);CHKERRQ(ierr);
    /* If users do not set an inner pc type, we need to set a default value */
    if (!hmg->innerpctype) {
    /* If hypre is available, use hypre, otherwise, use gamg */
#if PETSC_HAVE_HYPRE
      ierr = PetscStrallocpy(PCHYPRE,&(hmg->innerpctype));CHKERRQ(ierr);
#else
      ierr = PetscStrallocpy(PCGAMG,&(hmg->innerpctype));CHKERRQ(ierr);
#endif
    }
    ierr = PCSetType(hmg->innerpc,hmg->innerpctype);CHKERRQ(ierr);
  }
  ierr = PCGetOperators(pc,NULL,&PA);CHKERRQ(ierr);
  /* Users need to correctly set a block size of matrix in order to use subspace coarsening */
  ierr = MatGetBlockSize(PA,&blocksize);CHKERRQ(ierr);
  if (blocksize<=1) hmg->subcoarsening = PETSC_FALSE;
  /* Extract a submatrix for constructing subinterpolations */
  if (hmg->subcoarsening) {
    ierr = PCHMGExtractSubMatrix_Private(PA,&submat,MAT_INITIAL_MATRIX,hmg->component,blocksize);CHKERRQ(ierr);
    PA = submat;
  }
  ierr = PCSetOperators(hmg->innerpc,PA,PA);CHKERRQ(ierr);
  if (hmg->subcoarsening) {
   ierr = MatDestroy(&PA);CHKERRQ(ierr);
  }
  /* Setup inner PC correctly. During this step, matrix will be coarsened */
  ierr = PCSetUseAmat(hmg->innerpc,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)pc,&prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)hmg->innerpc,prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)hmg->innerpc,"hmg_inner_");CHKERRQ(ierr);
  ierr = PCSetFromOptions(hmg->innerpc);CHKERRQ(ierr);
  ierr = PCSetUp(hmg->innerpc);

  /* Obtain interpolations IN PLACE. For BoomerAMG, (I,J,data) is reused to avoid memory overhead */
  ierr = PCGetInterpolations(hmg->innerpc,&num_levels,&interpolations);CHKERRQ(ierr);
  /* We can reuse the coarse operators when we do the full space coarsening */
  if (!hmg->subcoarsening) {
    ierr = PCGetCoarseOperators(hmg->innerpc,&num_levels,&operators);CHKERRQ(ierr);
  }

  ierr = PCDestroy(&hmg->innerpc);CHKERRQ(ierr);
  hmg->innerpc = NULL;
  ierr = PCMGSetLevels_MG(pc,num_levels,NULL);CHKERRQ(ierr);
  /* Set coarse matrices and interpolations to PCMG */
  for (level=num_levels-1; level>0; level--) {
    Mat P=NULL, pmat=NULL;
    Vec b, x,r;
    if (hmg->subcoarsening) {
      if (hmg->usematmaij) {
        ierr = MatCreateMAIJ(interpolations[level-1],blocksize,&P);CHKERRQ(ierr);
        ierr = MatDestroy(&interpolations[level-1]);CHKERRQ(ierr);
      } else {
        /* Grow interpolation. In the future, we should use MAIJ */
        ierr = PCHMGExpandInterpolation_Private(interpolations[level-1],&P,blocksize);CHKERRQ(ierr);
        ierr = MatDestroy(&interpolations[level-1]);CHKERRQ(ierr);
      }
    } else {
      P = interpolations[level-1];
    }
    ierr = MatCreateVecs(P,&b,&r);CHKERRQ(ierr);
    ierr = PCMGSetInterpolation(pc,level,P);CHKERRQ(ierr);
    ierr = PCMGSetRestriction(pc,level,P);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    /* We reuse the matrices when we do not do subspace coarsening */
    if ((level-1)>=0 && !hmg->subcoarsening) {
      pmat = operators[level-1];
      ierr = PCMGSetOperators(pc,level-1,pmat,pmat);CHKERRQ(ierr);
      ierr = MatDestroy(&pmat);CHKERRQ(ierr);
    }
    ierr = PCMGSetRhs(pc,level-1,b);CHKERRQ(ierr);

    ierr = PCMGSetR(pc,level,r);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);

    ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
    ierr = PCMGSetX(pc,level-1,x);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
  }
  ierr = PetscFree(interpolations);CHKERRQ(ierr);
  if (!hmg->subcoarsening) {
    ierr = PetscFree(operators);CHKERRQ(ierr);
  }
  /* Turn Galerkin off when we already have coarse operators */
  ierr = PCMGSetGalerkin(pc,hmg->subcoarsening ? PC_MG_GALERKIN_PMAT:PC_MG_GALERKIN_NONE);CHKERRQ(ierr);
  ierr = PCSetDM(pc,NULL);CHKERRQ(ierr);
  ierr = PCSetUseAmat(pc,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)pc);CHKERRQ(ierr);
  ierr = PCSetFromOptions_MG(PetscOptionsObject,pc);CHKERRQ(ierr); /* should be called in PCSetFromOptions_HMG(), but cannot be called prior to PCMGSetLevels() */
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PCSetUp_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCDestroy_HMG(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg  = (PC_MG*)pc->data;
  PC_HMG         *hmg = (PC_HMG*) mg->innerctx;

  PetscFunctionBegin;
  ierr = PCDestroy(&hmg->innerpc);CHKERRQ(ierr);
  ierr = PetscFree(hmg->innerpctype);CHKERRQ(ierr);
  ierr = PetscFree(hmg);CHKERRQ(ierr);
  ierr = PCDestroy_MG(pc);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetReuseInterpolation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetUseSubspaceCoarsening_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetInnerPCType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetCoarseningComponent_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCView_HMG(PC pc,PetscViewer viewer)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_HMG         *hmg = (PC_HMG*) mg->innerctx;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer," Reuse interpolation: %s\n",hmg->reuseinterp? "true":"false");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer," Use subspace coarsening: %s\n",hmg->subcoarsening? "true":"false");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer," Coarsening component: %D \n",hmg->component);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer," Use MatMAIJ: %s \n",hmg->usematmaij? "true":"false");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer," Inner PC type: %s \n",hmg->innerpctype);CHKERRQ(ierr);
  }
  ierr = PCView_MG(pc,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_HMG(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_HMG         *hmg = (PC_HMG*) mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"HMG");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_hmg_reuse_interpolation","Reuse the interpolation operators when possible (cheaper, weaker when matrix entries change a lot)","PCHMGSetReuseInterpolation",hmg->reuseinterp,&hmg->reuseinterp,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_hmg_use_subspace_coarsening","Use the subspace coarsening to compute the interpolations","PCHMGSetUseSubspaceCoarsening",hmg->subcoarsening,&hmg->subcoarsening,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_hmg_use_matmaij","Use MatMAIJ store interpolation for saving memory","PCHMGSetInnerPCType",hmg->usematmaij,&hmg->usematmaij,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hmg_coarsening_component","Which component is chosen for the subspace-based coarsening algorithm","PCHMGSetCoarseningComponent",hmg->component,&hmg->component,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCHMGSetReuseInterpolation_C",(PC,PetscBool),(pc,reuse));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCHMGSetUseSubspaceCoarsening_C",(PC,PetscBool),(pc,subspace));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHMGSetInnerPCType_HMG(PC pc, PCType type)
{
  PC_MG           *mg  = (PC_MG*)pc->data;
  PC_HMG          *hmg = (PC_HMG*) mg->innerctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(type,&(hmg->innerpctype));CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCHMGSetInnerPCType_C",(PC,PCType),(pc,type));CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCHMGSetCoarseningComponent_C",(PC,PetscInt),(pc,component));CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCHMGUseMatMAIJ_C",(PC,PetscBool),(pc,usematmaij));CHKERRQ(ierr);
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
.   1. - Fande Kong, Yaqi Wang, Derek R Gaston, Cody J Permann, Andrew E Slaughter, Alexander D Lindsay, Richard C Martineau, A highly parallel multilevel
    Newton-Krylov-Schwarz method with subspace-based coarsening and partition-based balancing for the multigroup neutron transport equations on
    3D unstructured meshes, arXiv preprint arXiv:1903.03659, 2019

.seealso:  PCCreate(), PCSetType(), PCType, PC, PCMG, PCHYPRE, PCHMG, PCGetCoarseOperators(), PCGetInterpolations(), PCHMGSetReuseInterpolation(), PCHMGSetUseSubspaceCoarsening(),
           PCHMGSetInnerPCType()

M*/
PETSC_EXTERN PetscErrorCode PCCreate_HMG(PC pc)
{
  PetscErrorCode ierr;
  PC_HMG         *hmg;
  PC_MG          *mg;

  PetscFunctionBegin;
  /* if type was previously mg; must manually destroy it because call to PCSetType(pc,PCMG) will not destroy it */
  if (pc->ops->destroy) {
    ierr =  (*pc->ops->destroy)(pc);CHKERRQ(ierr);
    pc->data = NULL;
  }
  ierr = PetscFree(((PetscObject)pc)->type_name);CHKERRQ(ierr);

  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)pc, PCHMG);CHKERRQ(ierr);
  ierr = PetscNew(&hmg);CHKERRQ(ierr);

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

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetReuseInterpolation_C",PCHMGSetReuseInterpolation_HMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetUseSubspaceCoarsening_C",PCHMGSetUseSubspaceCoarsening_HMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetInnerPCType_C",PCHMGSetInnerPCType_HMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHMGSetCoarseningComponent_C",PCHMGSetCoarseningComponent_HMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHMGUseMatMAIJ_C",PCHMGUseMatMAIJ_HMG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
