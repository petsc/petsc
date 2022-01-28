
/*
  This file defines a "solve the problem redundantly on each subgroup of processor" preconditioner.
*/
#include <petsc/private/pcimpl.h>
#include <petscksp.h>           /*I "petscksp.h" I*/

typedef struct {
  KSP                ksp;
  PC                 pc;                   /* actual preconditioner used on each processor */
  Vec                xsub,ysub;            /* vectors of a subcommunicator to hold parallel vectors of PetscObjectComm((PetscObject)pc) */
  Vec                xdup,ydup;            /* parallel vector that congregates xsub or ysub facilitating vector scattering */
  Mat                pmats;                /* matrix and optional preconditioner matrix belong to a subcommunicator */
  VecScatter         scatterin,scatterout; /* scatter used to move all values to each processor group (subcommunicator) */
  PetscBool          useparallelmat;
  PetscSubcomm       psubcomm;
  PetscInt           nsubcomm;             /* num of data structure PetscSubcomm */
  PetscBool          shifttypeset;
  MatFactorShiftType shifttype;
} PC_Redundant;

PetscErrorCode  PCFactorSetShiftType_Redundant(PC pc,MatFactorShiftType shifttype)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (red->ksp) {
    PC pc;
    ierr = KSPGetPC(red->ksp,&pc);CHKERRQ(ierr);
    ierr = PCFactorSetShiftType(pc,shifttype);CHKERRQ(ierr);
  } else {
    red->shifttypeset = PETSC_TRUE;
    red->shifttype    = shifttype;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_Redundant(PC pc,PetscViewer viewer)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii,isstring;
  PetscViewer    subviewer;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    if (!red->psubcomm) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Not yet setup\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  First (color=0) of %D PCs follows\n",red->nsubcomm);CHKERRQ(ierr);
      ierr = PetscViewerGetSubViewer(viewer,((PetscObject)red->pc)->comm,&subviewer);CHKERRQ(ierr);
      if (!red->psubcomm->color) { /* only view first redundant pc */
        ierr = PetscViewerASCIIPushTab(subviewer);CHKERRQ(ierr);
        ierr = KSPView(red->ksp,subviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(subviewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerRestoreSubViewer(viewer,((PetscObject)red->pc)->comm,&subviewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," Redundant solver preconditioner");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/mpi/mpiaij.h>
static PetscErrorCode PCSetUp_Redundant(PC pc)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;
  PetscInt       mstart,mend,mlocal,M;
  PetscMPIInt    size;
  MPI_Comm       comm,subcomm;
  Vec            x;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);

  /* if pmatrix set by user is sequential then we do not need to gather the parallel matrix */
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size == 1) red->useparallelmat = PETSC_FALSE;

  if (!pc->setupcalled) {
    PetscInt mloc_sub;
    if (!red->psubcomm) { /* create red->psubcomm, new ksp and pc over subcomm */
      KSP ksp;
      ierr = PCRedundantGetKSP(pc,&ksp);CHKERRQ(ierr);
    }
    subcomm = PetscSubcommChild(red->psubcomm);

    if (red->useparallelmat) {
      /* grab the parallel matrix and put it into processors of a subcomminicator */
      ierr = MatCreateRedundantMatrix(pc->pmat,red->psubcomm->n,subcomm,MAT_INITIAL_MATRIX,&red->pmats);CHKERRQ(ierr);

      ierr = MPI_Comm_size(subcomm,&size);CHKERRMPI(ierr);
      if (size > 1) {
        PetscBool foundpack,issbaij;
        ierr = PetscObjectTypeCompare((PetscObject)red->pmats,MATMPISBAIJ,&issbaij);CHKERRQ(ierr);
        if (!issbaij) {
          ierr = MatGetFactorAvailable(red->pmats,NULL,MAT_FACTOR_LU,&foundpack);CHKERRQ(ierr);
        } else {
          ierr = MatGetFactorAvailable(red->pmats,NULL,MAT_FACTOR_CHOLESKY,&foundpack);CHKERRQ(ierr);
        }
        if (!foundpack) { /* reset default ksp and pc */
          ierr = KSPSetType(red->ksp,KSPGMRES);CHKERRQ(ierr);
          ierr = PCSetType(red->pc,PCBJACOBI);CHKERRQ(ierr);
        } else {
          ierr = PCFactorSetMatSolverType(red->pc,NULL);CHKERRQ(ierr);
        }
      }

      ierr = KSPSetOperators(red->ksp,red->pmats,red->pmats);CHKERRQ(ierr);

      /* get working vectors xsub and ysub */
      ierr = MatCreateVecs(red->pmats,&red->xsub,&red->ysub);CHKERRQ(ierr);

      /* create working vectors xdup and ydup.
       xdup concatenates all xsub's contigously to form a mpi vector over dupcomm  (see PetscSubcommCreate_interlaced())
       ydup concatenates all ysub and has empty local arrays because ysub's arrays will be place into it.
       Note: we use communicator dupcomm, not PetscObjectComm((PetscObject)pc)! */
      ierr = MatGetLocalSize(red->pmats,&mloc_sub,NULL);CHKERRQ(ierr);
      ierr = VecCreateMPI(PetscSubcommContiguousParent(red->psubcomm),mloc_sub,PETSC_DECIDE,&red->xdup);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(PetscSubcommContiguousParent(red->psubcomm),1,mloc_sub,PETSC_DECIDE,NULL,&red->ydup);CHKERRQ(ierr);

      /* create vecscatters */
      if (!red->scatterin) { /* efficiency of scatterin is independent from psubcomm_type! */
        IS       is1,is2;
        PetscInt *idx1,*idx2,i,j,k;

        ierr = MatCreateVecs(pc->pmat,&x,NULL);CHKERRQ(ierr);
        ierr = VecGetSize(x,&M);CHKERRQ(ierr);
        ierr = VecGetOwnershipRange(x,&mstart,&mend);CHKERRQ(ierr);
        mlocal = mend - mstart;
        ierr = PetscMalloc2(red->psubcomm->n*mlocal,&idx1,red->psubcomm->n*mlocal,&idx2);CHKERRQ(ierr);
        j    = 0;
        for (k=0; k<red->psubcomm->n; k++) {
          for (i=mstart; i<mend; i++) {
            idx1[j]   = i;
            idx2[j++] = i + M*k;
          }
        }
        ierr = ISCreateGeneral(comm,red->psubcomm->n*mlocal,idx1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
        ierr = ISCreateGeneral(comm,red->psubcomm->n*mlocal,idx2,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);
        ierr = VecScatterCreate(x,is1,red->xdup,is2,&red->scatterin);CHKERRQ(ierr);
        ierr = ISDestroy(&is1);CHKERRQ(ierr);
        ierr = ISDestroy(&is2);CHKERRQ(ierr);

        /* Impl below is good for PETSC_SUBCOMM_INTERLACED (no inter-process communication) and PETSC_SUBCOMM_CONTIGUOUS (communication within subcomm) */
        ierr = ISCreateStride(comm,mlocal,mstart+ red->psubcomm->color*M,1,&is1);CHKERRQ(ierr);
        ierr = ISCreateStride(comm,mlocal,mstart,1,&is2);CHKERRQ(ierr);
        ierr = VecScatterCreate(red->xdup,is1,x,is2,&red->scatterout);CHKERRQ(ierr);
        ierr = ISDestroy(&is1);CHKERRQ(ierr);
        ierr = ISDestroy(&is2);CHKERRQ(ierr);
        ierr = PetscFree2(idx1,idx2);CHKERRQ(ierr);
        ierr = VecDestroy(&x);CHKERRQ(ierr);
      }
    } else { /* !red->useparallelmat */
      ierr = KSPSetOperators(red->ksp,pc->mat,pc->pmat);CHKERRQ(ierr);
    }
  } else { /* pc->setupcalled */
    if (red->useparallelmat) {
      MatReuse       reuse;
      /* grab the parallel matrix and put it into processors of a subcomminicator */
      /*--------------------------------------------------------------------------*/
      if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
        /* destroy old matrices */
        ierr  = MatDestroy(&red->pmats);CHKERRQ(ierr);
        reuse = MAT_INITIAL_MATRIX;
      } else {
        reuse = MAT_REUSE_MATRIX;
      }
      ierr = MatCreateRedundantMatrix(pc->pmat,red->psubcomm->n,PetscSubcommChild(red->psubcomm),reuse,&red->pmats);CHKERRQ(ierr);
      ierr = KSPSetOperators(red->ksp,red->pmats,red->pmats);CHKERRQ(ierr);
    } else { /* !red->useparallelmat */
      ierr = KSPSetOperators(red->ksp,pc->mat,pc->pmat);CHKERRQ(ierr);
    }
  }

  if (pc->setfromoptionscalled) {
    ierr = KSPSetFromOptions(red->ksp);CHKERRQ(ierr);
  }
  ierr = KSPSetUp(red->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_Redundant(PC pc,Vec x,Vec y)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;
  PetscScalar    *array;

  PetscFunctionBegin;
  if (!red->useparallelmat) {
    ierr = KSPSolve(red->ksp,x,y);CHKERRQ(ierr);
    ierr = KSPCheckSolve(red->ksp,pc,y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* scatter x to xdup */
  ierr = VecScatterBegin(red->scatterin,x,red->xdup,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->scatterin,x,red->xdup,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* place xdup's local array into xsub */
  ierr = VecGetArray(red->xdup,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(red->xsub,(const PetscScalar*)array);CHKERRQ(ierr);

  /* apply preconditioner on each processor */
  ierr = KSPSolve(red->ksp,red->xsub,red->ysub);CHKERRQ(ierr);
  ierr = KSPCheckSolve(red->ksp,pc,red->ysub);CHKERRQ(ierr);
  ierr = VecResetArray(red->xsub);CHKERRQ(ierr);
  ierr = VecRestoreArray(red->xdup,&array);CHKERRQ(ierr);

  /* place ysub's local array into ydup */
  ierr = VecGetArray(red->ysub,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(red->ydup,(const PetscScalar*)array);CHKERRQ(ierr);

  /* scatter ydup to y */
  ierr = VecScatterBegin(red->scatterout,red->ydup,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->scatterout,red->ydup,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecResetArray(red->ydup);CHKERRQ(ierr);
  ierr = VecRestoreArray(red->ysub,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_Redundant(PC pc,Vec x,Vec y)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;
  PetscScalar    *array;

  PetscFunctionBegin;
  if (!red->useparallelmat) {
    ierr = KSPSolveTranspose(red->ksp,x,y);CHKERRQ(ierr);
    ierr = KSPCheckSolve(red->ksp,pc,y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* scatter x to xdup */
  ierr = VecScatterBegin(red->scatterin,x,red->xdup,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->scatterin,x,red->xdup,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* place xdup's local array into xsub */
  ierr = VecGetArray(red->xdup,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(red->xsub,(const PetscScalar*)array);CHKERRQ(ierr);

  /* apply preconditioner on each processor */
  ierr = KSPSolveTranspose(red->ksp,red->xsub,red->ysub);CHKERRQ(ierr);
  ierr = KSPCheckSolve(red->ksp,pc,red->ysub);CHKERRQ(ierr);
  ierr = VecResetArray(red->xsub);CHKERRQ(ierr);
  ierr = VecRestoreArray(red->xdup,&array);CHKERRQ(ierr);

  /* place ysub's local array into ydup */
  ierr = VecGetArray(red->ysub,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(red->ydup,(const PetscScalar*)array);CHKERRQ(ierr);

  /* scatter ydup to y */
  ierr = VecScatterBegin(red->scatterout,red->ydup,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->scatterout,red->ydup,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecResetArray(red->ydup);CHKERRQ(ierr);
  ierr = VecRestoreArray(red->ysub,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_Redundant(PC pc)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (red->useparallelmat) {
    ierr = VecScatterDestroy(&red->scatterin);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&red->scatterout);CHKERRQ(ierr);
    ierr = VecDestroy(&red->ysub);CHKERRQ(ierr);
    ierr = VecDestroy(&red->xsub);CHKERRQ(ierr);
    ierr = VecDestroy(&red->xdup);CHKERRQ(ierr);
    ierr = VecDestroy(&red->ydup);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&red->pmats);CHKERRQ(ierr);
  ierr = KSPReset(red->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Redundant(PC pc)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_Redundant(pc);CHKERRQ(ierr);
  ierr = KSPDestroy(&red->ksp);CHKERRQ(ierr);
  ierr = PetscSubcommDestroy(&red->psubcomm);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Redundant(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode ierr;
  PC_Redundant   *red = (PC_Redundant*)pc->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Redundant options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_redundant_number","Number of redundant pc","PCRedundantSetNumber",red->nsubcomm,&red->nsubcomm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCRedundantSetNumber_Redundant(PC pc,PetscInt nreds)
{
  PC_Redundant *red = (PC_Redundant*)pc->data;

  PetscFunctionBegin;
  red->nsubcomm = nreds;
  PetscFunctionReturn(0);
}

/*@
   PCRedundantSetNumber - Sets the number of redundant preconditioner contexts.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  nredundant - number of redundant preconditioner contexts; for example if you are using 64 MPI processes and
                              use an nredundant of 4 there will be 4 parallel solves each on 16 = 64/4 processes.

   Level: advanced

@*/
PetscErrorCode PCRedundantSetNumber(PC pc,PetscInt nredundant)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (nredundant <= 0) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG, "num of redundant pc %D must be positive",nredundant);
  ierr = PetscTryMethod(pc,"PCRedundantSetNumber_C",(PC,PetscInt),(pc,nredundant));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCRedundantSetScatter_Redundant(PC pc,VecScatter in,VecScatter out)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)in);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&red->scatterin);CHKERRQ(ierr);

  red->scatterin  = in;

  ierr = PetscObjectReference((PetscObject)out);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&red->scatterout);CHKERRQ(ierr);
  red->scatterout = out;
  PetscFunctionReturn(0);
}

/*@
   PCRedundantSetScatter - Sets the scatter used to copy values into the
     redundant local solve and the scatter to move them back into the global
     vector.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  in - the scatter to move the values in
-  out - the scatter to move them out

   Level: advanced

@*/
PetscErrorCode PCRedundantSetScatter(PC pc,VecScatter in,VecScatter out)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(in,PETSCSF_CLASSID,2);
  PetscValidHeaderSpecific(out,PETSCSF_CLASSID,3);
  ierr = PetscTryMethod(pc,"PCRedundantSetScatter_C",(PC,VecScatter,VecScatter),(pc,in,out));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCRedundantGetKSP_Redundant(PC pc,KSP *innerksp)
{
  PetscErrorCode ierr;
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  MPI_Comm       comm,subcomm;
  const char     *prefix;
  PetscBool      issbaij;

  PetscFunctionBegin;
  if (!red->psubcomm) {
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);

    ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
    ierr = PetscSubcommCreate(comm,&red->psubcomm);CHKERRQ(ierr);
    ierr = PetscSubcommSetNumber(red->psubcomm,red->nsubcomm);CHKERRQ(ierr);
    ierr = PetscSubcommSetType(red->psubcomm,PETSC_SUBCOMM_CONTIGUOUS);CHKERRQ(ierr);

    ierr = PetscSubcommSetOptionsPrefix(red->psubcomm,prefix);CHKERRQ(ierr);
    ierr = PetscSubcommSetFromOptions(red->psubcomm);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)pc,sizeof(PetscSubcomm));CHKERRQ(ierr);

    /* create a new PC that processors in each subcomm have copy of */
    subcomm = PetscSubcommChild(red->psubcomm);

    ierr = KSPCreate(subcomm,&red->ksp);CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(red->ksp,pc->erroriffailure);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)red->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)red->ksp);CHKERRQ(ierr);
    ierr = KSPSetType(red->ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(red->ksp,&red->pc);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);
    if (!issbaij) {
      ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATMPISBAIJ,&issbaij);CHKERRQ(ierr);
    }
    if (!issbaij) {
      ierr = PCSetType(red->pc,PCLU);CHKERRQ(ierr);
    } else {
      ierr = PCSetType(red->pc,PCCHOLESKY);CHKERRQ(ierr);
    }
    if (red->shifttypeset) {
      ierr = PCFactorSetShiftType(red->pc,red->shifttype);CHKERRQ(ierr);
      red->shifttypeset = PETSC_FALSE;
    }
    ierr = KSPSetOptionsPrefix(red->ksp,prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(red->ksp,"redundant_");CHKERRQ(ierr);
  }
  *innerksp = red->ksp;
  PetscFunctionReturn(0);
}

/*@
   PCRedundantGetKSP - Gets the less parallel KSP created by the redundant PC.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  innerksp - the KSP on the smaller set of processes

   Level: advanced

@*/
PetscErrorCode PCRedundantGetKSP(PC pc,KSP *innerksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(innerksp,2);
  ierr = PetscUseMethod(pc,"PCRedundantGetKSP_C",(PC,KSP*),(pc,innerksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCRedundantGetOperators_Redundant(PC pc,Mat *mat,Mat *pmat)
{
  PC_Redundant *red = (PC_Redundant*)pc->data;

  PetscFunctionBegin;
  if (mat)  *mat  = red->pmats;
  if (pmat) *pmat = red->pmats;
  PetscFunctionReturn(0);
}

/*@
   PCRedundantGetOperators - gets the sequential matrix and preconditioner matrix

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  mat - the matrix
-  pmat - the (possibly different) preconditioner matrix

   Level: advanced

@*/
PetscErrorCode PCRedundantGetOperators(PC pc,Mat *mat,Mat *pmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (mat)  PetscValidPointer(mat,2);
  if (pmat) PetscValidPointer(pmat,3);
  ierr = PetscUseMethod(pc,"PCRedundantGetOperators_C",(PC,Mat*,Mat*),(pc,mat,pmat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
     PCREDUNDANT - Runs a KSP solver with preconditioner for the entire problem on subgroups of processors

     Options for the redundant preconditioners can be set with -redundant_pc_xxx for the redundant KSP with -redundant_ksp_xxx

  Options Database:
.  -pc_redundant_number <n> - number of redundant solves, for example if you are using 64 MPI processes and
                              use an n of 4 there will be 4 parallel solves each on 16 = 64/4 processes.

   Level: intermediate

   Notes:
    The default KSP is preonly and the default PC is LU or CHOLESKY if Pmat is of type MATSBAIJ.

   PCFactorSetShiftType() applied to this PC will convey they shift type into the inner PC if it is factorization based.

   Developer Notes:
    Note that PCSetInitialGuessNonzero()  is not used by this class but likely should be.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCRedundantSetScatter(),
           PCRedundantGetKSP(), PCRedundantGetOperators(), PCRedundantSetNumber()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Redundant(PC pc)
{
  PetscErrorCode ierr;
  PC_Redundant   *red;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&red);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRMPI(ierr);

  red->nsubcomm       = size;
  red->useparallelmat = PETSC_TRUE;
  pc->data            = (void*)red;

  pc->ops->apply          = PCApply_Redundant;
  pc->ops->applytranspose = PCApplyTranspose_Redundant;
  pc->ops->setup          = PCSetUp_Redundant;
  pc->ops->destroy        = PCDestroy_Redundant;
  pc->ops->reset          = PCReset_Redundant;
  pc->ops->setfromoptions = PCSetFromOptions_Redundant;
  pc->ops->view           = PCView_Redundant;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCRedundantSetScatter_C",PCRedundantSetScatter_Redundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCRedundantSetNumber_C",PCRedundantSetNumber_Redundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCRedundantGetKSP_C",PCRedundantGetKSP_Redundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCRedundantGetOperators_C",PCRedundantGetOperators_Redundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetShiftType_C",PCFactorSetShiftType_Redundant);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

