#define PETSCKSP_DLL

/*
    Defines the multigrid preconditioner interface.
*/
#include "src/ksp/pc/impls/mg/mgimpl.h"                    /*I "petscmg.h" I*/


#undef __FUNCT__  
#define __FUNCT__ "PCMGMCycle_Private"
PetscErrorCode PCMGMCycle_Private(PC_MG **mglevels,PetscTruth *converged)
{
  PC_MG          *mg = *mglevels,*mgc;
  PetscErrorCode ierr;
  PetscInt       cycles = mg->cycles;

  PetscFunctionBegin;
  if (converged) *converged = PETSC_FALSE;

  if (mg->eventsolve) {ierr = PetscLogEventBegin(mg->eventsolve,0,0,0,0);CHKERRQ(ierr);}
  ierr = KSPSolve(mg->smoothd,mg->b,mg->x);CHKERRQ(ierr);
  if (mg->eventsolve) {ierr = PetscLogEventEnd(mg->eventsolve,0,0,0,0);CHKERRQ(ierr);}
  if (mg->level) {  /* not the coarsest grid */
    ierr = (*mg->residual)(mg->A,mg->b,mg->x,mg->r);CHKERRQ(ierr);

    /* if on finest level and have convergence criteria set */
    if (mg->level == mg->levels-1 && mg->ttol) {
      PetscReal rnorm;
      ierr = VecNorm(mg->r,NORM_2,&rnorm);CHKERRQ(ierr);
      if (rnorm <= mg->ttol) {
        *converged = PETSC_TRUE;
        if (rnorm < mg->abstol) {
          ierr = PetscLogInfo((0,"PCMGMCycle_Private:Linear solver has converged. Residual norm %g is less than absolute tolerance %g\n",rnorm,mg->abstol));CHKERRQ(ierr);
        } else {
          ierr = PetscLogInfo((0,"PCMGMCycle_Private:Linear solver has converged. Residual norm %g is less than relative tolerance times initial residual norm %g\n",rnorm,mg->ttol));CHKERRQ(ierr);
        }
        PetscFunctionReturn(0);
      }
    }

    mgc = *(mglevels - 1);
    ierr = MatRestrict(mg->restrct,mg->r,mgc->b);CHKERRQ(ierr);
    ierr = VecSet(mgc->x,0.0);CHKERRQ(ierr);
    while (cycles--) {
      ierr = PCMGMCycle_Private(mglevels-1,converged);CHKERRQ(ierr); 
    }
    ierr = MatInterpolateAdd(mg->interpolate,mgc->x,mg->x,mg->x);CHKERRQ(ierr);
    if (mg->eventsolve) {ierr = PetscLogEventBegin(mg->eventsolve,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSolve(mg->smoothu,mg->b,mg->x);CHKERRQ(ierr); 
    if (mg->eventsolve) {ierr = PetscLogEventEnd(mg->eventsolve,0,0,0,0);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*
       PCMGCreate_Private - Creates a PC_MG structure for use with the
               multigrid code. Level 0 is the coarsest. (But the 
               finest level is stored first in the array).

*/
#undef __FUNCT__  
#define __FUNCT__ "PCMGCreate_Private"
static PetscErrorCode PCMGCreate_Private(MPI_Comm comm,PetscInt levels,PC pc,MPI_Comm *comms,PC_MG ***result)
{
  PC_MG          **mg;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscMPIInt    size;
  const char     *prefix;
  PC             ipc;

  PetscFunctionBegin;
  ierr = PetscMalloc(levels*sizeof(PC_MG*),&mg);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,levels*(sizeof(PC_MG*)+sizeof(PC_MG)));CHKERRQ(ierr);

  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);

  for (i=0; i<levels; i++) {
    ierr = PetscNew(PC_MG,&mg[i]);CHKERRQ(ierr);
    mg[i]->level           = i;
    mg[i]->levels          = levels;
    mg[i]->cycles          = 1;
    mg[i]->galerkin        = PETSC_FALSE;
    mg[i]->galerkinused    = PETSC_FALSE;
    mg[i]->default_smoothu = 1;
    mg[i]->default_smoothd = 1;

    if (comms) comm = comms[i];
    ierr = KSPCreate(comm,&mg[i]->smoothd);CHKERRQ(ierr);
    ierr = KSPSetTolerances(mg[i]->smoothd,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT, mg[i]->default_smoothd);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(mg[i]->smoothd,prefix);CHKERRQ(ierr);

    /* do special stuff for coarse grid */
    if (!i && levels > 1) {
      ierr = KSPAppendOptionsPrefix(mg[0]->smoothd,"mg_coarse_");CHKERRQ(ierr);

      /* coarse solve is (redundant) LU by default */
      ierr = KSPSetType(mg[0]->smoothd,KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(mg[0]->smoothd,&ipc);CHKERRQ(ierr);
      ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
      if (size > 1) {
        ierr = PCSetType(ipc,PCREDUNDANT);CHKERRQ(ierr);
        ierr = PCRedundantGetPC(ipc,&ipc);CHKERRQ(ierr);
      }
      ierr = PCSetType(ipc,PCLU);CHKERRQ(ierr);

    } else {
      char tprefix[128];
      sprintf(tprefix,"mg_levels_%d_",(int)i);
      ierr = KSPAppendOptionsPrefix(mg[i]->smoothd,tprefix);CHKERRQ(ierr);
    }
    ierr = PetscLogObjectParent(pc,mg[i]->smoothd);CHKERRQ(ierr);
    mg[i]->smoothu         = mg[i]->smoothd;
    mg[i]->rtol = 0.0;
    mg[i]->abstol = 0.0;
    mg[i]->dtol = 0.0;
    mg[i]->ttol = 0.0;
    mg[i]->eventsetup = 0;
    mg[i]->eventsolve = 0;
  }
  *result = mg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_MG"
static PetscErrorCode PCDestroy_MG(PC pc)
{
  PC_MG          **mg = (PC_MG**)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,n = mg[0]->levels;

  PetscFunctionBegin;
  if (mg[0]->galerkinused) {
    Mat B;
    for (i=0; i<n-1; i++) {
      ierr = KSPGetOperators(mg[i]->smoothd,0,&B,0);CHKERRQ(ierr);
      ierr = MatDestroy(B);CHKERRQ(ierr);
    }
  }

  for (i=0; i<n-1; i++) {
    if (mg[i+1]->r) {ierr = VecDestroy(mg[i+1]->r);CHKERRQ(ierr);}
    if (mg[i]->b) {ierr = VecDestroy(mg[i]->b);CHKERRQ(ierr);}
    if (mg[i]->x) {ierr = VecDestroy(mg[i]->x);CHKERRQ(ierr);}
    if (mg[i+1]->restrct) {ierr = MatDestroy(mg[i+1]->restrct);CHKERRQ(ierr);}
    if (mg[i+1]->interpolate) {ierr = MatDestroy(mg[i+1]->interpolate);CHKERRQ(ierr);}
  }

  for (i=0; i<n; i++) {
    if (mg[i]->smoothd != mg[i]->smoothu) {
      ierr = KSPDestroy(mg[i]->smoothd);CHKERRQ(ierr);
    }
    ierr = KSPDestroy(mg[i]->smoothu);CHKERRQ(ierr);
    ierr = PetscFree(mg[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(mg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



EXTERN PetscErrorCode PCMGACycle_Private(PC_MG**);
EXTERN PetscErrorCode PCMGFCycle_Private(PC_MG**);
EXTERN PetscErrorCode PCMGKCycle_Private(PC_MG**);

/*
   PCApply_MG - Runs either an additive, multiplicative, Kaskadic
             or full cycle of multigrid. 

  Note: 
  A simple wrapper which calls PCMGMCycle(),PCMGACycle(), or PCMGFCycle(). 
*/ 
#undef __FUNCT__  
#define __FUNCT__ "PCApply_MG"
static PetscErrorCode PCApply_MG(PC pc,Vec b,Vec x)
{
  PC_MG          **mg = (PC_MG**)pc->data;
  PetscErrorCode ierr;
  PetscInt       levels = mg[0]->levels;

  PetscFunctionBegin;
  mg[levels-1]->b = b; 
  mg[levels-1]->x = x;
  if (!mg[levels-1]->r && mg[0]->am == PC_MG_ADDITIVE) {
    Vec tvec;
    ierr = VecDuplicate(mg[levels-1]->b,&tvec);CHKERRQ(ierr);
    ierr = PCMGSetR(pc,levels-1,tvec);CHKERRQ(ierr);
    ierr = VecDestroy(tvec);CHKERRQ(ierr);
  }
  if (mg[0]->am == PC_MG_MULTIPLICATIVE) {
    ierr = VecSet(x,0.0);CHKERRQ(ierr);
    ierr = PCMGMCycle_Private(mg+levels-1,PETSC_NULL);CHKERRQ(ierr);
  } 
  else if (mg[0]->am == PC_MG_ADDITIVE) {
    ierr = PCMGACycle_Private(mg);CHKERRQ(ierr);
  }
  else if (mg[0]->am == PC_MG_KASKADE) {
    ierr = PCMGKCycle_Private(mg);CHKERRQ(ierr);
  }
  else {
    ierr = PCMGFCycle_Private(mg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyRichardson_MG"
static PetscErrorCode PCApplyRichardson_MG(PC pc,Vec b,Vec x,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its)
{
  PC_MG          **mg = (PC_MG**)pc->data;
  PetscErrorCode ierr;
  PetscInt       levels = mg[0]->levels;
  PetscTruth     converged = PETSC_FALSE;

  PetscFunctionBegin;
  mg[levels-1]->b    = b; 
  mg[levels-1]->x    = x;

  mg[levels-1]->rtol = rtol;
  mg[levels-1]->abstol = abstol;
  mg[levels-1]->dtol = dtol;
  if (rtol) {
    /* compute initial residual norm for relative convergence test */
    PetscReal rnorm;
    ierr               = (*mg[levels-1]->residual)(mg[levels-1]->A,b,x,w);CHKERRQ(ierr);
    ierr               = VecNorm(w,NORM_2,&rnorm);CHKERRQ(ierr);
    mg[levels-1]->ttol = PetscMax(rtol*rnorm,abstol);
  } else if (abstol) {
    mg[levels-1]->ttol = abstol;
  } else {
    mg[levels-1]->ttol = 0.0;
  }

  while (its-- && !converged) {
    ierr = PCMGMCycle_Private(mg+levels-1,&converged);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_MG"
PetscErrorCode PCSetFromOptions_MG(PC pc)
{
  PetscErrorCode ierr;
  PetscInt       m,levels = 1;
  PetscTruth     flg;
  PC_MG          **mg = (PC_MG**)pc->data;
  PCMGType       mgtype = mg[0]->am;;

  PetscFunctionBegin;

  ierr = PetscOptionsHead("Multigrid options");CHKERRQ(ierr);
    if (!pc->data) {
      ierr = PetscOptionsInt("-pc_mg_levels","Number of Levels","PCMGSetLevels",levels,&levels,&flg);CHKERRQ(ierr);
      ierr = PCMGSetLevels(pc,levels,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-pc_mg_cycles","1 for V cycle, 2 for W-cycle","PCMGSetCycles",1,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCMGSetCycles(pc,m);CHKERRQ(ierr);
    } 
    ierr = PetscOptionsName("-pc_mg_galerkin","Use Galerkin process to compute coarser operators","PCMGSetGalerkin",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCMGSetGalerkin(pc);CHKERRQ(ierr);
    } 
    ierr = PetscOptionsInt("-pc_mg_smoothup","Number of post-smoothing steps","PCMGSetNumberSmoothUp",1,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCMGSetNumberSmoothUp(pc,m);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-pc_mg_smoothdown","Number of pre-smoothing steps","PCMGSetNumberSmoothDown",1,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCMGSetNumberSmoothDown(pc,m);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-pc_mg_type","Multigrid type","PCMGSetType",PCMGTypes,(PetscEnum)mgtype,(PetscEnum*)&mgtype,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PCMGSetType(pc,mgtype);CHKERRQ(ierr);}
    ierr = PetscOptionsName("-pc_mg_log","Log times for each multigrid level","None",&flg);CHKERRQ(ierr);
    if (flg) {
      PetscInt i;
      char     eventname[128];
      if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
      levels = mg[0]->levels;
      for (i=0; i<levels; i++) {  
        sprintf(eventname,"MSetup Level %d",(int)i);
        ierr = PetscLogEventRegister(&mg[i]->eventsetup,eventname,pc->cookie);CHKERRQ(ierr);
        sprintf(eventname,"MGSolve Level %d to 0",(int)i);
        ierr = PetscLogEventRegister(&mg[i]->eventsolve,eventname,pc->cookie);CHKERRQ(ierr);
      }
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

const char *PCMGTypes[] = {"MULTIPLICATIVE","ADDITIVE","FULL","KASKADE","PCMGType","PC_MG",0};

#undef __FUNCT__  
#define __FUNCT__ "PCView_MG"
static PetscErrorCode PCView_MG(PC pc,PetscViewer viewer)
{
  PC_MG          **mg = (PC_MG**)pc->data;
  PetscErrorCode ierr;
  PetscInt       levels = mg[0]->levels,i;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  MG: type is %s, levels=%D cycles=%D, pre-smooths=%D, post-smooths=%D\n",
                      PCMGTypes[mg[0]->am],levels,mg[0]->cycles,mg[0]->default_smoothd,mg[0]->default_smoothu);CHKERRQ(ierr);
    if (mg[0]->galerkin) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Using Galerkin computed coarse grid matrices\n");CHKERRQ(ierr);
    }
    for (i=0; i<levels; i++) {
      if (!i) {
        ierr = PetscViewerASCIIPrintf(viewer,"Coarse gride solver -- level %D -------------------------------\n",i);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Down solver (pre-smoother) on level %D -------------------------------\n",i);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = KSPView(mg[i]->smoothd,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      if (i && mg[i]->smoothd == mg[i]->smoothu) {
        ierr = PetscViewerASCIIPrintf(viewer,"Up solver (post-smoother) same as down solver (pre-smoother)\n");CHKERRQ(ierr);
      } else if (i){
        ierr = PetscViewerASCIIPrintf(viewer,"Up solver (post-smoother) on level %D -------------------------------\n",i);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = KSPView(mg[i]->smoothu,viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    }
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCMG",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*
    Calls setup for the KSP on each level
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_MG"
static PetscErrorCode PCSetUp_MG(PC pc)
{
  PC_MG          **mg = (PC_MG**)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,n = mg[0]->levels;
  PC             cpc;
  PetscTruth     preonly,lu,redundant,cholesky,monitor = PETSC_FALSE,dump;
  PetscViewer    ascii;
  MPI_Comm       comm;
  Mat            dA,dB;
  MatStructure   uflag;
  Vec            tvec;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    ierr = PetscOptionsHasName(0,"-pc_mg_monitor",&monitor);CHKERRQ(ierr);
     
    for (i=0; i<n; i++) {
      if (monitor) {
        ierr = PetscObjectGetComm((PetscObject)mg[i]->smoothd,&comm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
        ierr = PetscViewerASCIISetTab(ascii,n-i);CHKERRQ(ierr);
        ierr = KSPSetMonitor(mg[i]->smoothd,KSPDefaultMonitor,ascii,(PetscErrorCode(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
      }
      ierr = KSPSetFromOptions(mg[i]->smoothd);CHKERRQ(ierr);
    }
    for (i=1; i<n; i++) {
      if (mg[i]->smoothu && mg[i]->smoothu != mg[i]->smoothd) {
        if (monitor) {
          ierr = PetscObjectGetComm((PetscObject)mg[i]->smoothu,&comm);CHKERRQ(ierr);
          ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
          ierr = PetscViewerASCIISetTab(ascii,n-i);CHKERRQ(ierr);
          ierr = KSPSetMonitor(mg[i]->smoothu,KSPDefaultMonitor,ascii,(PetscErrorCode(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
        }
        ierr = KSPSetFromOptions(mg[i]->smoothu);CHKERRQ(ierr);
      }
    }
    for (i=1; i<n; i++) {
      if (mg[i]->restrct && !mg[i]->interpolate) {
        ierr = PCMGSetInterpolate(pc,i,mg[i]->restrct);CHKERRQ(ierr);
      }
      if (!mg[i]->restrct && mg[i]->interpolate) {
        ierr = PCMGSetRestriction(pc,i,mg[i]->interpolate);CHKERRQ(ierr);
      }
#if defined(PETSC_USE_DEBUG)
      if (!mg[i]->restrct || !mg[i]->interpolate) {
        SETERRQ1(PETSC_ERR_ARG_WRONGSTATE,"Need to set restriction or interpolation on level %d",(int)i);
      }
#endif
    }
    for (i=0; i<n-1; i++) {
      if (!mg[i]->b) {
        Mat mat;
        ierr = KSPGetOperators(mg[i]->smoothd,PETSC_NULL,&mat,PETSC_NULL);CHKERRQ(ierr);

      }
      if (!mg[i]->r && i) {
        ierr = VecDuplicate(mg[i]->b,&tvec);CHKERRQ(ierr);
        ierr = PCMGSetR(pc,i,tvec);CHKERRQ(ierr);
        ierr = VecDestroy(tvec);CHKERRQ(ierr);
      }
      if (!mg[i]->x) {
        ierr = VecDuplicate(mg[i]->b,&tvec);CHKERRQ(ierr);
        ierr = PCMGSetX(pc,i,tvec);CHKERRQ(ierr);
        ierr = VecDestroy(tvec);CHKERRQ(ierr);
      }
    }
  }

  /* If user did not provide fine grid operators, use those from PC */
  /* BUG BUG BUG This will work ONLY the first time called: hence if the user changes
     the PC matrices between solves PCMG will continue to use first set provided */
  ierr = KSPGetOperators(mg[n-1]->smoothd,&dA,&dB,&uflag);CHKERRQ(ierr);
  if (!dA  && !dB) {
    ierr = PetscLogInfo((pc,"PCSetUp_MG: Using outer operators to define finest grid operator \n  because PCMGGetSmoother(pc,nlevels-1,&ksp);KSPSetOperators(ksp,...); was not called.\n"));
    ierr = KSPSetOperators(mg[n-1]->smoothd,pc->mat,pc->pmat,uflag);CHKERRQ(ierr);
  }

  if (mg[0]->galerkin) {
    Mat B;
    mg[0]->galerkinused = PETSC_TRUE;
    /* currently only handle case where mat and pmat are the same on coarser levels */
    ierr = KSPGetOperators(mg[n-1]->smoothd,&dA,&dB,&uflag);CHKERRQ(ierr);
    if (!pc->setupcalled) {
      for (i=n-2; i>-1; i--) {
        ierr = MatPtAP(dB,mg[i+1]->interpolate,MAT_INITIAL_MATRIX,1.0,&B);CHKERRQ(ierr);
        ierr = KSPSetOperators(mg[i]->smoothd,B,B,uflag);CHKERRQ(ierr);
        dB   = B;
      }
    } else {
      for (i=n-2; i>-1; i--) {
        ierr = KSPGetOperators(mg[i]->smoothd,0,&B,0);CHKERRQ(ierr);
        ierr = MatPtAP(dB,mg[i]->interpolate,MAT_REUSE_MATRIX,1.0,&B);CHKERRQ(ierr);
        ierr = KSPSetOperators(mg[i]->smoothd,B,B,uflag);CHKERRQ(ierr);
        dB   = B;
      }
    }
  }

  for (i=1; i<n; i++) {
    if (mg[i]->smoothu == mg[i]->smoothd) {
      /* if doing only down then initial guess is zero */
      ierr = KSPSetInitialGuessNonzero(mg[i]->smoothd,PETSC_TRUE);CHKERRQ(ierr);
    }
    if (mg[i]->eventsetup) {ierr = PetscLogEventBegin(mg[i]->eventsetup,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSetUp(mg[i]->smoothd);CHKERRQ(ierr);
    if (mg[i]->eventsetup) {ierr = PetscLogEventEnd(mg[i]->eventsetup,0,0,0,0);CHKERRQ(ierr);}
  }
  for (i=1; i<n; i++) {
    if (mg[i]->smoothu && mg[i]->smoothu != mg[i]->smoothd) {
      PC           uppc,downpc;
      Mat          downmat,downpmat,upmat,uppmat;
      MatStructure matflag;

      /* check if operators have been set for up, if not use down operators to set them */
      ierr = KSPGetPC(mg[i]->smoothu,&uppc);CHKERRQ(ierr);
      ierr = PCGetOperators(uppc,&upmat,&uppmat,PETSC_NULL);CHKERRQ(ierr);
      if (!upmat) {
        ierr = KSPGetPC(mg[i]->smoothd,&downpc);CHKERRQ(ierr);
        ierr = PCGetOperators(downpc,&downmat,&downpmat,&matflag);CHKERRQ(ierr);
        ierr = KSPSetOperators(mg[i]->smoothu,downmat,downpmat,matflag);CHKERRQ(ierr);
      }

      ierr = KSPSetInitialGuessNonzero(mg[i]->smoothu,PETSC_TRUE);CHKERRQ(ierr);
      if (mg[i]->eventsetup) {ierr = PetscLogEventBegin(mg[i]->eventsetup,0,0,0,0);CHKERRQ(ierr);}
      ierr = KSPSetUp(mg[i]->smoothu);CHKERRQ(ierr);
      if (mg[i]->eventsetup) {ierr = PetscLogEventEnd(mg[i]->eventsetup,0,0,0,0);CHKERRQ(ierr);}
    }
  }

  /*
      If coarse solver is not direct method then DO NOT USE preonly 
  */
  ierr = PetscTypeCompare((PetscObject)mg[0]->smoothd,KSPPREONLY,&preonly);CHKERRQ(ierr);
  if (preonly) {
    ierr = KSPGetPC(mg[0]->smoothd,&cpc);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)cpc,PCLU,&lu);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)cpc,PCREDUNDANT,&redundant);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)cpc,PCCHOLESKY,&cholesky);CHKERRQ(ierr);
    if (!lu && !redundant && !cholesky) {
      ierr = KSPSetType(mg[0]->smoothd,KSPGMRES);CHKERRQ(ierr);
    }
  }

  if (!pc->setupcalled) {
    if (monitor) {
      ierr = PetscObjectGetComm((PetscObject)mg[0]->smoothd,&comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
      ierr = PetscViewerASCIISetTab(ascii,n);CHKERRQ(ierr);
      ierr = KSPSetMonitor(mg[0]->smoothd,KSPDefaultMonitor,ascii,(PetscErrorCode(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
    }
    ierr = KSPSetFromOptions(mg[0]->smoothd);CHKERRQ(ierr);
  }

  if (mg[0]->eventsetup) {ierr = PetscLogEventBegin(mg[0]->eventsetup,0,0,0,0);CHKERRQ(ierr);}
  ierr = KSPSetUp(mg[0]->smoothd);CHKERRQ(ierr);
  if (mg[0]->eventsetup) {ierr = PetscLogEventEnd(mg[0]->eventsetup,0,0,0,0);CHKERRQ(ierr);}

#if defined(PETSC_USE_SOCKET_VIEWER)
  /*
     Dump the interpolation/restriction matrices to matlab plus the 
   Jacobian/stiffness on each level. This allows Matlab users to 
   easily check if the Galerkin condition A_c = R A_f R^T is satisfied */
  ierr = PetscOptionsHasName(pc->prefix,"-pc_mg_dump_matlab",&dump);CHKERRQ(ierr);
  if (dump) {
    for (i=1; i<n; i++) {
      ierr = MatView(mg[i]->restrct,PETSC_VIEWER_SOCKET_(pc->comm));CHKERRQ(ierr);
    }
    for (i=0; i<n; i++) {
      ierr = KSPGetPC(mg[i]->smoothd,&pc);CHKERRQ(ierr);
      ierr = MatView(pc->mat,PETSC_VIEWER_SOCKET_(pc->comm));CHKERRQ(ierr);
    }
  }
#endif

  ierr = PetscOptionsHasName(pc->prefix,"-pc_mg_dump_binary",&dump);CHKERRQ(ierr);
  if (dump) {
    for (i=1; i<n; i++) {
      ierr = MatView(mg[i]->restrct,PETSC_VIEWER_BINARY_(pc->comm));CHKERRQ(ierr);
    }
    for (i=0; i<n; i++) {
      ierr = KSPGetPC(mg[i]->smoothd,&pc);CHKERRQ(ierr);
      ierr = MatView(pc->mat,PETSC_VIEWER_BINARY_(pc->comm));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetLevels"
/*@C
   PCMGSetLevels - Sets the number of levels to use with MG.
   Must be called before any other MG routine.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  levels - the number of levels
-  comms - optional communicators for each level; this is to allow solving the coarser problems
           on smaller sets of processors. Use PETSC_NULL_OBJECT for default in Fortran

   Level: intermediate

   Notes:
     If the number of levels is one then the multigrid uses the -mg_levels prefix
  for setting the level options rather than the -mg_coarse prefix.

.keywords: MG, set, levels, multigrid

.seealso: PCMGSetType(), PCMGGetLevels()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetLevels(PC pc,PetscInt levels,MPI_Comm *comms)
{
  PetscErrorCode ierr;
  PC_MG          **mg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);

  if (pc->data) {
    SETERRQ(PETSC_ERR_ORDER,"Number levels already set for MG\n\
    make sure that you call PCMGSetLevels() before KSPSetFromOptions()");
  }
  ierr                     = PCMGCreate_Private(pc->comm,levels,pc,comms,&mg);CHKERRQ(ierr);
  mg[0]->am                = PC_MG_MULTIPLICATIVE;
  pc->data                 = (void*)mg;
  pc->ops->applyrichardson = PCApplyRichardson_MG;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGGetLevels"
/*@
   PCMGGetLevels - Gets the number of levels to use with MG.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output parameter:
.  levels - the number of levels

   Level: advanced

.keywords: MG, get, levels, multigrid

.seealso: PCMGSetLevels()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetLevels(PC pc,PetscInt *levels)
{
  PC_MG  **mg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidIntPointer(levels,2);

  mg      = (PC_MG**)pc->data;
  *levels = mg[0]->levels;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetType"
/*@
   PCMGSetType - Determines the form of multigrid to use:
   multiplicative, additive, full, or the Kaskade algorithm.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  form - multigrid form, one of PC_MG_MULTIPLICATIVE, PC_MG_ADDITIVE,
   PC_MG_FULL, PC_MG_KASKADE

   Options Database Key:
.  -pc_mg_type <form> - Sets <form>, one of multiplicative,
   additive, full, kaskade   

   Level: advanced

.keywords: MG, set, method, multiplicative, additive, full, Kaskade, multigrid

.seealso: PCMGSetLevels()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetType(PC pc,PCMGType form)
{
  PC_MG **mg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  mg = (PC_MG**)pc->data;

  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  mg[0]->am = form;
  if (form == PC_MG_MULTIPLICATIVE) pc->ops->applyrichardson = PCApplyRichardson_MG;
  else pc->ops->applyrichardson = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetCycles"
/*@
   PCMGSetCycles - Sets the type cycles to use.  Use PCMGSetCyclesOnLevel() for more 
   complicated cycling.

   Collective on PC

   Input Parameters:
+  pc - the multigrid context 
-  n - the number of cycles

   Options Database Key:
$  -pc_mg_cycles n - 1 denotes a V-cycle; 2 denotes a W-cycle.

   Level: advanced

.keywords: MG, set, cycles, V-cycle, W-cycle, multigrid

.seealso: PCMGSetCyclesOnLevel()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetCycles(PC pc,PetscInt n)
{ 
  PC_MG    **mg;
  PetscInt i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  mg     = (PC_MG**)pc->data;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  levels = mg[0]->levels;

  for (i=0; i<levels; i++) {  
    mg[i]->cycles  = n; 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetGalerkin"
/*@
   PCMGSetGalerkin - Causes the coarser grid matrices to be computed from the
      finest grid via the Galerkin process: A_i-1 = r_i * A_i * r_i^t

   Collective on PC

   Input Parameters:
+  pc - the multigrid context 
-  n - the number of cycles

   Options Database Key:
$  -pc_mg_galerkin

   Level: intermediate

.keywords: MG, set, Galerkin

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetGalerkin(PC pc)
{ 
  PC_MG    **mg;
  PetscInt i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  mg     = (PC_MG**)pc->data;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  levels = mg[0]->levels;

  for (i=0; i<levels; i++) {  
    mg[i]->galerkin = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetNumberSmoothDown"
/*@
   PCMGSetNumberSmoothDown - Sets the number of pre-smoothing steps to
   use on all levels. Use PCMGGetSmootherDown() to set different 
   pre-smoothing steps on different levels.

   Collective on PC

   Input Parameters:
+  mg - the multigrid context 
-  n - the number of smoothing steps

   Options Database Key:
.  -pc_mg_smoothdown <n> - Sets number of pre-smoothing steps

   Level: advanced

.keywords: MG, smooth, down, pre-smoothing, steps, multigrid

.seealso: PCMGSetNumberSmoothUp()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetNumberSmoothDown(PC pc,PetscInt n)
{ 
  PC_MG          **mg;
  PetscErrorCode ierr;
  PetscInt       i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  mg     = (PC_MG**)pc->data;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  levels = mg[0]->levels;

  for (i=1; i<levels; i++) {  
    /* make sure smoother up and down are different */
    ierr = PCMGGetSmootherUp(pc,i,PETSC_NULL);CHKERRQ(ierr);
    ierr = KSPSetTolerances(mg[i]->smoothd,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,n);CHKERRQ(ierr);
    mg[i]->default_smoothd = n;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetNumberSmoothUp"
/*@
   PCMGSetNumberSmoothUp - Sets the number of post-smoothing steps to use 
   on all levels. Use PCMGGetSmootherUp() to set different numbers of 
   post-smoothing steps on different levels.

   Collective on PC

   Input Parameters:
+  mg - the multigrid context 
-  n - the number of smoothing steps

   Options Database Key:
.  -pc_mg_smoothup <n> - Sets number of post-smoothing steps

   Level: advanced

   Note: this does not set a value on the coarsest grid, since we assume that
    there is no seperate smooth up on the coarsest grid.

.keywords: MG, smooth, up, post-smoothing, steps, multigrid

.seealso: PCMGSetNumberSmoothDown()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetNumberSmoothUp(PC pc,PetscInt n)
{ 
  PC_MG          **mg;
  PetscErrorCode ierr;
  PetscInt       i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  mg     = (PC_MG**)pc->data;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  levels = mg[0]->levels;

  for (i=1; i<levels; i++) {  
    /* make sure smoother up and down are different */
    ierr = PCMGGetSmootherUp(pc,i,PETSC_NULL);CHKERRQ(ierr);
    ierr = KSPSetTolerances(mg[i]->smoothu,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,n);CHKERRQ(ierr);
    mg[i]->default_smoothu = n;
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------*/

/*MC
   PCMG - Use geometric multigrid preconditioning. This preconditioner requires you provide additional
    information about the coarser grid matrices and restriction/interpolation operators.

   Options Database Keys:
+  -pc_mg_levels <nlevels> - number of levels including finest
.  -pc_mg_cycles 1 or 2 - for V or W-cycle
.  -pc_mg_smoothup <n> - number of smoothing steps after interpolation
.  -pc_mg_smoothdown <n> - number of smoothing steps before applying restriction operator
.  -pc_mg_type <additive,multiplicative,full,cascade> - multiplicative is the default
.  -pc_mg_log - log information about time spent on each level of the solver
.  -pc_mg_monitor - print information on the multigrid convergence
.  -pc_mg_galerkin - use Galerkin process to compute coarser operators
-  -pc_mg_dump_matlab - dumps the matrices for each level and the restriction/interpolation matrices
                        to the Socket viewer for reading from Matlab.

   Notes:

   Level: intermediate

   Concepts: multigrid

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCMGType, 
           PCMGSetLevels(), PCMGGetLevels(), PCMGSetType(), PCMGSetCycles(), PCMGSetNumberSmoothDown(),
           PCMGSetNumberSmoothUp(), PCMGGetCoarseSolve(), PCMGSetResidual(), PCMGSetInterpolation(),
           PCMGSetRestriction(), PCMGGetSmoother(), PCMGGetSmootherUp(), PCMGGetSmootherDown(),
           PCMGSetCyclesOnLevel(), PCMGSetRhs(), PCMGSetX(), PCMGSetR()           
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_MG"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_MG(PC pc)
{
  PetscFunctionBegin;
  pc->ops->apply          = PCApply_MG;
  pc->ops->setup          = PCSetUp_MG;
  pc->ops->destroy        = PCDestroy_MG;
  pc->ops->setfromoptions = PCSetFromOptions_MG;
  pc->ops->view           = PCView_MG;

  pc->data                = (void*)0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
