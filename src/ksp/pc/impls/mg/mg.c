/*
    Defines the multigrid preconditioner interface.
*/
#include "src/ksp/pc/impls/mg/mgimpl.h"                    /*I "petscmg.h" I*/


/*
       MGMCycle_Private - Given an MG structure created with MGCreate() runs 
                  one multiplicative cycle down through the levels and
                  back up.

    Input Parameter:
.   mg - structure created with  MGCreate().
*/
#undef __FUNCT__  
#define __FUNCT__ "MGMCycle_Private"
PetscErrorCode MGMCycle_Private(MG *mglevels,PetscTruth *converged)
{
  MG             mg = *mglevels,mgc;
  PetscErrorCode ierr;
  PetscInt       cycles = mg->cycles;
  PetscScalar    zero = 0.0;

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
          PetscLogInfo(0,"MGMCycle_Private:Linear solver has converged. Residual norm %g is less than absolute tolerance %g\n",rnorm,mg->abstol);
        } else {
          PetscLogInfo(0,"MGMCycle_Private:Linear solver has converged. Residual norm %g is less than relative tolerance times initial residual norm %g\n",rnorm,mg->ttol);
        }
        PetscFunctionReturn(0);
      }
    }

    mgc = *(mglevels - 1);
    ierr = MatRestrict(mg->restrct,mg->r,mgc->b);CHKERRQ(ierr);
    ierr = VecSet(&zero,mgc->x);CHKERRQ(ierr);
    while (cycles--) {
      ierr = MGMCycle_Private(mglevels-1,converged);CHKERRQ(ierr); 
    }
    ierr = MatInterpolateAdd(mg->interpolate,mgc->x,mg->x,mg->x);CHKERRQ(ierr);
    if (mg->eventsolve) {ierr = PetscLogEventBegin(mg->eventsolve,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSolve(mg->smoothu,mg->b,mg->x);CHKERRQ(ierr); 
    if (mg->eventsolve) {ierr = PetscLogEventEnd(mg->eventsolve,0,0,0,0);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*
       MGCreate_Private - Creates a MG structure for use with the
               multigrid code. Level 0 is the coarsest. (But the 
               finest level is stored first in the array).

*/
#undef __FUNCT__  
#define __FUNCT__ "MGCreate_Private"
static PetscErrorCode MGCreate_Private(MPI_Comm comm,PetscInt levels,PC pc,MPI_Comm *comms,MG **result)
{
  MG             *mg;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscMPIInt    size;
  char           *prefix;
  PC             ipc;

  PetscFunctionBegin;
  ierr = PetscMalloc(levels*sizeof(MG),&mg);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,levels*(sizeof(MG)+sizeof(struct _MG)));CHKERRQ(ierr);

  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);

  for (i=0; i<levels; i++) {
    ierr = PetscNew(struct _MG,&mg[i]);CHKERRQ(ierr);
    mg[i]->level  = i;
    mg[i]->levels = levels;
    mg[i]->cycles = 1;
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
  MG             *mg = (MG*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,n = mg[0]->levels;

  PetscFunctionBegin;
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



EXTERN PetscErrorCode MGACycle_Private(MG*);
EXTERN PetscErrorCode MGFCycle_Private(MG*);
EXTERN PetscErrorCode MGKCycle_Private(MG*);

/*
   PCApply_MG - Runs either an additive, multiplicative, Kaskadic
             or full cycle of multigrid. 

  Note: 
  A simple wrapper which calls MGMCycle(),MGACycle(), or MGFCycle(). 
*/ 
#undef __FUNCT__  
#define __FUNCT__ "PCApply_MG"
static PetscErrorCode PCApply_MG(PC pc,Vec b,Vec x)
{
  MG             *mg = (MG*)pc->data;
  PetscScalar    zero = 0.0;
  PetscErrorCode ierr;
  PetscInt       levels = mg[0]->levels;

  PetscFunctionBegin;
  mg[levels-1]->b = b; 
  mg[levels-1]->x = x;
  if (mg[0]->am == MGMULTIPLICATIVE) {
    ierr = VecSet(&zero,x);CHKERRQ(ierr);
    ierr = MGMCycle_Private(mg+levels-1,PETSC_NULL);CHKERRQ(ierr);
  } 
  else if (mg[0]->am == MGADDITIVE) {
    ierr = MGACycle_Private(mg);CHKERRQ(ierr);
  }
  else if (mg[0]->am == MGKASKADE) {
    ierr = MGKCycle_Private(mg);CHKERRQ(ierr);
  }
  else {
    ierr = MGFCycle_Private(mg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyRichardson_MG"
static PetscErrorCode PCApplyRichardson_MG(PC pc,Vec b,Vec x,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its)
{
  MG             *mg = (MG*)pc->data;
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
    ierr = MGMCycle_Private(mg+levels-1,&converged);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_MG"
static PetscErrorCode PCSetFromOptions_MG(PC pc)
{
  PetscErrorCode ierr;
  PetscInt       indx,m,levels = 1;
  PetscTruth     flg;
  const char     *type[] = {"additive","multiplicative","full","cascade","kascade"};

  PetscFunctionBegin;

  ierr = PetscOptionsHead("Multigrid options");CHKERRQ(ierr);
    if (!pc->data) {
      ierr = PetscOptionsInt("-pc_mg_levels","Number of Levels","MGSetLevels",levels,&levels,&flg);CHKERRQ(ierr);
      ierr = MGSetLevels(pc,levels,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-pc_mg_cycles","1 for V cycle, 2 for W-cycle","MGSetCycles",1,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MGSetCycles(pc,m);CHKERRQ(ierr);
    } 
    ierr = PetscOptionsInt("-pc_mg_smoothup","Number of post-smoothing steps","MGSetNumberSmoothUp",1,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MGSetNumberSmoothUp(pc,m);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-pc_mg_smoothdown","Number of pre-smoothing steps","MGSetNumberSmoothDown",1,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MGSetNumberSmoothDown(pc,m);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEList("-pc_mg_type","Multigrid type","MGSetType",type,5,type[1],&indx,&flg);CHKERRQ(ierr);
    if (flg) {
      MGType mg = (MGType) 0;
      switch (indx) {
      case 0:
        mg = MGADDITIVE;
        break;
      case 1:
        mg = MGMULTIPLICATIVE;
        break;
      case 2:
        mg = MGFULL;
        break;
      case 3:
        mg = MGKASKADE;
        break;
      case 4:
        mg = MGKASKADE;
        break;
      }
      ierr = MGSetType(pc,mg);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-pc_mg_log","Log times for each multigrid level","None",&flg);CHKERRQ(ierr);
    if (flg) {
      MG   *mg = (MG*)pc->data;
      PetscInt i;
      char eventname[128];
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

#undef __FUNCT__  
#define __FUNCT__ "PCView_MG"
static PetscErrorCode PCView_MG(PC pc,PetscViewer viewer)
{
  MG             *mg = (MG*)pc->data;
  PetscErrorCode ierr;
  PetscInt       levels = mg[0]->levels,i;
  const char     *cstring;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (mg[0]->am == MGMULTIPLICATIVE) cstring = "multiplicative";
    else if (mg[0]->am == MGADDITIVE)  cstring = "additive";
    else if (mg[0]->am == MGFULL)      cstring = "full";
    else if (mg[0]->am == MGKASKADE)   cstring = "Kaskade";
    else cstring = "unknown";
    ierr = PetscViewerASCIIPrintf(viewer,"  MG: type is %s, levels=%D cycles=%D, pre-smooths=%D, post-smooths=%D\n",
                      cstring,levels,mg[0]->cycles,mg[0]->default_smoothd,mg[0]->default_smoothu);CHKERRQ(ierr);
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
  MG             *mg = (MG*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,n = mg[0]->levels;
  PC             cpc;
  PetscTruth     preonly,lu,redundant,monitor = PETSC_FALSE,dump;
  PetscViewer    ascii;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    ierr = PetscOptionsHasName(0,"-pc_mg_monitor",&monitor);CHKERRQ(ierr);
     
    for (i=0; i<n; i++) {
      if (mg[i]->smoothd) {
        if (monitor) {
          ierr = PetscObjectGetComm((PetscObject)mg[i]->smoothd,&comm);CHKERRQ(ierr);
          ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
          ierr = PetscViewerASCIISetTab(ascii,n-i);CHKERRQ(ierr);
          ierr = KSPSetMonitor(mg[i]->smoothd,KSPDefaultMonitor,ascii,(PetscErrorCode(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
        }
        ierr = KSPSetFromOptions(mg[i]->smoothd);CHKERRQ(ierr);
      }
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
  }

  for (i=1; i<n; i++) {
    if (mg[i]->smoothd) {
      if (mg[i]->smoothu == mg[i]->smoothd) {
        /* if doing only down then initial guess is zero */
        ierr = KSPSetInitialGuessNonzero(mg[i]->smoothd,PETSC_TRUE);CHKERRQ(ierr);
      }
      if (mg[i]->eventsetup) {ierr = PetscLogEventBegin(mg[i]->eventsetup,0,0,0,0);CHKERRQ(ierr);}
      ierr = KSPSetUp(mg[i]->smoothd);CHKERRQ(ierr);
      if (mg[i]->eventsetup) {ierr = PetscLogEventEnd(mg[i]->eventsetup,0,0,0,0);CHKERRQ(ierr);}
    }
  }
  for (i=1; i<n; i++) {
    if (mg[i]->smoothu && mg[i]->smoothu != mg[i]->smoothd) {
        PC           downpc,uppc;
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
    if (!lu && !redundant) {
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

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "MGSetLevels"
/*@C
   MGSetLevels - Sets the number of levels to use with MG.
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

.seealso: MGSetType(), MGGetLevels()
@*/
PetscErrorCode MGSetLevels(PC pc,PetscInt levels,MPI_Comm *comms)
{
  PetscErrorCode ierr;
  MG             *mg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);

  if (pc->data) {
    SETERRQ(PETSC_ERR_ORDER,"Number levels already set for MG\n\
    make sure that you call MGSetLevels() before KSPSetFromOptions()");
  }
  ierr                     = MGCreate_Private(pc->comm,levels,pc,comms,&mg);CHKERRQ(ierr);
  mg[0]->am                = MGMULTIPLICATIVE;
  pc->data                 = (void*)mg;
  pc->ops->applyrichardson = PCApplyRichardson_MG;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGGetLevels"
/*@
   MGGetLevels - Gets the number of levels to use with MG.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output parameter:
.  levels - the number of levels

   Level: advanced

.keywords: MG, get, levels, multigrid

.seealso: MGSetLevels()
@*/
PetscErrorCode MGGetLevels(PC pc,PetscInt *levels)
{
  MG  *mg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidIntPointer(levels,2);

  mg      = (MG*)pc->data;
  *levels = mg[0]->levels;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGSetType"
/*@
   MGSetType - Determines the form of multigrid to use:
   multiplicative, additive, full, or the Kaskade algorithm.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  form - multigrid form, one of MGMULTIPLICATIVE, MGADDITIVE,
   MGFULL, MGKASKADE

   Options Database Key:
.  -pc_mg_type <form> - Sets <form>, one of multiplicative,
   additive, full, kaskade   

   Level: advanced

.keywords: MG, set, method, multiplicative, additive, full, Kaskade, multigrid

.seealso: MGSetLevels()
@*/
PetscErrorCode MGSetType(PC pc,MGType form)
{
  MG *mg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  mg = (MG*)pc->data;

  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  mg[0]->am = form;
  if (form == MGMULTIPLICATIVE) pc->ops->applyrichardson = PCApplyRichardson_MG;
  else pc->ops->applyrichardson = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGSetCycles"
/*@
   MGSetCycles - Sets the type cycles to use.  Use MGSetCyclesOnLevel() for more 
   complicated cycling.

   Collective on PC

   Input Parameters:
+  mg - the multigrid context 
-  n - the number of cycles

   Options Database Key:
$  -pc_mg_cycles n - 1 denotes a V-cycle; 2 denotes a W-cycle.

   Level: advanced

.keywords: MG, set, cycles, V-cycle, W-cycle, multigrid

.seealso: MGSetCyclesOnLevel()
@*/
PetscErrorCode MGSetCycles(PC pc,PetscInt n)
{ 
  MG       *mg;
  PetscInt i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  mg     = (MG*)pc->data;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  levels = mg[0]->levels;

  for (i=0; i<levels; i++) {  
    mg[i]->cycles  = n; 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGCheck"
/*@
   MGCheck - Checks that all components of the MG structure have 
   been set.

   Collective on PC

   Input Parameters:
.  mg - the MG structure

   Level: advanced

.keywords: MG, check, set, multigrid
@*/
PetscErrorCode MGCheck(PC pc)
{
  MG       *mg;
  PetscInt i,n,count = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  mg = (MG*)pc->data;

  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");

  n = mg[0]->levels;

  for (i=1; i<n; i++) {
    if (!mg[i]->restrct) {
      (*PetscErrorPrintf)("No restrict set level %D \n",n-i); count++;
    }    
    if (!mg[i]->interpolate) {
      (*PetscErrorPrintf)("No interpolate set level %D \n",n-i); count++;
    }
    if (!mg[i]->residual) {
      (*PetscErrorPrintf)("No residual set level %D \n",n-i); count++;
    }
    if (!mg[i]->smoothu) {
      (*PetscErrorPrintf)("No smoothup set level %D \n",n-i); count++;
    }  
    if (!mg[i]->smoothd) {
      (*PetscErrorPrintf)("No smoothdown set level %D \n",n-i); count++;
    }
    if (!mg[i]->r) {
      (*PetscErrorPrintf)("No r set level %D \n",n-i); count++;
    } 
    if (!mg[i-1]->x) {
      (*PetscErrorPrintf)("No x set level %D \n",n-i); count++;
    }
    if (!mg[i-1]->b) {
      (*PetscErrorPrintf)("No b set level %D \n",n-i); count++;
    }
  }
  PetscFunctionReturn(count);
}


#undef __FUNCT__  
#define __FUNCT__ "MGSetNumberSmoothDown"
/*@
   MGSetNumberSmoothDown - Sets the number of pre-smoothing steps to
   use on all levels. Use MGGetSmootherDown() to set different 
   pre-smoothing steps on different levels.

   Collective on PC

   Input Parameters:
+  mg - the multigrid context 
-  n - the number of smoothing steps

   Options Database Key:
.  -pc_mg_smoothdown <n> - Sets number of pre-smoothing steps

   Level: advanced

.keywords: MG, smooth, down, pre-smoothing, steps, multigrid

.seealso: MGSetNumberSmoothUp()
@*/
PetscErrorCode MGSetNumberSmoothDown(PC pc,PetscInt n)
{ 
  MG             *mg;
  PetscErrorCode ierr;
  PetscInt       i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  mg     = (MG*)pc->data;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  levels = mg[0]->levels;

  for (i=0; i<levels; i++) {  
    /* make sure smoother up and down are different */
    ierr = MGGetSmootherUp(pc,i,PETSC_NULL);CHKERRQ(ierr);
    ierr = KSPSetTolerances(mg[i]->smoothd,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,n);CHKERRQ(ierr);
    mg[i]->default_smoothd = n;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MGSetNumberSmoothUp"
/*@
   MGSetNumberSmoothUp - Sets the number of post-smoothing steps to use 
   on all levels. Use MGGetSmootherUp() to set different numbers of 
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

.seealso: MGSetNumberSmoothDown()
@*/
PetscErrorCode  MGSetNumberSmoothUp(PC pc,PetscInt n)
{ 
  MG             *mg;
  PetscErrorCode ierr;
  PetscInt       i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  mg     = (MG*)pc->data;
  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  levels = mg[0]->levels;

  for (i=1; i<levels; i++) {  
    /* make sure smoother up and down are different */
    ierr = MGGetSmootherUp(pc,i,PETSC_NULL);CHKERRQ(ierr);
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
-  -pc_mg_dump_matlab - dumps the matrices for each level and the restriction/interpolation matrices
                        to the Socket viewer for reading from Matlab.

   Notes:

   Level: intermediate

   Concepts: multigrid

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCMGType, 
           MGSetLevels(), MGGetLevels(), MGSetType(), MPSetCycles(), MGSetNumberSmoothDown(),
           MGSetNumberSmoothUp(), MGGetCoarseSolve(), MGSetResidual(), MGSetInterpolation(),
           MGSetRestriction(), MGGetSmoother(), MGGetSmootherUp(), MGGetSmootherDown(),
           MGSetCyclesOnLevel(), MGSetRhs(), MGSetX(), MGSetR()           
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_MG"
PetscErrorCode PCCreate_MG(PC pc)
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
