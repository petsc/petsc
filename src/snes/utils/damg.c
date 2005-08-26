#define PETSCSNES_DLL
 
#include "petscda.h"            /*I "petscda.h"   I*/
#include "petscksp.h"           /*I "petscksp.h"  I*/
#include "petscmg.h"            /*I "petscmg.h"   I*/
#include "petscdmmg.h"          /*I "petscdmmg.h" I*/
#include "src/ksp/pc/pcimpl.h"  /*I "petscpc.h"   I*/

/*
   Code for almost fully managing multigrid/multi-level linear solvers for DA grids
*/

#undef __FUNCT__  
#define __FUNCT__ "DMMGCreate"
/*@C
    DMMGCreate - Creates a DA based multigrid solver object. This allows one to 
      easily implement MG methods on regular grids.

    Collective on MPI_Comm

    Input Parameter:
+   comm - the processors that will share the grids and solution process
.   nlevels - number of multigrid levels 
-   user - an optional user context

    Output Parameters:
.    - the context

    Notes:
      To provide a different user context for each level call DMMGSetUser() after calling
      this routine

    Level: advanced

.seealso DMMGDestroy(), DMMGSetUser(), DMMGGetUser()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGCreate(MPI_Comm comm,PetscInt nlevels,void *user,DMMG **dmmg)
{
  PetscErrorCode ierr;
  PetscInt       i;
  DMMG           *p;
  PetscTruth     galerkin;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(0,"-dmmg_nlevels",&nlevels,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(0,"-dmmg_galerkin",&galerkin);CHKERRQ(ierr);

  ierr = PetscMalloc(nlevels*sizeof(DMMG),&p);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    ierr           = PetscNew(struct _n_DMMG,&p[i]);CHKERRQ(ierr);
    p[i]->nlevels  = nlevels - i;
    p[i]->comm     = comm;
    p[i]->user     = user;
    p[i]->galerkin = galerkin;
  }
  *dmmg = p;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetUseGalerkinCoarse"
/*@C
    DMMGSetUseGalerkinCoarse - Courses the DMMG to use R*A_f*R^T to form
       the coarser matrices from finest 

    Collective on DMMG

    Input Parameter:
.    - the context

    Options Database Keys:
.    -dmmg_galerkin

    Level: advanced

.seealso DMMGCreate()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGSetUseGalerkinCoarse(DMMG* dmmg)
{
  PetscInt  i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_ERR_ARG_NULL,"Passing null as DMMG");

  for (i=0; i<nlevels; i++) {
    dmmg[i]->galerkin = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGDestroy"
/*@C
    DMMGDestroy - Destroys a DA based multigrid solver object. 

    Collective on DMMG

    Input Parameter:
.    - the context

    Level: advanced

.seealso DMMGCreate()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGDestroy(DMMG *dmmg)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_ERR_ARG_NULL,"Passing null as DMMG");

  for (i=1; i<nlevels; i++) {
    if (dmmg[i]->R) {ierr = MatDestroy(dmmg[i]->R);CHKERRQ(ierr);}
  }
  for (i=0; i<nlevels; i++) {
    if (dmmg[i]->dm)      {ierr = DMDestroy(dmmg[i]->dm);CHKERRQ(ierr);}
    if (dmmg[i]->x)       {ierr = VecDestroy(dmmg[i]->x);CHKERRQ(ierr);}
    if (dmmg[i]->b)       {ierr = VecDestroy(dmmg[i]->b);CHKERRQ(ierr);}
    if (dmmg[i]->r)       {ierr = VecDestroy(dmmg[i]->r);CHKERRQ(ierr);}
    if (dmmg[i]->work1)   {ierr = VecDestroy(dmmg[i]->work1);CHKERRQ(ierr);}
    if (dmmg[i]->w)       {ierr = VecDestroy(dmmg[i]->w);CHKERRQ(ierr);}
    if (dmmg[i]->work2)   {ierr = VecDestroy(dmmg[i]->work2);CHKERRQ(ierr);}
    if (dmmg[i]->lwork1)  {ierr = VecDestroy(dmmg[i]->lwork1);CHKERRQ(ierr);}
    if (dmmg[i]->B && dmmg[i]->B != dmmg[i]->J) {ierr = MatDestroy(dmmg[i]->B);CHKERRQ(ierr);}
    if (dmmg[i]->J)         {ierr = MatDestroy(dmmg[i]->J);CHKERRQ(ierr);}
    if (dmmg[i]->Rscale)    {ierr = VecDestroy(dmmg[i]->Rscale);CHKERRQ(ierr);}
    if (dmmg[i]->fdcoloring){ierr = MatFDColoringDestroy(dmmg[i]->fdcoloring);CHKERRQ(ierr);}
    if (dmmg[i]->ksp && !dmmg[i]->snes) {ierr = KSPDestroy(dmmg[i]->ksp);CHKERRQ(ierr);}
    if (dmmg[i]->snes)      {ierr = PetscObjectDestroy((PetscObject)dmmg[i]->snes);CHKERRQ(ierr);} 
    if (dmmg[i]->inject)    {ierr = VecScatterDestroy(dmmg[i]->inject);CHKERRQ(ierr);} 
    ierr = PetscFree(dmmg[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dmmg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetDM"
/*@C
    DMMGSetDM - Sets the coarse grid information for the grids

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
-   dm - the DA or VecPack object

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGSetDM(DMMG *dmmg,DM dm)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_ERR_ARG_NULL,"Passing null as DMMG");

  /* Create DA data structure for all the levels */
  dmmg[0]->dm = dm;
  ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
  for (i=1; i<nlevels; i++) {
    ierr = DMRefine(dmmg[i-1]->dm,dmmg[i]->comm,&dmmg[i]->dm);CHKERRQ(ierr);
  }
  ierr = DMMGSetUp(dmmg);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetUp"
/*@C
    DMMGSetUp - Prepares the DMMG to solve a system

    Collective on DMMG

    Input Parameter:
.   dmmg - the context

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy(), DMMG, DMMGSetSNES(), DMMGSetKSP(), DMMGSolve()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGSetUp(DMMG *dmmg)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;

  /* Create work vectors and matrix for each level */
  for (i=0; i<nlevels; i++) {
    ierr = DMCreateGlobalVector(dmmg[i]->dm,&dmmg[i]->x);CHKERRQ(ierr);
    ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->b);CHKERRQ(ierr);
    ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->r);CHKERRQ(ierr);
  }

  /* Create interpolation/restriction between levels */
  for (i=1; i<nlevels; i++) {
    ierr = DMGetInterpolation(dmmg[i-1]->dm,dmmg[i]->dm,&dmmg[i]->R,PETSC_NULL);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSolve"
/*@C
    DMMGSolve - Actually solves the (non)linear system defined with the DMMG

    Collective on DMMG

    Input Parameter:
.   dmmg - the context

    Level: advanced

    Options Database:
+   -dmmg_grid_sequence - use grid sequencing to get the initial solution for each level from the previous
-   -dmmg_vecmonitor - display the solution at each iteration

     Notes: For linear (KSP) problems may be called more than once, uses the same 
    matrices but recomputes the right hand side for each new solve. Call DMMGSetKSP()
    to generate new matrices.
 
.seealso DMMGCreate(), DMMGDestroy(), DMMG, DMMGSetSNES(), DMMGSetKSP(), DMMGSetUp()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGSolve(DMMG *dmmg)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;
  PetscTruth     gridseq,vecmonitor,flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(0,"-dmmg_grid_sequence",&gridseq);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(0,"-dmmg_vecmonitor",&vecmonitor);CHKERRQ(ierr);
  if (gridseq) {
    if (dmmg[0]->initialguess) {
      ierr = (*dmmg[0]->initialguess)(dmmg[0],dmmg[0]->x);CHKERRQ(ierr);
      if (dmmg[0]->ksp && !dmmg[0]->snes) {
        ierr = KSPSetInitialGuessNonzero(dmmg[0]->ksp,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
    for (i=0; i<nlevels-1; i++) {
      ierr = (*dmmg[i]->solve)(dmmg,i);CHKERRQ(ierr);
      if (vecmonitor) {
        ierr = VecView(dmmg[i]->x,PETSC_VIEWER_DRAW_(dmmg[i]->comm));CHKERRQ(ierr);
      }
      ierr = MatInterpolate(dmmg[i+1]->R,dmmg[i]->x,dmmg[i+1]->x);CHKERRQ(ierr);
      if (dmmg[i+1]->ksp && !dmmg[i+1]->ksp) {
        ierr = KSPSetInitialGuessNonzero(dmmg[i+1]->ksp,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
  } else {
    if (dmmg[nlevels-1]->initialguess) {
      ierr = (*dmmg[nlevels-1]->initialguess)(dmmg[nlevels-1],dmmg[nlevels-1]->x);CHKERRQ(ierr);
    }
  }
  ierr = (*DMMGGetFine(dmmg)->solve)(dmmg,nlevels-1);CHKERRQ(ierr);
  if (vecmonitor) {
     ierr = VecView(dmmg[nlevels-1]->x,PETSC_VIEWER_DRAW_(dmmg[nlevels-1]->comm));CHKERRQ(ierr);
  }

  ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_view",&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = DMMGView(dmmg,PETSC_VIEWER_STDOUT_(dmmg[0]->comm));CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_view_binary",&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = DMMGView(dmmg,PETSC_VIEWER_BINARY_(dmmg[0]->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveKSP"
PetscErrorCode PETSCSNES_DLLEXPORT DMMGSolveKSP(DMMG *dmmg,PetscInt level)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dmmg[level]->rhs) {
    ierr = (*dmmg[level]->rhs)(dmmg[level],dmmg[level]->b);CHKERRQ(ierr); 
  }
  if (dmmg[level]->matricesset) {
    ierr = KSPSetOperators(dmmg[level]->ksp,dmmg[level]->J,dmmg[level]->B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    dmmg[level]->matricesset = PETSC_FALSE;
  }
  ierr = KSPSolve(dmmg[level]->ksp,dmmg[level]->b,dmmg[level]->x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Sets each of the linear solvers to use multigrid 
*/
#undef __FUNCT__  
#define __FUNCT__ "DMMGSetUpLevel"
PetscErrorCode PETSCSNES_DLLEXPORT DMMGSetUpLevel(DMMG *dmmg,KSP ksp,PetscInt nlevels)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PC             pc;
  PetscTruth     ismg,monitor,ismf,isshell,ismffd;
  KSP            lksp; /* solver internal to the multigrid preconditioner */
  MPI_Comm       *comms,comm;
  PetscViewer    ascii;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_ERR_ARG_NULL,"Passing null as DMMG");

  ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_ksp_monitor",&monitor);CHKERRQ(ierr);
  if (monitor) {
    ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
    ierr = PetscViewerASCIISetTab(ascii,1+dmmg[0]->nlevels-nlevels);CHKERRQ(ierr);
    ierr = KSPSetMonitor(ksp,KSPDefaultMonitor,ascii,(PetscErrorCode(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
  }

  /* use fgmres on outer iteration by default */
  ierr  = KSPSetType(ksp,KSPFGMRES);CHKERRQ(ierr);
  ierr  = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr  = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr  = PetscMalloc(nlevels*sizeof(MPI_Comm),&comms);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    comms[i] = dmmg[i]->comm;
  }
  ierr  = PCMGSetLevels(pc,nlevels,comms);CHKERRQ(ierr);
  ierr  = PetscFree(comms);CHKERRQ(ierr); 
  ierr =  PCMGSetType(pc,PC_MG_FULL);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    /* set solvers for each level */
    for (i=0; i<nlevels; i++) {
      ierr = PCMGGetSmoother(pc,i,&lksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(lksp,dmmg[i]->J,dmmg[i]->B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      if (i < nlevels-1) { /* don't set for finest level, they are set in PCApply_MG()*/
	ierr = PCMGSetX(pc,i,dmmg[i]->x);CHKERRQ(ierr); 
	ierr = PCMGSetRhs(pc,i,dmmg[i]->b);CHKERRQ(ierr); 
      }
      if (i > 0) {
        ierr = PCMGSetR(pc,i,dmmg[i]->r);CHKERRQ(ierr); 
        ierr = PCMGSetResidual(pc,i,PCMGDefaultResidual,dmmg[i]->J);CHKERRQ(ierr);
      }
      if (monitor) {
        ierr = PetscObjectGetComm((PetscObject)lksp,&comm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
        ierr = PetscViewerASCIISetTab(ascii,1+dmmg[0]->nlevels-i);CHKERRQ(ierr);
        ierr = KSPSetMonitor(lksp,KSPDefaultMonitor,ascii,(PetscErrorCode(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
      }
      /* If using a matrix free multiply and did not provide an explicit matrix to build
         the preconditioner then must use no preconditioner 
      */
      ierr = PetscTypeCompare((PetscObject)dmmg[i]->B,MATSHELL,&isshell);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)dmmg[i]->B,MATDAAD,&ismf);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)dmmg[i]->B,MATMFFD,&ismffd);CHKERRQ(ierr);
      if (isshell || ismf || ismffd) {
        PC  lpc;
        ierr = KSPGetPC(lksp,&lpc);CHKERRQ(ierr);
        ierr = PCSetType(lpc,PCNONE);CHKERRQ(ierr);
      }
    }

    /* Set interpolation/restriction between levels */
    for (i=1; i<nlevels; i++) {
      ierr = PCMGSetInterpolate(pc,i,dmmg[i]->R);CHKERRQ(ierr); 
      ierr = PCMGSetRestriction(pc,i,dmmg[i]->R);CHKERRQ(ierr); 
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetKSP"
/*@C
    DMMGSetKSP - Sets the linear solver object that will use the grid hierarchy

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   func - function to compute linear system matrix on each grid level
-   rhs - function to compute right hand side on each level (need only work on the finest grid
          if you do not use grid sequencing)

    Level: advanced

    Notes: For linear problems my be called more than once, reevaluates the matrices if it is called more
       than once. Call DMMGSolve() directly several times to solve with the same matrix but different 
       right hand sides.
   
.seealso DMMGCreate(), DMMGDestroy, DMMGSetDM(), DMMGSolve()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGSetKSP(DMMG *dmmg,PetscErrorCode (*rhs)(DMMG,Vec),PetscErrorCode (*func)(DMMG,Mat,Mat))
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;
  PetscTruth     galerkin;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_ERR_ARG_NULL,"Passing null as DMMG");
  galerkin = dmmg[0]->galerkin;  

  if (galerkin) {
    ierr = DMGetMatrix(dmmg[nlevels-1]->dm,MATAIJ,&dmmg[nlevels-1]->B);CHKERRQ(ierr);
    if (!dmmg[nlevels-1]->J) {
      dmmg[nlevels-1]->J = dmmg[nlevels-1]->B;
    }
    ierr = (*func)(dmmg[nlevels-1],dmmg[nlevels-1]->J,dmmg[nlevels-1]->B);CHKERRQ(ierr);
    for (i=nlevels-2; i>-1; i--) {
      ierr = MatPtAP(dmmg[i+1]->B,dmmg[i+1]->R,MAT_INITIAL_MATRIX,1.0,&dmmg[i]->B);CHKERRQ(ierr);
      if (!dmmg[i]->J) {
        dmmg[i]->J = dmmg[i]->B;
      }
    }
  }

  if (!dmmg[0]->ksp) {
    /* create solvers for each level if they don't already exist*/
    for (i=0; i<nlevels; i++) {

      if (!dmmg[i]->B && !galerkin) {
        ierr = DMGetMatrix(dmmg[i]->dm,MATAIJ,&dmmg[i]->B);CHKERRQ(ierr);
      } 
      if (!dmmg[i]->J) {
        dmmg[i]->J = dmmg[i]->B;
      }

      ierr = KSPCreate(dmmg[i]->comm,&dmmg[i]->ksp);CHKERRQ(ierr);
      ierr = DMMGSetUpLevel(dmmg,dmmg[i]->ksp,i+1);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(dmmg[i]->ksp);CHKERRQ(ierr);
      dmmg[i]->solve = DMMGSolveKSP;
      dmmg[i]->rhs   = rhs;
    }
  }

  /* evalute matrix on each level */
  for (i=0; i<nlevels; i++) {
    if (!galerkin) {
      ierr = (*func)(dmmg[i],dmmg[i]->J,dmmg[i]->B);CHKERRQ(ierr);
    }
    dmmg[i]->matricesset = PETSC_TRUE;
  }

  for (i=0; i<nlevels-1; i++) {
    ierr = KSPSetOptionsPrefix(dmmg[i]->ksp,"dmmg_");CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGView"
/*@C
    DMMGView - prints information on a DA based multi-level preconditioner

    Collective on DMMG and PetscViewer

    Input Parameter:
+   dmmg - the context
-   viewer - the viewer

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGView(DMMG *dmmg,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;
  PetscMPIInt    flag;
  MPI_Comm       comm;
  PetscTruth     iascii,isbinary;

  PetscFunctionBegin;
  PetscValidPointer(dmmg,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm,dmmg[0]->comm,&flag);CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) {
    SETERRQ(PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the DMMG and the PetscViewer");
  }

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (isbinary) {
    for (i=0; i<nlevels; i++) {
      ierr = MatView(dmmg[i]->J,viewer);CHKERRQ(ierr);
    }
    for (i=1; i<nlevels; i++) {
      ierr = MatView(dmmg[i]->R,viewer);CHKERRQ(ierr);
    }
  } else {
    if (iascii) {
      ierr = PetscViewerASCIIPrintf(viewer,"DMMG Object with %D levels\n",nlevels);CHKERRQ(ierr);
    }
    for (i=0; i<nlevels; i++) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = DMView(dmmg[i]->dm,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (iascii) {
      ierr = PetscViewerASCIIPrintf(viewer,"%s Object on finest level\n",dmmg[nlevels-1]->ksp ? "KSP" : "SNES");CHKERRQ(ierr);
      if (dmmg[nlevels-1]->galerkin) {
	ierr = PetscViewerASCIIPrintf(viewer,"Using Galerkin R^T*A*R process to compute coarser matrices");CHKERRQ(ierr);
      }
    }
    if (dmmg[nlevels-1]->ksp) {
      ierr = KSPView(dmmg[nlevels-1]->ksp,viewer);CHKERRQ(ierr);
    } else {
      /* use of PetscObjectView() means we do not have to link with libpetscsnes if SNES is not being used */
      ierr = PetscObjectView((PetscObject)dmmg[nlevels-1]->snes,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetNullSpace"
/*@C
    DMMGSetNullSpace - Indicates the null space in the linear operator (this is needed by the linear solver)

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   has_cnst - is the constant vector in the null space
.   n - number of null vectors (excluding the possible constant vector)
-   func - a function that fills an array of vectors with the null vectors (must be orthonormal), may be PETSC_NULL

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetDM(), DMMGSolve(), MatNullSpaceCreate(), KSPSetNullSpace()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGSetNullSpace(DMMG *dmmg,PetscTruth has_cnst,PetscInt n,PetscErrorCode (*func)(DMMG,Vec[]))
{
  PetscErrorCode ierr;
  PetscInt       i,j,nlevels = dmmg[0]->nlevels;
  Vec            *nulls = 0;
  MatNullSpace   nullsp;
  KSP            iksp;
  PC             pc,ipc;
  PetscTruth     ismg,isred;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_ERR_ARG_NULL,"Passing null as DMMG");
  if (!dmmg[0]->ksp) SETERRQ(PETSC_ERR_ORDER,"Must call AFTER DMMGSetKSP() or DMMGSetSNES()");
  if ((n && !func) || (!n && func)) SETERRQ(PETSC_ERR_ARG_INCOMP,"Both n and func() must be set together");
  if (n < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Cannot have negative number of vectors in null space n = %D",n)

  for (i=0; i<nlevels; i++) {
    if (n) {
      ierr = VecDuplicateVecs(dmmg[i]->b,n,&nulls);CHKERRQ(ierr);
      ierr = (*func)(dmmg[i],nulls);CHKERRQ(ierr);
    }
    ierr = MatNullSpaceCreate(dmmg[i]->comm,has_cnst,n,nulls,&nullsp);CHKERRQ(ierr);
    ierr = KSPSetNullSpace(dmmg[i]->ksp,nullsp);CHKERRQ(ierr);
    for (j=i; j<nlevels; j++) {
      ierr = KSPGetPC(dmmg[j]->ksp,&pc);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
      if (ismg) {
        ierr = PCMGGetSmoother(pc,i,&iksp);CHKERRQ(ierr);
        ierr = KSPSetNullSpace(iksp, nullsp);CHKERRQ(ierr);
      }
    }
    ierr = MatNullSpaceDestroy(nullsp);CHKERRQ(ierr);
    if (n) {
      ierr = PetscFree(nulls);CHKERRQ(ierr);
    }
  }
  /* make all the coarse grid solvers have LU shift since they are singular */
  for (i=0; i<nlevels; i++) {
    ierr = KSPGetPC(dmmg[i]->ksp,&pc);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
    if (ismg) {
      ierr = PCMGGetSmoother(pc,0,&iksp);CHKERRQ(ierr);
      ierr = KSPGetPC(iksp,&ipc);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)ipc,PCREDUNDANT,&isred);CHKERRQ(ierr);
      if (isred) {
        ierr = PCRedundantGetPC(ipc,&ipc);CHKERRQ(ierr);
      }
      ierr = PCFactorSetShiftPd(ipc,PETSC_TRUE);CHKERRQ(ierr); 
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGInitialGuessCurrent"
/*@C
    DMMGInitialGuessCurrent - Use with DMMGSetInitialGuess() to use the current value in the 
       solution vector (obtainable with DMMGGetx() as the initial guess)

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
-   vec - dummy argument

    Level: intermediate

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNES(), DMMGSetInitialGuess()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGInitialGuessCurrent(DMMG dmmg,Vec vec)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetInitialGuess"
/*@C
    DMMGSetInitialGuess - Sets the function that computes an initial guess.

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
-   guess - the function

    Notes: For nonlinear problems, if this is not set, then the current value in the 
             solution vector (obtained with DMMGGetX()) is used. Thus is if you doing 'time
             stepping' it will use your current solution as the guess for the next timestep.
           If grid sequencing is used (via -dmmg_grid_sequence) then the "guess" function
             is used only on the coarsest grid.
           For linear problems, if this is not set, then 0 is used as an initial guess.
             If you would like the linear solver to also (like the nonlinear solver) use
             the current solution vector as the initial guess then use DMMGInitialGuessCurrent()
             as the function you pass in

    Level: intermediate


.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNES(), DMMGInitialGuessCurrent()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGSetInitialGuess(DMMG *dmmg,PetscErrorCode (*guess)(DMMG,Vec))
{
  PetscInt i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  for (i=0; i<nlevels; i++) {
    dmmg[i]->initialguess = guess;
  }
  PetscFunctionReturn(0);
}







