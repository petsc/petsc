/*$Id: damg.c,v 1.35 2001/07/20 21:25:12 bsmith Exp $*/
 
#include "petscda.h"      /*I      "petscda.h"     I*/
#include "petscksp.h"    /*I      "petscksp.h"    I*/
#include "petscmg.h"      /*I      "petscmg.h"    I*/

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

.seealso DMMGDestroy() 

@*/
int DMMGCreate(MPI_Comm comm,int nlevels,void *user,DMMG **dmmg)
{
  int        ierr,i;
  DMMG       *p;
  PetscTruth galerkin;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(0,"-dmmg_nlevels",&nlevels,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(0,"-dmmg_galerkin",&galerkin);CHKERRQ(ierr);

  ierr = PetscMalloc(nlevels*sizeof(DMMG),&p);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    ierr           = PetscNew(struct _p_DMMG,&p[i]);CHKERRQ(ierr);
    ierr           = PetscMemzero(p[i],sizeof(struct _p_DMMG));CHKERRQ(ierr);
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
int DMMGSetUseGalerkinCoarse(DMMG* dmmg)
{
  int  i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as DMMG");

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
int DMMGDestroy(DMMG *dmmg)
{
  int     ierr,i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as DMMG");

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
    if (dmmg[i]->ksp)      {ierr = KSPDestroy(dmmg[i]->ksp);CHKERRQ(ierr);}
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
int DMMGSetDM(DMMG *dmmg,DM dm)
{
  int        ierr,i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as DMMG");

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
int DMMGSetUp(DMMG *dmmg)
{
  int        ierr,i,nlevels = dmmg[0]->nlevels;

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
int DMMGSolve(DMMG *dmmg)
{
  int        i,ierr,nlevels = dmmg[0]->nlevels;
  PetscTruth gridseq,vecmonitor,flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(0,"-dmmg_grid_sequence",&gridseq);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(0,"-dmmg_vecmonitor",&vecmonitor);CHKERRQ(ierr);
  if (gridseq) {
    if (dmmg[0]->initialguess) {
      ierr = (*dmmg[0]->initialguess)(dmmg[0]->snes,dmmg[0]->x,dmmg[0]);CHKERRQ(ierr);
      if (dmmg[0]->ksp) {
        ierr = KSPSetInitialGuessNonzero(dmmg[0]->ksp,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
    for (i=0; i<nlevels-1; i++) {
      ierr = (*dmmg[i]->solve)(dmmg,i);CHKERRQ(ierr);
      if (vecmonitor) {
        ierr = VecView(dmmg[i]->x,PETSC_VIEWER_DRAW_(dmmg[i]->comm));CHKERRQ(ierr);
      }
      ierr = MatInterpolate(dmmg[i+1]->R,dmmg[i]->x,dmmg[i+1]->x);CHKERRQ(ierr);
      if (dmmg[i+1]->ksp) {
        ierr = KSPSetInitialGuessNonzero(dmmg[i+1]->ksp,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
  } else {
    if (dmmg[nlevels-1]->initialguess) {
      ierr = (*dmmg[nlevels-1]->initialguess)(dmmg[nlevels-1]->snes,dmmg[nlevels-1]->x,dmmg[nlevels-1]);CHKERRQ(ierr);
      if (dmmg[nlevels-1]->ksp) {
        ierr = KSPSetInitialGuessNonzero(dmmg[nlevels-1]->ksp,PETSC_TRUE);CHKERRQ(ierr);
      }
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveKSP"
int DMMGSolveKSP(DMMG *dmmg,int level)
{
  int        ierr;

  PetscFunctionBegin;
  ierr = (*dmmg[level]->rhs)(dmmg[level],dmmg[level]->b);CHKERRQ(ierr); 
  if (dmmg[level]->matricesset) {
    ierr = KSPSetOperators(dmmg[level]->ksp,dmmg[level]->J,dmmg[level]->J,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    dmmg[level]->matricesset = PETSC_FALSE;
  }
  ierr = KSPSetRhs(dmmg[level]->ksp,dmmg[level]->b);CHKERRQ(ierr);
  ierr = KSPSetSolution(dmmg[level]->ksp,dmmg[level]->x);CHKERRQ(ierr);
  ierr = KSPSolve(dmmg[level]->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Sets each of the linear solvers to use multigrid 
*/
#undef __FUNCT__  
#define __FUNCT__ "DMMGSetUpLevel"
int DMMGSetUpLevel(DMMG *dmmg,KSP ksp,int nlevels)
{
  int         ierr,i;
  PC          pc;
  PetscTruth  ismg,monitor,ismf,isshell,ismffd;
  KSP        lksp; /* solver internal to the multigrid preconditioner */
  MPI_Comm    *comms,comm;
  PetscViewer ascii;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as DMMG");

  ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_ksp_monitor",&monitor);CHKERRQ(ierr);
  if (monitor) {
    ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
    ierr = PetscViewerASCIISetTab(ascii,1+dmmg[0]->nlevels-nlevels);CHKERRQ(ierr);
    ierr = KSPSetMonitor(ksp,KSPDefaultMonitor,ascii,(int(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
  }

  /* use fgmres on outer iteration by default */
  ierr  = KSPSetType(ksp,KSPFGMRES);CHKERRQ(ierr);
  ierr  = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr  = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr  = PetscMalloc(nlevels*sizeof(MPI_Comm),&comms);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    comms[i] = dmmg[i]->comm;
  }
  ierr  = MGSetLevels(pc,nlevels,comms);CHKERRQ(ierr);
  ierr  = PetscFree(comms);CHKERRQ(ierr); 
  ierr =  MGSetType(pc,MGFULL);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    /* set solvers for each level */
    for (i=0; i<nlevels; i++) {
      ierr = MGGetSmoother(pc,i,&lksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(lksp,dmmg[i]->J,dmmg[i]->B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MGSetX(pc,i,dmmg[i]->x);CHKERRQ(ierr); 
      ierr = MGSetRhs(pc,i,dmmg[i]->b);CHKERRQ(ierr); 
      ierr = MGSetR(pc,i,dmmg[i]->r);CHKERRQ(ierr); 
      ierr = MGSetResidual(pc,i,MGDefaultResidual,dmmg[i]->J);CHKERRQ(ierr);
      if (monitor) {
        ierr = PetscObjectGetComm((PetscObject)lksp,&comm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
        ierr = PetscViewerASCIISetTab(ascii,1+dmmg[0]->nlevels-i);CHKERRQ(ierr);
        ierr = KSPSetMonitor(lksp,KSPDefaultMonitor,ascii,(int(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
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
      ierr = MGSetInterpolate(pc,i,dmmg[i]->R);CHKERRQ(ierr); 
      ierr = MGSetRestriction(pc,i,dmmg[i]->R);CHKERRQ(ierr); 
    }
  }
  PetscFunctionReturn(0);
}

extern int MatSeqAIJPtAP(Mat,Mat,Mat*);

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetKSP"
/*@C
    DMMGSetKSP - Sets the linear solver object that will use the grid hierarchy

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   func - function to compute linear system matrix on each grid level
-   rhs - function to compute right hand side on each level (need only work on the finest grid
          if you do not use grid sequencing

    Level: advanced

    Notes: For linear problems my be called more than once, reevaluates the matrices if it is called more
       than once. Call DMMGSolve() directly several times to solve with the same matrix but different 
       right hand sides.
   
.seealso DMMGCreate(), DMMGDestroy, DMMGSetDM(), DMMGSolve()

@*/
int DMMGSetKSP(DMMG *dmmg,int (*rhs)(DMMG,Vec),int (*func)(DMMG,Mat))
{
  int        ierr,size,i,nlevels = dmmg[0]->nlevels;
  PetscTruth galerkin;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as DMMG");
  galerkin = dmmg[0]->galerkin;  

  if (galerkin) {
    ierr = MPI_Comm_size(dmmg[nlevels-1]->comm,&size);CHKERRQ(ierr);
    ierr = DMGetMatrix(dmmg[nlevels-1]->dm,MATAIJ,&dmmg[nlevels-1]->B);CHKERRQ(ierr);
    ierr = (*func)(dmmg[nlevels-1],dmmg[nlevels-1]->B);CHKERRQ(ierr);
    for (i=nlevels-2; i>-1; i--) {
      ierr = MatSeqAIJPtAP(dmmg[i+1]->B,dmmg[i+1]->R,&dmmg[i]->B);CHKERRQ(ierr);
    }
  }

  if (!dmmg[0]->ksp) {
    /* create solvers for each level */
    for (i=0; i<nlevels; i++) {

      if (!dmmg[i]->B && !galerkin) {
        ierr = MPI_Comm_size(dmmg[i]->comm,&size);CHKERRQ(ierr);
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
      ierr = (*func)(dmmg[i],dmmg[i]->J);CHKERRQ(ierr);
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
int DMMGView(DMMG *dmmg,PetscViewer viewer)
{
  int            ierr,i,nlevels = dmmg[0]->nlevels,flag;
  MPI_Comm       comm;
  PetscTruth     isascii;

  PetscFunctionBegin;
  PetscValidPointer(dmmg,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm,dmmg[0]->comm,&flag);CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) {
    SETERRQ(PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the DMMG and the PetscViewer");
  }

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"DMMG Object with %d levels\n",nlevels);CHKERRQ(ierr);
  }
  for (i=0; i<nlevels; i++) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = DMView(dmmg[i]->dm,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  if (isascii) {
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
  PetscFunctionReturn(0);
}










