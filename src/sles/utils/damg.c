/*$Id: damg.c,v 1.22 2000/09/28 21:14:03 bsmith Exp bsmith $*/
 
#include "petscda.h"      /*I      "petscda.h"     I*/
#include "petscsles.h"    /*I      "petscsles.h"    I*/
#include "petscmg.h"      /*I      "petscmg.h"    I*/

/*
   Code for almost fully managing multigrid/multi-level linear solvers for DA grids
*/

#undef __FUNC__  
#define __FUNC__ "DMMGCreate"
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
      To provide a different user context for each level, or to change the 
    ratio in the grid spacing simply change the DMMG structure after calling this routine

    Level: advanced

.seealso DMMGDestroy() 

@*/
int DMMGCreate(MPI_Comm comm,int nlevels,void *user,DMMG **dmmg)
{
  int        ierr,i,array[3],narray = 3,ratiox = 2, ratioy = 2, ratioz = 2;
  DMMG       *p;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(0,"-dmmg_nlevels",&nlevels,0);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(0,"-dmmg_ratio",array,&narray,&flg);CHKERRQ(ierr);
  if (flg) {
    if (narray > 0) ratiox = array[0];
    if (narray > 1) ratioy = array[1]; else ratioy = ratiox;
    if (narray > 2) ratioz = array[2]; else ratioz = ratioy;
  }

ierr = PetscMalloc(nlevels*sizeof(DMMG),&(  p ));CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    p[i]             = (DMMG)PetscMalloc(sizeof(struct _p_DMMG));CHKERRQ(ierr);
    ierr             = PetscMemzero(p[i],sizeof(struct _p_DMMG));CHKERRQ(ierr);
    p[i]->nlevels    = nlevels - i;
    p[i]->ratiox     = ratiox;
    p[i]->ratioy     = ratioy;
    p[i]->ratioz     = ratioz;
    p[i]->comm       = comm;
    p[i]->user       = user;
    p[i]->matrixfree = PETSC_FALSE;
  }
  *dmmg = p;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DMMGSetUseMatrixFree"
/*@C
    DMMGSetUseMatrixFree - Use matrix-free version of operator

    Collective on DMMG

    Input Parameter:
.    - the context

    Level: advanced

.seealso DMMGCreate()

@*/
int DMMGSetUseMatrixFree(DMMG *dmmg)
{
  int i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as DMMG");
  for (i=0; i<nlevels; i++) {
    dmmg[i]->matrixfree = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DMMGDestroy"
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
    if (dmmg[i]->R) {ierr = MatDestroy(dmmg[i]->R);CHKERRA(ierr);}
  }
  for (i=0; i<nlevels; i++) {
    if (dmmg[i]->dm) {ierr = DMDestroy(dmmg[i]->dm);CHKERRQ(ierr);}
    if (dmmg[i]->x)  {ierr = VecDestroy(dmmg[i]->x);CHKERRQ(ierr);}
    if (dmmg[i]->b)  {ierr = VecDestroy(dmmg[i]->b);CHKERRQ(ierr);}
    if (dmmg[i]->r)  {ierr = VecDestroy(dmmg[i]->r);CHKERRQ(ierr);}
    if (dmmg[i]->work1)  {ierr = VecDestroy(dmmg[i]->work1);CHKERRQ(ierr);}
    if (dmmg[i]->work2)  {ierr = VecDestroy(dmmg[i]->work2);CHKERRQ(ierr);}
    if (dmmg[i]->B && dmmg[i]->B != dmmg[i]->J) {ierr = MatDestroy(dmmg[i]->B);CHKERRQ(ierr);}
    if (dmmg[i]->J)  {ierr = MatDestroy(dmmg[i]->J);CHKERRQ(ierr);}
    if (dmmg[i]->Rscale)  {ierr = VecDestroy(dmmg[i]->Rscale);CHKERRQ(ierr);}
    if (dmmg[i]->fdcoloring)  {ierr = MatFDColoringDestroy(dmmg[i]->fdcoloring);CHKERRQ(ierr);}
    if (dmmg[i]->sles)  {ierr = SLESDestroy(dmmg[i]->sles);CHKERRQ(ierr);}
    if (dmmg[i]->snes)  {ierr = PetscObjectDestroy((PetscObject)dmmg[i]->snes);CHKERRQ(ierr);} 
    ierr = PetscFree(dmmg[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dmmg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DMMGSetDM"
/*@C
    DMMGSetDM - Sets the coarse grid information for the grids

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
-   dm - the DA or VecPack object

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy

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

#undef __FUNC__  
#define __FUNC__ "DMMGSetDA"
/*@C
    DMMGSetDA - Sets the DA information for the grids

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   dim - 1, 2, or 3
.   pt - DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, DA_XYPERIODIC, DA_XYZPERIODIC, 
         DA_XZPERIODIC, or DA_YZPERIODIC
.   st - DA_STENCIL_STAR or DA_STENCIL_BOX
.   M - grid points in x
.   N - grid points in y
.   P - grid points in z
.   sw - stencil width, often 1
+   dof - number of degrees of freedom per node

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy

@*/
int DMMGSetDA(DMMG *dmmg,int dim,DAPeriodicType pt,DAStencilType st,int M,int N,int P,int dof,int sw)
{
  int        ierr,i,nlevels = dmmg[0]->nlevels,m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE;
  int        array[3],narray = 3;
  PetscTruth flg,split = PETSC_FALSE;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as DMMG");

  ierr = PetscOptionsGetIntArray(PETSC_NULL,"-dmmg_mnp",array,&narray,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscOptionsGetIntArray(PETSC_NULL,"-dmmg_mn",array,&narray,&flg);CHKERRQ(ierr);
  }
  if (flg) {
    if (narray > 0) M = array[0];
    if (narray > 1) N = array[1]; else N = M;
    if (narray > 2) P = array[2]; else P = N;
  }
  ierr = PetscOptionsGetLogical(PETSC_NULL,"-dmmg_split",&split,PETSC_NULL);CHKERRQ(ierr);

  /* Create DA data structure for all the levels */
  for (i=0; i<nlevels; i++) {
    if (dim == 3) {
      ierr = DACreate3d(dmmg[i]->comm,pt,st,M,N,P,m,n,p,dof,sw,0,0,0,(DA*)&dmmg[i]->dm);CHKERRQ(ierr);
    } else if (dim == 2) {
      if (split) {
        ierr = DASplitComm2d(dmmg[i]->comm,M,N,sw,&dmmg[i]->comm);CHKERRQ(ierr);
      }
      ierr = DACreate2d(dmmg[i]->comm,pt,st,M,N,m,n,dof,sw,0,0,(DA*)&dmmg[i]->dm);CHKERRQ(ierr);
    } else if (dim == 1) {
      ierr = DACreate1d(dmmg[i]->comm,pt,M,dof,sw,0,(DA*)&dmmg[i]->dm);CHKERRQ(ierr);
    } else {
      SETERRQ1(1,"Cannot handle dimension %d",dim);
    }

    if (DAXPeriodic(pt)) {
      M = dmmg[i]->ratiox*M;
    } else {
      M = dmmg[i]->ratiox*(M-1) + 1;
    }
    if (DAYPeriodic(pt)) {
      N = dmmg[i]->ratioy*N + 1;
    } else {
      N = dmmg[i]->ratioy*(N-1) + 1;
    }
    if (DAZPeriodic(pt)) {
      P = dmmg[i]->ratioz*P + 1;
    } else {
      P = dmmg[i]->ratioz*(P-1) + 1;
    }
  }
  ierr = DMMGSetUp(dmmg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DMMGSetUp"
int DMMGSetUp(DMMG *dmmg)
{
  int        ierr,i,nlevels = dmmg[0]->nlevels;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_snes_mf",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMMGSetUseMatrixFree(dmmg);CHKERRQ(ierr);
  }

  /* Create work vectors and matrix for each level */
  for (i=0; i<nlevels; i++) {
    ierr = DMCreateGlobalVector(dmmg[i]->dm,&dmmg[i]->x);CHKERRA(ierr);
    ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->b);CHKERRA(ierr);
    ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->r);CHKERRA(ierr);
    if (!dmmg[i]->matrixfree) {
      ierr = DMGetColoring(dmmg[i]->dm,PETSC_NULL,&dmmg[i]->J);CHKERRQ(ierr);
    } 
    dmmg[i]->B = dmmg[i]->J;
  }

  /* Create interpolation/restriction between levels */
  for (i=1; i<nlevels; i++) {
    ierr = DMGetInterpolation(dmmg[i-1]->dm,dmmg[i]->dm,&dmmg[i]->R,PETSC_NULL);CHKERRA(ierr);
  }

  ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMMGView(dmmg,PETSC_VIEWER_STDOUT_(dmmg[0]->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DMMGSolve"
int DMMGSolve(DMMG *dmmg)
{
  int        i,ierr,nlevels = dmmg[0]->nlevels;
  PetscTruth gridseq,vecmonitor;
  KSP        ksp;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(0,"-dmmg_grid_sequence",&gridseq);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(0,"-dmmg_vecmonitor",&vecmonitor);CHKERRQ(ierr);
  if (gridseq) {
    if (dmmg[0]->initialguess) {
      ierr = (*dmmg[0]->initialguess)(dmmg[0]->snes,dmmg[0]->x,dmmg[0]);CHKERRQ(ierr);
      if (dmmg[0]->sles) {
        ierr = SLESGetKSP(dmmg[0]->sles,&ksp);CHKERRQ(ierr);
        ierr = KSPSetInitialGuessNonzero(ksp);CHKERRQ(ierr);
      }
    }
    for (i=0; i<nlevels-1; i++) {
      ierr = (*dmmg[i]->solve)(dmmg,i);CHKERRQ(ierr);
      if (vecmonitor) {
        ierr = VecView(dmmg[i]->x,PETSC_VIEWER_DRAW_(dmmg[i]->comm));CHKERRQ(ierr);
      }
      ierr = MatInterpolate(dmmg[i+1]->R,dmmg[i]->x,dmmg[i+1]->x);CHKERRQ(ierr);
      if (dmmg[i+1]->sles) {
        ierr = SLESGetKSP(dmmg[i+1]->sles,&ksp);CHKERRQ(ierr);
        ierr = KSPSetInitialGuessNonzero(ksp);CHKERRQ(ierr);
      }
    }
  } else {
    if (dmmg[nlevels-1]->initialguess) {
      ierr = (*dmmg[nlevels-1]->initialguess)(dmmg[nlevels-1]->snes,dmmg[nlevels-1]->x,dmmg[nlevels-1]);CHKERRQ(ierr);
      if (dmmg[nlevels-1]->sles) {
        ierr = SLESGetKSP(dmmg[nlevels-1]->sles,&ksp);CHKERRQ(ierr);
        ierr = KSPSetInitialGuessNonzero(ksp);CHKERRQ(ierr);
      }
    }
  }
  ierr = (*DMMGGetFine(dmmg)->solve)(dmmg,nlevels-1);CHKERRQ(ierr);
  if (vecmonitor) {
     ierr = VecView(dmmg[nlevels-1]->x,PETSC_VIEWER_DRAW_(dmmg[nlevels-1]->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DMMGSolveSLES"
int DMMGSolveSLES(DMMG *dmmg,int level)
{
  int        ierr,its;

  PetscFunctionBegin;
  ierr = (*dmmg[level]->rhs)(dmmg[level],dmmg[level]->b);CHKERRQ(ierr); 
  ierr = SLESSetOperators(dmmg[level]->sles,dmmg[level]->J,dmmg[level]->J,DIFFERENT_NONZERO_PATTERN);
  ierr = SLESSolve(dmmg[level]->sles,dmmg[level]->b,dmmg[level]->x,&its);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Sets each of the linear solvers to use multigrid 
*/
#undef __FUNC__  
#define __FUNC__ "DMMGSetUpLevel"
int DMMGSetUpLevel(DMMG *dmmg,SLES sles,int nlevels)
{
  int        ierr,i;
  PC         pc;
  PetscTruth ismg,monitor;
  SLES       lsles; /* solver internal to the multigrid preconditioner */
  MPI_Comm   *comms,comm;
  PetscViewer     ascii;
  KSP        ksp;


  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as DMMG");

  ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_ksp_monitor",&monitor);CHKERRQ(ierr);
  if (monitor) {
    ierr = SLESGetKSP(sles,&ksp);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
    ierr = PetscViewerASCIISetTab(ascii,1+dmmg[0]->nlevels-nlevels);CHKERRQ(ierr);
    ierr = KSPSetMonitor(ksp,KSPDefaultMonitor,ascii,(int(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
  }

  ierr  = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr  = PCSetType(pc,PCMG);CHKERRA(ierr);
ierr = PetscMalloc(nlevels*sizeof(MPI_Comm),&(  comms ));CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    comms[i] = dmmg[i]->comm;
  }
  ierr  = MGSetLevels(pc,nlevels,comms);CHKERRA(ierr);
  ierr  = PetscFree(comms);CHKERRQ(ierr); 

  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    /* set solvers for each level */
    for (i=0; i<nlevels; i++) {
      ierr = MGGetSmoother(pc,i,&lsles);CHKERRA(ierr);
      ierr = SLESSetOperators(lsles,dmmg[i]->J,dmmg[i]->J,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
      ierr = MGSetX(pc,i,dmmg[i]->x);CHKERRA(ierr); 
      ierr = MGSetRhs(pc,i,dmmg[i]->b);CHKERRA(ierr); 
      ierr = MGSetR(pc,i,dmmg[i]->r);CHKERRA(ierr); 
      ierr = MGSetResidual(pc,i,MGDefaultResidual,dmmg[i]->J);CHKERRA(ierr);
      if (monitor) {
        ierr = SLESGetKSP(lsles,&ksp);CHKERRQ(ierr);
        ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
        ierr = PetscViewerASCIISetTab(ascii,1+dmmg[0]->nlevels-i);CHKERRQ(ierr);
        ierr = KSPSetMonitor(ksp,KSPDefaultMonitor,ascii,(int(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
      }
    }

    /* Set interpolation/restriction between levels */
    for (i=1; i<nlevels; i++) {
      ierr = MGSetInterpolate(pc,i,dmmg[i]->R);CHKERRA(ierr); 
      ierr = MGSetRestriction(pc,i,dmmg[i]->R);CHKERRA(ierr); 
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DMMGSetSLES"
/*@C
    DMMGSetSLES - Sets the linear solver object that will use the grid hierarchy

    Collective on DMMG and SLES

    Input Parameter:
+   dmmg - the context
.   func - function to compute linear system matrix on each grid level
-   rhs - function to compute right hand side on each level

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetDA()

@*/
int DMMGSetSLES(DMMG *dmmg,int (*rhs)(DMMG,Vec),int (*func)(DMMG,Mat))
{
  int        ierr,i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as DMMG");

  /* create solvers for each level */
  for (i=0; i<nlevels; i++) {
    ierr = SLESCreate(dmmg[i]->comm,&dmmg[i]->sles);CHKERRQ(ierr);
    ierr = DMMGSetUpLevel(dmmg,dmmg[i]->sles,i+1);CHKERRQ(ierr);
    ierr = SLESSetFromOptions(dmmg[i]->sles);CHKERRA(ierr);
    dmmg[i]->solve = DMMGSolveSLES;
    dmmg[i]->rhs   = rhs;
  }

  /* evalute matrix on each level */
  for (i=0; i<nlevels; i++) {
    ierr = (*func)(dmmg[i],dmmg[i]->J);CHKERRQ(ierr);
  }

  for (i=0; i<nlevels-1; i++) {
    ierr = SLESSetOptionsPrefix(dmmg[i]->sles,"dmmg_");CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DMMGView"
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
  if (!dmmg) SETERRQ(1,"Passing null as DMMG");
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm,dmmg[0]->comm,&flag);CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) {
    SETERRQ(PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the DMMG and the PetscViewer");
  }

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"DMMG Object with %d levels\n",nlevels);CHKERRQ(ierr);
    for (i=0; i<nlevels; i++) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = DMView(dmmg[i]->dm,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(1,"Viewer type %s not supported",*((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}










