/*$Id: damg.c,v 1.19 2000/07/20 20:44:57 bsmith Exp balay $*/
 
#include "petscda.h"      /*I      "petscda.h"     I*/
#include "petscsles.h"    /*I      "petscsles.h"    I*/
#include "petscmg.h"      /*I      "petscmg.h"    I*/

/*
   Code for almost fully managing multigrid/multi-level linear solvers for DA grids
*/

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGCreate"></a>*/"DAMGCreate"
/*@C
    DAMGCreate - Creates a DA based multigrid solver object. This allows one to 
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
    ratio in the grid spacing simply change the DAMG structure after calling this routine

    Level: advanced

.seealso DAMGDestroy() 

@*/
int DAMGCreate(MPI_Comm comm,int nlevels,void *user,DAMG **damg)
{
  int        ierr,i,array[3],narray = 3,ratiox = 2, ratioy = 2, ratioz = 2;
  DAMG       *p;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = OptionsGetInt(0,"-damg_nlevels",&nlevels,0);CHKERRQ(ierr);
  ierr = OptionsGetIntArray(0,"-damg_ratio",array,&narray,&flg);CHKERRQ(ierr);
  if (flg) {
    if (narray > 0) ratiox = array[0];
    if (narray > 1) ratioy = array[1]; else ratioy = ratiox;
    if (narray > 2) ratioz = array[2]; else ratioz = ratioy;
  }

  p    = (DAMG *)PetscMalloc(nlevels*sizeof(DAMG));CHKPTRQ(p);
  for (i=0; i<nlevels; i++) {
    p[i]          = (DAMG)PetscMalloc(sizeof(struct _p_DAMG));CHKPTRQ(p[i]);
    ierr          = PetscMemzero(p[i],sizeof(struct _p_DAMG));CHKERRQ(ierr);
    p[i]->nlevels = nlevels - i;
    p[i]->ratiox  = ratiox;
    p[i]->ratioy  = ratioy;
    p[i]->ratioz  = ratioz;
    p[i]->comm    = comm;
    p[i]->user    = user;
  }
  *damg = p;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGDestroy"></a>*/"DAMGDestroy"
/*@C
    DAMGDestroy - Destroys a DA based multigrid solver object. 

    Collective on DAMG

    Input Parameter:
.    - the context

    Level: advanced

.seealso DAMGCreate()

@*/
int DAMGDestroy(DAMG *damg)
{
  int     ierr,i,nlevels = damg[0]->nlevels;

  PetscFunctionBegin;
  if (!damg) SETERRQ(1,1,"Passing null as DAMG");

  for (i=1; i<nlevels; i++) {
    if (damg[i]->R) {ierr = MatDestroy(damg[i]->R);CHKERRA(ierr);}
  }
  for (i=0; i<nlevels; i++) {
    if (damg[i]->da) {ierr = DADestroy(damg[i]->da);CHKERRQ(ierr);}
    if (damg[i]->x)  {ierr = VecDestroy(damg[i]->x);CHKERRQ(ierr);}
    if (damg[i]->b)  {ierr = VecDestroy(damg[i]->b);CHKERRQ(ierr);}
    if (damg[i]->r)  {ierr = VecDestroy(damg[i]->r);CHKERRQ(ierr);}
    if (damg[i]->B && damg[i]->B != damg[i]->J) {ierr = MatDestroy(damg[i]->B);CHKERRQ(ierr);}
    if (damg[i]->J)  {ierr = MatDestroy(damg[i]->J);CHKERRQ(ierr);}
    if (damg[i]->Rscale)  {ierr = VecDestroy(damg[i]->Rscale);CHKERRQ(ierr);}
    if (damg[i]->localX)  {ierr = VecDestroy(damg[i]->localX);CHKERRQ(ierr);}
    if (damg[i]->localF)  {ierr = VecDestroy(damg[i]->localF);CHKERRQ(ierr);}
    if (damg[i]->fdcoloring)  {ierr = MatFDColoringDestroy(damg[i]->fdcoloring);CHKERRQ(ierr);}
    if (damg[i]->sles)  {ierr = SLESDestroy(damg[i]->sles);CHKERRQ(ierr);}
    if (damg[i]->snes)  {ierr = PetscObjectDestroy((PetscObject)damg[i]->snes);CHKERRQ(ierr);} 
    ierr = PetscFree(damg[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(damg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGSetGrid"></a>*/"DAMGSetGrid"
/*@C
    DAMGSetGrid - Sets the grid information for the grids

    Collective on DAMG

    Input Parameter:
+   damg - the context
.   dim - 1, 2, or 3
.   pt - DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, DA_XYPERIODIC, DA_XYZPERIODIC, DA_XZPERIODIC, or DA_YZPERIODIC
.   st - DA_STENCIL_STAR or DA_STENCIL_BOX
.   M - grid points in x
.   N - grid points in y
.   P - grid points in z
.   sw - stencil width, often 1
+   dof - number of degrees of freedom per node

    Level: advanced

.seealso DAMGCreate(), DAMGDestroy

@*/
int DAMGSetGrid(DAMG *damg,int dim,DAPeriodicType pt,DAStencilType st,int M,int N,int P,int dof,int sw)
{
  int            ierr,i,j,nlevels = damg[0]->nlevels,m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE, array[3],narray = 3;
  MPI_Comm       comm = damg[0]->comm;
  PetscTruth     flg,split = PETSC_FALSE;

  PetscFunctionBegin;
  if (!damg) SETERRQ(1,1,"Passing null as DAMG");

  ierr = OptionsGetIntArray(PETSC_NULL,"-damg_mnp",array,&narray,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = OptionsGetIntArray(PETSC_NULL,"-damg_mn",array,&narray,&flg);CHKERRQ(ierr);
  }
  if (flg) {
    if (narray > 0) M = array[0];
    if (narray > 1) N = array[1]; else N = M;
    if (narray > 2) P = array[2]; else P = N;
  }
  ierr = OptionsGetLogical(PETSC_NULL,"-damg_split",&split,PETSC_NULL);CHKERRQ(ierr);

  /* Create DA data structure for all the levels */
  for (i=0; i<nlevels; i++) {
    if (dim == 3) {
      ierr = DACreate3d(damg[i]->comm,pt,st,M,N,P,m,n,p,dof,sw,0,0,0,&damg[i]->da);CHKERRQ(ierr);
    } else if (dim == 2) {
      if (split) {
        ierr = DASplitComm2d(damg[i]->comm,M,N,sw,&damg[i]->comm);CHKERRQ(ierr);
      }
      ierr = DACreate2d(damg[i]->comm,pt,st,M,N,m,n,dof,sw,0,0,&damg[i]->da);CHKERRQ(ierr);
    } else {
      SETERRQ1(1,1,"Cannot handle dimension %d",dim);
    }

    M = damg[i]->ratiox*(M-1) + 1;
    N = damg[i]->ratioy*(N-1) + 1;
    P = damg[i]->ratioz*(P-1) + 1;
  }

  /* Create work vectors and matrix for each level */
  for (i=0; i<nlevels; i++) {
    ierr = DACreateGlobalVector(damg[i]->da,&damg[i]->x);CHKERRA(ierr);
    ierr = VecDuplicate(damg[i]->x,&damg[i]->b);CHKERRA(ierr);
    ierr = VecDuplicate(damg[i]->x,&damg[i]->r);CHKERRA(ierr);
    ierr = DAGetColoring(damg[i]->da,PETSC_NULL,&damg[i]->J);CHKERRQ(ierr);
    damg[i]->B = damg[i]->J;
  }

  /* Create interpolation/restriction between levels */
  for (i=1; i<nlevels; i++) {
    ierr = DAGetInterpolation(damg[i-1]->da,damg[i]->da,&damg[i]->R,PETSC_NULL);CHKERRA(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-damg_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DAMGView(damg,VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGSolve"></a>*/"DAMGSolve"
int DAMGSolve(DAMG *damg)
{
  int        i,ierr,nlevels = damg[0]->nlevels;
  PetscTruth gridseq;
  KSP        ksp;

  PetscFunctionBegin;
  ierr = OptionsHasName(0,"-damg_grid_sequence",&gridseq);CHKERRQ(ierr);
  if (gridseq) {
    if (damg[0]->initialguess) {
      ierr = (*damg[0]->initialguess)(damg[0]->snes,damg[0]->x,damg[0]);CHKERRQ(ierr);
    }
    for (i=0; i<nlevels-1; i++) {
      ierr = (*damg[i]->solve)(damg,i);CHKERRQ(ierr);
      ierr = MatInterpolate(damg[i+1]->R,damg[i]->x,damg[i+1]->x);CHKERRQ(ierr);
      if (damg[i+1]->sles) {
        ierr = SLESGetKSP(damg[i+1]->sles,&ksp);CHKERRQ(ierr);
        ierr = KSPSetInitialGuessNonzero(ksp);CHKERRQ(ierr);
      }
    }
  } else {
    if (damg[nlevels-1]->initialguess) {
      ierr = (*damg[nlevels-1]->initialguess)(damg[nlevels-1]->snes,damg[nlevels-1]->x,damg[nlevels-1]);CHKERRQ(ierr);
    }
  }
  ierr = (*DAMGGetFine(damg)->solve)(damg,nlevels-1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGSolveSLES"></a>*/"DAMGSolveSLES"
int DAMGSolveSLES(DAMG *damg,int level)
{
  int        ierr,its;

  PetscFunctionBegin;
  ierr = (*damg[level]->rhs)(damg[level],damg[level]->b);CHKERRQ(ierr); 
  ierr = SLESSetOperators(damg[level]->sles,damg[level]->J,damg[level]->J,DIFFERENT_NONZERO_PATTERN);
  ierr = SLESSolve(damg[level]->sles,damg[level]->b,damg[level]->x,&its);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
    Sets each of the linear solvers to use multigrid 
*/
#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGSetUpLevel"></a>*/"DAMGSetUpLevel"
int DAMGSetUpLevel(DAMG *damg,SLES sles,int nlevels)
{
  int        ierr,i;
  PC         pc;
  PetscTruth ismg;
  SLES       lsles; /* solver internal to the multigrid preconditioner */
  MPI_Comm   *comms;

  PetscFunctionBegin;
  if (!damg) SETERRQ(1,1,"Passing null as DAMG");

  ierr  = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr  = PCSetType(pc,PCMG);CHKERRA(ierr);
  comms = (MPI_Comm*)PetscMalloc(nlevels*sizeof(MPI_Comm));CHKPTRQ(comms);
  for (i=0; i<nlevels; i++) {
    comms[i] = damg[i]->comm;
  }
  ierr  = MGSetLevels(pc,nlevels,comms);CHKERRA(ierr);
  ierr  = PetscFree(comms);CHKERRQ(ierr); 
  ierr  = SLESSetFromOptions(sles);CHKERRA(ierr);

  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    /* set solvers for each level */
    for (i=0; i<nlevels; i++) {
      ierr = MGGetSmoother(pc,i,&lsles);CHKERRA(ierr);
      ierr = SLESSetFromOptions(lsles);CHKERRA(ierr);
      ierr = SLESSetOperators(lsles,damg[i]->J,damg[i]->J,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
      ierr = MGSetX(pc,i,damg[i]->x);CHKERRA(ierr); 
      ierr = MGSetRhs(pc,i,damg[i]->b);CHKERRA(ierr); 
      ierr = MGSetR(pc,i,damg[i]->r);CHKERRA(ierr); 
      ierr = MGSetResidual(pc,i,MGDefaultResidual,damg[i]->J);CHKERRA(ierr);
    }

    /* Set interpolation/restriction between levels */
    for (i=1; i<nlevels; i++) {
      ierr = MGSetInterpolate(pc,i,damg[i]->R);CHKERRA(ierr); 
      ierr = MGSetRestriction(pc,i,damg[i]->R);CHKERRA(ierr); 
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGSetSLES"></a>*/"DAMGSetSLES"
/*@C
    DAMGSetSLES - Sets the linear solver object that will use the grid hierarchy

    Collective on DAMG and SLES

    Input Parameter:
+   damg - the context
.   func - function to compute linear system matrix on each grid level
-   rhs - function to compute right hand side on each level

    Level: advanced

.seealso DAMGCreate(), DAMGDestroy, DAMGSetGrid()

@*/
int DAMGSetSLES(DAMG *damg,int (*rhs)(DAMG,Vec),int (*func)(DAMG,Mat))
{
  int        ierr,i,nlevels = damg[0]->nlevels;

  PetscFunctionBegin;
  if (!damg) SETERRQ(1,1,"Passing null as DAMG");

  /* create solvers for each level */
  for (i=0; i<nlevels; i++) {
    ierr = SLESCreate(damg[i]->comm,&damg[i]->sles);CHKERRQ(ierr);
    ierr = DAMGSetUpLevel(damg,damg[i]->sles,i+1);CHKERRQ(ierr);
    damg[i]->solve = DAMGSolveSLES;
    damg[i]->rhs   = rhs;
  }

  /* evalute matrix on each level */
  for (i=0; i<nlevels; i++) {
    ierr = (*func)(damg[i],damg[i]->J);CHKERRQ(ierr);
  }

  for (i=0; i<nlevels-1; i++) {
    ierr = SLESSetOptionsPrefix(damg[i]->sles,"damg_");CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGView"></a>*/"DAMGView"
/*@C
    DAMGView - prints information on a DA based multi-level preconditioner

    Collective on DAMG and Viewer

    Input Parameter:
+   damg - the context
-   viewer - the viewer

    Level: advanced

.seealso DAMGCreate(), DAMGDestroy

@*/
int DAMGView(DAMG *damg,Viewer viewer)
{
  int            ierr,i,nlevels = damg[0]->nlevels,flag;
  MPI_Comm       comm;
  PetscTruth     isascii;

  PetscFunctionBegin;
  if (!damg) SETERRQ(1,1,"Passing null as DAMG");
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm,damg[0]->comm,&flag);CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) {
    SETERRQ(PETSC_ERR_ARG_NOTSAMECOMM,0,"Different communicators in the DAMG and the Viewer");
  }

  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"DAMG Object with %d levels\n",nlevels);CHKERRQ(ierr);
    for (i=0; i<nlevels; i++) {
      ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = DAView(damg[i]->da,viewer);CHKERRQ(ierr);
      ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported",*((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}










