 
#include <petscdm.h>            /*I "petscdm.h"   I*/
#include <petscksp.h>           /*I "petscksp.h"  I*/
#include <petscpcmg.h>            /*I "petscpcmg.h"   I*/
#include <petscdmmg.h>          /*I "petscdmmg.h" I*/
#include <petsc-private/pcimpl.h>     /*I "petscpc.h"   I*/

/*
   Code for almost fully managing multigrid/multi-level linear solvers for DM grids
*/

#undef __FUNCT__  
#define __FUNCT__ "DMMGCreate"
/*@C
    DMMGCreate - Creates a DM based multigrid solver object. This allows one to 
      easily implement MG methods on regular grids.

     This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Collective on MPI_Comm

    Input Parameter:
+   comm - the processors that will share the grids and solution process
.   nlevels - number of multigrid levels (if this is negative it CANNOT be reset with -dmmg_nlevels
-   user - an optional user context

    Output Parameters:
.    - the context

    Options Database:
+     -dmmg_nlevels <levels> - number of levels to use
.     -pc_mg_galerkin - use Galerkin approach to compute coarser matrices
-     -dmmg_mat_type <type> - matrix type that DMMG should create, defaults to MATAIJ

    Notes:
      To provide a different user context for each level call DMMGSetUser() after calling
      this routine

    Level: advanced

.seealso DMMGDestroy(), DMMGSetUser(), DMMGGetUser(), DMMGSetMatType(),  DMMGSetNullSpace(), DMMGSetInitialGuess(),
         DMMGSetISColoringType()

@*/
PetscErrorCode  DMMGCreate(MPI_Comm comm,PetscInt nlevels,void *user,DMMG **dmmg)
{
  PetscErrorCode ierr;
  PetscInt       i;
  DMMG           *p;
  PetscBool      ftype;
  char           mtype[256];

  PetscFunctionBegin;
  if (nlevels < 0) {
    nlevels = -nlevels;
  } else {
    ierr = PetscOptionsGetInt(0,"-dmmg_nlevels",&nlevels,PETSC_IGNORE);CHKERRQ(ierr);
  }
  if (nlevels < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot set levels less than 1");

  ierr = PetscMalloc(nlevels*sizeof(DMMG),&p);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    ierr           = PetscNew(struct _n_DMMG,&p[i]);CHKERRQ(ierr);
    p[i]->nlevels  = nlevels - i;
    p[i]->comm     = comm;
    p[i]->user     = user;
    p[i]->updatejacobianperiod = 1;
    p[i]->updatejacobian       = PETSC_TRUE;
    p[i]->isctype  = IS_COLORING_GLOBAL; 
    ierr           = PetscStrallocpy(MATAIJ,&p[i]->mtype);CHKERRQ(ierr);
  }
  *dmmg = p;

  ierr = PetscOptionsGetString(PETSC_NULL,"-dmmg_mat_type",mtype,256,&ftype);CHKERRQ(ierr);
  if (ftype) {
    ierr = DMMGSetMatType(*dmmg,mtype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetMatType"
/*@C
    DMMGSetMatType - Sets the type of matrices that DMMG will create for its solvers.

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Logically Collective on MPI_Comm 

    Input Parameters:
+    dmmg - the DMMG object created with DMMGCreate()
-    mtype - the matrix type, defaults to MATAIJ

    Level: intermediate

.seealso DMMGDestroy(), DMMGSetUser(), DMMGGetUser(), DMMGCreate(), DMMGSetNullSpace()

@*/
PetscErrorCode  DMMGSetMatType(DMMG *dmmg,const MatType mtype)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<dmmg[0]->nlevels; i++) {
    ierr = PetscFree(dmmg[i]->mtype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(mtype,&dmmg[i]->mtype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetOptionsPrefix"
/*@C
    DMMGSetOptionsPrefix - Sets the prefix used for the solvers inside a DMMG

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Logically Collective on MPI_Comm 

    Input Parameters:
+    dmmg - the DMMG object created with DMMGCreate()
-    prefix - the prefix string

    Level: intermediate

.seealso DMMGDestroy(), DMMGSetUser(), DMMGGetUser(), DMMGCreate(), DMMGSetNullSpace()

@*/
PetscErrorCode  DMMGSetOptionsPrefix(DMMG *dmmg,const char prefix[])
{
  PetscInt       i;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  for (i=0; i<dmmg[0]->nlevels; i++) {
    ierr = PetscStrallocpy(prefix,&dmmg[i]->prefix);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGDestroy"
/*@C
    DMMGDestroy - Destroys a DM based multigrid solver object. 

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Collective on DMMG

    Input Parameter:
.    - the context

    Level: advanced

.seealso DMMGCreate()

@*/
PetscErrorCode  DMMGDestroy(DMMG *dmmg)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Passing null as DMMG");

  for (i=1; i<nlevels; i++) {
    ierr = MatDestroy(&dmmg[i]->R);CHKERRQ(ierr);
  }
  for (i=0; i<nlevels; i++) {
    ierr = PetscFree(dmmg[i]->prefix);CHKERRQ(ierr);
    ierr = PetscFree(dmmg[i]->mtype);CHKERRQ(ierr);
    ierr = DMDestroy(&dmmg[i]->dm);CHKERRQ(ierr);
    ierr = VecDestroy(&dmmg[i]->x);CHKERRQ(ierr);
    ierr = VecDestroy(&dmmg[i]->b);CHKERRQ(ierr);
    ierr = VecDestroy(&dmmg[i]->r);CHKERRQ(ierr);
    ierr = VecDestroy(&dmmg[i]->work1);CHKERRQ(ierr);
    ierr = VecDestroy(&dmmg[i]->w);CHKERRQ(ierr);
    ierr = VecDestroy(&dmmg[i]->work2);CHKERRQ(ierr);
    ierr = VecDestroy(&dmmg[i]->lwork1);CHKERRQ(ierr);
    ierr = MatDestroy(&dmmg[i]->B);CHKERRQ(ierr);
    ierr = MatDestroy(&dmmg[i]->J);CHKERRQ(ierr);
    ierr = VecDestroy(&dmmg[i]->Rscale);CHKERRQ(ierr);
    ierr = MatFDColoringDestroy(&dmmg[i]->fdcoloring);CHKERRQ(ierr);
    if (dmmg[i]->ksp && !dmmg[i]->snes) {ierr = KSPDestroy(&dmmg[i]->ksp);CHKERRQ(ierr);}
    ierr = PetscObjectDestroy((PetscObject*)&dmmg[i]->snes);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&dmmg[i]->inject);CHKERRQ(ierr);
    ierr = PetscFree(dmmg[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dmmg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetDM"
/*@C
    DMMGSetDM - Sets the coarse grid information for the grids

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Logically Collective on DMMG and DM

    Input Parameter:
+   dmmg - the context
-   dm - the DMDA or DMComposite object

    Options Database Keys:
.   -dmmg_refine: Use the input problem as the coarse level and refine.
.   -dmmg_refine false: Use the input problem as the fine level and coarsen.

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy(), DMMGSetMatType()

@*/
PetscErrorCode  DMMGSetDM(DMMG *dmmg, DM dm)
{
  PetscInt       nlevels     = dmmg[0]->nlevels;
  PetscBool      doRefine    = PETSC_TRUE;
  PetscInt       i;
  DM             *hierarchy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Passing null as DMMG");

  /* Create DM data structure for all the levels */
  ierr = PetscOptionsGetBool(PETSC_NULL, "-dmmg_refine", &doRefine, PETSC_IGNORE);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
  ierr = PetscMalloc(nlevels*sizeof(DM),&hierarchy);CHKERRQ(ierr);
  if (doRefine) {
    ierr = DMRefineHierarchy(dm,nlevels-1,hierarchy);CHKERRQ(ierr);
    dmmg[0]->dm = dm;
    for(i=1; i<nlevels; ++i) {
      dmmg[i]->dm = hierarchy[i-1];
    }
  } else {
    dmmg[nlevels-1]->dm = dm;
    ierr = DMCoarsenHierarchy(dm,nlevels-1,hierarchy);CHKERRQ(ierr);
    for(i=0; i<nlevels-1; ++i) {
      dmmg[nlevels-2-i]->dm = hierarchy[i];
    }
  }
  ierr = PetscFree(hierarchy);CHKERRQ(ierr);
  /* Cleanup old structures (should use some private Destroy() instead) */
  for(i = 0; i < nlevels; ++i) {
    ierr = MatDestroy(&dmmg[i]->B);CHKERRQ(ierr);
    ierr = MatDestroy(&dmmg[i]->J);CHKERRQ(ierr);
  }

  /* Create work vectors and matrix for each level */
  for (i=0; i<nlevels; i++) {
    ierr = DMCreateGlobalVector(dmmg[i]->dm,&dmmg[i]->x);CHKERRQ(ierr);
    ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->b);CHKERRQ(ierr);
    ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->r);CHKERRQ(ierr);
  }

  /* Create interpolation/restriction between levels */
  for (i=1; i<nlevels; i++) {
    ierr = DMCreateInterpolation(dmmg[i-1]->dm,dmmg[i]->dm,&dmmg[i]->R,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMMGSolve"
/*@C
    DMMGSolve - Actually solves the (non)linear system defined with the DMMG

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Collective on DMMG

    Input Parameter:
.   dmmg - the context

    Level: advanced

    Options Database:
+   -dmmg_grid_sequence - use grid sequencing to get the initial solution for each level from the previous
-   -dmmg_monitor_solution - display the solution at each iteration

     Notes: For linear (KSP) problems may be called more than once, uses the same 
    matrices but recomputes the right hand side for each new solve. Call DMMGSetKSP()
    to generate new matrices.
 
.seealso DMMGCreate(), DMMGDestroy(), DMMG, DMMGSetSNES(), DMMGSetKSP(), DMMGSetUp(), DMMGSetMatType()

@*/
PetscErrorCode  DMMGSolve(DMMG *dmmg)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;
  PetscBool      gridseq = PETSC_FALSE,vecmonitor = PETSC_FALSE,flg;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(0,"-dmmg_grid_sequence",&gridseq,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(0,"-dmmg_monitor_solution",&vecmonitor,PETSC_NULL);CHKERRQ(ierr);
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
      if (dmmg[i+1]->ksp && !dmmg[i+1]->snes) {
        ierr = KSPSetInitialGuessNonzero(dmmg[i+1]->ksp,PETSC_TRUE);CHKERRQ(ierr);
     }
    }
  } else {
    if (dmmg[nlevels-1]->initialguess) {
      ierr = (*dmmg[nlevels-1]->initialguess)(dmmg[nlevels-1],dmmg[nlevels-1]->x);CHKERRQ(ierr);
    }
  }

  /*ierr = VecView(dmmg[nlevels-1]->x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);*/

  ierr = (*DMMGGetFine(dmmg)->solve)(dmmg,nlevels-1);CHKERRQ(ierr);
  if (vecmonitor) {
     ierr = VecView(dmmg[nlevels-1]->x,PETSC_VIEWER_DRAW_(dmmg[nlevels-1]->comm));CHKERRQ(ierr);
  }

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-dmmg_view",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIGetStdout(dmmg[0]->comm,&viewer);CHKERRQ(ierr);
    ierr = DMMGView(dmmg,viewer);CHKERRQ(ierr);
  }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-dmmg_view_binary",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = DMMGView(dmmg,PETSC_VIEWER_BINARY_(dmmg[0]->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveKSP"
PetscErrorCode  DMMGSolveKSP(DMMG *dmmg,PetscInt level)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dmmg[level]->rhs) {
    CHKMEMQ;
    ierr = (*dmmg[level]->rhs)(dmmg[level],dmmg[level]->b);CHKERRQ(ierr); 
    CHKMEMQ;
  }
  ierr = KSPSolve(dmmg[level]->ksp,dmmg[level]->b,dmmg[level]->x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    For each level (of grid sequencing) this sets the interpolation/restriction and 
    work vectors needed by the multigrid preconditioner within the KSP 
    (for nonlinear problems the KSP inside the SNES) of that level.

    Also sets the KSP monitoring on all the levels if requested by user.

*/
#undef __FUNCT__  
#define __FUNCT__ "DMMGSetUpLevel"
PetscErrorCode  DMMGSetUpLevel(DMMG *dmmg,KSP ksp,PetscInt nlevels)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PC                      pc;
  PetscBool               ismg,ismf,isshell,ismffd;
  KSP                     lksp; /* solver internal to the multigrid preconditioner */
  MPI_Comm                *comms;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Passing null as DMMG");

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
      if (i < nlevels-1) { /* don't set for finest level, they are set in PCApply_MG()*/
	ierr = PCMGSetX(pc,i,dmmg[i]->x);CHKERRQ(ierr); 
	ierr = PCMGSetRhs(pc,i,dmmg[i]->b);CHKERRQ(ierr); 
      }
      if (i > 0) {
        ierr = PCMGSetR(pc,i,dmmg[i]->r);CHKERRQ(ierr); 
      }
      /* If using a matrix free multiply and did not provide an explicit matrix to build
         the preconditioner then must use no preconditioner 
      */
      ierr = PetscTypeCompare((PetscObject)dmmg[i]->B,MATSHELL,&isshell);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)dmmg[i]->B,MATDAAD,&ismf);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)dmmg[i]->B,MATMFFD,&ismffd);CHKERRQ(ierr);
      if (isshell || ismf || ismffd) {
        PC  lpc;
        ierr = PCMGGetSmoother(pc,i,&lksp);CHKERRQ(ierr); 
        ierr = KSPGetPC(lksp,&lpc);CHKERRQ(ierr);
        ierr = PCSetType(lpc,PCNONE);CHKERRQ(ierr);
      }
    }

    /* Set interpolation/restriction between levels */
    for (i=1; i<nlevels; i++) {
      ierr = PCMGSetInterpolation(pc,i,dmmg[i]->R);CHKERRQ(ierr); 
      ierr = PCMGSetRestriction(pc,i,dmmg[i]->R);CHKERRQ(ierr); 
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetKSP"
/*@C
    DMMGSetKSP - Sets the linear solver object that will use the grid hierarchy

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Logically Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   func - function to compute linear system matrix on each grid level
-   rhs - function to compute right hand side on each level (need only work on the finest grid
          if you do not use grid sequencing)

    Level: advanced

    Notes: For linear problems my be called more than once, reevaluates the matrices if it is called more
       than once. Call DMMGSolve() directly several times to solve with the same matrix but different 
       right hand sides.
   
.seealso DMMGCreate(), DMMGDestroy, DMMGSetDM(), DMMGSolve(), DMMGSetMatType()

@*/
PetscErrorCode  DMMGSetKSP(DMMG *dmmg,PetscErrorCode (*rhs)(DMMG,Vec),PetscErrorCode (*func)(DMMG,Mat,Mat))
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels,level;
  PetscBool      ismg,galerkin=PETSC_FALSE;
  PC             pc;
  KSP            lksp;
  
  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Passing null as DMMG");

  if (!dmmg[0]->ksp) {
    /* create solvers for each level if they don't already exist*/
    for (i=0; i<nlevels; i++) {

      ierr = KSPCreate(dmmg[i]->comm,&dmmg[i]->ksp);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)dmmg[i]->ksp,PETSC_NULL,nlevels-i);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(dmmg[i]->ksp,dmmg[i]->prefix);CHKERRQ(ierr);
      ierr = DMMGSetUpLevel(dmmg,dmmg[i]->ksp,i+1);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(dmmg[i]->ksp);CHKERRQ(ierr);

      /*  if the multigrid is being run with Galerkin then these matrices do not need to be created except on the finest level
          we do not take advantage of this because it may be that Galerkin has not yet been selected for the KSP object 
          These are also used if grid sequencing is selected for the linear problem. We should probably turn off grid sequencing
          for the linear problem */
      if (!dmmg[i]->B) {
	ierr = DMCreateMatrix(dmmg[i]->dm,dmmg[nlevels-1]->mtype,&dmmg[i]->B);CHKERRQ(ierr);
      } 
      if (!dmmg[i]->J) {
	dmmg[i]->J = dmmg[i]->B;
	ierr = PetscObjectReference((PetscObject) dmmg[i]->J);CHKERRQ(ierr);
      }

      dmmg[i]->solve = DMMGSolveKSP;
      dmmg[i]->rhs   = rhs;
    }
  }

  /* evalute matrix on each level */
  ierr = KSPGetPC(dmmg[nlevels-1]->ksp,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {
    ierr = PCMGGetGalerkin(pc,&galerkin);CHKERRQ(ierr);
  }
  if (func) {
    if (galerkin) {
      ierr = (*func)(dmmg[nlevels-1],dmmg[nlevels-1]->J,dmmg[nlevels-1]->B);CHKERRQ(ierr);
    } else {
      for (i=0; i<nlevels; i++) {
        ierr = (*func)(dmmg[i],dmmg[i]->J,dmmg[i]->B);CHKERRQ(ierr);
      }
    }
  }

  for (i=0; i<nlevels-1; i++) {
    ierr = KSPSetOptionsPrefix(dmmg[i]->ksp,"dmmg_");CHKERRQ(ierr);
  }

  for (level=0; level<nlevels; level++) {
    ierr = KSPSetOperators(dmmg[level]->ksp,dmmg[level]->J,dmmg[level]->B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPGetPC(dmmg[level]->ksp,&pc);CHKERRQ(ierr);
    if (ismg) {
      for (i=0; i<=level; i++) {
        ierr = PCMGGetSmoother(pc,i,&lksp);CHKERRQ(ierr); 
        ierr = KSPSetOperators(lksp,dmmg[i]->J,dmmg[i]->B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      }
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGView"
/*@C
    DMMGView - prints information on a DM based multi-level preconditioner

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Collective on DMMG and PetscViewer

    Input Parameter:
+   dmmg - the context
-   viewer - the viewer

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy(), DMMGSetMatType()

@*/
PetscErrorCode  DMMGView(DMMG *dmmg,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;
  PetscMPIInt    flag;
  MPI_Comm       comm;
  PetscBool      iascii,isbinary;

  PetscFunctionBegin;
  PetscValidPointer(dmmg,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm,dmmg[0]->comm,&flag);CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the DMMG and the PetscViewer");

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
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
      if (dmmg[0]->isctype == IS_COLORING_GLOBAL) {
        ierr = PetscViewerASCIIPrintf(viewer,"Using global (nonghosted) Jacobian coloring computation\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Using ghosted Jacobian coloring computation\n");CHKERRQ(ierr);
      }
    }
    for (i=0; i<nlevels; i++) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = DMView(dmmg[i]->dm,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (iascii) {
      ierr = PetscViewerASCIIPrintf(viewer,"Using matrix type %s\n",dmmg[nlevels-1]->mtype);CHKERRQ(ierr);
    }
    if (DMMGGetKSP(dmmg)) {
      ierr = KSPView(DMMGGetKSP(dmmg),viewer);CHKERRQ(ierr);
    } else if (DMMGGetSNES(dmmg)) {
      ierr = SNESView(DMMGGetSNES(dmmg),viewer);CHKERRQ(ierr);
    } else if (iascii) {
      ierr = PetscViewerASCIIPrintf(viewer,"DMMG does not have a SNES or KSP set\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetNullSpace"
/*@C
    DMMGSetNullSpace - Indicates the null space in the linear operator (this is needed by the linear solver)

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Logically Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   has_cnst - is the constant vector in the null space
.   n - number of null vectors (excluding the possible constant vector)
-   func - a function that fills an array of vectors with the null vectors (must be orthonormal), may be PETSC_NULL

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetDM(), DMMGSolve(), MatNullSpaceCreate(), KSPSetNullSpace(), DMMGSetMatType()

@*/
PetscErrorCode  DMMGSetNullSpace(DMMG *dmmg,PetscBool  has_cnst,PetscInt n,PetscErrorCode (*func)(DMMG,Vec[]))
{
  PetscErrorCode ierr;
  PetscInt       i,j,nlevels = dmmg[0]->nlevels;
  Vec            *nulls = 0;
  MatNullSpace   nullsp;
  KSP            iksp;
  PC             pc,ipc;
  PetscBool      ismg,isred;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Passing null as DMMG");
  if (!dmmg[0]->ksp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call AFTER DMMGSetKSP() or DMMGSetSNES()");
  if ((n && !func) || (!n && func)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Both n and func() must be set together");
  if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot have negative number of vectors in null space n = %D",n);

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
    ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
    if (n) {
      ierr = VecDestroyVecs(n,&nulls);CHKERRQ(ierr);
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
        KSP iksp;
        ierr = PCRedundantGetKSP(ipc,&iksp);CHKERRQ(ierr);
        ierr = KSPGetPC(iksp,&ipc);CHKERRQ(ierr);
      }
      ierr = PCFactorSetShiftType(ipc,MAT_SHIFT_POSITIVE_DEFINITE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGInitialGuessCurrent"
/*@C
    DMMGInitialGuessCurrent - Use with DMMGSetInitialGuess() to use the current value in the 
       solution vector (obtainable with DMMGGetx()) as the initial guess. Otherwise for linear
       problems zero is used for the initial guess (unless grid sequencing is used). For nonlinear 
       problems this is not needed; it always uses the previous solution as the initial guess.

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Logically Collective on DMMG

    Input Parameter:
+   dmmg - the context
-   vec - dummy argument

    Level: intermediate

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNES(), DMMGSetInitialGuess()

@*/
PetscErrorCode  DMMGInitialGuessCurrent(DMMG dmmg,Vec vec)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetInitialGuess"
/*@C
    DMMGSetInitialGuess - Sets the function that computes an initial guess.

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Logically Collective on DMMG

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


.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNES(), DMMGInitialGuessCurrent(), DMMGSetGalekin(), DMMGSetMatType(), DMMGSetNullSpace()

@*/
PetscErrorCode  DMMGSetInitialGuess(DMMG *dmmg,PetscErrorCode (*guess)(DMMG,Vec))
{
  PetscInt       i,nlevels = dmmg[0]->nlevels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<nlevels; i++) {
    if (dmmg[i]->ksp && !dmmg[i]->snes) {
      ierr = KSPSetInitialGuessNonzero(dmmg[i]->ksp,PETSC_TRUE);CHKERRQ(ierr);
    }
    dmmg[i]->initialguess = guess;
  }
  PetscFunctionReturn(0);
}







