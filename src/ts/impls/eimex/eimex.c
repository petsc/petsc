
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscdm.h>

static const PetscInt TSEIMEXDefault = 3;

typedef struct {
  PetscInt     row_ind;         /* Return the term T[row_ind][col_ind] */
  PetscInt     col_ind;         /* Return the term T[row_ind][col_ind] */
  PetscInt     nstages;         /* Numbers of stages in current scheme */
  PetscInt     max_rows;        /* Maximum number of rows */
  PetscInt     *N;              /* Harmonic sequence N[max_rows] */
  Vec          Y;               /* States computed during the step, used to complete the step */
  Vec          Z;               /* For shift*(Y-Z) */
  Vec          *T;              /* Working table, size determined by nstages */
  Vec          YdotRHS;         /* f(x) Work vector holding YdotRHS during residual evaluation */
  Vec          YdotI;           /* xdot-g(x) Work vector holding YdotI = G(t,x,xdot) when xdot =0 */
  Vec          Ydot;            /* f(x)+g(x) Work vector */
  Vec          VecSolPrev;      /* Work vector holding the solution from the previous step (used for interpolation) */
  PetscReal    shift;
  PetscReal    ctime;
  PetscBool    recompute_jacobian; /* Recompute the Jacobian at each stage, default is to freeze the Jacobian at the start of each step */
  PetscBool    ord_adapt;       /* order adapativity */
  TSStepStatus status;
} TS_EIMEX;

/* This function is pure */
static PetscInt Map(PetscInt i, PetscInt j, PetscInt s)
{
  return ((2*s-j+1)*j/2+i-j);
}

static PetscErrorCode TSEvaluateStep_EIMEX(TS ts,PetscInt order,Vec X,PetscBool *done)
{
  TS_EIMEX        *ext = (TS_EIMEX*)ts->data;
  const PetscInt  ns = ext->nstages;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecCopy(ext->T[Map(ext->row_ind,ext->col_ind,ns)],X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStage_EIMEX(TS ts,PetscInt istage)
{
  TS_EIMEX        *ext = (TS_EIMEX*)ts->data;
  PetscReal       h;
  Vec             Y=ext->Y, Z=ext->Z;
  SNES            snes;
  TSAdapt         adapt;
  PetscInt        i,its,lits;
  PetscBool       accept;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  h = ts->time_step/ext->N[istage];/* step size for the istage-th stage */
  ext->shift = 1./h;
  ierr = SNESSetLagJacobian(snes,-2);CHKERRQ(ierr); /* Recompute the Jacobian on this solve, but not again */
  ierr = VecCopy(ext->VecSolPrev,Y);CHKERRQ(ierr); /* Take the previous solution as initial step */

  for (i=0; i<ext->N[istage]; i++) {
    ext->ctime = ts->ptime + h*i;
    ierr = VecCopy(Y,Z);CHKERRQ(ierr);/* Save the solution of the previous substep */
    ierr = SNESSolve(snes,NULL,Y);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(snes,&lits);CHKERRQ(ierr);
    ts->snes_its += its; ts->ksp_its += lits;
    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
    ierr = TSAdaptCheckStage(adapt,ts,ext->ctime,Y,&accept);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_EIMEX(TS ts)
{
  TS_EIMEX        *ext = (TS_EIMEX*)ts->data;
  const PetscInt  ns = ext->nstages;
  Vec             *T=ext->T, Y=ext->Y;

  SNES            snes;
  PetscInt        i,j;
  PetscBool       accept = PETSC_FALSE;
  PetscErrorCode  ierr;
  PetscReal       alpha,local_error,local_error_a,local_error_r;
  PetscFunctionBegin;

  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetType(snes,"ksponly");CHKERRQ(ierr);
  ext->status = TS_STEP_INCOMPLETE;

  ierr = VecCopy(ts->vec_sol,ext->VecSolPrev);CHKERRQ(ierr);

  /* Apply n_j steps of the base method to obtain solutions of T(j,1),1<=j<=s */
  for (j=0; j<ns; j++) {
        ierr = TSStage_EIMEX(ts,j);CHKERRQ(ierr);
        ierr = VecCopy(Y,T[j]);CHKERRQ(ierr);
  }

  for (i=1;i<ns;i++) {
    for (j=i;j<ns;j++) {
      alpha = -(PetscReal)ext->N[j]/ext->N[j-i];
      ierr  = VecAXPBYPCZ(T[Map(j,i,ns)],alpha,1.0,0,T[Map(j,i-1,ns)],T[Map(j-1,i-1,ns)]);/* T[j][i]=alpha*T[j][i-1]+T[j-1][i-1] */CHKERRQ(ierr);
      alpha = 1.0/(1.0 + alpha);
      ierr  = VecScale(T[Map(j,i,ns)],alpha);CHKERRQ(ierr);
    }
  }

  ierr = TSEvaluateStep(ts,ns,ts->vec_sol,NULL);CHKERRQ(ierr);/*update ts solution */

  if (ext->ord_adapt && ext->nstages < ext->max_rows) {
        accept = PETSC_FALSE;
        while (!accept && ext->nstages < ext->max_rows) {
          ierr = TSErrorWeightedNorm(ts,ts->vec_sol,T[Map(ext->nstages-1,ext->nstages-2,ext->nstages)],ts->adapt->wnormtype,&local_error,&local_error_a,&local_error_r);CHKERRQ(ierr);
          accept = (local_error < 1.0)? PETSC_TRUE : PETSC_FALSE;

          if (!accept) {/* add one more stage*/
            ierr = TSStage_EIMEX(ts,ext->nstages);CHKERRQ(ierr);
            ext->nstages++; ext->row_ind++; ext->col_ind++;
            /*T table need to be recycled*/
            ierr = VecDuplicateVecs(ts->vec_sol,(1+ext->nstages)*ext->nstages/2,&ext->T);CHKERRQ(ierr);
            for (i=0; i<ext->nstages-1; i++) {
              for (j=0; j<=i; j++) {
                ierr = VecCopy(T[Map(i,j,ext->nstages-1)],ext->T[Map(i,j,ext->nstages)]);CHKERRQ(ierr);
              }
            }
            ierr = VecDestroyVecs(ext->nstages*(ext->nstages-1)/2,&T);CHKERRQ(ierr);
            T = ext->T; /*reset the pointer*/
            /*recycling finished, store the new solution*/
            ierr = VecCopy(Y,T[ext->nstages-1]);CHKERRQ(ierr);
            /*extrapolation for the newly added stage*/
            for (i=1;i<ext->nstages;i++) {
              alpha = -(PetscReal)ext->N[ext->nstages-1]/ext->N[ext->nstages-1-i];
              ierr  = VecAXPBYPCZ(T[Map(ext->nstages-1,i,ext->nstages)],alpha,1.0,0,T[Map(ext->nstages-1,i-1,ext->nstages)],T[Map(ext->nstages-1-1,i-1,ext->nstages)]);/*T[ext->nstages-1][i]=alpha*T[ext->nstages-1][i-1]+T[ext->nstages-1-1][i-1]*/CHKERRQ(ierr);
              alpha = 1.0/(1.0 + alpha);
              ierr  = VecScale(T[Map(ext->nstages-1,i,ext->nstages)],alpha);CHKERRQ(ierr);
            }
            /*update ts solution */
            ierr = TSEvaluateStep(ts,ext->nstages,ts->vec_sol,NULL);CHKERRQ(ierr);
          }/*end if !accept*/
        }/*end while*/

        if (ext->nstages == ext->max_rows) {
          ierr = PetscInfo(ts,"Max number of rows has been used\n");CHKERRQ(ierr);
        }
  }/*end if ext->ord_adapt*/
  ts->ptime += ts->time_step;
  ext->status = TS_STEP_COMPLETE;

  if (ext->status != TS_STEP_COMPLETE && !ts->reason) ts->reason = TS_DIVERGED_STEP_REJECTED;
  PetscFunctionReturn(0);
}

/* cubic Hermit spline */
static PetscErrorCode TSInterpolate_EIMEX(TS ts,PetscReal itime,Vec X)
{
  TS_EIMEX       *ext = (TS_EIMEX*)ts->data;
  PetscReal      t,a,b;
  Vec            Y0=ext->VecSolPrev,Y1=ext->Y,Ydot=ext->Ydot,YdotI=ext->YdotI;
  const PetscReal h = ts->ptime - ts->ptime_prev;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  t = (itime -ts->ptime + h)/h;
  /* YdotI = -f(x)-g(x) */

  ierr = VecZeroEntries(Ydot);CHKERRQ(ierr);
  ierr = TSComputeIFunction(ts,ts->ptime-h,Y0,Ydot,YdotI,PETSC_FALSE);CHKERRQ(ierr);

  a    = 2.0*t*t*t - 3.0*t*t + 1.0;
  b    = -(t*t*t - 2.0*t*t + t)*h;
  ierr = VecAXPBYPCZ(X,a,b,0.0,Y0,YdotI);CHKERRQ(ierr);

  ierr = TSComputeIFunction(ts,ts->ptime,Y1,Ydot,YdotI,PETSC_FALSE);CHKERRQ(ierr);
  a    = -2.0*t*t*t+3.0*t*t;
  b    = -(t*t*t - t*t)*h;
  ierr = VecAXPBYPCZ(X,a,b,1.0,Y1,YdotI);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_EIMEX(TS ts)
{
  TS_EIMEX        *ext = (TS_EIMEX*)ts->data;
  PetscInt        ns;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ns = ext->nstages;
  ierr = VecDestroyVecs((1+ns)*ns/2,&ext->T);CHKERRQ(ierr);
  ierr = VecDestroy(&ext->Y);CHKERRQ(ierr);
  ierr = VecDestroy(&ext->Z);CHKERRQ(ierr);
  ierr = VecDestroy(&ext->YdotRHS);CHKERRQ(ierr);
  ierr = VecDestroy(&ext->YdotI);CHKERRQ(ierr);
  ierr = VecDestroy(&ext->Ydot);CHKERRQ(ierr);
  ierr = VecDestroy(&ext->VecSolPrev);CHKERRQ(ierr);
  ierr = PetscFree(ext->N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_EIMEX(TS ts)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSReset_EIMEX(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSEIMEXSetMaxRows_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSEIMEXSetRowCol_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSEIMEXSetOrdAdapt_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEIMEXGetVecs(TS ts,DM dm,Vec *Z,Vec *Ydot,Vec *YdotI, Vec *YdotRHS)
{
  TS_EIMEX       *ext = (TS_EIMEX*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Z) {
    if (dm && dm != ts->dm) {
      ierr = DMGetNamedGlobalVector(dm,"TSEIMEX_Z",Z);CHKERRQ(ierr);
    } else *Z = ext->Z;
  }
  if (Ydot) {
    if (dm && dm != ts->dm) {
      ierr = DMGetNamedGlobalVector(dm,"TSEIMEX_Ydot",Ydot);CHKERRQ(ierr);
    } else *Ydot = ext->Ydot;
  }
  if (YdotI) {
    if (dm && dm != ts->dm) {
      ierr = DMGetNamedGlobalVector(dm,"TSEIMEX_YdotI",YdotI);CHKERRQ(ierr);
    } else *YdotI = ext->YdotI;
  }
  if (YdotRHS) {
    if (dm && dm != ts->dm) {
      ierr = DMGetNamedGlobalVector(dm,"TSEIMEX_YdotRHS",YdotRHS);CHKERRQ(ierr);
    } else *YdotRHS = ext->YdotRHS;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEIMEXRestoreVecs(TS ts,DM dm,Vec *Z,Vec *Ydot,Vec *YdotI,Vec *YdotRHS)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Z) {
    if (dm && dm != ts->dm) {
      ierr = DMRestoreNamedGlobalVector(dm,"TSEIMEX_Z",Z);CHKERRQ(ierr);
    }
  }
  if (Ydot) {
    if (dm && dm != ts->dm) {
      ierr = DMRestoreNamedGlobalVector(dm,"TSEIMEX_Ydot",Ydot);CHKERRQ(ierr);
    }
  }
  if (YdotI) {
    if (dm && dm != ts->dm) {
      ierr = DMRestoreNamedGlobalVector(dm,"TSEIMEX_YdotI",YdotI);CHKERRQ(ierr);
    }
  }
  if (YdotRHS) {
    if (dm && dm != ts->dm) {
      ierr = DMRestoreNamedGlobalVector(dm,"TSEIMEX_YdotRHS",YdotRHS);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
  Fn[t0+Theta*dt, U, (U-U0)*shift] = 0
  In the case of Backward Euler, Fn = (U-U0)/h-g(t1,U))
  Since FormIFunction calculates G = ydot - g(t,y), ydot will be set to (U-U0)/h
*/
static PetscErrorCode SNESTSFormFunction_EIMEX(SNES snes,Vec X,Vec G,TS ts)
{
  TS_EIMEX        *ext = (TS_EIMEX*)ts->data;
  PetscErrorCode  ierr;
  Vec             Ydot,Z;
  DM              dm,dmsave;

  PetscFunctionBegin;
  ierr = VecZeroEntries(G);CHKERRQ(ierr);

  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = TSEIMEXGetVecs(ts,dm,&Z,&Ydot,NULL,NULL);CHKERRQ(ierr);
  ierr = VecZeroEntries(Ydot);CHKERRQ(ierr);
  dmsave = ts->dm;
  ts->dm = dm;
  ierr = TSComputeIFunction(ts,ext->ctime,X,Ydot,G,PETSC_FALSE);CHKERRQ(ierr);
  /* PETSC_FALSE indicates non-imex, adding explicit RHS to the implicit I function.  */
  ierr = VecCopy(G,Ydot);CHKERRQ(ierr);
  ts->dm = dmsave;
  ierr = TSEIMEXRestoreVecs(ts,dm,&Z,&Ydot,NULL,NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
 This defined the Jacobian matrix for SNES. Jn = (I/h-g'(t,y))
 */
static PetscErrorCode SNESTSFormJacobian_EIMEX(SNES snes,Vec X,Mat A,Mat B,TS ts)
{
  TS_EIMEX        *ext = (TS_EIMEX*)ts->data;
  Vec             Ydot;
  PetscErrorCode  ierr;
  DM              dm,dmsave;
  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = TSEIMEXGetVecs(ts,dm,NULL,&Ydot,NULL,NULL);CHKERRQ(ierr);
  /*  ierr = VecZeroEntries(Ydot);CHKERRQ(ierr); */
  /* ext->Ydot have already been computed in SNESTSFormFunction_EIMEX (SNES guarantees this) */
  dmsave = ts->dm;
  ts->dm = dm;
  ierr = TSComputeIJacobian(ts,ts->ptime,X,Ydot,ext->shift,A,B,PETSC_TRUE);CHKERRQ(ierr);
  ts->dm = dmsave;
  ierr = TSEIMEXRestoreVecs(ts,dm,NULL,&Ydot,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_TSEIMEX(DM fine,DM coarse,void *ctx)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRestrictHook_TSEIMEX(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  TS ts = (TS)ctx;
  PetscErrorCode ierr;
  Vec Z,Z_c;

  PetscFunctionBegin;
  ierr = TSEIMEXGetVecs(ts,fine,&Z,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = TSEIMEXGetVecs(ts,coarse,&Z_c,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = MatRestrict(restrct,Z,Z_c);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Z_c,rscale,Z_c);CHKERRQ(ierr);
  ierr = TSEIMEXRestoreVecs(ts,fine,&Z,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = TSEIMEXRestoreVecs(ts,coarse,&Z_c,NULL,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_EIMEX(TS ts)
{
  TS_EIMEX       *ext = (TS_EIMEX*)ts->data;
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  if (!ext->N) { /* ext->max_rows not set */
    ierr = TSEIMEXSetMaxRows(ts,TSEIMEXDefault);CHKERRQ(ierr);
  }
  if (-1 == ext->row_ind && -1 == ext->col_ind) {
        ierr = TSEIMEXSetRowCol(ts,ext->max_rows,ext->max_rows);CHKERRQ(ierr);
  } else{/* ext->row_ind and col_ind already set */
    if (ext->ord_adapt) {
      ierr = PetscInfo(ts,"Order adaptivity is enabled and TSEIMEXSetRowCol or -ts_eimex_row_col option will take no effect\n");CHKERRQ(ierr);
    }
  }

  if (ext->ord_adapt) {
    ext->nstages = 2; /* Start with the 2-stage scheme */
    ierr = TSEIMEXSetRowCol(ts,ext->nstages,ext->nstages);CHKERRQ(ierr);
  } else{
    ext->nstages = ext->max_rows; /* by default nstages is the same as max_rows, this can be changed by setting order adaptivity */
  }

  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);

  ierr = VecDuplicateVecs(ts->vec_sol,(1+ext->nstages)*ext->nstages/2,&ext->T);CHKERRQ(ierr);/* full T table */
  ierr = VecDuplicate(ts->vec_sol,&ext->YdotI);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ext->YdotRHS);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ext->Ydot);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ext->VecSolPrev);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ext->Y);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ext->Z);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  if (dm) {
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_TSEIMEX,DMRestrictHook_TSEIMEX,ts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_EIMEX(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_EIMEX       *ext = (TS_EIMEX*)ts->data;
  PetscErrorCode ierr;
  PetscInt       tindex[2];
  PetscInt       np = 2, nrows=TSEIMEXDefault;

  PetscFunctionBegin;
  tindex[0] = TSEIMEXDefault;
  tindex[1] = TSEIMEXDefault;
  ierr = PetscOptionsHead(PetscOptionsObject,"EIMEX ODE solver options");CHKERRQ(ierr);
  {
    PetscBool flg;
    ierr = PetscOptionsInt("-ts_eimex_max_rows","Define the maximum number of rows used","TSEIMEXSetMaxRows",nrows,&nrows,&flg);CHKERRQ(ierr); /* default value 3 */
    if (flg) {
      ierr = TSEIMEXSetMaxRows(ts,nrows);CHKERRQ(ierr);
    }
    ierr = PetscOptionsIntArray("-ts_eimex_row_col","Return the specific term in the T table","TSEIMEXSetRowCol",tindex,&np,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = TSEIMEXSetRowCol(ts,tindex[0],tindex[1]);CHKERRQ(ierr);
    }
    ierr = PetscOptionsBool("-ts_eimex_order_adapt","Solve the problem with adaptive order","TSEIMEXSetOrdAdapt",ext->ord_adapt,&ext->ord_adapt,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_EIMEX(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*@C
  TSEIMEXSetMaxRows - Set the maximum number of rows for EIMEX schemes

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  nrows - maximum number of rows

  Level: intermediate

.seealso: TSEIMEXSetRowCol(), TSEIMEXSetOrdAdapt(), TSEIMEX
@*/
PetscErrorCode TSEIMEXSetMaxRows(TS ts, PetscInt nrows)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSEIMEXSetMaxRows_C",(TS,PetscInt),(ts,nrows));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSEIMEXSetRowCol - Set the type index in the T table for the return value

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  tindex - index in the T table

  Level: intermediate

.seealso: TSEIMEXSetMaxRows(), TSEIMEXSetOrdAdapt(), TSEIMEX
@*/
PetscErrorCode TSEIMEXSetRowCol(TS ts, PetscInt row, PetscInt col)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSEIMEXSetRowCol_C",(TS,PetscInt, PetscInt),(ts,row,col));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSEIMEXSetOrdAdapt - Set the order adaptativity

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  tindex - index in the T table

  Level: intermediate

.seealso: TSEIMEXSetRowCol(), TSEIMEXSetOrdAdapt(), TSEIMEX
@*/
PetscErrorCode TSEIMEXSetOrdAdapt(TS ts, PetscBool flg)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSEIMEXSetOrdAdapt_C",(TS,PetscBool),(ts,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEIMEXSetMaxRows_EIMEX(TS ts,PetscInt nrows)
{
  TS_EIMEX *ext = (TS_EIMEX*)ts->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (nrows < 0 || nrows > 100) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Max number of rows (current value %D) should be an integer number between 1 and 100",nrows);
  ierr = PetscFree(ext->N);CHKERRQ(ierr);
  ext->max_rows = nrows;
  ierr = PetscMalloc1(nrows,&ext->N);CHKERRQ(ierr);
  for (i=0;i<nrows;i++) ext->N[i]=i+1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEIMEXSetRowCol_EIMEX(TS ts,PetscInt row,PetscInt col)
{
  TS_EIMEX *ext = (TS_EIMEX*)ts->data;

  PetscFunctionBegin;
  if (row < 1 || col < 1) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"The row or column index (current value %d,%d) should not be less than 1 ",row,col);
  if (row > ext->max_rows || col > ext->max_rows) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"The row or column index (current value %d,%d) exceeds the maximum number of rows %d",row,col,ext->max_rows);
  if (col > row) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"The column index (%d) exceeds the row index (%d)",col,row);

  ext->row_ind = row - 1;
  ext->col_ind = col - 1; /* Array index in C starts from 0 */
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEIMEXSetOrdAdapt_EIMEX(TS ts,PetscBool flg)
{
  TS_EIMEX *ext = (TS_EIMEX*)ts->data;
  PetscFunctionBegin;
  ext->ord_adapt = flg;
  PetscFunctionReturn(0);
}

/*MC
      TSEIMEX - Time stepping with Extrapolated IMEX methods.

   These methods are intended for problems with well-separated time scales, especially when a slow scale is strongly nonlinear such that it
   is expensive to solve with a fully implicit method. The user should provide the stiff part of the equation using TSSetIFunction() and the
   non-stiff part with TSSetRHSFunction().

   Notes:
  The default is a 3-stage scheme, it can be changed with TSEIMEXSetMaxRows() or -ts_eimex_max_rows

  This method currently only works with ODE, for which the stiff part G(t,X,Xdot) has the form Xdot + Ghat(t,X).

  The general system is written as

  G(t,X,Xdot) = F(t,X)

  where G represents the stiff part and F represents the non-stiff part. The user should provide the stiff part
  of the equation using TSSetIFunction() and the non-stiff part with TSSetRHSFunction().
  This method is designed to be linearly implicit on G and can use an approximate and lagged Jacobian.

  Another common form for the system is

  y'=f(x)+g(x)

  The relationship between F,G and f,g is

  G = y'-g(x), F = f(x)

 References
  E. Constantinescu and A. Sandu, Extrapolated implicit-explicit time stepping, SIAM Journal on Scientific
Computing, 31 (2010), pp. 4452-4477.

      Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSEIMEXSetMaxRows(), TSEIMEXSetRowCol(), TSEIMEXSetOrdAdapt()

 M*/
PETSC_EXTERN PetscErrorCode TSCreate_EIMEX(TS ts)
{
  TS_EIMEX       *ext;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ts->ops->reset          = TSReset_EIMEX;
  ts->ops->destroy        = TSDestroy_EIMEX;
  ts->ops->view           = TSView_EIMEX;
  ts->ops->setup          = TSSetUp_EIMEX;
  ts->ops->step           = TSStep_EIMEX;
  ts->ops->interpolate    = TSInterpolate_EIMEX;
  ts->ops->evaluatestep   = TSEvaluateStep_EIMEX;
  ts->ops->setfromoptions = TSSetFromOptions_EIMEX;
  ts->ops->snesfunction   = SNESTSFormFunction_EIMEX;
  ts->ops->snesjacobian   = SNESTSFormJacobian_EIMEX;
  ts->default_adapt_type  = TSADAPTNONE;

  ts->usessnes = PETSC_TRUE;

  ierr = PetscNewLog(ts,&ext);CHKERRQ(ierr);
  ts->data = (void*)ext;

  ext->ord_adapt = PETSC_FALSE; /* By default, no order adapativity */
  ext->row_ind   = -1;
  ext->col_ind   = -1;
  ext->max_rows  = TSEIMEXDefault;
  ext->nstages   = TSEIMEXDefault;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSEIMEXSetMaxRows_C", TSEIMEXSetMaxRows_EIMEX);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSEIMEXSetRowCol_C",  TSEIMEXSetRowCol_EIMEX);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSEIMEXSetOrdAdapt_C",TSEIMEXSetOrdAdapt_EIMEX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
