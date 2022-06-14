
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
  PetscFunctionBegin;
  PetscCall(VecCopy(ext->T[Map(ext->row_ind,ext->col_ind,ns)],X));
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

  PetscFunctionBegin;
  PetscCall(TSGetSNES(ts,&snes));
  h = ts->time_step/ext->N[istage];/* step size for the istage-th stage */
  ext->shift = 1./h;
  PetscCall(SNESSetLagJacobian(snes,-2)); /* Recompute the Jacobian on this solve, but not again */
  PetscCall(VecCopy(ext->VecSolPrev,Y)); /* Take the previous solution as initial step */

  for (i=0; i<ext->N[istage]; i++) {
    ext->ctime = ts->ptime + h*i;
    PetscCall(VecCopy(Y,Z));/* Save the solution of the previous substep */
    PetscCall(SNESSolve(snes,NULL,Y));
    PetscCall(SNESGetIterationNumber(snes,&its));
    PetscCall(SNESGetLinearSolveIterations(snes,&lits));
    ts->snes_its += its; ts->ksp_its += lits;
    PetscCall(TSGetAdapt(ts,&adapt));
    PetscCall(TSAdaptCheckStage(adapt,ts,ext->ctime,Y,&accept));
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
  PetscReal       alpha,local_error,local_error_a,local_error_r;

  PetscFunctionBegin;
  PetscCall(TSGetSNES(ts,&snes));
  PetscCall(SNESSetType(snes,"ksponly"));
  ext->status = TS_STEP_INCOMPLETE;

  PetscCall(VecCopy(ts->vec_sol,ext->VecSolPrev));

  /* Apply n_j steps of the base method to obtain solutions of T(j,1),1<=j<=s */
  for (j=0; j<ns; j++) {
        PetscCall(TSStage_EIMEX(ts,j));
        PetscCall(VecCopy(Y,T[j]));
  }

  for (i=1;i<ns;i++) {
    for (j=i;j<ns;j++) {
      alpha = -(PetscReal)ext->N[j]/ext->N[j-i];
      PetscCall(VecAXPBYPCZ(T[Map(j,i,ns)],alpha,1.0,0,T[Map(j,i-1,ns)],T[Map(j-1,i-1,ns)]));/* T[j][i]=alpha*T[j][i-1]+T[j-1][i-1] */
      alpha = 1.0/(1.0 + alpha);
      PetscCall(VecScale(T[Map(j,i,ns)],alpha));
    }
  }

  PetscCall(TSEvaluateStep(ts,ns,ts->vec_sol,NULL));/*update ts solution */

  if (ext->ord_adapt && ext->nstages < ext->max_rows) {
        accept = PETSC_FALSE;
        while (!accept && ext->nstages < ext->max_rows) {
          PetscCall(TSErrorWeightedNorm(ts,ts->vec_sol,T[Map(ext->nstages-1,ext->nstages-2,ext->nstages)],ts->adapt->wnormtype,&local_error,&local_error_a,&local_error_r));
          accept = (local_error < 1.0)? PETSC_TRUE : PETSC_FALSE;

          if (!accept) {/* add one more stage*/
            PetscCall(TSStage_EIMEX(ts,ext->nstages));
            ext->nstages++; ext->row_ind++; ext->col_ind++;
            /*T table need to be recycled*/
            PetscCall(VecDuplicateVecs(ts->vec_sol,(1+ext->nstages)*ext->nstages/2,&ext->T));
            for (i=0; i<ext->nstages-1; i++) {
              for (j=0; j<=i; j++) {
                PetscCall(VecCopy(T[Map(i,j,ext->nstages-1)],ext->T[Map(i,j,ext->nstages)]));
              }
            }
            PetscCall(VecDestroyVecs(ext->nstages*(ext->nstages-1)/2,&T));
            T = ext->T; /*reset the pointer*/
            /*recycling finished, store the new solution*/
            PetscCall(VecCopy(Y,T[ext->nstages-1]));
            /*extrapolation for the newly added stage*/
            for (i=1;i<ext->nstages;i++) {
              alpha = -(PetscReal)ext->N[ext->nstages-1]/ext->N[ext->nstages-1-i];
              PetscCall(VecAXPBYPCZ(T[Map(ext->nstages-1,i,ext->nstages)],alpha,1.0,0,T[Map(ext->nstages-1,i-1,ext->nstages)],T[Map(ext->nstages-1-1,i-1,ext->nstages)]));/*T[ext->nstages-1][i]=alpha*T[ext->nstages-1][i-1]+T[ext->nstages-1-1][i-1]*/
              alpha = 1.0/(1.0 + alpha);
              PetscCall(VecScale(T[Map(ext->nstages-1,i,ext->nstages)],alpha));
            }
            /*update ts solution */
            PetscCall(TSEvaluateStep(ts,ext->nstages,ts->vec_sol,NULL));
          }/*end if !accept*/
        }/*end while*/

        if (ext->nstages == ext->max_rows) {
          PetscCall(PetscInfo(ts,"Max number of rows has been used\n"));
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
  PetscFunctionBegin;
  t = (itime -ts->ptime + h)/h;
  /* YdotI = -f(x)-g(x) */

  PetscCall(VecZeroEntries(Ydot));
  PetscCall(TSComputeIFunction(ts,ts->ptime-h,Y0,Ydot,YdotI,PETSC_FALSE));

  a    = 2.0*t*t*t - 3.0*t*t + 1.0;
  b    = -(t*t*t - 2.0*t*t + t)*h;
  PetscCall(VecAXPBYPCZ(X,a,b,0.0,Y0,YdotI));

  PetscCall(TSComputeIFunction(ts,ts->ptime,Y1,Ydot,YdotI,PETSC_FALSE));
  a    = -2.0*t*t*t+3.0*t*t;
  b    = -(t*t*t - t*t)*h;
  PetscCall(VecAXPBYPCZ(X,a,b,1.0,Y1,YdotI));

  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_EIMEX(TS ts)
{
  TS_EIMEX        *ext = (TS_EIMEX*)ts->data;
  PetscInt        ns;

  PetscFunctionBegin;
  ns = ext->nstages;
  PetscCall(VecDestroyVecs((1+ns)*ns/2,&ext->T));
  PetscCall(VecDestroy(&ext->Y));
  PetscCall(VecDestroy(&ext->Z));
  PetscCall(VecDestroy(&ext->YdotRHS));
  PetscCall(VecDestroy(&ext->YdotI));
  PetscCall(VecDestroy(&ext->Ydot));
  PetscCall(VecDestroy(&ext->VecSolPrev));
  PetscCall(PetscFree(ext->N));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_EIMEX(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSReset_EIMEX(ts));
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSEIMEXSetMaxRows_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSEIMEXSetRowCol_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSEIMEXSetOrdAdapt_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEIMEXGetVecs(TS ts,DM dm,Vec *Z,Vec *Ydot,Vec *YdotI, Vec *YdotRHS)
{
  TS_EIMEX       *ext = (TS_EIMEX*)ts->data;

  PetscFunctionBegin;
  if (Z) {
    if (dm && dm != ts->dm) {
      PetscCall(DMGetNamedGlobalVector(dm,"TSEIMEX_Z",Z));
    } else *Z = ext->Z;
  }
  if (Ydot) {
    if (dm && dm != ts->dm) {
      PetscCall(DMGetNamedGlobalVector(dm,"TSEIMEX_Ydot",Ydot));
    } else *Ydot = ext->Ydot;
  }
  if (YdotI) {
    if (dm && dm != ts->dm) {
      PetscCall(DMGetNamedGlobalVector(dm,"TSEIMEX_YdotI",YdotI));
    } else *YdotI = ext->YdotI;
  }
  if (YdotRHS) {
    if (dm && dm != ts->dm) {
      PetscCall(DMGetNamedGlobalVector(dm,"TSEIMEX_YdotRHS",YdotRHS));
    } else *YdotRHS = ext->YdotRHS;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEIMEXRestoreVecs(TS ts,DM dm,Vec *Z,Vec *Ydot,Vec *YdotI,Vec *YdotRHS)
{
  PetscFunctionBegin;
  if (Z) {
    if (dm && dm != ts->dm) {
      PetscCall(DMRestoreNamedGlobalVector(dm,"TSEIMEX_Z",Z));
    }
  }
  if (Ydot) {
    if (dm && dm != ts->dm) {
      PetscCall(DMRestoreNamedGlobalVector(dm,"TSEIMEX_Ydot",Ydot));
    }
  }
  if (YdotI) {
    if (dm && dm != ts->dm) {
      PetscCall(DMRestoreNamedGlobalVector(dm,"TSEIMEX_YdotI",YdotI));
    }
  }
  if (YdotRHS) {
    if (dm && dm != ts->dm) {
      PetscCall(DMRestoreNamedGlobalVector(dm,"TSEIMEX_YdotRHS",YdotRHS));
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
  Vec             Ydot,Z;
  DM              dm,dmsave;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(G));

  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(TSEIMEXGetVecs(ts,dm,&Z,&Ydot,NULL,NULL));
  PetscCall(VecZeroEntries(Ydot));
  dmsave = ts->dm;
  ts->dm = dm;
  PetscCall(TSComputeIFunction(ts,ext->ctime,X,Ydot,G,PETSC_FALSE));
  /* PETSC_FALSE indicates non-imex, adding explicit RHS to the implicit I function.  */
  PetscCall(VecCopy(G,Ydot));
  ts->dm = dmsave;
  PetscCall(TSEIMEXRestoreVecs(ts,dm,&Z,&Ydot,NULL,NULL));

  PetscFunctionReturn(0);
}

/*
 This defined the Jacobian matrix for SNES. Jn = (I/h-g'(t,y))
 */
static PetscErrorCode SNESTSFormJacobian_EIMEX(SNES snes,Vec X,Mat A,Mat B,TS ts)
{
  TS_EIMEX        *ext = (TS_EIMEX*)ts->data;
  Vec             Ydot;
  DM              dm,dmsave;
  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(TSEIMEXGetVecs(ts,dm,NULL,&Ydot,NULL,NULL));
  /*  PetscCall(VecZeroEntries(Ydot)); */
  /* ext->Ydot have already been computed in SNESTSFormFunction_EIMEX (SNES guarantees this) */
  dmsave = ts->dm;
  ts->dm = dm;
  PetscCall(TSComputeIJacobian(ts,ts->ptime,X,Ydot,ext->shift,A,B,PETSC_TRUE));
  ts->dm = dmsave;
  PetscCall(TSEIMEXRestoreVecs(ts,dm,NULL,&Ydot,NULL,NULL));
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
  Vec Z,Z_c;

  PetscFunctionBegin;
  PetscCall(TSEIMEXGetVecs(ts,fine,&Z,NULL,NULL,NULL));
  PetscCall(TSEIMEXGetVecs(ts,coarse,&Z_c,NULL,NULL,NULL));
  PetscCall(MatRestrict(restrct,Z,Z_c));
  PetscCall(VecPointwiseMult(Z_c,rscale,Z_c));
  PetscCall(TSEIMEXRestoreVecs(ts,fine,&Z,NULL,NULL,NULL));
  PetscCall(TSEIMEXRestoreVecs(ts,coarse,&Z_c,NULL,NULL,NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_EIMEX(TS ts)
{
  TS_EIMEX       *ext = (TS_EIMEX*)ts->data;
  DM             dm;

  PetscFunctionBegin;
  if (!ext->N) { /* ext->max_rows not set */
    PetscCall(TSEIMEXSetMaxRows(ts,TSEIMEXDefault));
  }
  if (-1 == ext->row_ind && -1 == ext->col_ind) {
        PetscCall(TSEIMEXSetRowCol(ts,ext->max_rows,ext->max_rows));
  } else{/* ext->row_ind and col_ind already set */
    if (ext->ord_adapt) {
      PetscCall(PetscInfo(ts,"Order adaptivity is enabled and TSEIMEXSetRowCol or -ts_eimex_row_col option will take no effect\n"));
    }
  }

  if (ext->ord_adapt) {
    ext->nstages = 2; /* Start with the 2-stage scheme */
    PetscCall(TSEIMEXSetRowCol(ts,ext->nstages,ext->nstages));
  } else{
    ext->nstages = ext->max_rows; /* by default nstages is the same as max_rows, this can be changed by setting order adaptivity */
  }

  PetscCall(TSGetAdapt(ts,&ts->adapt));

  PetscCall(VecDuplicateVecs(ts->vec_sol,(1+ext->nstages)*ext->nstages/2,&ext->T));/* full T table */
  PetscCall(VecDuplicate(ts->vec_sol,&ext->YdotI));
  PetscCall(VecDuplicate(ts->vec_sol,&ext->YdotRHS));
  PetscCall(VecDuplicate(ts->vec_sol,&ext->Ydot));
  PetscCall(VecDuplicate(ts->vec_sol,&ext->VecSolPrev));
  PetscCall(VecDuplicate(ts->vec_sol,&ext->Y));
  PetscCall(VecDuplicate(ts->vec_sol,&ext->Z));
  PetscCall(TSGetDM(ts,&dm));
  if (dm) PetscCall(DMCoarsenHookAdd(dm,DMCoarsenHook_TSEIMEX,DMRestrictHook_TSEIMEX,ts));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_EIMEX(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_EIMEX       *ext = (TS_EIMEX*)ts->data;
  PetscInt       tindex[2];
  PetscInt       np = 2, nrows=TSEIMEXDefault;

  PetscFunctionBegin;
  tindex[0] = TSEIMEXDefault;
  tindex[1] = TSEIMEXDefault;
  PetscOptionsHeadBegin(PetscOptionsObject,"EIMEX ODE solver options");
  {
    PetscBool flg;
    PetscCall(PetscOptionsInt("-ts_eimex_max_rows","Define the maximum number of rows used","TSEIMEXSetMaxRows",nrows,&nrows,&flg)); /* default value 3 */
    if (flg) PetscCall(TSEIMEXSetMaxRows(ts,nrows));
    PetscCall(PetscOptionsIntArray("-ts_eimex_row_col","Return the specific term in the T table","TSEIMEXSetRowCol",tindex,&np,&flg));
    if (flg) {
      PetscCall(TSEIMEXSetRowCol(ts,tindex[0],tindex[1]));
    }
    PetscCall(PetscOptionsBool("-ts_eimex_order_adapt","Solve the problem with adaptive order","TSEIMEXSetOrdAdapt",ext->ord_adapt,&ext->ord_adapt,NULL));
  }
  PetscOptionsHeadEnd();
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

.seealso: `TSEIMEXSetRowCol()`, `TSEIMEXSetOrdAdapt()`, `TSEIMEX`
@*/
PetscErrorCode TSEIMEXSetMaxRows(TS ts, PetscInt nrows)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscTryMethod(ts,"TSEIMEXSetMaxRows_C",(TS,PetscInt),(ts,nrows));
  PetscFunctionReturn(0);
}

/*@C
  TSEIMEXSetRowCol - Set the type index in the T table for the return value

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  tindex - index in the T table

  Level: intermediate

.seealso: `TSEIMEXSetMaxRows()`, `TSEIMEXSetOrdAdapt()`, `TSEIMEX`
@*/
PetscErrorCode TSEIMEXSetRowCol(TS ts, PetscInt row, PetscInt col)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscTryMethod(ts,"TSEIMEXSetRowCol_C",(TS,PetscInt, PetscInt),(ts,row,col));
  PetscFunctionReturn(0);
}

/*@C
  TSEIMEXSetOrdAdapt - Set the order adaptativity

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  tindex - index in the T table

  Level: intermediate

.seealso: `TSEIMEXSetRowCol()`, `TSEIMEXSetOrdAdapt()`, `TSEIMEX`
@*/
PetscErrorCode TSEIMEXSetOrdAdapt(TS ts, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscTryMethod(ts,"TSEIMEXSetOrdAdapt_C",(TS,PetscBool),(ts,flg));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEIMEXSetMaxRows_EIMEX(TS ts,PetscInt nrows)
{
  TS_EIMEX *ext = (TS_EIMEX*)ts->data;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheck(nrows >= 0 && nrows <= 100,((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Max number of rows (current value %" PetscInt_FMT ") should be an integer number between 1 and 100",nrows);
  PetscCall(PetscFree(ext->N));
  ext->max_rows = nrows;
  PetscCall(PetscMalloc1(nrows,&ext->N));
  for (i=0;i<nrows;i++) ext->N[i]=i+1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEIMEXSetRowCol_EIMEX(TS ts,PetscInt row,PetscInt col)
{
  TS_EIMEX *ext = (TS_EIMEX*)ts->data;

  PetscFunctionBegin;
  PetscCheck(row >= 1 && col >= 1,((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"The row or column index (current value %" PetscInt_FMT ",%" PetscInt_FMT ") should not be less than 1 ",row,col);
  PetscCheck(row <= ext->max_rows && col <= ext->max_rows,((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"The row or column index (current value %" PetscInt_FMT ",%" PetscInt_FMT ") exceeds the maximum number of rows %" PetscInt_FMT,row,col,ext->max_rows);
  PetscCheck(col <= row,((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"The column index (%" PetscInt_FMT ") exceeds the row index (%" PetscInt_FMT ")",col,row);

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

.seealso: `TSCreate()`, `TS`, `TSSetType()`, `TSEIMEXSetMaxRows()`, `TSEIMEXSetRowCol()`, `TSEIMEXSetOrdAdapt()`

 M*/
PETSC_EXTERN PetscErrorCode TSCreate_EIMEX(TS ts)
{
  TS_EIMEX       *ext;

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

  PetscCall(PetscNewLog(ts,&ext));
  ts->data = (void*)ext;

  ext->ord_adapt = PETSC_FALSE; /* By default, no order adapativity */
  ext->row_ind   = -1;
  ext->col_ind   = -1;
  ext->max_rows  = TSEIMEXDefault;
  ext->nstages   = TSEIMEXDefault;

  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSEIMEXSetMaxRows_C", TSEIMEXSetMaxRows_EIMEX));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSEIMEXSetRowCol_C",  TSEIMEXSetRowCol_EIMEX));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSEIMEXSetOrdAdapt_C",TSEIMEXSetOrdAdapt_EIMEX));
  PetscFunctionReturn(0);
}
