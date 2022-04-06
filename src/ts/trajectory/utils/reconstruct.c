#include <petsc/private/tshistoryimpl.h>
#include <petscts.h>

/* these two functions have been stolen from bdf.c */
static inline void LagrangeBasisVals(PetscInt n,PetscReal t,const PetscReal T[],PetscScalar L[])
{
  PetscInt k,j;
  for (k=0; k<n; k++) {
    for (L[k]=1, j=0; j<n; j++) {
      if (j != k) L[k] *= (t - T[j])/(T[k] - T[j]);
    }
  }
}

static inline void LagrangeBasisDers(PetscInt n,PetscReal t,const PetscReal T[],PetscScalar dL[])
{
  PetscInt k,j,i;
  for (k=0; k<n; k++) {
    for (dL[k]=0, j=0; j<n; j++) {
      if (j != k) {
        PetscReal L = 1/(T[k] - T[j]);
        for (i=0; i<n; i++) {
          if (i != j && i != k) L *= (t - T[i])/(T[k] - T[i]);
        }
        dL[k] += L;
      }
    }
  }
}

static inline PetscInt LagrangeGetId(PetscReal t, PetscInt n, const PetscReal T[], const PetscBool Taken[])
{
  PetscInt _tid = 0;
  while (_tid < n && PetscAbsReal(t-T[_tid]) > PETSC_SMALL) _tid++;
  if (_tid < n && !Taken[_tid]) {
    return _tid;
  } else { /* we get back a negative id, where the maximum time is stored, since we use usually reconstruct backward in time */
    PetscReal max = PETSC_MIN_REAL;
    PetscInt  maxloc = n;
    _tid = 0;
    while (_tid < n) { maxloc = (max < T[_tid] && !Taken[_tid]) ? (max = T[_tid],_tid) : maxloc; _tid++; }
    return -maxloc-1;
  }
}

PetscErrorCode TSTrajectoryReconstruct_Private(TSTrajectory tj,TS ts,PetscReal t,Vec U,Vec Udot)
{
  TSHistory       tsh = tj->tsh;
  const PetscReal *tshhist;
  const PetscInt  *tshhist_id;
  PetscInt        id, cnt, i, tshn;

  PetscFunctionBegin;
  PetscCall(TSHistoryGetLocFromTime(tsh,t,&id));
  PetscCall(TSHistoryGetHistory(tsh,&tshn,&tshhist,&tshhist_id,NULL));
  if (id == -1 || id == -tshn - 1) {
    PetscReal t0 = tshn ? tshhist[0]      : 0.0;
    PetscReal tf = tshn ? tshhist[tshn-1] : 0.0;
    SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_PLIB,"Requested time %g is outside the history interval [%g, %g] (%" PetscInt_FMT ")",(double)t,(double)t0,(double)tf,tshn);
  }
  if (tj->monitor) {
    PetscCall(PetscViewerASCIIPrintf(tj->monitor,"Reconstructing at time %g, order %" PetscInt_FMT "\n",(double)t,tj->lag.order));
  }
  if (!tj->lag.T) {
    PetscInt o = tj->lag.order+1;
    PetscCall(PetscMalloc5(o,&tj->lag.L,o,&tj->lag.T,o,&tj->lag.WW,2*o,&tj->lag.TT,o,&tj->lag.TW));
    for (i = 0; i < o; i++) tj->lag.T[i] = PETSC_MAX_REAL;
    PetscCall(VecDuplicateVecs(U ? U : Udot,o,&tj->lag.W));
  }
  cnt = 0;
  PetscCall(PetscArrayzero(tj->lag.TT,2*(tj->lag.order+1)));
  if (id < 0 || Udot) { /* populate snapshots for interpolation */
    PetscInt s,nid = id < 0 ? -(id+1) : id;

    PetscInt up = PetscMin(nid + tj->lag.order/2+1,tshn);
    PetscInt low = PetscMax(up-tj->lag.order-1,0);
    up = PetscMin(PetscMax(low + tj->lag.order + 1,up),tshn);
    if (tj->monitor) {
      PetscCall(PetscViewerASCIIPushTab(tj->monitor));
    }

    /* first see if we can reuse any */
    for (s = up-1; s >= low; s--) {
      PetscReal t = tshhist[s];
      PetscInt tid = LagrangeGetId(t,tj->lag.order+1,tj->lag.T,tj->lag.TT);
      if (tid < 0) continue;
      if (tj->monitor) {
        PetscCall(PetscViewerASCIIPrintf(tj->monitor,"Reusing snapshot %" PetscInt_FMT ", step %" PetscInt_FMT ", time %g\n",tid,tshhist_id[s],(double)t));
      }
      tj->lag.TT[tid] = PETSC_TRUE;
      tj->lag.WW[cnt] = tj->lag.W[tid];
      tj->lag.TW[cnt] = t;
      tj->lag.TT[tj->lag.order+1 + s-low] = PETSC_TRUE; /* tell the next loop to skip it */
      cnt++;
    }

    /* now load the missing ones */
    for (s = up-1; s >= low; s--) {
      PetscReal t = tshhist[s];
      PetscInt tid;

      if (tj->lag.TT[tj->lag.order+1 + s-low]) continue;
      tid = LagrangeGetId(t,tj->lag.order+1,tj->lag.T,tj->lag.TT);
      PetscCheck(tid < 0,PetscObjectComm((PetscObject)tj),PETSC_ERR_PLIB,"This should not happen");
      tid = -tid-1;
      if (tj->monitor) {
        if (tj->lag.T[tid] < PETSC_MAX_REAL) {
          PetscCall(PetscViewerASCIIPrintf(tj->monitor,"Discarding snapshot %" PetscInt_FMT " at time %g\n",tid,(double)tj->lag.T[tid]));
        } else {
          PetscCall(PetscViewerASCIIPrintf(tj->monitor,"New snapshot %" PetscInt_FMT "\n",tid));
        }
        PetscCall(PetscViewerASCIIPushTab(tj->monitor));
      }
      PetscCall(TSTrajectoryGetVecs(tj,ts,tshhist_id[s],&t,tj->lag.W[tid],NULL));
      tj->lag.T[tid] = t;
      if (tj->monitor) {
        PetscCall(PetscViewerASCIIPopTab(tj->monitor));
      }
      tj->lag.TT[tid] = PETSC_TRUE;
      tj->lag.WW[cnt] = tj->lag.W[tid];
      tj->lag.TW[cnt] = t;
      tj->lag.TT[tj->lag.order+1 + s-low] = PETSC_TRUE;
      cnt++;
    }
    if (tj->monitor) {
      PetscCall(PetscViewerASCIIPopTab(tj->monitor));
    }
  }
  PetscCall(PetscArrayzero(tj->lag.TT,tj->lag.order+1));
  if (id >=0 && U) { /* requested time match */
    PetscInt tid = LagrangeGetId(t,tj->lag.order+1,tj->lag.T,tj->lag.TT);
    if (tj->monitor) {
      PetscCall(PetscViewerASCIIPrintf(tj->monitor,"Retrieving solution from exact step\n"));
      PetscCall(PetscViewerASCIIPushTab(tj->monitor));
    }
    if (tid < 0) {
      tid = -tid-1;
      if (tj->monitor) {
        if (tj->lag.T[tid] < PETSC_MAX_REAL) {
          PetscCall(PetscViewerASCIIPrintf(tj->monitor,"Discarding snapshot %" PetscInt_FMT " at time %g\n",tid,(double)tj->lag.T[tid]));
        } else {
          PetscCall(PetscViewerASCIIPrintf(tj->monitor,"New snapshot %" PetscInt_FMT "\n",tid));
        }
        PetscCall(PetscViewerASCIIPushTab(tj->monitor));
      }
      PetscCall(TSTrajectoryGetVecs(tj,ts,tshhist_id[id],&t,tj->lag.W[tid],NULL));
      if (tj->monitor) {
        PetscCall(PetscViewerASCIIPopTab(tj->monitor));
      }
      tj->lag.T[tid] = t;
    } else if (tj->monitor) {
      PetscCall(PetscViewerASCIIPrintf(tj->monitor,"Reusing snapshot %" PetscInt_FMT " step %" PetscInt_FMT ", time %g\n",tid,tshhist_id[id],(double)t));
    }
    PetscCall(VecCopy(tj->lag.W[tid],U));
    PetscCall(PetscObjectStateGet((PetscObject)U,&tj->lag.Ucached.state));
    PetscCall(PetscObjectGetId((PetscObject)U,&tj->lag.Ucached.id));
    tj->lag.Ucached.time = t;
    tj->lag.Ucached.step = tshhist_id[id];
    if (tj->monitor) {
      PetscCall(PetscViewerASCIIPopTab(tj->monitor));
    }
  }
  if (id < 0 && U) {
    if (tj->monitor) {
      PetscCall(PetscViewerASCIIPrintf(tj->monitor,"Interpolating solution with %" PetscInt_FMT " snapshots\n",cnt));
    }
    LagrangeBasisVals(cnt,t,tj->lag.TW,tj->lag.L);
    PetscCall(VecZeroEntries(U));
    PetscCall(VecMAXPY(U,cnt,tj->lag.L,tj->lag.WW));
    PetscCall(PetscObjectStateGet((PetscObject)U,&tj->lag.Ucached.state));
    PetscCall(PetscObjectGetId((PetscObject)U,&tj->lag.Ucached.id));
    tj->lag.Ucached.time = t;
    tj->lag.Ucached.step = PETSC_MIN_INT;
  }
  if (Udot) {
    if (tj->monitor) {
      PetscCall(PetscViewerASCIIPrintf(tj->monitor,"Interpolating derivative with %" PetscInt_FMT " snapshots\n",cnt));
    }
    LagrangeBasisDers(cnt,t,tj->lag.TW,tj->lag.L);
    PetscCall(VecZeroEntries(Udot));
    PetscCall(VecMAXPY(Udot,cnt,tj->lag.L,tj->lag.WW));
    PetscCall(PetscObjectStateGet((PetscObject)Udot,&tj->lag.Udotcached.state));
    PetscCall(PetscObjectGetId((PetscObject)Udot,&tj->lag.Udotcached.id));
    tj->lag.Udotcached.time = t;
    tj->lag.Udotcached.step = PETSC_MIN_INT;
  }
  PetscFunctionReturn(0);
}
