#include <petsc/private/tshistoryimpl.h>
#include <petscts.h>

/* these two functions have been stolen from bdf.c */
PETSC_STATIC_INLINE void LagrangeBasisVals(PetscInt n,PetscReal t,const PetscReal T[],PetscScalar L[])
{
  PetscInt k,j;
  for (k=0; k<n; k++) {
    for (L[k]=1, j=0; j<n; j++) {
      if (j != k) L[k] *= (t - T[j])/(T[k] - T[j]);
    }
  }
}

PETSC_STATIC_INLINE void LagrangeBasisDers(PetscInt n,PetscReal t,const PetscReal T[],PetscScalar dL[])
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

PETSC_STATIC_INLINE PetscInt LagrangeGetId(PetscReal t, PetscInt n, const PetscReal T[], const PetscBool Taken[])
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSHistoryGetLocFromTime(tsh,t,&id);CHKERRQ(ierr);
  ierr = TSHistoryGetHistory(tsh,&tshn,&tshhist,&tshhist_id,NULL);CHKERRQ(ierr);
  if (id == -1 || id == -tshn - 1) {
    PetscReal t0 = tshn ? tshhist[0]      : 0.0;
    PetscReal tf = tshn ? tshhist[tshn-1] : 0.0;
    SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_PLIB,"Requested time %g is outside the history interval [%g, %g] (%d)",(double)t,(double)t0,(double)tf,tshn);
  }
  if (tj->monitor) {
    ierr = PetscViewerASCIIPrintf(tj->monitor,"Reconstructing at time %g, order %D\n",(double)t,tj->lag.order);CHKERRQ(ierr);
  }
  if (!tj->lag.T) {
    PetscInt o = tj->lag.order+1;
    ierr = PetscMalloc5(o,&tj->lag.L,o,&tj->lag.T,o,&tj->lag.WW,2*o,&tj->lag.TT,o,&tj->lag.TW);CHKERRQ(ierr);
    for (i = 0; i < o; i++) tj->lag.T[i] = PETSC_MAX_REAL;
    ierr = VecDuplicateVecs(U ? U : Udot,o,&tj->lag.W);CHKERRQ(ierr);
  }
  cnt = 0;
  ierr = PetscArrayzero(tj->lag.TT,2*(tj->lag.order+1));CHKERRQ(ierr);
  if (id < 0 || Udot) { /* populate snapshots for interpolation */
    PetscInt s,nid = id < 0 ? -(id+1) : id;

    PetscInt up = PetscMin(nid + tj->lag.order/2+1,tshn);
    PetscInt low = PetscMax(up-tj->lag.order-1,0);
    up = PetscMin(PetscMax(low + tj->lag.order + 1,up),tshn);
    if (tj->monitor) {
      ierr = PetscViewerASCIIPushTab(tj->monitor);CHKERRQ(ierr);
    }

    /* first see if we can reuse any */
    for (s = up-1; s >= low; s--) {
      PetscReal t = tshhist[s];
      PetscInt tid = LagrangeGetId(t,tj->lag.order+1,tj->lag.T,tj->lag.TT);
      if (tid < 0) continue;
      if (tj->monitor) {
        ierr = PetscViewerASCIIPrintf(tj->monitor,"Reusing snapshot %D, step %D, time %g\n",tid,tshhist_id[s],(double)t);CHKERRQ(ierr);
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
      if (tid >= 0) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_PLIB,"This should not happen");
      tid = -tid-1;
      if (tj->monitor) {
        if (tj->lag.T[tid] < PETSC_MAX_REAL) {
          ierr = PetscViewerASCIIPrintf(tj->monitor,"Discarding snapshot %D at time %g\n",tid,(double)tj->lag.T[tid]);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(tj->monitor,"New snapshot %D\n",tid);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPushTab(tj->monitor);CHKERRQ(ierr);
      }
      ierr = TSTrajectoryGetVecs(tj,ts,tshhist_id[s],&t,tj->lag.W[tid],NULL);CHKERRQ(ierr);
      tj->lag.T[tid] = t;
      if (tj->monitor) {
        ierr = PetscViewerASCIIPopTab(tj->monitor);CHKERRQ(ierr);
      }
      tj->lag.TT[tid] = PETSC_TRUE;
      tj->lag.WW[cnt] = tj->lag.W[tid];
      tj->lag.TW[cnt] = t;
      tj->lag.TT[tj->lag.order+1 + s-low] = PETSC_TRUE;
      cnt++;
    }
    if (tj->monitor) {
      ierr = PetscViewerASCIIPopTab(tj->monitor);CHKERRQ(ierr);
    }
  }
  ierr = PetscArrayzero(tj->lag.TT,tj->lag.order+1);CHKERRQ(ierr);
  if (id >=0 && U) { /* requested time match */
    PetscInt tid = LagrangeGetId(t,tj->lag.order+1,tj->lag.T,tj->lag.TT);
    if (tj->monitor) {
      ierr = PetscViewerASCIIPrintf(tj->monitor,"Retrieving solution from exact step\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(tj->monitor);CHKERRQ(ierr);
    }
    if (tid < 0) {
      tid = -tid-1;
      if (tj->monitor) {
        if (tj->lag.T[tid] < PETSC_MAX_REAL) {
          ierr = PetscViewerASCIIPrintf(tj->monitor,"Discarding snapshot %D at time %g\n",tid,(double)tj->lag.T[tid]);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(tj->monitor,"New snapshot %D\n",tid);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPushTab(tj->monitor);CHKERRQ(ierr);
      }
      ierr = TSTrajectoryGetVecs(tj,ts,tshhist_id[id],&t,tj->lag.W[tid],NULL);CHKERRQ(ierr);
      if (tj->monitor) {
        ierr = PetscViewerASCIIPopTab(tj->monitor);CHKERRQ(ierr);
      }
      tj->lag.T[tid] = t;
    } else if (tj->monitor) {
      ierr = PetscViewerASCIIPrintf(tj->monitor,"Reusing snapshot %D step %D, time %g\n",tid,tshhist_id[id],(double)t);CHKERRQ(ierr);
    }
    ierr = VecCopy(tj->lag.W[tid],U);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)U,&tj->lag.Ucached.state);CHKERRQ(ierr);
    ierr = PetscObjectGetId((PetscObject)U,&tj->lag.Ucached.id);CHKERRQ(ierr);
    tj->lag.Ucached.time = t;
    tj->lag.Ucached.step = tshhist_id[id];
    if (tj->monitor) {
      ierr = PetscViewerASCIIPopTab(tj->monitor);CHKERRQ(ierr);
    }
  }
  if (id < 0 && U) {
    if (tj->monitor) {
      ierr = PetscViewerASCIIPrintf(tj->monitor,"Interpolating solution with %D snapshots\n",cnt);CHKERRQ(ierr);
    }
    LagrangeBasisVals(cnt,t,tj->lag.TW,tj->lag.L);
    ierr = VecZeroEntries(U);CHKERRQ(ierr);
    ierr = VecMAXPY(U,cnt,tj->lag.L,tj->lag.WW);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)U,&tj->lag.Ucached.state);CHKERRQ(ierr);
    ierr = PetscObjectGetId((PetscObject)U,&tj->lag.Ucached.id);CHKERRQ(ierr);
    tj->lag.Ucached.time = t;
    tj->lag.Ucached.step = PETSC_MIN_INT;
  }
  if (Udot) {
    if (tj->monitor) {
      ierr = PetscViewerASCIIPrintf(tj->monitor,"Interpolating derivative with %D snapshots\n",cnt);CHKERRQ(ierr);
    }
    LagrangeBasisDers(cnt,t,tj->lag.TW,tj->lag.L);
    ierr = VecZeroEntries(Udot);CHKERRQ(ierr);
    ierr = VecMAXPY(Udot,cnt,tj->lag.L,tj->lag.WW);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)Udot,&tj->lag.Udotcached.state);CHKERRQ(ierr);
    ierr = PetscObjectGetId((PetscObject)Udot,&tj->lag.Udotcached.id);CHKERRQ(ierr);
    tj->lag.Udotcached.time = t;
    tj->lag.Udotcached.step = PETSC_MIN_INT;
  }
  PetscFunctionReturn(0);
}
