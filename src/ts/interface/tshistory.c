#include <petsc/private/tshistoryimpl.h>

struct _n_TSHistory {
  MPI_Comm   comm;    /* used for runtime collective checks */
  PetscReal *hist;    /* time history */
  PetscInt  *hist_id; /* stores the stepid in time history */
  PetscCount n;       /* current number of steps registered */
  PetscBool  sorted;  /* if the history is sorted in ascending order */
  PetscCount c;       /* current capacity of history */
  PetscCount s;       /* reallocation size */
};

PetscErrorCode TSHistoryGetNumSteps(TSHistory tsh, PetscInt *n)
{
  PetscFunctionBegin;
  PetscAssertPointer(n, 2);
  PetscCall(PetscIntCast(tsh->n, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSHistoryUpdate(TSHistory tsh, PetscInt id, PetscReal time)
{
  PetscFunctionBegin;
  if (tsh->n == tsh->c) { /* reallocation */
    tsh->c += tsh->s;
    PetscCall(PetscRealloc(tsh->c * sizeof(*tsh->hist), &tsh->hist));
    PetscCall(PetscRealloc(tsh->c * sizeof(*tsh->hist_id), &tsh->hist_id));
  }
  tsh->sorted = (PetscBool)(tsh->sorted && (tsh->n ? (PetscBool)(time >= tsh->hist[tsh->n - 1]) : PETSC_TRUE));
#if defined(PETSC_USE_DEBUG)
  if (tsh->n) { /* id should be unique */
    PetscInt loc, *ids;

    PetscCall(PetscMalloc1(tsh->n, &ids));
    PetscCall(PetscArraycpy(ids, tsh->hist_id, tsh->n));
    PetscCall(PetscSortInt(tsh->n, ids));
    PetscCall(PetscFindInt(id, tsh->n, ids, &loc));
    PetscCall(PetscFree(ids));
    PetscCheck(loc < 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "History id should be unique");
  }
#endif
  tsh->hist[tsh->n]    = time;
  tsh->hist_id[tsh->n] = id;
  tsh->n += 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSHistoryGetTime(TSHistory tsh, PetscBool backward, PetscInt step, PetscReal *t)
{
  PetscFunctionBegin;
  if (!t) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(t, 4);
  if (!tsh->sorted) {
    PetscCall(PetscSortRealWithArrayInt(tsh->n, tsh->hist, tsh->hist_id));
    tsh->sorted = PETSC_TRUE;
  }
  PetscCheck(step >= 0 && step < tsh->n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Given time step %" PetscInt_FMT " does not match any in history [0,%" PetscCount_FMT "]", step, tsh->n);
  if (!backward) *t = tsh->hist[step];
  else *t = tsh->hist[tsh->n - step - 1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSHistoryGetTimeStep(TSHistory tsh, PetscBool backward, PetscInt step, PetscReal *dt)
{
  PetscFunctionBegin;
  if (!dt) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(dt, 4);
  if (!tsh->sorted) {
    PetscCall(PetscSortRealWithArrayInt(tsh->n, tsh->hist, tsh->hist_id));
    tsh->sorted = PETSC_TRUE;
  }
  PetscCheck(step >= 0 && step <= tsh->n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Given time step %" PetscInt_FMT " does not match any in history [0,%" PetscCount_FMT "]", step, tsh->n);
  if (!backward) *dt = tsh->hist[PetscMin(step + 1, tsh->n - 1)] - tsh->hist[PetscMin(step, tsh->n - 1)];
  else *dt = tsh->hist[PetscMax(tsh->n - step - 1, 0)] - tsh->hist[PetscMax(tsh->n - step - 2, 0)];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSHistoryGetLocFromTime(TSHistory tsh, PetscReal time, PetscInt *loc)
{
  PetscFunctionBegin;
  PetscAssertPointer(loc, 3);
  if (!tsh->sorted) {
    PetscCall(PetscSortRealWithArrayInt(tsh->n, tsh->hist, tsh->hist_id));
    tsh->sorted = PETSC_TRUE;
  }
  PetscCall(PetscFindReal(time, tsh->n, tsh->hist, PETSC_SMALL, loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSHistorySetHistory(TSHistory tsh, PetscInt n, PetscReal hist[], PetscInt hist_id[], PetscBool sorted)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveIntComm(tsh->comm, n, 2);
  PetscCheck(n >= 0, tsh->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot request a negative size for history storage");
  if (n) PetscAssertPointer(hist, 3);
  PetscCall(PetscFree(tsh->hist));
  PetscCall(PetscFree(tsh->hist_id));
  tsh->n = (size_t)n;
  tsh->c = (size_t)n;
  PetscCall(PetscMalloc1(tsh->n, &tsh->hist));
  PetscCall(PetscMalloc1(tsh->n, &tsh->hist_id));
  for (PetscInt i = 0; i < n; i++) {
    tsh->hist[i]    = hist[i];
    tsh->hist_id[i] = hist_id ? hist_id[i] : i;
  }
  if (!sorted) PetscCall(PetscSortRealWithArrayInt(tsh->n, tsh->hist, tsh->hist_id));
  tsh->sorted = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSHistoryGetHistory(TSHistory tsh, PetscInt *n, const PetscReal *hist[], const PetscInt *hist_id[], PetscBool *sorted)
{
  PetscFunctionBegin;
  if (n) PetscCall(PetscIntCast(tsh->n, n));
  if (hist) *hist = tsh->hist;
  if (hist_id) *hist_id = tsh->hist_id;
  if (sorted) *sorted = tsh->sorted;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSHistoryDestroy(TSHistory *tsh)
{
  PetscFunctionBegin;
  if (!*tsh) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFree((*tsh)->hist));
  PetscCall(PetscFree((*tsh)->hist_id));
  PetscCall(PetscCommDestroy(&(*tsh)->comm));
  PetscCall(PetscFree(*tsh));
  *tsh = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSHistoryCreate(MPI_Comm comm, TSHistory *hst)
{
  TSHistory tsh;

  PetscFunctionBegin;
  PetscAssertPointer(hst, 2);
  PetscCall(PetscNew(&tsh));
  PetscCall(PetscCommDuplicate(comm, &tsh->comm, NULL));

  tsh->c      = 1024; /* capacity */
  tsh->s      = 1024; /* reallocation size */
  tsh->sorted = PETSC_TRUE;

  PetscCall(PetscMalloc1(tsh->c, &tsh->hist));
  PetscCall(PetscMalloc1(tsh->c, &tsh->hist_id));
  *hst = tsh;
  PetscFunctionReturn(PETSC_SUCCESS);
}
