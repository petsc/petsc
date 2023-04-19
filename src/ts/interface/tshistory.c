#include <petsc/private/tshistoryimpl.h>

/* These macros can be moved to petscimpl.h eventually */
#if defined(PETSC_USE_DEBUG)

  #define PetscValidLogicalCollectiveIntComm(a, b, c) \
    do { \
      PetscInt b1[2], b2[2]; \
      b1[0] = -b; \
      b1[1] = b; \
      PetscCall(MPIU_Allreduce(b1, b2, 2, MPIU_INT, MPI_MAX, a)); \
      PetscCheck(-b2[0] == b2[1], a, PETSC_ERR_ARG_WRONG, "Int value must be same on all processes, argument # %d", c); \
    } while (0)

  #define PetscValidLogicalCollectiveBoolComm(a, b, c) \
    do { \
      PetscMPIInt b1[2], b2[2]; \
      b1[0] = -(PetscMPIInt)b; \
      b1[1] = (PetscMPIInt)b; \
      PetscCall(MPIU_Allreduce(b1, b2, 2, MPI_INT, MPI_MAX, a)); \
      PetscCheck(-b2[0] == b2[1], a, PETSC_ERR_ARG_WRONG, "Bool value must be same on all processes, argument # %d", c); \
    } while (0)

  #define PetscValidLogicalCollectiveRealComm(a, b, c) \
    do { \
      PetscReal b1[3], b2[3]; \
      if (PetscIsNanReal(b)) { \
        b1[2] = 1; \
      } else { \
        b1[2] = 0; \
      }; \
      b1[0] = -b; \
      b1[1] = b; \
      PetscCall(MPIU_Allreduce(b1, b2, 3, MPIU_REAL, MPIU_MAX, a)); \
      PetscCheck((b2[2] == 1) || PetscEqualReal(-b2[0], b2[1]), a, PETSC_ERR_ARG_WRONG, "Real value must be same on all processes, argument # %d", c); \
    } while (0)

#else

  #define PetscValidLogicalCollectiveRealComm(a, b, c) \
    do { \
    } while (0)
  #define PetscValidLogicalCollectiveIntComm(a, b, c) \
    do { \
    } while (0)
  #define PetscValidLogicalCollectiveBoolComm(a, b, c) \
    do { \
    } while (0)

#endif

struct _n_TSHistory {
  MPI_Comm   comm;    /* used for runtime collective checks */
  PetscReal *hist;    /* time history */
  PetscInt  *hist_id; /* stores the stepid in time history */
  size_t     n;       /* current number of steps registered */
  PetscBool  sorted;  /* if the history is sorted in ascending order */
  size_t     c;       /* current capacity of history */
  size_t     s;       /* reallocation size */
};

PetscErrorCode TSHistoryGetNumSteps(TSHistory tsh, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidIntPointer(n, 2);
  *n = tsh->n;
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
  tsh->sorted = (PetscBool)(tsh->sorted && (tsh->n ? time >= tsh->hist[tsh->n - 1] : PETSC_TRUE));
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
  PetscValidRealPointer(t, 4);
  if (!tsh->sorted) {
    PetscCall(PetscSortRealWithArrayInt(tsh->n, tsh->hist, tsh->hist_id));
    tsh->sorted = PETSC_TRUE;
  }
  PetscCheck(step >= 0 && step < (PetscInt)tsh->n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Given time step %" PetscInt_FMT " does not match any in history [0,%" PetscInt_FMT "]", step, (PetscInt)tsh->n);
  if (!backward) *t = tsh->hist[step];
  else *t = tsh->hist[tsh->n - step - 1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSHistoryGetTimeStep(TSHistory tsh, PetscBool backward, PetscInt step, PetscReal *dt)
{
  PetscFunctionBegin;
  if (!dt) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidRealPointer(dt, 4);
  if (!tsh->sorted) {
    PetscCall(PetscSortRealWithArrayInt(tsh->n, tsh->hist, tsh->hist_id));
    tsh->sorted = PETSC_TRUE;
  }
  PetscCheck(step >= 0 && step <= (PetscInt)tsh->n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Given time step %" PetscInt_FMT " does not match any in history [0,%" PetscInt_FMT "]", step, (PetscInt)tsh->n);
  if (!backward) *dt = tsh->hist[PetscMin(step + 1, (PetscInt)tsh->n - 1)] - tsh->hist[PetscMin(step, (PetscInt)tsh->n - 1)];
  else *dt = tsh->hist[PetscMax((PetscInt)tsh->n - step - 1, 0)] - tsh->hist[PetscMax((PetscInt)tsh->n - step - 2, 0)];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSHistoryGetLocFromTime(TSHistory tsh, PetscReal time, PetscInt *loc)
{
  PetscFunctionBegin;
  PetscValidIntPointer(loc, 3);
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
  if (n) PetscValidRealPointer(hist, 3);
  PetscCall(PetscFree(tsh->hist));
  PetscCall(PetscFree(tsh->hist_id));
  tsh->n = (size_t)n;
  tsh->c = (size_t)n;
  PetscCall(PetscMalloc1(tsh->n, &tsh->hist));
  PetscCall(PetscMalloc1(tsh->n, &tsh->hist_id));
  for (PetscInt i = 0; i < (PetscInt)tsh->n; i++) {
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
  if (n) *n = tsh->n;
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
  PetscCall(PetscCommDestroy(&((*tsh)->comm)));
  PetscCall(PetscFree((*tsh)));
  *tsh = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSHistoryCreate(MPI_Comm comm, TSHistory *hst)
{
  TSHistory tsh;

  PetscFunctionBegin;
  PetscValidPointer(hst, 2);
  *hst = NULL;
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
