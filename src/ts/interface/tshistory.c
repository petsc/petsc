#include <petsc/private/tshistoryimpl.h>

/* These macros can be moved to petscimpl.h eventually */
#if defined(PETSC_USE_DEBUG)

#define PetscValidLogicalCollectiveIntComm(a,b,c)                       \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscInt b1[2],b2[2];                                               \
    b1[0] = -b; b1[1] = b;                                              \
    _7_ierr = MPIU_Allreduce(b1,b2,2,MPIU_INT,MPI_MAX,a);CHKERRMPI(_7_ierr); \
    PetscCheckFalse(-b2[0] != b2[1],a,PETSC_ERR_ARG_WRONG,"Int value must be same on all processes, argument # %d",c); \
  } while (0)

#define PetscValidLogicalCollectiveBoolComm(a,b,c)                      \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscMPIInt b1[2],b2[2];                                            \
    b1[0] = -(PetscMPIInt)b; b1[1] = (PetscMPIInt)b;                    \
    _7_ierr = MPIU_Allreduce(b1,b2,2,MPI_INT,MPI_MAX,a);CHKERRMPI(_7_ierr); \
    PetscCheckFalse(-b2[0] != b2[1],a,PETSC_ERR_ARG_WRONG,"Bool value must be same on all processes, argument # %d",c); \
  } while (0)

#define PetscValidLogicalCollectiveRealComm(a,b,c)                      \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscReal b1[3],b2[3];                                              \
    if (PetscIsNanReal(b)) {b1[2] = 1;} else {b1[2] = 0;};              \
    b1[0] = -b; b1[1] = b;                                              \
    _7_ierr = MPI_Allreduce(b1,b2,3,MPIU_REAL,MPIU_MAX,a);CHKERRMPI(_7_ierr); \
    PetscCheckFalse(!(b2[2] > 0) && !PetscEqualReal(-b2[0],b2[1]),a,PETSC_ERR_ARG_WRONG,"Real value must be same on all processes, argument # %d",c); \
  } while (0)

#else

#define PetscValidLogicalCollectiveRealComm(a,b,c) do {} while (0)
#define PetscValidLogicalCollectiveIntComm(a,b,c) do {} while (0)
#define PetscValidLogicalCollectiveBoolComm(a,b,c) do {} while (0)

#endif

struct _n_TSHistory {
  MPI_Comm  comm;     /* used for runtime collective checks */
  PetscReal *hist;    /* time history */
  PetscInt  *hist_id; /* stores the stepid in time history */
  PetscInt  n;        /* current number of steps registered */
  PetscBool sorted;   /* if the history is sorted in ascending order */
  PetscInt  c;        /* current capacity of hist */
  PetscInt  s;        /* reallocation size */
};

PetscErrorCode TSHistoryGetNumSteps(TSHistory tsh, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidPointer(n,2);
  *n = tsh->n;
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistoryUpdate(TSHistory tsh, PetscInt id, PetscReal time)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveIntComm(tsh->comm,id,2);
  PetscValidLogicalCollectiveRealComm(tsh->comm,time,3);
  if (tsh->n == tsh->c) { /* reallocation */
    tsh->c += tsh->s;
    ierr = PetscRealloc(tsh->c*sizeof(*tsh->hist),&tsh->hist);CHKERRQ(ierr);
    ierr = PetscRealloc(tsh->c*sizeof(*tsh->hist_id),&tsh->hist_id);CHKERRQ(ierr);
  }
  tsh->sorted = (PetscBool)(tsh->sorted && (tsh->n ? time >= tsh->hist[tsh->n-1] : PETSC_TRUE));
#if defined(PETSC_USE_DEBUG)
  if (tsh->n) { /* id should be unique */
    PetscInt loc,*ids;

    ierr = PetscMalloc1(tsh->n,&ids);CHKERRQ(ierr);
    ierr = PetscArraycpy(ids,tsh->hist_id,tsh->n);CHKERRQ(ierr);
    ierr = PetscSortInt(tsh->n,ids);CHKERRQ(ierr);
    ierr = PetscFindInt(id,tsh->n,ids,&loc);CHKERRQ(ierr);
    ierr = PetscFree(ids);CHKERRQ(ierr);
    PetscCheckFalse(loc >=0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"History id should be unique");
  }
#endif
  tsh->hist[tsh->n]    = time;
  tsh->hist_id[tsh->n] = id;
  tsh->n += 1;
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistoryGetTime(TSHistory tsh, PetscBool backward, PetscInt step, PetscReal *t)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveBoolComm(tsh->comm,backward,2);
  PetscValidLogicalCollectiveIntComm(tsh->comm,step,3);
  if (!t) PetscFunctionReturn(0);
  PetscValidRealPointer(t,4);
  if (!tsh->sorted) {
    PetscErrorCode ierr;

    ierr = PetscSortRealWithArrayInt(tsh->n,tsh->hist,tsh->hist_id);CHKERRQ(ierr);
    tsh->sorted = PETSC_TRUE;
  }
  PetscCheckFalse(step < 0 || step >= tsh->n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Given time step %D does not match any in history [0,%D]",step,tsh->n);
  if (!backward) *t = tsh->hist[step];
  else           *t = tsh->hist[tsh->n-step-1];
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistoryGetTimeStep(TSHistory tsh, PetscBool backward, PetscInt step, PetscReal *dt)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveBoolComm(tsh->comm,backward,2);
  PetscValidLogicalCollectiveIntComm(tsh->comm,step,3);
  if (!dt) PetscFunctionReturn(0);
  PetscValidRealPointer(dt,4);
  if (!tsh->sorted) {
    PetscErrorCode ierr;

    ierr = PetscSortRealWithArrayInt(tsh->n,tsh->hist,tsh->hist_id);CHKERRQ(ierr);
    tsh->sorted = PETSC_TRUE;
  }
  PetscCheckFalse(step < 0 || step > tsh->n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Given time step %D does not match any in history [0,%D]",step,tsh->n);
  if (!backward) *dt = tsh->hist[PetscMin(step+1,tsh->n-1)] - tsh->hist[PetscMin(step,tsh->n-1)];
  else           *dt = tsh->hist[PetscMax(tsh->n-step-1,0)] - tsh->hist[PetscMax(tsh->n-step-2,0)];
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistoryGetLocFromTime(TSHistory tsh, PetscReal time, PetscInt *loc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveRealComm(tsh->comm,time,2);
  PetscValidIntPointer(loc,3);
  if (!tsh->sorted) {
    ierr = PetscSortRealWithArrayInt(tsh->n,tsh->hist,tsh->hist_id);CHKERRQ(ierr);
    tsh->sorted = PETSC_TRUE;
  }
  ierr = PetscFindReal(time,tsh->n,tsh->hist,PETSC_SMALL,loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistorySetHistory(TSHistory tsh, PetscInt n, PetscReal hist[], PetscInt hist_id[], PetscBool sorted)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveIntComm(tsh->comm,n,2);
  if (n) PetscValidRealPointer(hist,3);
  ierr = PetscFree(tsh->hist);CHKERRQ(ierr);
  ierr = PetscFree(tsh->hist_id);CHKERRQ(ierr);
  tsh->n = n;
  tsh->c = n;
  ierr = PetscMalloc1(tsh->n,&tsh->hist);CHKERRQ(ierr);
  ierr = PetscMalloc1(tsh->n,&tsh->hist_id);CHKERRQ(ierr);
  for (i = 0; i < tsh->n; i++) {
    tsh->hist[i]    = hist[i];
    tsh->hist_id[i] = hist_id ? hist_id[i] : i;
  }
  if (!sorted) {
    ierr = PetscSortRealWithArrayInt(tsh->n,tsh->hist,tsh->hist_id);CHKERRQ(ierr);
  }
  tsh->sorted = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistoryGetHistory(TSHistory tsh, PetscInt *n, const PetscReal* hist[], const PetscInt* hist_id[], PetscBool *sorted)
{
  PetscFunctionBegin;
  if (n)             *n = tsh->n;
  if (hist)       *hist = tsh->hist;
  if (hist_id) *hist_id = tsh->hist_id;
  if (sorted)   *sorted = tsh->sorted;
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistoryDestroy(TSHistory *tsh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*tsh) PetscFunctionReturn(0);
  ierr = PetscFree((*tsh)->hist);CHKERRQ(ierr);
  ierr = PetscFree((*tsh)->hist_id);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&((*tsh)->comm));CHKERRQ(ierr);
  ierr = PetscFree((*tsh));CHKERRQ(ierr);
  *tsh = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode TSHistoryCreate(MPI_Comm comm, TSHistory *hst)
{
  TSHistory      tsh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(hst,2);
  *hst = NULL;
  ierr = PetscNew(&tsh);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(comm,&tsh->comm,NULL);CHKERRQ(ierr);

  tsh->c      = 1024; /* capacity */
  tsh->s      = 1024; /* reallocation size */
  tsh->sorted = PETSC_TRUE;

  ierr = PetscMalloc1(tsh->c,&tsh->hist);CHKERRQ(ierr);
  ierr = PetscMalloc1(tsh->c,&tsh->hist_id);CHKERRQ(ierr);
  *hst = tsh;
  PetscFunctionReturn(0);
}
