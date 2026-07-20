#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/

/*@
  DMPlexPointQueueCreate - Create a `DMPlexPointQueue`, a simple FIFO queue of `PetscInt` mesh points used by `DMPLEX` traversal routines.

  Not Collective

  Input Parameter:
. size - the initial capacity of the queue

  Output Parameter:
. queue - the newly created `DMPlexPointQueue`

  Level: developer

  Note:
  The queue grows automatically when full; see `DMPlexPointQueueEnsureSize()`.

.seealso: `DMPLEX`, `DMPlexPointQueue`, `DMPlexPointQueueDestroy()`, `DMPlexPointQueueEnqueue()`, `DMPlexPointQueueDequeue()`
@*/
PetscErrorCode DMPlexPointQueueCreate(PetscInt size, DMPlexPointQueue *queue)
{
  DMPlexPointQueue q;

  PetscFunctionBegin;
  PetscCheck(size >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Queue size %" PetscInt_FMT " must be non-negative", size);
  PetscCall(PetscCalloc1(1, &q));
  q->size = size;
  PetscCall(PetscMalloc1(q->size, &q->points));
  q->num   = 0;
  q->front = 0;
  q->back  = q->size - 1;
  *queue   = q;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexPointQueueDestroy - Destroy a `DMPlexPointQueue` previously created with `DMPlexPointQueueCreate()`.

  Not Collective

  Input Parameter:
. queue - the queue to destroy; set to `NULL` on return

  Level: developer

.seealso: `DMPLEX`, `DMPlexPointQueue`, `DMPlexPointQueueCreate()`
@*/
PetscErrorCode DMPlexPointQueueDestroy(DMPlexPointQueue *queue)
{
  DMPlexPointQueue q = *queue;

  PetscFunctionBegin;
  PetscCall(PetscFree(q->points));
  PetscCall(PetscFree(q));
  *queue = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexPointQueueEnsureSize - Ensure that a `DMPlexPointQueue` has room for at least one more entry, doubling its capacity if it is full.

  Not Collective

  Input Parameter:
. queue - the queue

  Level: developer

.seealso: `DMPLEX`, `DMPlexPointQueue`, `DMPlexPointQueueCreate()`, `DMPlexPointQueueEnqueue()`
@*/
PetscErrorCode DMPlexPointQueueEnsureSize(DMPlexPointQueue queue)
{
  PetscFunctionBegin;
  if (queue->num < queue->size) PetscFunctionReturn(PETSC_SUCCESS);
  queue->size *= 2;
  PetscCall(PetscRealloc(queue->size * sizeof(PetscInt), &queue->points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexPointQueueEnqueue - Add a mesh point to the back of a `DMPlexPointQueue`.

  Not Collective

  Input Parameters:
+ queue - the queue
- p     - the mesh point to enqueue

  Level: developer

.seealso: `DMPLEX`, `DMPlexPointQueue`, `DMPlexPointQueueDequeue()`, `DMPlexPointQueueBack()`
@*/
PetscErrorCode DMPlexPointQueueEnqueue(DMPlexPointQueue queue, PetscInt p)
{
  PetscFunctionBegin;
  PetscCall(DMPlexPointQueueEnsureSize(queue));
  queue->back                = (queue->back + 1) % queue->size;
  queue->points[queue->back] = p;
  ++queue->num;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexPointQueueDequeue - Remove and return the mesh point at the front of a `DMPlexPointQueue`.

  Not Collective

  Input Parameter:
. queue - the queue

  Output Parameter:
. p - the mesh point that was at the front of the queue

  Level: developer

  Note:
  It is an error to dequeue from an empty queue; use `DMPlexPointQueueEmpty()` to check first.

.seealso: `DMPLEX`, `DMPlexPointQueue`, `DMPlexPointQueueEnqueue()`, `DMPlexPointQueueFront()`, `DMPlexPointQueueEmpty()`
@*/
PetscErrorCode DMPlexPointQueueDequeue(DMPlexPointQueue queue, PetscInt *p)
{
  PetscFunctionBegin;
  PetscCheck(queue->num, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot dequeue from an empty queue");
  *p           = queue->points[queue->front];
  queue->front = (queue->front + 1) % queue->size;
  --queue->num;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexPointQueueFront - Return, without removing, the mesh point at the front of a `DMPlexPointQueue`.

  Not Collective

  Input Parameter:
. queue - the queue

  Output Parameter:
. p - the mesh point at the front of the queue

  Level: developer

.seealso: `DMPLEX`, `DMPlexPointQueue`, `DMPlexPointQueueBack()`, `DMPlexPointQueueDequeue()`, `DMPlexPointQueueEmpty()`
@*/
PetscErrorCode DMPlexPointQueueFront(DMPlexPointQueue queue, PetscInt *p)
{
  PetscFunctionBegin;
  PetscCheck(queue->num, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot get the front of an empty queue");
  *p = queue->points[queue->front];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexPointQueueBack - Return, without removing, the mesh point at the back of a `DMPlexPointQueue`.

  Not Collective

  Input Parameter:
. queue - the queue

  Output Parameter:
. p - the mesh point at the back of the queue

  Level: developer

.seealso: `DMPLEX`, `DMPlexPointQueue`, `DMPlexPointQueueFront()`, `DMPlexPointQueueEnqueue()`, `DMPlexPointQueueEmpty()`
@*/
PetscErrorCode DMPlexPointQueueBack(DMPlexPointQueue queue, PetscInt *p)
{
  PetscFunctionBegin;
  PetscCheck(queue->num, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot get the back of an empty queue");
  *p = queue->points[queue->back];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscBool DMPlexPointQueueEmpty(DMPlexPointQueue queue)
{
  if (!queue->num) return PETSC_TRUE;
  return PETSC_FALSE;
}

/*@
  DMPlexPointQueueEmptyCollective - Collectively determine whether a `DMPlexPointQueue` is empty on every rank of a communicator.

  Collective

  Input Parameters:
+ obj   - a `PetscObject` whose communicator is used for the reduction
- queue - the queue

  Output Parameter:
. empty - `PETSC_TRUE` if the queue is empty on every rank, `PETSC_FALSE` otherwise

  Level: developer

.seealso: `DMPLEX`, `DMPlexPointQueue`, `DMPlexPointQueueEmpty()`
@*/
PetscErrorCode DMPlexPointQueueEmptyCollective(PetscObject obj, DMPlexPointQueue queue, PetscBool *empty)
{
  PetscFunctionBeginHot;
  *empty = DMPlexPointQueueEmpty(queue);
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, empty, 1, MPI_C_BOOL, MPI_LAND, PetscObjectComm(obj)));
  PetscFunctionReturn(PETSC_SUCCESS);
}
