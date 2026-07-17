#include <petsc/private/petscimpl.h>
#include <petscviewer.h>

typedef struct {
  PetscInt id;
  PetscInt value;
} HeapNode;

struct _n_PetscHeap {
  PetscInt  end;   /* one past the last item */
  PetscInt  alloc; /* length of array */
  PetscInt  stash; /* stash grows down, this points to last item */
  HeapNode *base;
};

/*
  The arity of the heap can be changed via the parameter B below. Consider the B=2 (arity=4 case below)

  [00 (sentinel); 01 (min node); 10 (unused); 11 (unused); 0100 (first child); 0101; 0110; 0111; ...]

  Slots 10 and 11 are referred to as the "hole" below in the implementation.
*/

#define B     1        /* log2(ARITY) */
#define ARITY (1 << B) /* tree branching factor */
static inline PetscInt Parent(PetscInt loc)
{
  PetscInt p = loc >> B;
  if (p < ARITY) return (PetscInt)(loc != 1); /* Parent(1) is 0, otherwise fix entries ending up in the hole */
  return p;
}
#define Value(h, loc) ((h)->base[loc].value)
#define Id(h, loc)    ((h)->base[loc].id)

static inline void Swap(PetscHeap h, PetscInt loc, PetscInt loc2)
{
  PetscInt id, val;
  id                  = Id(h, loc);
  val                 = Value(h, loc);
  h->base[loc].id     = Id(h, loc2);
  h->base[loc].value  = Value(h, loc2);
  h->base[loc2].id    = id;
  h->base[loc2].value = val;
}
static inline PetscInt MinChild(PetscHeap h, PetscInt loc)
{
  PetscInt min, chld, left, right;
  left  = loc << B;
  right = PetscMin(left + ARITY - 1, h->end - 1);
  chld  = 0;
  min   = PETSC_INT_MAX;
  for (; left <= right; left++) {
    PetscInt val = Value(h, left);
    if (val < min) {
      min  = val;
      chld = left;
    }
  }
  return chld;
}

/*@
  PetscHeapCreate - Creates a `PetscHeap` object, a simple min-heap for `(id, value)` pairs.

  Not Collective

  Input Parameter:
. maxsize - the maximum number of items the heap can hold at once

  Output Parameter:
. heap - the newly created `PetscHeap` object

  Level: developer

  Note:
  The heap is ordered by `value`; items with equal values may be returned in any order.

.seealso: `PetscHeap`, `PetscHeapAdd()`, `PetscHeapPop()`, `PetscHeapPeek()`, `PetscHeapStash()`, `PetscHeapUnstash()`, `PetscHeapView()`, `PetscHeapDestroy()`
@*/
PetscErrorCode PetscHeapCreate(PetscInt maxsize, PetscHeap *heap)
{
  PetscHeap h;

  PetscFunctionBegin;
  *heap = NULL;
  PetscCall(PetscMalloc1(1, &h));
  h->end   = 1;
  h->alloc = maxsize + ARITY; /* We waste all but one slot (loc=1) in the first ARITY slots */
  h->stash = h->alloc;
  PetscCall(PetscCalloc1(h->alloc, &h->base));
  h->base[0].id    = -1;
  h->base[0].value = PETSC_INT_MIN;
  *heap            = h;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscHeapAdd - Insert an item into a `PetscHeap`.

  Not Collective

  Input Parameters:
+ h   - the `PetscHeap`
. id  - the item identifier
- val - the value used for heap ordering

  Level: developer

.seealso: `PetscHeap`, `PetscHeapCreate()`, `PetscHeapPop()`, `PetscHeapPeek()`, `PetscHeapStash()`, `PetscHeapUnstash()`, `PetscHeapDestroy()`
@*/
PetscErrorCode PetscHeapAdd(PetscHeap h, PetscInt id, PetscInt val)
{
  PetscInt loc, par;

  PetscFunctionBegin;
  if (1 < h->end && h->end < ARITY) h->end = ARITY;
  loc = h->end++;
  PetscCheck(h->end <= h->stash, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Addition would exceed allocation %" PetscInt_FMT " (%" PetscInt_FMT " stashed)", h->alloc, h->alloc - h->stash);
  h->base[loc].id    = id;
  h->base[loc].value = val;

  /* move up until heap condition is satisfied */
  while ((void)(par = Parent(loc)), Value(h, par) > val) {
    Swap(h, loc, par);
    loc = par;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscHeapPop - Remove and return the minimum item from a `PetscHeap`.

  Not Collective

  Input Parameter:
. h - the `PetscHeap`

  Output Parameters:
+ id  - identifier of the popped item, or `-1` if the heap is empty
- val - value of the popped item, or `PETSC_INT_MIN` if the heap is empty

  Level: developer

.seealso: `PetscHeap`, `PetscHeapCreate()`, `PetscHeapAdd()`, `PetscHeapPeek()`, `PetscHeapStash()`, `PetscHeapUnstash()`, `PetscHeapDestroy()`
@*/
PetscErrorCode PetscHeapPop(PetscHeap h, PetscInt *id, PetscInt *val)
{
  PetscInt loc, chld;

  PetscFunctionBegin;
  if (h->end == 1) {
    *id  = h->base[0].id;
    *val = h->base[0].value;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  *id  = h->base[1].id;
  *val = h->base[1].value;

  /* rotate last entry into first position */
  loc = --h->end;
  if (h->end == ARITY) h->end = 2; /* Skip over hole */
  h->base[1].id    = Id(h, loc);
  h->base[1].value = Value(h, loc);

  /* move down until min heap condition is satisfied */
  loc = 1;
  while ((chld = MinChild(h, loc)) && Value(h, loc) > Value(h, chld)) {
    Swap(h, loc, chld);
    loc = chld;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscHeapPeek - Return the minimum item of a `PetscHeap` without removing it.

  Not Collective

  Input Parameter:
. h - the `PetscHeap`

  Output Parameters:
+ id  - identifier of the minimum item, or `-1` if the heap is empty
- val - value of the minimum item, or `PETSC_INT_MIN` if the heap is empty

  Level: developer

.seealso: `PetscHeap`, `PetscHeapCreate()`, `PetscHeapAdd()`, `PetscHeapPop()`, `PetscHeapDestroy()`
@*/
PetscErrorCode PetscHeapPeek(PetscHeap h, PetscInt *id, PetscInt *val)
{
  PetscFunctionBegin;
  if (h->end == 1) {
    *id  = h->base[0].id;
    *val = h->base[0].value;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  *id  = h->base[1].id;
  *val = h->base[1].value;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscHeapStash - Set aside an item in a `PetscHeap` for later insertion via `PetscHeapUnstash()`.

  Not Collective

  Input Parameters:
+ h   - the `PetscHeap`
. id  - the item identifier
- val - the value used for heap ordering

  Level: developer

  Note:
  Stashed items are held in the trailing portion of the heap's storage and do not participate in
  heap ordering until `PetscHeapUnstash()` reinserts them.

.seealso: `PetscHeap`, `PetscHeapCreate()`, `PetscHeapAdd()`, `PetscHeapUnstash()`, `PetscHeapDestroy()`
@*/
PetscErrorCode PetscHeapStash(PetscHeap h, PetscInt id, PetscInt val)
{
  PetscInt loc;

  PetscFunctionBegin;
  loc                = --h->stash;
  h->base[loc].id    = id;
  h->base[loc].value = val;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscHeapUnstash - Reinsert all items previously stashed with `PetscHeapStash()` into the heap.

  Not Collective

  Input Parameter:
. h - the `PetscHeap`

  Level: developer

.seealso: `PetscHeap`, `PetscHeapCreate()`, `PetscHeapAdd()`, `PetscHeapStash()`, `PetscHeapDestroy()`
@*/
PetscErrorCode PetscHeapUnstash(PetscHeap h)
{
  PetscFunctionBegin;
  while (h->stash < h->alloc) {
    PetscInt id = Id(h, h->stash), value = Value(h, h->stash);
    h->stash++;
    PetscCall(PetscHeapAdd(h, id, value));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscHeapDestroy - Destroys a `PetscHeap` created with `PetscHeapCreate()`.

  Not Collective

  Input Parameter:
. heap - the `PetscHeap` to destroy; set to `NULL` on return

  Level: developer

.seealso: `PetscHeap`, `PetscHeapCreate()`
@*/
PetscErrorCode PetscHeapDestroy(PetscHeap *heap)
{
  PetscFunctionBegin;
  PetscCall(PetscFree((*heap)->base));
  PetscCall(PetscFree(*heap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscHeapView - View the contents of a `PetscHeap`, including any stashed items.

  Not Collective

  Input Parameters:
+ h      - the `PetscHeap`
- viewer - a `PetscViewer`, or `NULL` to use `PETSC_VIEWER_STDOUT_SELF`

  Level: developer

.seealso: `PetscHeap`, `PetscHeapCreate()`, `PetscHeapAdd()`, `PetscHeapPop()`
@*/
PetscErrorCode PetscHeapView(PetscHeap h, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_SELF, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Heap size %" PetscInt_FMT " with %" PetscInt_FMT " stashed\n", h->end - 1, h->alloc - h->stash));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Heap in (id,value) pairs\n"));
    PetscCall(PetscIntView(2 * (h->end - 1), (const PetscInt *)(h->base + 1), viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stash in (id,value) pairs\n"));
    PetscCall(PetscIntView(2 * (h->alloc - h->stash), (const PetscInt *)(h->base + h->stash), viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
