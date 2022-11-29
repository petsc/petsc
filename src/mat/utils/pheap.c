
#include <petsc/private/petscimpl.h>
#include <petscviewer.h>

typedef struct {
  PetscInt id;
  PetscInt value;
} HeapNode;

struct _PetscHeap {
  PetscInt  end;   /* one past the last item */
  PetscInt  alloc; /* length of array */
  PetscInt  stash; /* stash grows down, this points to last item */
  HeapNode *base;
};

/*
 * The arity of the heap can be changed via the parameter B below. Consider the B=2 (arity=4 case below)
 *
 * [00 (sentinel); 01 (min node); 10 (unused); 11 (unused); 0100 (first child); 0101; 0110; 0111; ...]
 *
 * Slots 10 and 11 are referred to as the "hole" below in the implementation.
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
  min   = PETSC_MAX_INT;
  for (; left <= right; left++) {
    PetscInt val = Value(h, left);
    if (val < min) {
      min  = val;
      chld = left;
    }
  }
  return chld;
}

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
  h->base[0].value = PETSC_MIN_INT;
  *heap            = h;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHeapAdd(PetscHeap h, PetscInt id, PetscInt val)
{
  PetscInt loc, par;

  PetscFunctionBegin;
  if (1 < h->end && h->end < ARITY) h->end = ARITY;
  loc = h->end++;
  PetscCheck(h->end <= h->stash, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Addition would exceed allocation %" PetscInt_FMT " (%" PetscInt_FMT " stashed)", h->alloc, (h->alloc - h->stash));
  h->base[loc].id    = id;
  h->base[loc].value = val;

  /* move up until heap condition is satisfied */
  while ((void)(par = Parent(loc)), Value(h, par) > val) {
    Swap(h, loc, par);
    loc = par;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHeapPop(PetscHeap h, PetscInt *id, PetscInt *val)
{
  PetscInt loc, chld;

  PetscFunctionBegin;
  if (h->end == 1) {
    *id  = h->base[0].id;
    *val = h->base[0].value;
    PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHeapPeek(PetscHeap h, PetscInt *id, PetscInt *val)
{
  PetscFunctionBegin;
  if (h->end == 1) {
    *id  = h->base[0].id;
    *val = h->base[0].value;
    PetscFunctionReturn(0);
  }

  *id  = h->base[1].id;
  *val = h->base[1].value;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHeapStash(PetscHeap h, PetscInt id, PetscInt val)
{
  PetscInt loc;

  PetscFunctionBegin;
  loc                = --h->stash;
  h->base[loc].id    = id;
  h->base[loc].value = val;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHeapUnstash(PetscHeap h)
{
  PetscFunctionBegin;
  while (h->stash < h->alloc) {
    PetscInt id = Id(h, h->stash), value = Value(h, h->stash);
    h->stash++;
    PetscCall(PetscHeapAdd(h, id, value));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHeapDestroy(PetscHeap *heap)
{
  PetscFunctionBegin;
  PetscCall(PetscFree((*heap)->base));
  PetscCall(PetscFree(*heap));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHeapView(PetscHeap h, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_SELF, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Heap size %" PetscInt_FMT " with %" PetscInt_FMT " stashed\n", h->end - 1, h->alloc - h->stash));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Heap in (id,value) pairs\n"));
    PetscCall(PetscIntView(2 * (h->end - 1), (const PetscInt *)(h->base + 1), viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stash in (id,value) pairs\n"));
    PetscCall(PetscIntView(2 * (h->alloc - h->stash), (const PetscInt *)(h->base + h->stash), viewer));
  }
  PetscFunctionReturn(0);
}
