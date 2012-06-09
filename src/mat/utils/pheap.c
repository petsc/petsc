#include <../src/mat/utils/petscheap.h>

typedef struct {
  PetscInt id;
  PetscInt value;
} HeapNode;

struct _PetscHeap {
  PetscInt end;                 /* one past the last item */
  PetscInt alloc;               /* length of array */
  PetscInt stash;               /* stash grows down, this points to last item */
  HeapNode *base;
};

#define Parent(loc) ((loc) >> 1)
#define Left(loc) ((loc) << 1)
#define Right(loc) (Left((loc))+1)
#define Value(h,loc) ((h)->base[loc].value)
#define Id(h,loc)  ((h)->base[loc].id)

PETSC_STATIC_INLINE void Swap(PetscHeap h,PetscInt loc,PetscInt loc2) {
  PetscInt id,val;
  id = Id(h,loc);
  val = Value(h,loc);
  h->base[loc].id = Id(h,loc2);
  h->base[loc].value = Value(h,loc2);
  h->base[loc2].id = id;
  h->base[loc2].value = val;
}
PETSC_STATIC_INLINE PetscInt MinChild(PetscHeap h,PetscInt loc) {
  if (Left(loc) >= h->end) return 0;
  if (Right(loc) >= h->end) return Left(loc);
  return Value(h,Left(loc)) <= Value(h,Right(loc)) ? Left(loc) : Right(loc);
}

#undef __FUNCT__
#define __FUNCT__ "PetscHeapCreate"
PetscErrorCode PetscHeapCreate(PetscInt maxsize,PetscHeap *heap)
{
  PetscErrorCode ierr;
  PetscHeap h;

  PetscFunctionBegin;
  *heap = PETSC_NULL;
  ierr = PetscMalloc(sizeof *h,&h);CHKERRQ(ierr);
  h->end = 1;
  h->alloc = maxsize+1;
  h->stash = h->alloc;
  ierr = PetscMalloc((maxsize+1)*sizeof(HeapNode),&h->base);CHKERRQ(ierr);
  h->base[0].id    = -1;
  h->base[0].value = PETSC_MIN_INT;
  *heap = h;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscHeapAdd"
PetscErrorCode PetscHeapAdd(PetscHeap h,PetscInt id,PetscInt val)
{
  PetscInt loc;

  PetscFunctionBegin;
  loc = h->end++;
  if (h->end > h->stash) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Addition would exceed allocation %D (%D stashed)",h->alloc,(h->alloc-h->stash));
  h->base[loc].id    = id;
  h->base[loc].value = val;

  /* move up until heap condition is satisfied */
  while (Value(h,Parent(loc)) > val) {
    Swap(h,loc,Parent(loc));
    loc = Parent(loc);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscHeapPop"
PetscErrorCode PetscHeapPop(PetscHeap h,PetscInt *id,PetscInt *val)
{
  PetscInt loc,chld;

  PetscFunctionBegin;
  if (h->end == 1) {
    *id = h->base[0].id;
    *val = h->base[0].value;
    PetscFunctionReturn(0);
  }

  *id = h->base[1].id;
  *val = h->base[1].value;

  /* rotate last entry into first position */
  loc = --h->end;
  h->base[1].id   = Id(h,loc);
  h->base[1].value = Value(h,loc);

  /* move down until min heap condition is satisfied */
  loc = 1;
  while ((chld = MinChild(h,loc)) && Value(h,loc) > Value(h,chld)) {
    Swap(h,loc,chld);
    loc = chld;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscHeapPeek"
PetscErrorCode PetscHeapPeek(PetscHeap h,PetscInt *id,PetscInt *val)
{
  PetscFunctionBegin;
  if (h->end == 1) {
    *id = h->base[0].id;
    *val = h->base[0].value;
    PetscFunctionReturn(0);
  }

  *id = h->base[1].id;
  *val = h->base[1].value;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscHeapStash"
PetscErrorCode PetscHeapStash(PetscHeap h,PetscInt id,PetscInt val)
{
  PetscInt loc;

  PetscFunctionBegin;
  loc = --h->stash;
  h->base[loc].id = id;
  h->base[loc].value = val;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscHeapUnstash"
PetscErrorCode PetscHeapUnstash(PetscHeap h)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (h->stash < h->alloc) {
    PetscInt id = Id(h,h->stash),value = Value(h,h->stash);
    h->stash++;
    ierr = PetscHeapAdd(h,id,value);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscHeapDestroy"
PetscErrorCode PetscHeapDestroy(PetscHeap *heap)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*heap)->base);CHKERRQ(ierr);
  ierr = PetscFree(*heap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscHeapView"
PetscErrorCode PetscHeapView(PetscHeap h,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool iascii;

  PetscFunctionBegin;
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PETSC_COMM_SELF,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Heap size %D with %D stashed\n",h->end-1,h->alloc-h->stash);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Heap in (id,value) pairs\n");CHKERRQ(ierr);
    ierr = PetscIntView(2*(h->end-1),(const PetscInt*)(h->base+1),viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Stash in (id,value) pairs\n");CHKERRQ(ierr);
    ierr = PetscIntView(2*(h->alloc-h->stash),(const PetscInt*)(h->base+h->stash),viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
