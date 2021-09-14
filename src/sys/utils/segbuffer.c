#include <petscsys.h>

struct _PetscSegBufferLink {
  struct _PetscSegBufferLink *tail;
  size_t alloc;
  size_t used;
  size_t tailused;
  union {                       /* Dummy types to ensure alignment */
    PetscReal dummy_real;
    PetscInt  dummy_int;
    char      array[1];         /* This array is over-allocated for the size of the link */
  } u;
};

/* Segmented (extendable) array implementation */
struct _n_PetscSegBuffer {
  struct _PetscSegBufferLink *head;
  size_t unitbytes;
};

static PetscErrorCode PetscSegBufferAlloc_Private(PetscSegBuffer seg,size_t count)
{
  PetscErrorCode     ierr;
  size_t             alloc;
  struct _PetscSegBufferLink *newlink,*s;

  PetscFunctionBegin;
  s = seg->head;
  /* Grow at least fast enough to hold next item, like Fibonacci otherwise (up to 1MB chunks) */
  alloc = PetscMax(s->used+count,PetscMin(1000000/seg->unitbytes+1,s->alloc+s->tailused));
  ierr  = PetscMalloc(offsetof(struct _PetscSegBufferLink,u)+alloc*seg->unitbytes,&newlink);CHKERRQ(ierr);
  ierr  = PetscMemzero(newlink,offsetof(struct _PetscSegBufferLink,u));CHKERRQ(ierr);

  newlink->tailused  = s->used + s->tailused;
  newlink->tail      = s;
  newlink->alloc     = alloc;
  seg->head = newlink;
  PetscFunctionReturn(0);
}

/*@C
   PetscSegBufferCreate - create segmented buffer

   Not Collective

   Input Parameters:
+  unitbytes - number of bytes that each entry will contain
-  expected - expected/typical number of entries

   Output Parameter:
.  seg - segmented buffer object

   Level: developer

.seealso: PetscSegBufferGet(), PetscSegBufferExtractAlloc(), PetscSegBufferExtractTo(), PetscSegBufferExtractInPlace(), PetscSegBufferDestroy()
@*/
PetscErrorCode PetscSegBufferCreate(size_t unitbytes,size_t expected,PetscSegBuffer *seg)
{
  PetscErrorCode ierr;
  struct _PetscSegBufferLink *head;

  PetscFunctionBegin;
  ierr = PetscNew(seg);CHKERRQ(ierr);
  ierr = PetscMalloc(offsetof(struct _PetscSegBufferLink,u)+expected*unitbytes,&head);CHKERRQ(ierr);
  ierr = PetscMemzero(head,offsetof(struct _PetscSegBufferLink,u));CHKERRQ(ierr);

  head->alloc       = expected;
  (*seg)->unitbytes = unitbytes;
  (*seg)->head      = head;
  PetscFunctionReturn(0);
}

/*@C
   PetscSegBufferGet - get new buffer space from a segmented buffer

   Not Collective

   Input Parameters:
+  seg - address of segmented buffer
-  count - number of entries needed

   Output Parameter:
.  buf - address of new buffer for contiguous data

   Level: developer

.seealso: PetscSegBufferCreate(), PetscSegBufferExtractAlloc(), PetscSegBufferExtractTo(), PetscSegBufferExtractInPlace(), PetscSegBufferDestroy()
@*/
PetscErrorCode PetscSegBufferGet(PetscSegBuffer seg,size_t count,void *buf)
{
  PetscErrorCode ierr;
  struct _PetscSegBufferLink *s;

  PetscFunctionBegin;
  s = seg->head;
  if (PetscUnlikely(s->used + count > s->alloc)) {ierr = PetscSegBufferAlloc_Private(seg,count);CHKERRQ(ierr);}
  s = seg->head;
  *(char**)buf = &s->u.array[s->used*seg->unitbytes];
  s->used += count;
  PetscFunctionReturn(0);
}

/*@C
   PetscSegBufferDestroy - destroy segmented buffer

   Not Collective

   Input Parameter:
.  seg - address of segmented buffer object

   Level: developer

.seealso: PetscSegBufferCreate()
@*/
PetscErrorCode PetscSegBufferDestroy(PetscSegBuffer *seg)
{
  PetscErrorCode             ierr;
  struct _PetscSegBufferLink *s;

  PetscFunctionBegin;
  if (!*seg) PetscFunctionReturn(0);
  for (s=(*seg)->head; s;) {
    struct _PetscSegBufferLink *tail = s->tail;
    ierr = PetscFree(s);CHKERRQ(ierr);
    s = tail;
  }
  ierr = PetscFree(*seg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSegBufferExtractTo - extract contiguous data to provided buffer and reset segmented buffer

   Not Collective

   Input Parameters:
+  seg - segmented buffer
-  contig - allocated buffer to hold contiguous data

   Level: developer

.seealso: PetscSegBufferCreate(), PetscSegBufferGet(), PetscSegBufferDestroy(), PetscSegBufferExtractAlloc(), PetscSegBufferExtractInPlace()
@*/
PetscErrorCode PetscSegBufferExtractTo(PetscSegBuffer seg,void *contig)
{
  PetscErrorCode             ierr;
  size_t                     unitbytes;
  struct _PetscSegBufferLink *s,*t;
  char                       *ptr;

  PetscFunctionBegin;
  unitbytes = seg->unitbytes;
  s = seg->head;
  ptr  = ((char*)contig) + s->tailused*unitbytes;
  ierr = PetscMemcpy(ptr,s->u.array,s->used*unitbytes);CHKERRQ(ierr);
  for (t=s->tail; t;) {
    struct _PetscSegBufferLink *tail = t->tail;
    ptr -= t->used*unitbytes;
    ierr = PetscMemcpy(ptr,t->u.array,t->used*unitbytes);CHKERRQ(ierr);
    ierr = PetscFree(t);CHKERRQ(ierr);
    t    = tail;
  }
  if (ptr != contig) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Tail count does not match");
  s->used             = 0;
  s->tailused         = 0;
  s->tail             = NULL;
  PetscFunctionReturn(0);
}

/*@C
   PetscSegBufferExtractAlloc - extract contiguous data to new allocation and reset segmented buffer

   Not Collective

   Input Parameter:
.  seg - segmented buffer

   Output Parameter:
.  contiguous - address of new array containing contiguous data, caller frees with PetscFree()

   Level: developer

   Developer Notes: 'seg' argument is a pointer so that implementation could reallocate, though this is not currently done

.seealso: PetscSegBufferCreate(), PetscSegBufferGet(), PetscSegBufferDestroy(), PetscSegBufferExtractTo(), PetscSegBufferExtractInPlace()
@*/
PetscErrorCode PetscSegBufferExtractAlloc(PetscSegBuffer seg,void *contiguous)
{
  PetscErrorCode             ierr;
  struct _PetscSegBufferLink *s;
  void                       *contig;

  PetscFunctionBegin;
  s = seg->head;

  ierr = PetscMalloc((s->used+s->tailused)*seg->unitbytes,&contig);CHKERRQ(ierr);
  ierr = PetscSegBufferExtractTo(seg,contig);CHKERRQ(ierr);
  *(void**)contiguous = contig;
  PetscFunctionReturn(0);
}

/*@C
   PetscSegBufferExtractInPlace - extract in-place contiguous representation of data and reset segmented buffer for reuse

   Not Collective

   Input Parameter:
.  seg - segmented buffer object

   Output Parameter:
.  contig - address of pointer to contiguous memory

   Level: developer

.seealso: PetscSegBufferExtractAlloc(), PetscSegBufferExtractTo()
@*/
PetscErrorCode PetscSegBufferExtractInPlace(PetscSegBuffer seg,void *contig)
{
  PetscErrorCode ierr;
  struct _PetscSegBufferLink *head;

  PetscFunctionBegin;
  head = seg->head;
  if (PetscUnlikely(head->tail)) {
    PetscSegBuffer newseg;

    ierr = PetscSegBufferCreate(seg->unitbytes,head->used+head->tailused,&newseg);CHKERRQ(ierr);
    ierr = PetscSegBufferExtractTo(seg,newseg->head->u.array);CHKERRQ(ierr);
    seg->head = newseg->head;
    newseg->head = head;
    ierr = PetscSegBufferDestroy(&newseg);CHKERRQ(ierr);
    head = seg->head;
  }
  *(char**)contig = head->u.array;
  head->used = 0;
  PetscFunctionReturn(0);
}

/*@C
   PetscSegBufferGetSize - get currently used size of segmented buffer

   Not Collective

   Input Parameter:
.  seg - segmented buffer object

   Output Parameter:
.  usedsize - number of used units

   Level: developer

.seealso: PetscSegBufferExtractAlloc(), PetscSegBufferExtractTo(), PetscSegBufferCreate(), PetscSegBufferGet()
@*/
PetscErrorCode PetscSegBufferGetSize(PetscSegBuffer seg,size_t *usedsize)
{

  PetscFunctionBegin;
  *usedsize = seg->head->tailused + seg->head->used;
  PetscFunctionReturn(0);
}

/*@C
   PetscSegBufferUnuse - return some unused entries obtained with an overzealous PetscSegBufferGet()

   Not Collective

   Input Parameters:
+  seg - segmented buffer object
-  unused - number of unused units

   Level: developer

.seealso: PetscSegBufferCreate(), PetscSegBufferGet()
@*/
PetscErrorCode PetscSegBufferUnuse(PetscSegBuffer seg,size_t unused)
{
  struct _PetscSegBufferLink *head;

  PetscFunctionBegin;
  head = seg->head;
  if (PetscUnlikely(head->used < unused)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Attempt to return more unused entries (%D) than previously gotten (%D)",unused,head->used);
  head->used -= unused;
  PetscFunctionReturn(0);
}
