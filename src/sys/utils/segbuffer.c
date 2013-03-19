#include <petscsys.h>

/* Segmented (extendable) array implementation */
struct _n_PetscSegBuffer {
  PetscInt unitbytes;
  PetscInt alloc;
  PetscInt used;
  PetscInt tailused;
  PetscSegBuffer tail;
  union {                       /* Dummy types to ensure alignment */
    PetscReal dummy_real;
    PetscInt  dummy_int;
    char      array[1];
  } u;
};

#undef __FUNCT__
#define __FUNCT__ "PetscSegBufferAlloc_Private"
static PetscErrorCode PetscSegBufferAlloc_Private(PetscSegBuffer *seg,PetscInt count)
{
  PetscErrorCode ierr;
  PetscSegBuffer newseg,s;
  PetscInt       alloc;

  PetscFunctionBegin;
  s = *seg;
  /* Grow at least fast enough to hold next item, like Fibonacci otherwise (up to 1MB chunks) */
  alloc = PetscMax(s->used+count,PetscMin(1000000/s->unitbytes+1,s->alloc+s->tailused));
  ierr  = PetscMalloc(offsetof(struct _n_PetscSegBuffer,u)+alloc*s->unitbytes,&newseg);CHKERRQ(ierr);
  ierr  = PetscMemzero(newseg,offsetof(struct _n_PetscSegBuffer,u));CHKERRQ(ierr);

  newseg->unitbytes = s->unitbytes;
  newseg->tailused  = s->used + s->tailused;
  newseg->tail      = s;
  newseg->alloc     = alloc;
  *seg              = newseg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSegBufferCreate"
/*@C
   PetscSegBufferCreate - create segmented buffer

   Not Collective

   Input Arguments:
+  unitbytes - number of bytes that each entry will contain
-  expected - expected/typical number of entries

   Output Argument:
.  seg - segmented buffer object

   Level: developer

.seealso: PetscSegBufferGet(), PetscSegBufferExtractAlloc(), PetscSegBufferExtractTo(), PetscSegBufferExtractInPlace(), PetscSegBufferDestroy()
@*/
PetscErrorCode PetscSegBufferCreate(PetscInt unitbytes,PetscInt expected,PetscSegBuffer *seg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(offsetof(struct _n_PetscSegBuffer,u)+expected*unitbytes,seg);CHKERRQ(ierr);
  ierr = PetscMemzero(*seg,offsetof(struct _n_PetscSegBuffer,u));CHKERRQ(ierr);

  (*seg)->unitbytes = unitbytes;
  (*seg)->alloc     = expected;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSegBufferGet"
/*@C
   PetscSegBufferGet - get new buffer space from a segmented buffer

   Not Collective

   Input Arguments:
+  seg - address of segmented buffer
-  count - number of entries needed

   Output Argument:
.  buf - address of new buffer for contiguous data

   Level: developer

.seealso: PetscSegBufferCreate(), PetscSegBufferExtractAlloc(), PetscSegBufferExtractTo(), PetscSegBufferExtractInPlace(), PetscSegBufferDestroy()
@*/
PetscErrorCode PetscSegBufferGet(PetscSegBuffer *seg,PetscInt count,void *buf)
{
  PetscErrorCode ierr;
  PetscSegBuffer s;

  PetscFunctionBegin;
  s = *seg;
  if (PetscUnlikely(s->used + count > s->alloc)) {ierr = PetscSegBufferAlloc_Private(seg,count);CHKERRQ(ierr);}
  s = *seg;
  *(char**)buf = &s->u.array[s->used*s->unitbytes];
  s->used += count;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSegBufferDestroy"
/*@C
   PetscSegBufferDestroy - destroy segmented buffer

   Not Collective

   Input Arguments:
.  seg - address of segmented buffer object

   Level: developer

.seealso: PetscSegBufferCreate()
@*/
PetscErrorCode PetscSegBufferDestroy(PetscSegBuffer *seg)
{
  PetscErrorCode ierr;
  PetscSegBuffer s;

  PetscFunctionBegin;
  for (s=*seg; s;) {
    PetscSegBuffer tail = s->tail;
    ierr = PetscFree(s);CHKERRQ(ierr);
    s = tail;
  }
  *seg = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSegBufferExtractTo"
/*@C
   PetscSegBufferExtractTo - extract contiguous data to provided buffer and reset segmented buffer

   Not Collective

   Input Argument:
+  seg - segmented buffer
-  contig - allocated buffer to hold contiguous data

   Level: developer

.seealso: PetscSegBufferCreate(), PetscSegBufferGet(), PetscSegBufferDestroy(), PetscSegBufferExtractAlloc(), PetscSegBufferExtractInPlace()
@*/
PetscErrorCode PetscSegBufferExtractTo(PetscSegBuffer *seg,void *contig)
{
  PetscErrorCode ierr;
  PetscInt       unitbytes;
  PetscSegBuffer s,t;
  char           *ptr;

  PetscFunctionBegin;
  s = *seg;

  unitbytes = s->unitbytes;

  ptr  = ((char*)contig) + s->tailused*unitbytes;
  ierr = PetscMemcpy(ptr,s->u.array,s->used*unitbytes);CHKERRQ(ierr);
  for (t=s->tail; t;) {
    PetscSegBuffer tail = t->tail;
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

#undef __FUNCT__
#define __FUNCT__ "PetscSegBufferExtractAlloc"
/*@C
   PetscSegBufferExtractAlloc - extract contiguous data to new allocation and reset segmented buffer

   Not Collective

   Input Argument:
.  seg - segmented buffer

   Output Argument:
.  contiguous - address of new array containing contiguous data, caller frees with PetscFree()

   Level: developer

   Developer Notes: 'seg' argument is a pointer so that implementation could reallocate, though this is not currently done

.seealso: PetscSegBufferCreate(), PetscSegBufferGet(), PetscSegBufferDestroy(), PetscSegBufferExtractTo(), PetscSegBufferExtractInPlace()
@*/
PetscErrorCode PetscSegBufferExtractAlloc(PetscSegBuffer *seg,void *contiguous)
{
  PetscErrorCode ierr;
  PetscSegBuffer s;
  void           *contig;

  PetscFunctionBegin;
  s = *seg;

  ierr = PetscMalloc((s->used+s->tailused)*s->unitbytes,&contig);CHKERRQ(ierr);
  ierr = PetscSegBufferExtractTo(seg,contig);CHKERRQ(ierr);
  *(void**)contiguous = contig;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSegBufferExtractInPlace"
/*@C
   PetscSegBufferExtractInPlace - extract in-place contiguous representation of data and reset segmented buffer for reuse

   Collective

   Input Arguments:
.  seg - segmented buffer object

   Output Arguments:
.  contig - address of pointer to contiguous memory

   Level: developer

.seealso: PetscSegBufferExtractAlloc(), PetscSegBufferExtractTo()
@*/
PetscErrorCode PetscSegBufferExtractInPlace(PetscSegBuffer *seg,void *contig)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(*seg)->tail) {
    *(char**)contig = (*seg)->u.array;
  } else {
    PetscSegBuffer s = *seg,newseg;

    ierr = PetscSegBufferCreate(s->unitbytes,s->used+s->tailused,&newseg);CHKERRQ(ierr);
    ierr = PetscSegBufferExtractTo(seg,newseg->u.array);CHKERRQ(ierr);
    ierr = PetscSegBufferDestroy(seg);CHKERRQ(ierr);
    *seg = newseg;
    *(void**)contig = newseg->u.array;
  }
  PetscFunctionReturn(0);
}
