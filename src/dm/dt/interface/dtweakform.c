#include <petsc/private/petscdsimpl.h> /*I "petscds.h" I*/

PetscClassId PETSCWEAKFORM_CLASSID = 0;

static PetscErrorCode PetscChunkBufferCreate(size_t unitbytes, size_t expected, PetscChunkBuffer **buffer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(buffer);CHKERRQ(ierr);
  ierr = PetscCalloc1(expected*unitbytes, &(*buffer)->array);CHKERRQ(ierr);
  (*buffer)->size      = expected;
  (*buffer)->unitbytes = unitbytes;
  (*buffer)->alloc     = expected*unitbytes;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscChunkBufferDestroy(PetscChunkBuffer **buffer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*buffer)->array);CHKERRQ(ierr);
  ierr = PetscFree(*buffer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscChunkBufferCreateChunk(PetscChunkBuffer *buffer, PetscInt size, PetscChunk *chunk)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((buffer->size + size)*buffer->unitbytes > buffer->alloc) {
    char *tmp;

    if (!buffer->alloc) buffer->alloc = (buffer->size + size)*buffer->unitbytes;
    while ((buffer->size + size)*buffer->unitbytes > buffer->alloc) buffer->alloc *= 2;
    ierr = PetscMalloc(buffer->alloc, &tmp);CHKERRQ(ierr);
    ierr = PetscMemcpy(tmp, buffer->array, buffer->size*buffer->unitbytes);CHKERRQ(ierr);
    ierr = PetscFree(buffer->array);CHKERRQ(ierr);
    buffer->array = tmp;
  }
  chunk->start    = buffer->size;
  chunk->size     = size;
  chunk->reserved = size;
  buffer->size   += size;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscChunkBufferEnlargeChunk(PetscChunkBuffer *buffer, PetscInt size, PetscChunk *chunk)
{
  size_t         siz = size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (chunk->size + size > chunk->reserved) {
    PetscChunk newchunk;
    PetscInt   reserved = chunk->size;

    /* TODO Here if we had a chunk list, we could update them all to reclaim unused space */
    while (reserved < chunk->size+size) reserved *= 2;
    ierr = PetscChunkBufferCreateChunk(buffer, (size_t) reserved, &newchunk);CHKERRQ(ierr);
    newchunk.size = chunk->size+size;
    ierr = PetscMemcpy(&buffer->array[newchunk.start], &buffer->array[chunk->start], chunk->size * buffer->unitbytes);CHKERRQ(ierr);
    *chunk = newchunk;
  } else {
    chunk->size += siz;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscHashFormKeySort - Sorts an array of PetscHashFormKey in place in increasing order.

  Not Collective

  Input Parameters:
+ n - number of values
- X - array of PetscHashFormKey

  Level: intermediate

.seealso: PetscIntSortSemiOrdered(), PetscSortInt()
@*/
PetscErrorCode PetscHashFormKeySort(PetscInt n, PetscHashFormKey arr[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n <= 1) PetscFunctionReturn(0);
  PetscValidPointer(arr, 2);
  ierr = PetscTimSort(n, arr, sizeof(PetscHashFormKey), Compare_PetscHashFormKey_Private, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt *n, void (***func)())
{
  PetscHashFormKey key;
  PetscChunk       chunk;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  key.label = label; key.value = value; key.field = f;
  ierr = PetscHMapFormGet(ht, key, &chunk);CHKERRQ(ierr);
  if (chunk.size < 0) {*n = 0;          *func = NULL;}
  else                {*n = chunk.size; *func = &((void (**)()) wf->funcs->array)[chunk.start];}
  PetscFunctionReturn(0);
}

/* A NULL argument for func causes this to clear the key */
PetscErrorCode PetscWeakFormSetFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt n, void (**func)())
{
  PetscHashFormKey key;
  PetscChunk       chunk;
  PetscInt         i;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  key.label = label; key.value = value; key.field = f;
  if (!func) {
    ierr = PetscHMapFormDel(ht, key);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    ierr = PetscHMapFormGet(ht, key, &chunk);CHKERRQ(ierr);
  }
  if (chunk.size < 0) {
    ierr = PetscChunkBufferCreateChunk(wf->funcs, n, &chunk);CHKERRQ(ierr);
    ierr = PetscHMapFormSet(ht, key, chunk);CHKERRQ(ierr);
  } else if (chunk.size <= n) {
    ierr = PetscChunkBufferEnlargeChunk(wf->funcs, n - chunk.size, &chunk);CHKERRQ(ierr);
    ierr = PetscHMapFormSet(ht, key, chunk);CHKERRQ(ierr);
  }
  for (i = 0; i < n; ++i) ((void (**)()) wf->funcs->array)[chunk.start+i] = func[i];
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, void (*func)())
{
  PetscHashFormKey key;
  PetscChunk       chunk;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!func) PetscFunctionReturn(0);
  key.label = label; key.value = value; key.field = f;
  ierr = PetscHMapFormGet(ht, key, &chunk);CHKERRQ(ierr);
  if (chunk.size < 0) {
    ierr = PetscChunkBufferCreateChunk(wf->funcs, 1, &chunk);CHKERRQ(ierr);
    ierr = PetscHMapFormSet(ht, key, chunk);CHKERRQ(ierr);
    ((void (**)()) wf->funcs->array)[chunk.start] = func;
  } else {
    ierr = PetscChunkBufferEnlargeChunk(wf->funcs, 1, &chunk);CHKERRQ(ierr);
    ierr = PetscHMapFormSet(ht, key, chunk);CHKERRQ(ierr);
    ((void (**)()) wf->funcs->array)[chunk.start+chunk.size-1] = func;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetIndexFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt ind, void (**func)())
{
  PetscHashFormKey key;
  PetscChunk       chunk;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  key.label = label; key.value = value; key.field = f;
  ierr = PetscHMapFormGet(ht, key, &chunk);CHKERRQ(ierr);
  if (chunk.size < 0) {*func = NULL;}
  else {
    if (ind >= chunk.size) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %D not in [0, %D)", ind, chunk.size);
    *func = ((void (**)()) wf->funcs->array)[chunk.start+ind];
  }
  PetscFunctionReturn(0);
}

/* A NULL argument for func causes this to clear the slot, and if there is nothing else, clear the key */
PetscErrorCode PetscWeakFormSetIndexFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt ind, void (*func)())
{
  PetscHashFormKey key;
  PetscChunk       chunk;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  key.label = label; key.value = value; key.field = f;
  ierr = PetscHMapFormGet(ht, key, &chunk);CHKERRQ(ierr);
  if (chunk.size < 0) {
    if (!func) PetscFunctionReturn(0);
    ierr = PetscChunkBufferCreateChunk(wf->funcs, ind+1, &chunk);CHKERRQ(ierr);
    ierr = PetscHMapFormSet(ht, key, chunk);CHKERRQ(ierr);
  } else if (!func && !ind && chunk.size == 1) {
    ierr = PetscHMapFormDel(ht, key);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else if (chunk.size <= ind) {
    ierr = PetscChunkBufferEnlargeChunk(wf->funcs, ind - chunk.size + 1, &chunk);CHKERRQ(ierr);
    ierr = PetscHMapFormSet(ht, key, chunk);CHKERRQ(ierr);
  }
  ((void (**)()) wf->funcs->array)[chunk.start+ind] = func;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt *n,
                                         void (***obj)(PetscInt, PetscInt, PetscInt,
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormGetFunction_Private(wf, wf->obj, label, val, f, n, (void (***)(void)) obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt n,
                                         void (**obj)(PetscInt, PetscInt, PetscInt,
                                                      const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const  PetscScalar[],
                                                      const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                      PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetFunction_Private(wf, wf->obj, label, val, f, n, (void (**)(void)) obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f,
                                         void (*obj)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormAddFunction_Private(wf, wf->obj, label, val, f, (void (*)(void)) obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetIndexObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt ind,
                                              void (**obj)(PetscInt, PetscInt, PetscInt,
                                                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                           PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormGetIndexFunction_Private(wf, wf->obj, label, val, f, ind, (void (**)(void)) obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt ind,
                                              void (*obj)(PetscInt, PetscInt, PetscInt,
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->obj, label, val, f, ind, (void (*)(void)) obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f,
                                        PetscInt *n0,
                                        void (***f0)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt *n1,
                                        void (***f1)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormGetFunction_Private(wf, wf->f0, label, val, f, n0, (void (***)(void)) f0);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->f1, label, val, f, n1, (void (***)(void)) f1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f,
                                        void (*f0)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        void (*f1)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormAddFunction_Private(wf, wf->f0, label, val, f, (void (*)(void)) f0);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->f1, label, val, f, (void (*)(void)) f1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f,
                                        PetscInt n0,
                                        void (**f0)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt n1,
                                        void (**f1)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetFunction_Private(wf, wf->f0, label, val, f, n0, (void (**)(void)) f0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->f1, label, val, f, n1, (void (**)(void)) f1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f,
                                        PetscInt i0,
                                        void (*f0)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt i1,
                                        void (*f1)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->f0, label, val, f, i0, (void (*)(void)) f0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->f1, label, val, f, i1, (void (*)(void)) f1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetBdResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f,
                                          PetscInt *n0,
                                        void (***f0)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt *n1,
                                        void (***f1)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormGetFunction_Private(wf, wf->bdf0, label, val, f, n0, (void (***)(void)) f0);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->bdf1, label, val, f, n1, (void (***)(void)) f1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddBdResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f,
                                          void (*f0)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          void (*f1)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormAddFunction_Private(wf, wf->bdf0, label, val, f, (void (*)(void)) f0);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->bdf1, label, val, f, (void (*)(void)) f1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetBdResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f,
                                          PetscInt n0,
                                          void (**f0)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          PetscInt n1,
                                          void (**f1)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetFunction_Private(wf, wf->bdf0, label, val, f, n0, (void (**)(void)) f0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->bdf1, label, val, f, n1, (void (**)(void)) f1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexBdResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f,
                                          PetscInt i0,
                                          void (*f0)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          PetscInt i1,
                                          void (*f1)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->bdf0, label, val, f, i0, (void (*)(void)) f0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->bdf1, label, val, f, i1, (void (*)(void)) f1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormHasJacobian(PetscWeakForm wf, PetscBool *hasJac)
{
  PetscInt       n0, n1, n2, n3;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscValidBoolPointer(hasJac, 2);
  ierr = PetscHMapFormGetSize(wf->g0, &n0);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->g1, &n1);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->g2, &n2);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->g3, &n3);CHKERRQ(ierr);
  *hasJac = n0+n1+n2+n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                        PetscInt *n0,
                                        void (***g0)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt *n1,
                                        void (***g1)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt *n2,
                                        void (***g2)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt *n3,
                                        void (***g3)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormGetFunction_Private(wf, wf->g0, label, val, find, n0, (void (***)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->g1, label, val, find, n1, (void (***)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->g2, label, val, find, n2, (void (***)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->g3, label, val, find, n3, (void (***)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                        void (*g0)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        void (*g1)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        void (*g2)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        void (*g3)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormAddFunction_Private(wf, wf->g0, label, val, find, (void (*)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->g1, label, val, find, (void (*)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->g2, label, val, find, (void (*)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->g3, label, val, find, (void (*)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                        PetscInt n0,
                                        void (**g0)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt n1,
                                        void (**g1)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt n2,
                                        void (**g2)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt n3,
                                        void (**g3)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetFunction_Private(wf, wf->g0, label, val, find, n0, (void (**)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->g1, label, val, find, n1, (void (**)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->g2, label, val, find, n2, (void (**)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->g3, label, val, find, n3, (void (**)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                        PetscInt i0,
                                        void (*g0)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt i1,
                                        void (*g1)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt i2,
                                        void (*g2)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt i3,
                                        void (*g3)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->g0, label, val, find, i0, (void (*)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->g1, label, val, find, i1, (void (*)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->g2, label, val, find, i2, (void (*)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->g3, label, val, find, i3, (void (*)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormHasJacobianPreconditioner(PetscWeakForm wf, PetscBool *hasJacPre)
{
  PetscInt       n0, n1, n2, n3;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscValidBoolPointer(hasJacPre, 2);
  ierr = PetscHMapFormGetSize(wf->gp0, &n0);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->gp1, &n1);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->gp2, &n2);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->gp3, &n3);CHKERRQ(ierr);
  *hasJacPre = n0+n1+n2+n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                                      PetscInt *n0,
                                                      void (***g0)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                      PetscInt *n1,
                                                      void (***g1)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                      PetscInt *n2,
                                                      void (***g2)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                      PetscInt *n3,
                                                      void (***g3)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormGetFunction_Private(wf, wf->gp0, label, val, find, n0, (void (***)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->gp1, label, val, find, n1, (void (***)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->gp2, label, val, find, n2, (void (***)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->gp3, label, val, find, n3, (void (***)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                        void (*g0)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        void (*g1)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        void (*g2)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        void (*g3)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormAddFunction_Private(wf, wf->gp0, label, val, find, (void (*)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->gp1, label, val, find, (void (*)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->gp2, label, val, find, (void (*)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->gp3, label, val, find, (void (*)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                                      PetscInt n0,
                                                      void (**g0)(PetscInt, PetscInt, PetscInt,
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                      PetscInt n1,
                                                      void (**g1)(PetscInt, PetscInt, PetscInt,
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                      PetscInt n2,
                                                      void (**g2)(PetscInt, PetscInt, PetscInt,
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                      PetscInt n3,
                                                      void (**g3)(PetscInt, PetscInt, PetscInt,
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetFunction_Private(wf, wf->gp0, label, val, find, n0, (void (**)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->gp1, label, val, find, n1, (void (**)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->gp2, label, val, find, n2, (void (**)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->gp3, label, val, find, n3, (void (**)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                                      PetscInt i0,
                                                      void (*g0)(PetscInt, PetscInt, PetscInt,
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                      PetscInt i1,
                                                      void (*g1)(PetscInt, PetscInt, PetscInt,
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                      PetscInt i2,
                                                      void (*g2)(PetscInt, PetscInt, PetscInt,
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                      PetscInt i3,
                                                      void (*g3)(PetscInt, PetscInt, PetscInt,
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                 PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->gp0, label, val, find, i0, (void (*)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->gp1, label, val, find, i1, (void (*)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->gp2, label, val, find, i2, (void (*)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->gp3, label, val, find, i3, (void (*)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormHasBdJacobian(PetscWeakForm wf, PetscBool *hasJac)
{
  PetscInt       n0, n1, n2, n3;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscValidBoolPointer(hasJac, 2);
  ierr = PetscHMapFormGetSize(wf->bdg0, &n0);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->bdg1, &n1);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->bdg2, &n2);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->bdg3, &n3);CHKERRQ(ierr);
  *hasJac = n0+n1+n2+n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetBdJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                          PetscInt *n0,
                                          void (***g0)(PetscInt, PetscInt, PetscInt,
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          PetscInt *n1,
                                          void (***g1)(PetscInt, PetscInt, PetscInt,
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          PetscInt *n2,
                                          void (***g2)(PetscInt, PetscInt, PetscInt,
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          PetscInt *n3,
                                          void (***g3)(PetscInt, PetscInt, PetscInt,
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormGetFunction_Private(wf, wf->bdg0, label, val, find, n0, (void (***)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->bdg1, label, val, find, n1, (void (***)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->bdg2, label, val, find, n2, (void (***)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->bdg3, label, val, find, n3, (void (***)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddBdJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                          void (*g0)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          void (*g1)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          void (*g2)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          void (*g3)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormAddFunction_Private(wf, wf->bdg0, label, val, find, (void (*)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->bdg1, label, val, find, (void (*)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->bdg2, label, val, find, (void (*)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->bdg3, label, val, find, (void (*)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetBdJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                          PetscInt n0,
                                          void (**g0)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          PetscInt n1,
                                          void (**g1)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          PetscInt n2,
                                          void (**g2)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          PetscInt n3,
                                          void (**g3)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetFunction_Private(wf, wf->bdg0, label, val, find, n0, (void (**)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->bdg1, label, val, find, n1, (void (**)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->bdg2, label, val, find, n2, (void (**)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->bdg3, label, val, find, n3, (void (**)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexBdJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                          PetscInt i0,
                                          void (*g0)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          PetscInt i1,
                                          void (*g1)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          PetscInt i2,
                                          void (*g2)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          PetscInt i3,
                                          void (*g3)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->bdg0, label, val, find, i0, (void (*)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->bdg1, label, val, find, i1, (void (*)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->bdg2, label, val, find, i2, (void (*)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->bdg3, label, val, find, i3, (void (*)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormHasBdJacobianPreconditioner(PetscWeakForm wf, PetscBool *hasJacPre)
{
  PetscInt       n0, n1, n2, n3;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscValidBoolPointer(hasJacPre, 2);
  ierr = PetscHMapFormGetSize(wf->bdgp0, &n0);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->bdgp1, &n1);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->bdgp2, &n2);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->bdgp3, &n3);CHKERRQ(ierr);
  *hasJacPre = n0+n1+n2+n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetBdJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                                        PetscInt *n0,
                                                        void (***g0)(PetscInt, PetscInt, PetscInt,
                                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                        PetscInt *n1,
                                                        void (***g1)(PetscInt, PetscInt, PetscInt,
                                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                        PetscInt *n2,
                                                        void (***g2)(PetscInt, PetscInt, PetscInt,
                                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                        PetscInt *n3,
                                                        void (***g3)(PetscInt, PetscInt, PetscInt,
                                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                     PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormGetFunction_Private(wf, wf->bdgp0, label, val, find, n0, (void (***)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->bdgp1, label, val, find, n1, (void (***)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->bdgp2, label, val, find, n2, (void (***)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->bdgp3, label, val, find, n3, (void (***)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddBdJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                                        void (*g0)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                        void (*g1)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                        void (*g2)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                        void (*g3)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormAddFunction_Private(wf, wf->bdgp0, label, val, find, (void (*)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->bdgp1, label, val, find, (void (*)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->bdgp2, label, val, find, (void (*)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->bdgp3, label, val, find, (void (*)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetBdJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                                        PetscInt n0,
                                                        void (**g0)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                        PetscInt n1,
                                                        void (**g1)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                        PetscInt n2,
                                                        void (**g2)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                        PetscInt n3,
                                                        void (**g3)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetFunction_Private(wf, wf->bdgp0, label, val, find, n0, (void (**)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->bdgp1, label, val, find, n1, (void (**)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->bdgp2, label, val, find, n2, (void (**)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->bdgp3, label, val, find, n3, (void (**)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexBdJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                                        PetscInt i0,
                                                        void (*g0)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                        PetscInt i1,
                                                        void (*g1)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                        PetscInt i2,
                                                        void (*g2)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                        PetscInt i3,
                                                        void (*g3)(PetscInt, PetscInt, PetscInt,
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                   PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->bdgp0, label, val, find, i0, (void (*)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->bdgp1, label, val, find, i1, (void (*)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->bdgp2, label, val, find, i2, (void (*)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->bdgp3, label, val, find, i3, (void (*)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormHasDynamicJacobian(PetscWeakForm wf, PetscBool *hasDynJac)
{
  PetscInt       n0, n1, n2, n3;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscValidBoolPointer(hasDynJac, 2);
  ierr = PetscHMapFormGetSize(wf->gt0, &n0);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->gt1, &n1);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->gt2, &n2);CHKERRQ(ierr);
  ierr = PetscHMapFormGetSize(wf->gt3, &n3);CHKERRQ(ierr);
  *hasDynJac = n0+n1+n2+n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetDynamicJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                        PetscInt *n0,
                                        void (***g0)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt *n1,
                                        void (***g1)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt *n2,
                                        void (***g2)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        PetscInt *n3,
                                        void (***g3)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormGetFunction_Private(wf, wf->gt0, label, val, find, n0, (void (***)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->gt1, label, val, find, n1, (void (***)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->gt2, label, val, find, n2, (void (***)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormGetFunction_Private(wf, wf->gt3, label, val, find, n3, (void (***)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddDynamicJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                        void (*g0)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        void (*g1)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        void (*g2)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        void (*g3)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormAddFunction_Private(wf, wf->gt0, label, val, find, (void (*)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->gt1, label, val, find, (void (*)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->gt2, label, val, find, (void (*)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormAddFunction_Private(wf, wf->gt3, label, val, find, (void (*)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetDynamicJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                               PetscInt n0,
                                               void (**g0)(PetscInt, PetscInt, PetscInt,
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                               PetscInt n1,
                                               void (**g1)(PetscInt, PetscInt, PetscInt,
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                               PetscInt n2,
                                               void (**g2)(PetscInt, PetscInt, PetscInt,
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                               PetscInt n3,
                                               void (**g3)(PetscInt, PetscInt, PetscInt,
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetFunction_Private(wf, wf->gt0, label, val, find, n0, (void (**)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->gt1, label, val, find, n1, (void (**)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->gt2, label, val, find, n2, (void (**)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormSetFunction_Private(wf, wf->gt3, label, val, find, n3, (void (**)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexDynamicJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g,
                                               PetscInt i0,
                                               void (*g0)(PetscInt, PetscInt, PetscInt,
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                               PetscInt i1,
                                               void (*g1)(PetscInt, PetscInt, PetscInt,
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                               PetscInt i2,
                                               void (*g2)(PetscInt, PetscInt, PetscInt,
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                               PetscInt i3,
                                               void (*g3)(PetscInt, PetscInt, PetscInt,
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt       find = f*wf->Nf + g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->gt0, label, val, find, i0, (void (*)(void)) g0);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->gt1, label, val, find, i1, (void (*)(void)) g1);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->gt2, label, val, find, i2, (void (*)(void)) g2);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->gt3, label, val, find, i3, (void (*)(void)) g3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetRiemannSolver(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt *n,
                                             void (***r)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormGetFunction_Private(wf, wf->r, label, val, f, n, (void (***)(void)) r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetRiemannSolver(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f,
                                             PetscInt n,
                                             void (**r)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetFunction_Private(wf, wf->r, label, val, f, n, (void (**)(void)) r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexRiemannSolver(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f,
                                                  PetscInt i,
                                                  void (*r)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormSetIndexFunction_Private(wf, wf->r, label, val, f, i, (void (*)(void)) r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscWeakFormGetNumFields - Returns the number of fields

  Not collective

  Input Parameter:
. wf - The PetscWeakForm object

  Output Parameter:
. Nf - The nubmer of fields

  Level: beginner

.seealso: PetscWeakFormSetNumFields(), PetscWeakFormCreate()
@*/
PetscErrorCode PetscWeakFormGetNumFields(PetscWeakForm wf, PetscInt *Nf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscValidPointer(Nf, 2);
  *Nf = wf->Nf;
  PetscFunctionReturn(0);
}

/*@
  PetscWeakFormSetNumFields - Sets the number of fields

  Not collective

  Input Parameters:
+ wf - The PetscWeakForm object
- Nf - The number of fields

  Level: beginner

.seealso: PetscWeakFormGetNumFields(), PetscWeakFormCreate()
@*/
PetscErrorCode PetscWeakFormSetNumFields(PetscWeakForm wf, PetscInt Nf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  wf->Nf = Nf;
  PetscFunctionReturn(0);
}

/*@
  PetscWeakFormDestroy - Destroys a PetscWeakForm object

  Collective on wf

  Input Parameter:
. wf - the PetscWeakForm object to destroy

  Level: developer

.seealso PetscWeakFormCreate(), PetscWeakFormView()
@*/
PetscErrorCode PetscWeakFormDestroy(PetscWeakForm *wf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*wf) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*wf), PETSCWEAKFORM_CLASSID, 1);

  if (--((PetscObject)(*wf))->refct > 0) {*wf = NULL; PetscFunctionReturn(0);}
  ((PetscObject) (*wf))->refct = 0;
  ierr = PetscChunkBufferDestroy(&(*wf)->funcs);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->obj);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->f0);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->f1);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->g0);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->g1);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->g2);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->g3);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->gp0);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->gp1);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->gp2);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->gp3);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->gt0);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->gt1);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->gt2);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->gt3);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->bdf0);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->bdf1);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->bdg0);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->bdg1);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->bdg2);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->bdg3);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->bdgp0);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->bdgp1);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->bdgp2);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->bdgp3);CHKERRQ(ierr);
  ierr = PetscHMapFormDestroy(&(*wf)->r);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(wf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscWeakFormViewTable_Ascii(PetscWeakForm wf, PetscViewer viewer, const char tableName[], PetscHMapForm map)
{
  PetscInt       Nk, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscHMapFormGetSize(map, &Nk);CHKERRQ(ierr);
  if (Nk) {
    PetscHashFormKey *keys;
    void           (**funcs)(void);
    const char       *name;
    PetscInt          off = 0, n, i;

    ierr = PetscMalloc1(Nk, &keys);CHKERRQ(ierr);
    ierr = PetscHMapFormGetKeys(map, &off, keys);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "%s\n", tableName);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (k = 0; k < Nk; ++k) {
      if (keys[k].label) {ierr = PetscObjectGetName((PetscObject) keys[k].label, &name);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "Key (%s, %D, %D) ", keys[k].label ? name : "None", keys[k].value, keys[k].field);CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer, PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscWeakFormGetFunction_Private(wf, map, keys[k].label, keys[k].value, keys[k].field, &n, &funcs);CHKERRQ(ierr);
      for (i = 0; i < n; ++i) {
        if (i > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
        ierr = PetscDLAddr(funcs[i], &name);CHKERRQ(ierr);
        if (name) {ierr = PetscViewerASCIIPrintf(viewer, "%s", name);CHKERRQ(ierr);}
        else      {ierr = PetscViewerASCIIPrintf(viewer, "%p", funcs[i]);CHKERRQ(ierr);}
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer, PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = PetscFree(keys);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscWeakFormView_Ascii(PetscWeakForm wf, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Weak Form System with %d fields\n", wf->Nf);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Objective", wf->obj);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Residual f0", wf->f0);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Residual f1", wf->f1);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Boudnary Residual f0", wf->bdf0);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Boundary Residual f1", wf->bdf1);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Jacobian g0", wf->g0);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Jacobian g1", wf->g1);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Jacobian g2", wf->g2);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Jacobian g3", wf->g3);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Jacobian Preconditioner g0", wf->gp0);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Jacobian Preconditioner g1", wf->gp1);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Jacobian Preconditioner g2", wf->gp2);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Jacobian Preconditioner g3", wf->gp3);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Dynamic Jacobian g0", wf->gt0);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Dynamic Jacobian g1", wf->gt1);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Dynamic Jacobian g2", wf->gt2);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Dynamic Jacobian g3", wf->gt3);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Boundary Jacobian g0", wf->bdg0);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Boundary Jacobian g1", wf->bdg1);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Boundary Jacobian g2", wf->bdg2);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Boundary Jacobian g3", wf->bdg3);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Boundary Jacobian Preconditioner g0", wf->bdgp0);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Boundary Jacobian Preconditioner g1", wf->bdgp1);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Boundary Jacobian Preconditioner g2", wf->bdgp2);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Boundary Jacobian Preconditioner g3", wf->bdgp3);CHKERRQ(ierr);
  ierr = PetscWeakFormViewTable_Ascii(wf, viewer, "Riemann Solver", wf->r);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscWeakFormView - Views a PetscWeakForm

  Collective on wf

  Input Parameter:
+ wf - the PetscWeakForm object to view
- v  - the viewer

  Level: developer

.seealso PetscWeakFormDestroy(), PetscWeakFormCreate()
@*/
PetscErrorCode PetscWeakFormView(PetscWeakForm wf, PetscViewer v)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  if (!v) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) wf), &v);CHKERRQ(ierr);}
  else    {PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);}
  ierr = PetscObjectTypeCompare((PetscObject) v, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscWeakFormView_Ascii(wf, v);CHKERRQ(ierr);}
  if (wf->ops->view) {ierr = (*wf->ops->view)(wf, v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscWeakFormCreate - Creates an empty PetscWeakForm object.

  Collective

  Input Parameter:
. comm - The communicator for the PetscWeakForm object

  Output Parameter:
. wf - The PetscWeakForm object

  Level: beginner

.seealso: PetscDS, PetscWeakFormDestroy()
@*/
PetscErrorCode PetscWeakFormCreate(MPI_Comm comm, PetscWeakForm *wf)
{
  PetscWeakForm  p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(wf, 2);
  *wf  = NULL;
  ierr = PetscDSInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(p, PETSCWEAKFORM_CLASSID, "PetscWeakForm", "Weak Form System", "PetscWeakForm", comm, PetscWeakFormDestroy, PetscWeakFormView);CHKERRQ(ierr);

  p->Nf = 0;
  ierr = PetscChunkBufferCreate(sizeof(&PetscWeakFormCreate), 2, &p->funcs);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->obj);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->f0);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->f1);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->g0);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->g1);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->g2);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->g3);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->gp0);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->gp1);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->gp2);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->gp3);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->gt0);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->gt1);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->gt2);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->gt3);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->bdf0);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->bdf1);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->bdg0);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->bdg1);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->bdg2);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->bdg3);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->bdgp0);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->bdgp1);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->bdgp2);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->bdgp3);CHKERRQ(ierr);
  ierr = PetscHMapFormCreate(&p->r);CHKERRQ(ierr);
  *wf = p;
  PetscFunctionReturn(0);
}
