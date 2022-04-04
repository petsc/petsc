#include <petsc/private/petscdsimpl.h> /*I "petscds.h" I*/

PetscClassId PETSCWEAKFORM_CLASSID = 0;

const char *const PetscWeakFormKinds[] = {"objective", "residual_f0", "residual_f1", "jacobian_g0", "jacobian_g1", "jacobian_g2", "jacobian_g3", "jacobian_preconditioner_g0", "jacobian_preconditioner_g1", "jacobian_preconditioner_g2", "jacobian_preconditioner_g3", "dynamic_jacobian_g0", "dynamic_jacobian_g1", "dynamic_jacobian_g2", "dynamic_jacobian_g3", "boundary_residual_f0", "boundary_residual_f1", "boundary_jacobian_g0", "boundary_jacobian_g1", "boundary_jacobian_g2", "boundary_jacobian_g3", "boundary_jacobian_preconditioner_g0", "boundary_jacobian_preconditioner_g1", "boundary_jacobian_preconditioner_g2", "boundary_jacobian_preconditioner_g3", "riemann_solver", "PetscWeakFormKind", "PETSC_WF_", NULL};

static PetscErrorCode PetscChunkBufferCreate(size_t unitbytes, size_t expected, PetscChunkBuffer **buffer)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(buffer));
  PetscCall(PetscCalloc1(expected*unitbytes, &(*buffer)->array));
  (*buffer)->size      = expected;
  (*buffer)->unitbytes = unitbytes;
  (*buffer)->alloc     = expected*unitbytes;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscChunkBufferDuplicate(PetscChunkBuffer *buffer, PetscChunkBuffer **bufferNew)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(bufferNew));
  PetscCall(PetscCalloc1(buffer->size*buffer->unitbytes, &(*bufferNew)->array));
  PetscCall(PetscMemcpy((*bufferNew)->array, buffer->array, buffer->size*buffer->unitbytes));
  (*bufferNew)->size      = buffer->size;
  (*bufferNew)->unitbytes = buffer->unitbytes;
  (*bufferNew)->alloc     = buffer->size*buffer->unitbytes;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscChunkBufferDestroy(PetscChunkBuffer **buffer)
{
  PetscFunctionBegin;
  PetscCall(PetscFree((*buffer)->array));
  PetscCall(PetscFree(*buffer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscChunkBufferCreateChunk(PetscChunkBuffer *buffer, PetscInt size, PetscChunk *chunk)
{
  PetscFunctionBegin;
  if ((buffer->size + size)*buffer->unitbytes > buffer->alloc) {
    char *tmp;

    if (!buffer->alloc) buffer->alloc = (buffer->size + size)*buffer->unitbytes;
    while ((buffer->size + size)*buffer->unitbytes > buffer->alloc) buffer->alloc *= 2;
    PetscCall(PetscMalloc(buffer->alloc, &tmp));
    PetscCall(PetscMemcpy(tmp, buffer->array, buffer->size*buffer->unitbytes));
    PetscCall(PetscFree(buffer->array));
    buffer->array = tmp;
  }
  chunk->start    = buffer->size*buffer->unitbytes;
  chunk->size     = size;
  chunk->reserved = size;
  buffer->size   += size;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscChunkBufferEnlargeChunk(PetscChunkBuffer *buffer, PetscInt size, PetscChunk *chunk)
{
  size_t         siz = size;

  PetscFunctionBegin;
  if (chunk->size + size > chunk->reserved) {
    PetscChunk newchunk;
    PetscInt   reserved = chunk->size;

    /* TODO Here if we had a chunk list, we could update them all to reclaim unused space */
    while (reserved < chunk->size+size) reserved *= 2;
    PetscCall(PetscChunkBufferCreateChunk(buffer, (size_t) reserved, &newchunk));
    newchunk.size = chunk->size+size;
    PetscCall(PetscMemcpy(&buffer->array[newchunk.start], &buffer->array[chunk->start], chunk->size * buffer->unitbytes));
    *chunk = newchunk;
  } else {
    chunk->size += siz;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscFormKeySort - Sorts an array of PetscFormKey in place in increasing order.

  Not Collective

  Input Parameters:
+ n - number of values
- X - array of PetscFormKey

  Level: intermediate

.seealso: PetscIntSortSemiOrdered(), PetscSortInt()
@*/
PetscErrorCode PetscFormKeySort(PetscInt n, PetscFormKey arr[])
{
  PetscFunctionBegin;
  if (n <= 1) PetscFunctionReturn(0);
  PetscValidPointer(arr, 2);
  PetscCall(PetscTimSort(n, arr, sizeof(PetscFormKey), Compare_PetscFormKey_Private, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt part, PetscInt *n, void (***func)())
{
  PetscFormKey   key;
  PetscChunk     chunk;

  PetscFunctionBegin;
  key.label = label; key.value = value; key.field = f; key.part = part;
  PetscCall(PetscHMapFormGet(ht, key, &chunk));
  if (chunk.size < 0) {*n = 0;          *func = NULL;}
  else                {*n = chunk.size; *func = (void (**)()) &wf->funcs->array[chunk.start];}
  PetscFunctionReturn(0);
}

/* A NULL argument for func causes this to clear the key */
PetscErrorCode PetscWeakFormSetFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt part, PetscInt n, void (**func)())
{
  PetscFormKey   key;
  PetscChunk     chunk;
  PetscInt       i;

  PetscFunctionBegin;
  key.label = label; key.value = value; key.field = f; key.part = part;
  if (!func) {
    PetscCall(PetscHMapFormDel(ht, key));
    PetscFunctionReturn(0);
  } else {
    PetscCall(PetscHMapFormGet(ht, key, &chunk));
  }
  if (chunk.size < 0) {
    PetscCall(PetscChunkBufferCreateChunk(wf->funcs, n, &chunk));
    PetscCall(PetscHMapFormSet(ht, key, chunk));
  } else if (chunk.size <= n) {
    PetscCall(PetscChunkBufferEnlargeChunk(wf->funcs, n - chunk.size, &chunk));
    PetscCall(PetscHMapFormSet(ht, key, chunk));
  }
  for (i = 0; i < n; ++i) ((void (**)()) &wf->funcs->array[chunk.start])[i] = func[i];
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt part, void (*func)())
{
  PetscFormKey   key;
  PetscChunk     chunk;

  PetscFunctionBegin;
  if (!func) PetscFunctionReturn(0);
  key.label = label; key.value = value; key.field = f; key.part = part;
  PetscCall(PetscHMapFormGet(ht, key, &chunk));
  if (chunk.size < 0) {
    PetscCall(PetscChunkBufferCreateChunk(wf->funcs, 1, &chunk));
    PetscCall(PetscHMapFormSet(ht, key, chunk));
    ((void (**)()) &wf->funcs->array[chunk.start])[0] = func;
  } else {
    PetscCall(PetscChunkBufferEnlargeChunk(wf->funcs, 1, &chunk));
    PetscCall(PetscHMapFormSet(ht, key, chunk));
    ((void (**)()) &wf->funcs->array[chunk.start])[chunk.size-1] = func;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetIndexFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt part, PetscInt ind, void (**func)())
{
  PetscFormKey key;
  PetscChunk       chunk;

  PetscFunctionBegin;
  key.label = label; key.value = value; key.field = f; key.part = part;
  PetscCall(PetscHMapFormGet(ht, key, &chunk));
  if (chunk.size < 0) {*func = NULL;}
  else {
    PetscCheck(ind < chunk.size,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %D not in [0, %D)", ind, chunk.size);
    *func = ((void (**)()) &wf->funcs->array[chunk.start])[ind];
  }
  PetscFunctionReturn(0);
}

/* Ignore a NULL func */
PetscErrorCode PetscWeakFormSetIndexFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt part, PetscInt ind, void (*func)())
{
  PetscFormKey key;
  PetscChunk       chunk;

  PetscFunctionBegin;
  if (!func) PetscFunctionReturn(0);
  key.label = label; key.value = value; key.field = f; key.part = part;
  PetscCall(PetscHMapFormGet(ht, key, &chunk));
  if (chunk.size < 0) {
    PetscCall(PetscChunkBufferCreateChunk(wf->funcs, ind+1, &chunk));
    PetscCall(PetscHMapFormSet(ht, key, chunk));
  } else if (chunk.size <= ind) {
    PetscCall(PetscChunkBufferEnlargeChunk(wf->funcs, ind - chunk.size + 1, &chunk));
    PetscCall(PetscHMapFormSet(ht, key, chunk));
  }
  ((void (**)()) &wf->funcs->array[chunk.start])[ind] = func;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormClearIndexFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt part, PetscInt ind)
{
  PetscFormKey key;
  PetscChunk       chunk;

  PetscFunctionBegin;
  key.label = label; key.value = value; key.field = f; key.part = part;
  PetscCall(PetscHMapFormGet(ht, key, &chunk));
  if (chunk.size < 0) {
    PetscFunctionReturn(0);
  } else if (!ind && chunk.size == 1) {
    PetscCall(PetscHMapFormDel(ht, key));
    PetscFunctionReturn(0);
  } else if (chunk.size <= ind) {
    PetscFunctionReturn(0);
  }
  ((void (**)()) &wf->funcs->array[chunk.start])[ind] = NULL;
  PetscFunctionReturn(0);
}

/*@
  PetscWeakFormCopy - Copy the pointwise functions to another PetscWeakForm

  Not Collective

  Input Parameter:
. wf - The original PetscWeakForm

  Output Parameter:
. wfNew - The copy PetscWeakForm

  Level: intermediate

.seealso: PetscWeakFormCreate(), PetscWeakFormDestroy()
@*/
PetscErrorCode PetscWeakFormCopy(PetscWeakForm wf, PetscWeakForm wfNew)
{
  PetscInt       f;

  PetscFunctionBegin;
  wfNew->Nf = wf->Nf;
  PetscCall(PetscChunkBufferDestroy(&wfNew->funcs));
  PetscCall(PetscChunkBufferDuplicate(wf->funcs, &wfNew->funcs));
  for (f = 0; f < PETSC_NUM_WF; ++f) {
    PetscCall(PetscHMapFormDestroy(&wfNew->form[f]));
    PetscCall(PetscHMapFormDuplicate(wf->form[f], &wfNew->form[f]));
  }
  PetscFunctionReturn(0);
}

/*@
  PetscWeakFormClear - Clear all functions from the PetscWeakForm

  Not Collective

  Input Parameter:
. wf - The original PetscWeakForm

  Level: intermediate

.seealso: PetscWeakFormCopy(), PetscWeakFormCreate(), PetscWeakFormDestroy()
@*/
PetscErrorCode PetscWeakFormClear(PetscWeakForm wf)
{
  PetscInt       f;

  PetscFunctionBegin;
  for (f = 0; f < PETSC_NUM_WF; ++f) PetscCall(PetscHMapFormClear(wf->form[f]));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscWeakFormRewriteKeys_Internal(PetscWeakForm wf, PetscHMapForm hmap, DMLabel label, PetscInt Nv, const PetscInt values[])
{
  PetscFormKey  *keys;
  PetscInt       n, i, v, off = 0;

  PetscFunctionBegin;
  PetscCall(PetscHMapFormGetSize(hmap, &n));
  PetscCall(PetscMalloc1(n, &keys));
  PetscCall(PetscHMapFormGetKeys(hmap, &off, keys));
  for (i = 0; i < n; ++i) {
    if (keys[i].label == label) {
      PetscBool clear = PETSC_TRUE;
      void   (**funcs)();
      PetscInt  Nf;

      PetscCall(PetscWeakFormGetFunction_Private(wf, hmap, keys[i].label, keys[i].value, keys[i].field, keys[i].part, &Nf, &funcs));
      for (v = 0; v < Nv; ++v) {
        PetscCall(PetscWeakFormSetFunction_Private(wf, hmap, keys[i].label, values[v], keys[i].field, keys[i].part, Nf, funcs));
        if (values[v] == keys[i].value) clear = PETSC_FALSE;
      }
      if (clear) PetscCall(PetscWeakFormSetFunction_Private(wf, hmap, keys[i].label, keys[i].value, keys[i].field, keys[i].part, 0, NULL));
    }
  }
  PetscCall(PetscFree(keys));
  PetscFunctionReturn(0);
}

/*@C
  PetscWeakFormRewriteKeys - Change any key on the given label to use the new set of label values

  Not Collective

  Input Parameters:
+ wf     - The original PetscWeakForm
. label  - The label to change keys for
. Nv     - The number of new label values
- values - The set of new values to relabel keys with

  Note: This is used internally when boundary label values are specified from the command line.

  Level: intermediate

.seealso: PetscWeakFormReplaceLabel(), PetscWeakFormCreate(), PetscWeakFormDestroy()
@*/
PetscErrorCode PetscWeakFormRewriteKeys(PetscWeakForm wf, DMLabel label, PetscInt Nv, const PetscInt values[])
{
  PetscInt       f;

  PetscFunctionBegin;
  for (f = 0; f < PETSC_NUM_WF; ++f) PetscCall(PetscWeakFormRewriteKeys_Internal(wf, wf->form[f], label, Nv, values));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscWeakFormReplaceLabel_Internal(PetscWeakForm wf, PetscHMapForm hmap, DMLabel label)
{
  PetscFormKey  *keys;
  PetscInt       n, i, off = 0, maxFuncs = 0;
  void       (**tmpf)();
  const char    *name = NULL;

  PetscFunctionBegin;
  if (label) PetscCall(PetscObjectGetName((PetscObject) label, &name));
  PetscCall(PetscHMapFormGetSize(hmap, &n));
  PetscCall(PetscMalloc1(n, &keys));
  PetscCall(PetscHMapFormGetKeys(hmap, &off, keys));
  for (i = 0; i < n; ++i) {
    PetscBool   match = PETSC_FALSE;
    const char *lname = NULL;

    if (label == keys[i].label) continue;
    if (keys[i].label) PetscCall(PetscObjectGetName((PetscObject) keys[i].label, &lname));
    PetscCall(PetscStrcmp(name, lname, &match));
    if ((!name && !lname) || match) {
      void  (**funcs)();
      PetscInt Nf;

      PetscCall(PetscWeakFormGetFunction_Private(wf, hmap, keys[i].label, keys[i].value, keys[i].field, keys[i].part, &Nf, &funcs));
      maxFuncs = PetscMax(maxFuncs, Nf);
    }
  }
  /* Need temp space because chunk buffer can be reallocated in SetFunction() call */
  PetscCall(PetscMalloc1(maxFuncs, &tmpf));
  for (i = 0; i < n; ++i) {
    PetscBool   match = PETSC_FALSE;
    const char *lname = NULL;

    if (label == keys[i].label) continue;
    if (keys[i].label) PetscCall(PetscObjectGetName((PetscObject) keys[i].label, &lname));
    PetscCall(PetscStrcmp(name, lname, &match));
    if ((!name && !lname) || match) {
      void  (**funcs)();
      PetscInt Nf, j;

      PetscCall(PetscWeakFormGetFunction_Private(wf, hmap, keys[i].label, keys[i].value, keys[i].field, keys[i].part, &Nf, &funcs));
      for (j = 0; j < Nf; ++j) tmpf[j] = funcs[j];
      PetscCall(PetscWeakFormSetFunction_Private(wf, hmap, label,         keys[i].value, keys[i].field, keys[i].part,  Nf,  tmpf));
      PetscCall(PetscWeakFormSetFunction_Private(wf, hmap, keys[i].label, keys[i].value, keys[i].field, keys[i].part,  0,   NULL));
    }
  }
  PetscCall(PetscFree(tmpf));
  PetscCall(PetscFree(keys));
  PetscFunctionReturn(0);
}

/*@C
  PetscWeakFormReplaceLabel - Change any key on a label of the same name to use the new label

  Not Collective

  Input Parameters:
+ wf    - The original PetscWeakForm
- label - The label to change keys for

  Note: This is used internally when meshes are modified

  Level: intermediate

.seealso: PetscWeakFormRewriteKeys(), PetscWeakFormCreate(), PetscWeakFormDestroy()
@*/
PetscErrorCode PetscWeakFormReplaceLabel(PetscWeakForm wf, DMLabel label)
{
  PetscInt       f;

  PetscFunctionBegin;
  for (f = 0; f < PETSC_NUM_WF; ++f) PetscCall(PetscWeakFormReplaceLabel_Internal(wf, wf->form[f], label));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormClearIndex(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscWeakFormKind kind, PetscInt ind)
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormClearIndexFunction_Private(wf, wf->form[kind], label, val, f, part, ind));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt *n,
                                         void (***obj)(PetscInt, PetscInt, PetscInt,
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_OBJECTIVE], label, val, f, part, n, (void (***)(void)) obj));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt n,
                                         void (**obj)(PetscInt, PetscInt, PetscInt,
                                                      const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const  PetscScalar[],
                                                      const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                      PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_OBJECTIVE], label, val, f, part, n, (void (**)(void)) obj));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part,
                                         void (*obj)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_OBJECTIVE], label, val, f, part, (void (*)(void)) obj));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetIndexObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt ind,
                                              void (**obj)(PetscInt, PetscInt, PetscInt,
                                                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                           PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetIndexFunction_Private(wf, wf->form[PETSC_WF_OBJECTIVE], label, val, f, part, ind, (void (**)(void)) obj));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt ind,
                                              void (*obj)(PetscInt, PetscInt, PetscInt,
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                          PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_OBJECTIVE], label, val, f, part, ind, (void (*)(void)) obj));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part,
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
  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_F0], label, val, f, part, n0, (void (***)(void)) f0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_F1], label, val, f, part, n1, (void (***)(void)) f1));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part,
                                        void (*f0)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        void (*f1)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_F0], label, val, f, part, (void (*)(void)) f0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_F1], label, val, f, part, (void (*)(void)) f1));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part,
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
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_F0], label, val, f, part, n0, (void (**)(void)) f0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_F1], label, val, f, part, n1, (void (**)(void)) f1));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part,
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
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_F0], label, val, f, part, i0, (void (*)(void)) f0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_F1], label, val, f, part, i1, (void (*)(void)) f1));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetBdResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part,
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
  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDF0], label, val, f, part, n0, (void (***)(void)) f0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDF1], label, val, f, part, n1, (void (***)(void)) f1));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddBdResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part,
                                          void (*f0)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          void (*f1)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDF0], label, val, f, part, (void (*)(void)) f0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDF1], label, val, f, part, (void (*)(void)) f1));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetBdResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part,
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
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDF0], label, val, f, part, n0, (void (**)(void)) f0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDF1], label, val, f, part, n1, (void (**)(void)) f1));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexBdResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part,
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
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDF0], label, val, f, part, i0, (void (*)(void)) f0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDF1], label, val, f, part, i1, (void (*)(void)) f1));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormHasJacobian(PetscWeakForm wf, PetscBool *hasJac)
{
  PetscInt       n0, n1, n2, n3;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscValidBoolPointer(hasJac, 2);
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_G0], &n0));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_G1], &n1));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_G2], &n2));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_G3], &n3));
  *hasJac = n0+n1+n2+n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_G0], label, val, find, part, n0, (void (***)(void)) g0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_G1], label, val, find, part, n1, (void (***)(void)) g1));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_G2], label, val, find, part, n2, (void (***)(void)) g2));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_G3], label, val, find, part, n3, (void (***)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_G0], label, val, find, part, (void (*)(void)) g0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_G1], label, val, find, part, (void (*)(void)) g1));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_G2], label, val, find, part, (void (*)(void)) g2));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_G3], label, val, find, part, (void (*)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_G0], label, val, find, part, n0, (void (**)(void)) g0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_G1], label, val, find, part, n1, (void (**)(void)) g1));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_G2], label, val, find, part, n2, (void (**)(void)) g2));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_G3], label, val, find, part, n3, (void (**)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_G0], label, val, find, part, i0, (void (*)(void)) g0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_G1], label, val, find, part, i1, (void (*)(void)) g1));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_G2], label, val, find, part, i2, (void (*)(void)) g2));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_G3], label, val, find, part, i3, (void (*)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormHasJacobianPreconditioner(PetscWeakForm wf, PetscBool *hasJacPre)
{
  PetscInt       n0, n1, n2, n3;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscValidBoolPointer(hasJacPre, 2);
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GP0], &n0));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GP1], &n1));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GP2], &n2));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GP3], &n3));
  *hasJacPre = n0+n1+n2+n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GP0], label, val, find, part, n0, (void (***)(void)) g0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GP1], label, val, find, part, n1, (void (***)(void)) g1));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GP2], label, val, find, part, n2, (void (***)(void)) g2));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GP3], label, val, find, part, n3, (void (***)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GP0], label, val, find, part, (void (*)(void)) g0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GP1], label, val, find, part, (void (*)(void)) g1));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GP2], label, val, find, part, (void (*)(void)) g2));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GP3], label, val, find, part, (void (*)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GP0], label, val, find, part, n0, (void (**)(void)) g0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GP1], label, val, find, part, n1, (void (**)(void)) g1));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GP2], label, val, find, part, n2, (void (**)(void)) g2));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GP3], label, val, find, part, n3, (void (**)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GP0], label, val, find, part, i0, (void (*)(void)) g0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GP1], label, val, find, part, i1, (void (*)(void)) g1));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GP2], label, val, find, part, i2, (void (*)(void)) g2));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GP3], label, val, find, part, i3, (void (*)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormHasBdJacobian(PetscWeakForm wf, PetscBool *hasJac)
{
  PetscInt       n0, n1, n2, n3;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscValidBoolPointer(hasJac, 2);
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDG0], &n0));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDG1], &n1));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDG2], &n2));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDG3], &n3));
  *hasJac = n0+n1+n2+n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetBdJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDG0], label, val, find, part, n0, (void (***)(void)) g0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDG1], label, val, find, part, n1, (void (***)(void)) g1));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDG2], label, val, find, part, n2, (void (***)(void)) g2));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDG3], label, val, find, part, n3, (void (***)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddBdJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDG0], label, val, find, part, (void (*)(void)) g0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDG1], label, val, find, part, (void (*)(void)) g1));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDG2], label, val, find, part, (void (*)(void)) g2));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDG3], label, val, find, part, (void (*)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetBdJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDG0], label, val, find, part, n0, (void (**)(void)) g0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDG1], label, val, find, part, n1, (void (**)(void)) g1));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDG2], label, val, find, part, n2, (void (**)(void)) g2));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDG3], label, val, find, part, n3, (void (**)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexBdJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDG0], label, val, find, part, i0, (void (*)(void)) g0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDG1], label, val, find, part, i1, (void (*)(void)) g1));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDG2], label, val, find, part, i2, (void (*)(void)) g2));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDG3], label, val, find, part, i3, (void (*)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormHasBdJacobianPreconditioner(PetscWeakForm wf, PetscBool *hasJacPre)
{
  PetscInt       n0, n1, n2, n3;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscValidBoolPointer(hasJacPre, 2);
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDGP0], &n0));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDGP1], &n1));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDGP2], &n2));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDGP3], &n3));
  *hasJacPre = n0+n1+n2+n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetBdJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDGP0], label, val, find, part, n0, (void (***)(void)) g0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDGP1], label, val, find, part, n1, (void (***)(void)) g1));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDGP2], label, val, find, part, n2, (void (***)(void)) g2));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDGP3], label, val, find, part, n3, (void (***)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddBdJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDGP0], label, val, find, part, (void (*)(void)) g0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDGP1], label, val, find, part, (void (*)(void)) g1));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDGP2], label, val, find, part, (void (*)(void)) g2));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDGP3], label, val, find, part, (void (*)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetBdJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDGP0], label, val, find, part, n0, (void (**)(void)) g0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDGP1], label, val, find, part, n1, (void (**)(void)) g1));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDGP2], label, val, find, part, n2, (void (**)(void)) g2));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDGP3], label, val, find, part, n3, (void (**)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexBdJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDGP0], label, val, find, part, i0, (void (*)(void)) g0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDGP1], label, val, find, part, i1, (void (*)(void)) g1));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDGP2], label, val, find, part, i2, (void (*)(void)) g2));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDGP3], label, val, find, part, i3, (void (*)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormHasDynamicJacobian(PetscWeakForm wf, PetscBool *hasDynJac)
{
  PetscInt       n0, n1, n2, n3;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscValidBoolPointer(hasDynJac, 2);
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GT0], &n0));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GT1], &n1));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GT2], &n2));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GT3], &n3));
  *hasDynJac = n0+n1+n2+n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetDynamicJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GT0], label, val, find, part, n0, (void (***)(void)) g0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GT1], label, val, find, part, n1, (void (***)(void)) g1));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GT2], label, val, find, part, n2, (void (***)(void)) g2));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GT3], label, val, find, part, n3, (void (***)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormAddDynamicJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GT0], label, val, find, part, (void (*)(void)) g0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GT1], label, val, find, part, (void (*)(void)) g1));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GT2], label, val, find, part, (void (*)(void)) g2));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GT3], label, val, find, part, (void (*)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetDynamicJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GT0], label, val, find, part, n0, (void (**)(void)) g0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GT1], label, val, find, part, n1, (void (**)(void)) g1));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GT2], label, val, find, part, n2, (void (**)(void)) g2));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GT3], label, val, find, part, n3, (void (**)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexDynamicJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part,
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

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GT0], label, val, find, part, i0, (void (*)(void)) g0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GT1], label, val, find, part, i1, (void (*)(void)) g1));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GT2], label, val, find, part, i2, (void (*)(void)) g2));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GT3], label, val, find, part, i3, (void (*)(void)) g3));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormGetRiemannSolver(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt *n,
                                             void (***r)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_R], label, val, f, part, n, (void (***)(void)) r));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetRiemannSolver(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part,
                                             PetscInt n,
                                             void (**r)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_R], label, val, f, part, n, (void (**)(void)) r));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscWeakFormSetIndexRiemannSolver(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part,
                                                  PetscInt i,
                                                  void (*r)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_R], label, val, f, part, i, (void (*)(void)) r));
  PetscFunctionReturn(0);
}

/*@
  PetscWeakFormGetNumFields - Returns the number of fields

  Not collective

  Input Parameter:
. wf - The PetscWeakForm object

  Output Parameter:
. Nf - The number of fields

  Level: beginner

.seealso: PetscWeakFormSetNumFields(), PetscWeakFormCreate()
@*/
PetscErrorCode PetscWeakFormGetNumFields(PetscWeakForm wf, PetscInt *Nf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscValidIntPointer(Nf, 2);
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
  PetscInt       f;

  PetscFunctionBegin;
  if (!*wf) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*wf), PETSCWEAKFORM_CLASSID, 1);

  if (--((PetscObject)(*wf))->refct > 0) {*wf = NULL; PetscFunctionReturn(0);}
  ((PetscObject) (*wf))->refct = 0;
  PetscCall(PetscChunkBufferDestroy(&(*wf)->funcs));
  for (f = 0; f < PETSC_NUM_WF; ++f) PetscCall(PetscHMapFormDestroy(&(*wf)->form[f]));
  PetscCall(PetscFree((*wf)->form));
  PetscCall(PetscHeaderDestroy(wf));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscWeakFormViewTable_Ascii(PetscWeakForm wf, PetscViewer viewer, PetscBool splitField, const char tableName[], PetscHMapForm map)
{
  PetscInt       Nf = wf->Nf, Nk, k;

  PetscFunctionBegin;
  PetscCall(PetscHMapFormGetSize(map, &Nk));
  if (Nk) {
    PetscFormKey *keys;
    void       (**funcs)(void);
    const char  **names;
    PetscInt     *values, *idx1, *idx2, *idx;
    PetscBool     showPart = PETSC_FALSE, showPointer = PETSC_FALSE;
    PetscInt      off = 0;

    PetscCall(PetscMalloc6(Nk, &keys, Nk, &names, Nk, &values, Nk, &idx1, Nk, &idx2, Nk, &idx));
    PetscCall(PetscHMapFormGetKeys(map, &off, keys));
    /* Sort keys by label name and value */
    {
      /* First sort values */
      for (k = 0; k < Nk; ++k) {values[k] = keys[k].value; idx1[k] = k;}
      PetscCall(PetscSortIntWithPermutation(Nk, values, idx1));
      /* If the string sort is stable, it will be sorted correctly overall */
      for (k = 0; k < Nk; ++k) {
        if (keys[idx1[k]].label) PetscCall(PetscObjectGetName((PetscObject) keys[idx1[k]].label, &names[k]));
        else                     {names[k] = "";}
        idx2[k] = k;
      }
      PetscCall(PetscSortStrWithPermutation(Nk, names, idx2));
      for (k = 0; k < Nk; ++k) {
        if (keys[k].label) PetscCall(PetscObjectGetName((PetscObject) keys[k].label, &names[k]));
        else               {names[k] = "";}
        idx[k] = idx1[idx2[k]];
      }
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s\n", tableName));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    for (k = 0; k < Nk; ++k) {
      if (keys[k].part != 0) showPart = PETSC_TRUE;
    }
    for (k = 0; k < Nk; ++k) {
      const PetscInt i = idx[k];
      PetscInt       n, f;

      if (keys[i].label) {
        if (showPointer) PetscCall(PetscViewerASCIIPrintf(viewer, "(%s:%p, %D) ", names[i], keys[i].label, keys[i].value));
        else             PetscCall(PetscViewerASCIIPrintf(viewer, "(%s, %D) ", names[i], keys[i].value));
      } else PetscCall(PetscViewerASCIIPrintf(viewer, ""));
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
      if (splitField) PetscCall(PetscViewerASCIIPrintf(viewer, "(%D, %D) ", keys[i].field/Nf, keys[i].field%Nf));
      else            PetscCall(PetscViewerASCIIPrintf(viewer, "(%D) ", keys[i].field));
      if (showPart)   PetscCall(PetscViewerASCIIPrintf(viewer, "(%D) ", keys[i].part));
      PetscCall(PetscWeakFormGetFunction_Private(wf, map, keys[i].label, keys[i].value, keys[i].field, keys[i].part, &n, &funcs));
      for (f = 0; f < n; ++f) {
        char  *fname;
        size_t len, l;

        if (f > 0) PetscCall(PetscViewerASCIIPrintf(viewer, ", "));
        PetscCall(PetscDLAddr(funcs[f], &fname));
        if (fname) {
          /* Eliminate argument types */
          PetscCall(PetscStrlen(fname, &len));
          for (l = 0; l < len; ++l) if (fname[l] == '(') {fname[l] = '\0'; break;}
          PetscCall(PetscViewerASCIIPrintf(viewer, "%s", fname));
        } else if (showPointer) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "%p", funcs[f]));
        }
        PetscCall(PetscFree(fname));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscFree6(keys, names, values, idx1, idx2, idx));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscWeakFormView_Ascii(PetscWeakForm wf, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscInt          f;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Weak Form System with %d fields\n", wf->Nf));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  for (f = 0; f < PETSC_NUM_WF; ++f) {
    PetscCall(PetscWeakFormViewTable_Ascii(wf, viewer, PETSC_TRUE, PetscWeakFormKinds[f], wf->form[f]));
  }
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

/*@C
  PetscWeakFormView - Views a PetscWeakForm

  Collective on wf

  Input Parameters:
+ wf - the PetscWeakForm object to view
- v  - the viewer

  Level: developer

.seealso PetscWeakFormDestroy(), PetscWeakFormCreate()
@*/
PetscErrorCode PetscWeakFormView(PetscWeakForm wf, PetscViewer v)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  if (!v) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) wf), &v));
  else    {PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);}
  PetscCall(PetscObjectTypeCompare((PetscObject) v, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscWeakFormView_Ascii(wf, v));
  if (wf->ops->view) PetscCall((*wf->ops->view)(wf, v));
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
  PetscInt       f;

  PetscFunctionBegin;
  PetscValidPointer(wf, 2);
  *wf  = NULL;
  PetscCall(PetscDSInitializePackage());

  PetscCall(PetscHeaderCreate(p, PETSCWEAKFORM_CLASSID, "PetscWeakForm", "Weak Form System", "PetscWeakForm", comm, PetscWeakFormDestroy, PetscWeakFormView));

  p->Nf = 0;
  PetscCall(PetscChunkBufferCreate(sizeof(&PetscWeakFormCreate), 2, &p->funcs));
  PetscCall(PetscMalloc1(PETSC_NUM_WF, &p->form));
  for (f = 0; f < PETSC_NUM_WF; ++f) PetscCall(PetscHMapFormCreate(&p->form[f]));
  *wf = p;
  PetscFunctionReturn(0);
}
