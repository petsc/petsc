#include <petsc/private/petscdsimpl.h> /*I "petscds.h" I*/

PetscClassId PETSCWEAKFORM_CLASSID = 0;

const char *const PetscWeakFormKinds[] = {"objective", "residual_f0", "residual_f1", "jacobian_g0", "jacobian_g1", "jacobian_g2", "jacobian_g3", "jacobian_preconditioner_g0", "jacobian_preconditioner_g1", "jacobian_preconditioner_g2", "jacobian_preconditioner_g3", "dynamic_jacobian_g0", "dynamic_jacobian_g1", "dynamic_jacobian_g2", "dynamic_jacobian_g3", "boundary_residual_f0", "boundary_residual_f1", "boundary_jacobian_g0", "boundary_jacobian_g1", "boundary_jacobian_g2", "boundary_jacobian_g3", "boundary_jacobian_preconditioner_g0", "boundary_jacobian_preconditioner_g1", "boundary_jacobian_preconditioner_g2", "boundary_jacobian_preconditioner_g3", "riemann_solver", "PetscWeakFormKind", "PETSC_WF_", NULL};

static PetscErrorCode PetscChunkBufferCreate(size_t unitbytes, PetscCount expected, PetscChunkBuffer *buffer[])
{
  PetscFunctionBegin;
  PetscCall(PetscNew(buffer));
  PetscCall(PetscCalloc1(expected * unitbytes, &(*buffer)->array));
  (*buffer)->size      = expected;
  (*buffer)->unitbytes = unitbytes;
  (*buffer)->alloc     = expected * unitbytes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscChunkBufferDuplicate(PetscChunkBuffer *buffer, PetscChunkBuffer *bufferNew[])
{
  PetscFunctionBegin;
  PetscCall(PetscNew(bufferNew));
  PetscCall(PetscCalloc1(buffer->size * buffer->unitbytes, &(*bufferNew)->array));
  PetscCall(PetscMemcpy((*bufferNew)->array, buffer->array, buffer->size * buffer->unitbytes));
  (*bufferNew)->size      = buffer->size;
  (*bufferNew)->unitbytes = buffer->unitbytes;
  (*bufferNew)->alloc     = buffer->size * buffer->unitbytes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscChunkBufferDestroy(PetscChunkBuffer **buffer)
{
  PetscFunctionBegin;
  PetscCall(PetscFree((*buffer)->array));
  PetscCall(PetscFree(*buffer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscChunkBufferCreateChunk(PetscChunkBuffer *buffer, PetscCount size, PetscChunk *chunk)
{
  PetscFunctionBegin;
  if ((buffer->size + size) * buffer->unitbytes > buffer->alloc) {
    char *tmp;

    if (!buffer->alloc) buffer->alloc = (buffer->size + size) * buffer->unitbytes;
    while ((buffer->size + size) * buffer->unitbytes > buffer->alloc) buffer->alloc *= 2;
    PetscCall(PetscMalloc(buffer->alloc, &tmp));
    PetscCall(PetscMemcpy(tmp, buffer->array, buffer->size * buffer->unitbytes));
    PetscCall(PetscFree(buffer->array));
    buffer->array = tmp;
  }
  chunk->start    = buffer->size * buffer->unitbytes;
  chunk->size     = size;
  chunk->reserved = size;
  buffer->size += size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscChunkBufferEnlargeChunk(PetscChunkBuffer *buffer, PetscCount size, PetscChunk *chunk)
{
  size_t siz = size;

  PetscFunctionBegin;
  if (chunk->size + size > chunk->reserved) {
    PetscChunk newchunk;
    PetscCount reserved = chunk->size;

    /* TODO Here if we had a chunk list, we could update them all to reclaim unused space */
    while (reserved < chunk->size + size) reserved *= 2;
    PetscCall(PetscChunkBufferCreateChunk(buffer, (size_t)reserved, &newchunk));
    newchunk.size = chunk->size + size;
    PetscCall(PetscMemcpy(&buffer->array[newchunk.start], &buffer->array[chunk->start], chunk->size * buffer->unitbytes));
    *chunk = newchunk;
  } else {
    chunk->size += siz;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFormKeySort - Sorts an array of `PetscFormKey` in place in increasing order.

  Not Collective

  Input Parameters:
+ n   - number of values
- arr - array of `PetscFormKey`

  Level: intermediate

.seealso: `PetscFormKey`, `PetscIntSortSemiOrdered()`, `PetscSortInt()`
@*/
PetscErrorCode PetscFormKeySort(PetscInt n, PetscFormKey arr[])
{
  PetscFunctionBegin;
  if (n <= 1) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(arr, 2);
  PetscCall(PetscTimSort(n, arr, sizeof(PetscFormKey), Compare_PetscFormKey_Private, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscWeakFormGetFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt part, PetscInt *n, void (***func)(void))
{
  PetscFormKey key;
  PetscChunk   chunk;

  PetscFunctionBegin;
  key.label = label;
  key.value = value;
  key.field = f;
  key.part  = part;
  PetscCall(PetscHMapFormGet(ht, key, &chunk));
  if (chunk.size < 0) {
    *n    = 0;
    *func = NULL;
  } else {
    PetscCall(PetscIntCast(chunk.size, n));
    *func = (PetscVoidFn **)&wf->funcs->array[chunk.start];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* A NULL argument for func causes this to clear the key */
static PetscErrorCode PetscWeakFormSetFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt part, PetscInt n, PetscVoidFn **func)
{
  PetscFormKey key;
  PetscChunk   chunk;

  PetscFunctionBegin;
  key.label = label;
  key.value = value;
  key.field = f;
  key.part  = part;
  if (!func) {
    PetscCall(PetscHMapFormDel(ht, key));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else PetscCall(PetscHMapFormGet(ht, key, &chunk));
  if (chunk.size < 0) {
    PetscCall(PetscChunkBufferCreateChunk(wf->funcs, n, &chunk));
    PetscCall(PetscHMapFormSet(ht, key, chunk));
  } else if (chunk.size <= n) {
    PetscCall(PetscChunkBufferEnlargeChunk(wf->funcs, n - chunk.size, &chunk));
    PetscCall(PetscHMapFormSet(ht, key, chunk));
  }
  for (PetscInt i = 0; i < n; ++i) ((PetscVoidFn **)&wf->funcs->array[chunk.start])[i] = func[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscWeakFormAddFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt part, PetscVoidFn *func)
{
  PetscFormKey key;
  PetscChunk   chunk;

  PetscFunctionBegin;
  if (!func) PetscFunctionReturn(PETSC_SUCCESS);
  key.label = label;
  key.value = value;
  key.field = f;
  key.part  = part;
  PetscCall(PetscHMapFormGet(ht, key, &chunk));
  if (chunk.size < 0) {
    PetscCall(PetscChunkBufferCreateChunk(wf->funcs, 1, &chunk));
    PetscCall(PetscHMapFormSet(ht, key, chunk));
    ((PetscVoidFn **)&wf->funcs->array[chunk.start])[0] = func;
  } else {
    PetscCall(PetscChunkBufferEnlargeChunk(wf->funcs, 1, &chunk));
    PetscCall(PetscHMapFormSet(ht, key, chunk));
    ((PetscVoidFn **)&wf->funcs->array[chunk.start])[chunk.size - 1] = func;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscWeakFormGetIndexFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt part, PetscInt ind, PetscVoidFn **func)
{
  PetscFormKey key;
  PetscChunk   chunk;

  PetscFunctionBegin;
  key.label = label;
  key.value = value;
  key.field = f;
  key.part  = part;
  PetscCall(PetscHMapFormGet(ht, key, &chunk));
  if (chunk.size < 0) {
    *func = NULL;
  } else {
    PetscCheck(ind < chunk.size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT " not in [0, %" PetscCount_FMT ")", ind, chunk.size);
    *func = ((PetscVoidFn **)&wf->funcs->array[chunk.start])[ind];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Ignore a NULL func */
static PetscErrorCode PetscWeakFormSetIndexFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt part, PetscInt ind, PetscVoidFn *func)
{
  PetscFormKey key;
  PetscChunk   chunk;

  PetscFunctionBegin;
  if (!func) PetscFunctionReturn(PETSC_SUCCESS);
  key.label = label;
  key.value = value;
  key.field = f;
  key.part  = part;
  PetscCall(PetscHMapFormGet(ht, key, &chunk));
  if (chunk.size < 0) {
    PetscCall(PetscChunkBufferCreateChunk(wf->funcs, ind + 1, &chunk));
    PetscCall(PetscHMapFormSet(ht, key, chunk));
  } else if (chunk.size <= ind) {
    PetscCall(PetscChunkBufferEnlargeChunk(wf->funcs, ind - chunk.size + 1, &chunk));
    PetscCall(PetscHMapFormSet(ht, key, chunk));
  }
  ((PetscVoidFn **)&wf->funcs->array[chunk.start])[ind] = func;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscWeakFormClearIndexFunction_Private(PetscWeakForm wf, PetscHMapForm ht, DMLabel label, PetscInt value, PetscInt f, PetscInt part, PetscInt ind)
{
  PetscFormKey key;
  PetscChunk   chunk;

  PetscFunctionBegin;
  key.label = label;
  key.value = value;
  key.field = f;
  key.part  = part;
  PetscCall(PetscHMapFormGet(ht, key, &chunk));
  if (chunk.size < 0) PetscFunctionReturn(PETSC_SUCCESS);
  else if (!ind && chunk.size == 1) {
    PetscCall(PetscHMapFormDel(ht, key));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else if (chunk.size <= ind) PetscFunctionReturn(PETSC_SUCCESS);
  ((PetscVoidFn **)&wf->funcs->array[chunk.start])[ind] = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormCopy - Copy the pointwise functions to another `PetscWeakForm`

  Not Collective

  Input Parameter:
. wf - The original `PetscWeakForm`

  Output Parameter:
. wfNew - The copy of the `PetscWeakForm`

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormCreate()`, `PetscWeakFormDestroy()`
@*/
PetscErrorCode PetscWeakFormCopy(PetscWeakForm wf, PetscWeakForm wfNew)
{
  PetscInt f;

  PetscFunctionBegin;
  wfNew->Nf = wf->Nf;
  PetscCall(PetscChunkBufferDestroy(&wfNew->funcs));
  PetscCall(PetscChunkBufferDuplicate(wf->funcs, &wfNew->funcs));
  for (f = 0; f < PETSC_NUM_WF; ++f) {
    PetscCall(PetscHMapFormDestroy(&wfNew->form[f]));
    PetscCall(PetscHMapFormDuplicate(wf->form[f], &wfNew->form[f]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormClear - Clear all functions from the `PetscWeakForm`

  Not Collective

  Input Parameter:
. wf - The original `PetscWeakForm`

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormCopy()`, `PetscWeakFormCreate()`, `PetscWeakFormDestroy()`
@*/
PetscErrorCode PetscWeakFormClear(PetscWeakForm wf)
{
  PetscInt f;

  PetscFunctionBegin;
  for (f = 0; f < PETSC_NUM_WF; ++f) PetscCall(PetscHMapFormClear(wf->form[f]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscWeakFormRewriteKeys_Internal(PetscWeakForm wf, PetscHMapForm hmap, DMLabel label, PetscInt Nv, const PetscInt values[])
{
  PetscFormKey *keys;
  PetscVoidFn **tmpfuncs;
  PetscInt      n, off = 0, maxNf = 0;

  PetscFunctionBegin;
  PetscCall(PetscHMapFormGetSize(hmap, &n));
  PetscCall(PetscMalloc1(n, &keys));
  PetscCall(PetscHMapFormGetKeys(hmap, &off, keys));
  // Need to make a copy since SetFunction() can invalidate the storage
  for (PetscInt i = 0; i < n; ++i) {
    if (keys[i].label == label) {
      PetscVoidFn **funcs;
      PetscInt      Nf;

      PetscCall(PetscWeakFormGetFunction_Private(wf, hmap, keys[i].label, keys[i].value, keys[i].field, keys[i].part, &Nf, &funcs));
      maxNf = PetscMax(maxNf, Nf);
    }
  }
  PetscCall(PetscMalloc1(maxNf, &tmpfuncs));
  for (PetscInt i = 0; i < n; ++i) {
    if (keys[i].label == label) {
      PetscBool     clear = PETSC_TRUE;
      PetscVoidFn **funcs;
      PetscInt      Nf;

      PetscCall(PetscWeakFormGetFunction_Private(wf, hmap, keys[i].label, keys[i].value, keys[i].field, keys[i].part, &Nf, &funcs));
      for (PetscInt f = 0; f < Nf; ++f) tmpfuncs[f] = funcs[f];
      for (PetscInt v = 0; v < Nv; ++v) {
        PetscCall(PetscWeakFormSetFunction_Private(wf, hmap, keys[i].label, values[v], keys[i].field, keys[i].part, Nf, tmpfuncs));
        if (values[v] == keys[i].value) clear = PETSC_FALSE;
      }
      if (clear) PetscCall(PetscWeakFormSetFunction_Private(wf, hmap, keys[i].label, keys[i].value, keys[i].field, keys[i].part, 0, NULL));
    }
  }
  PetscCall(PetscFree(tmpfuncs));
  PetscCall(PetscFree(keys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormRewriteKeys - Change any key on the given label to use the new set of label values

  Not Collective

  Input Parameters:
+ wf     - The original `PetscWeakForm`
. label  - The label to change keys for
. Nv     - The number of new label values
- values - The set of new values to relabel keys with

  Level: intermediate

  Note:
  This is used internally when boundary label values are specified from the command line.

.seealso: `PetscWeakForm`, `DMLabel`, `PetscWeakFormReplaceLabel()`, `PetscWeakFormCreate()`, `PetscWeakFormDestroy()`
@*/
PetscErrorCode PetscWeakFormRewriteKeys(PetscWeakForm wf, DMLabel label, PetscInt Nv, const PetscInt values[])
{
  PetscFunctionBegin;
  for (PetscInt f = 0; f < PETSC_NUM_WF; ++f) PetscCall(PetscWeakFormRewriteKeys_Internal(wf, wf->form[f], label, Nv, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscWeakFormReplaceLabel_Internal(PetscWeakForm wf, PetscHMapForm hmap, DMLabel label)
{
  PetscFormKey *keys;
  PetscInt      n, i, off = 0, maxFuncs = 0;
  PetscVoidFn **tmpf;
  const char   *name = NULL;

  PetscFunctionBegin;
  if (label) PetscCall(PetscObjectGetName((PetscObject)label, &name));
  PetscCall(PetscHMapFormGetSize(hmap, &n));
  PetscCall(PetscMalloc1(n, &keys));
  PetscCall(PetscHMapFormGetKeys(hmap, &off, keys));
  for (i = 0; i < n; ++i) {
    PetscBool   match = PETSC_FALSE;
    const char *lname = NULL;

    if (label == keys[i].label) continue;
    if (keys[i].label) PetscCall(PetscObjectGetName((PetscObject)keys[i].label, &lname));
    PetscCall(PetscStrcmp(name, lname, &match));
    if ((!name && !lname) || match) {
      PetscVoidFn **funcs;
      PetscInt      Nf;

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
    if (keys[i].label) PetscCall(PetscObjectGetName((PetscObject)keys[i].label, &lname));
    PetscCall(PetscStrcmp(name, lname, &match));
    if ((!name && !lname) || match) {
      PetscVoidFn **funcs;
      PetscInt      Nf;

      PetscCall(PetscWeakFormGetFunction_Private(wf, hmap, keys[i].label, keys[i].value, keys[i].field, keys[i].part, &Nf, &funcs));
      for (PetscInt j = 0; j < Nf; ++j) tmpf[j] = funcs[j];
      PetscCall(PetscWeakFormSetFunction_Private(wf, hmap, label, keys[i].value, keys[i].field, keys[i].part, Nf, tmpf));
      PetscCall(PetscWeakFormSetFunction_Private(wf, hmap, keys[i].label, keys[i].value, keys[i].field, keys[i].part, 0, NULL));
    }
  }
  PetscCall(PetscFree(tmpf));
  PetscCall(PetscFree(keys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormReplaceLabel - Change any key on a label of the same name to use the new label

  Not Collective

  Input Parameters:
+ wf    - The original `PetscWeakForm`
- label - The label to change keys for

  Level: intermediate

  Note:
  This is used internally when meshes are modified

.seealso: `PetscWeakForm`, `DMLabel`, `PetscWeakFormRewriteKeys()`, `PetscWeakFormCreate()`, `PetscWeakFormDestroy()`
@*/
PetscErrorCode PetscWeakFormReplaceLabel(PetscWeakForm wf, DMLabel label)
{
  PetscFunctionBegin;
  for (PetscInt f = 0; f < PETSC_NUM_WF; ++f) PetscCall(PetscWeakFormReplaceLabel_Internal(wf, wf->form[f], label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormClearIndex - Clear the pointwise function at a given index for the given key from a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
. part  - The equation part, or 0 if unused
. kind  - The kind of weak form, see `PetscWeakFormKind`
- ind   - The index of the function to clear in the function list for this key

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormKind`, `PetscWeakFormCreate()`
@*/
PetscErrorCode PetscWeakFormClearIndex(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscWeakFormKind kind, PetscInt ind)
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormClearIndexFunction_Private(wf, wf->form[kind], label, val, f, part, ind));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormGetObjective - Retrieve the list of objective pointwise functions for a given key from a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
- part  - The equation part, or 0 if unused

  Output Parameters:
+ n   - The number of objective pointwise functions registered for this key
- obj - The array of objective pointwise functions

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetObjective()`, `PetscWeakFormAddObjective()`, `PetscWeakFormSetIndexObjective()`, `PetscWeakFormGetIndexObjective()`
@*/
PetscErrorCode PetscWeakFormGetObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt *n, void (***obj)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_OBJECTIVE], label, val, f, part, n, (void (***)(void))obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetObjective - Set the list of objective pointwise functions for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
. part  - The equation part, or 0 if unused
. n     - The number of objective pointwise functions to set
- obj   - The array of objective pointwise functions, or `NULL` to clear the key

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormGetObjective()`, `PetscWeakFormAddObjective()`, `PetscWeakFormSetIndexObjective()`
@*/
PetscErrorCode PetscWeakFormSetObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt n, void (**obj)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_OBJECTIVE], label, val, f, part, n, (PetscVoidFn **)obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormAddObjective - Append an objective pointwise function to the list for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
. part  - The equation part, or 0 if unused
- obj   - The objective pointwise function to append; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetObjective()`, `PetscWeakFormGetObjective()`, `PetscWeakFormSetIndexObjective()`
@*/
PetscErrorCode PetscWeakFormAddObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, void (*obj)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_OBJECTIVE], label, val, f, part, (PetscVoidFn *)obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormGetIndexObjective - Retrieve a single objective pointwise function at the given index for a given key from a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
. part  - The equation part, or 0 if unused
- ind   - The index into the list of objective pointwise functions for this key

  Output Parameter:
. obj - The objective pointwise function at position `ind`, or `NULL` if no function is registered for this key

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetIndexObjective()`, `PetscWeakFormGetObjective()`, `PetscWeakFormSetObjective()`, `PetscWeakFormAddObjective()`
@*/
PetscErrorCode PetscWeakFormGetIndexObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt ind, void (**obj)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetIndexFunction_Private(wf, wf->form[PETSC_WF_OBJECTIVE], label, val, f, part, ind, (PetscVoidFn **)obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetIndexObjective - Set a single objective pointwise function at the given index for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
. part  - The equation part, or 0 if unused
. ind   - The index into the list of objective pointwise functions for this key
- obj   - The objective pointwise function to store at position `ind`; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormGetIndexObjective()`, `PetscWeakFormSetObjective()`, `PetscWeakFormAddObjective()`, `PetscWeakFormClearIndex()`
@*/
PetscErrorCode PetscWeakFormSetIndexObjective(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt ind, void (*obj)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_OBJECTIVE], label, val, f, part, ind, (PetscVoidFn *)obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormGetResidual - Retrieve the lists of residual pointwise functions `f0` and `f1` for a given key from a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
- part  - The equation part, or 0 if unused

  Output Parameters:
+ n0 - The number of `f0` pointwise functions registered for this key
. f0 - The array of `f0` residual pointwise functions
. n1 - The number of `f1` pointwise functions registered for this key
- f1 - The array of `f1` residual pointwise functions

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetResidual()`, `PetscWeakFormAddResidual()`, `PetscWeakFormGetBdResidual()`
@*/
PetscErrorCode PetscWeakFormGetResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt *n0, void (***f0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n1, void (***f1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_F0], label, val, f, part, n0, (void (***)(void))f0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_F1], label, val, f, part, n1, (void (***)(void))f1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormAddResidual - Append residual pointwise functions `f0` and `f1` to the lists for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
. part  - The equation part, or 0 if unused
. f0    - The `f0` residual pointwise function to append; a `NULL` is ignored
- f1    - The `f1` residual pointwise function to append; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetResidual()`, `PetscWeakFormGetResidual()`, `PetscWeakFormSetIndexResidual()`, `PetscWeakFormAddBdResidual()`
@*/
PetscErrorCode PetscWeakFormAddResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, void (*f0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*f1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_F0], label, val, f, part, (PetscVoidFn *)f0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_F1], label, val, f, part, (PetscVoidFn *)f1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetResidual - Set the lists of residual pointwise functions `f0` and `f1` for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
. part  - The equation part, or 0 if unused
. n0    - The number of `f0` pointwise functions to set
. f0    - The array of `f0` residual pointwise functions, or `NULL` to clear the key
. n1    - The number of `f1` pointwise functions to set
- f1    - The array of `f1` residual pointwise functions, or `NULL` to clear the key

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormGetResidual()`, `PetscWeakFormAddResidual()`, `PetscWeakFormSetIndexResidual()`
@*/
PetscErrorCode PetscWeakFormSetResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt n0, void (**f0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n1, void (**f1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_F0], label, val, f, part, n0, (PetscVoidFn **)f0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_F1], label, val, f, part, n1, (PetscVoidFn **)f1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetIndexResidual - Set the residual pointwise functions `f0` and `f1` at the given indices for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
. part  - The equation part, or 0 if unused
. i0    - The index at which to store `f0` in the `f0` list
. f0    - The `f0` residual pointwise function; a `NULL` is ignored
. i1    - The index at which to store `f1` in the `f1` list
- f1    - The `f1` residual pointwise function; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetResidual()`, `PetscWeakFormAddResidual()`, `PetscWeakFormGetResidual()`, `PetscWeakFormClearIndex()`
@*/
PetscErrorCode PetscWeakFormSetIndexResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt i0, void (*f0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i1, void (*f1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_F0], label, val, f, part, i0, (PetscVoidFn *)f0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_F1], label, val, f, part, i1, (PetscVoidFn *)f1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormGetBdResidual - Retrieve the lists of boundary residual pointwise functions `f0` and `f1` for a given key from a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the boundary region, or `NULL` for the entire boundary
. val   - The label value selecting the boundary region
. f     - The field number
- part  - The equation part, or 0 if unused

  Output Parameters:
+ n0 - The number of `f0` boundary pointwise functions registered for this key
. f0 - The array of `f0` boundary residual pointwise functions
. n1 - The number of `f1` boundary pointwise functions registered for this key
- f1 - The array of `f1` boundary residual pointwise functions

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetBdResidual()`, `PetscWeakFormAddBdResidual()`, `PetscWeakFormGetResidual()`
@*/
PetscErrorCode PetscWeakFormGetBdResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt *n0, void (***f0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n1, void (***f1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDF0], label, val, f, part, n0, (void (***)(void))f0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDF1], label, val, f, part, n1, (void (***)(void))f1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormAddBdResidual - Append boundary residual pointwise functions `f0` and `f1` to the lists for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the boundary region, or `NULL` for the entire boundary
. val   - The label value selecting the boundary region
. f     - The field number
. part  - The equation part, or 0 if unused
. f0    - The `f0` boundary residual pointwise function to append; a `NULL` is ignored
- f1    - The `f1` boundary residual pointwise function to append; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetBdResidual()`, `PetscWeakFormGetBdResidual()`, `PetscWeakFormAddResidual()`
@*/
PetscErrorCode PetscWeakFormAddBdResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, void (*f0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*f1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDF0], label, val, f, part, (PetscVoidFn *)f0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDF1], label, val, f, part, (PetscVoidFn *)f1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetBdResidual - Set the lists of boundary residual pointwise functions `f0` and `f1` for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the boundary region, or `NULL` for the entire boundary
. val   - The label value selecting the boundary region
. f     - The field number
. part  - The equation part, or 0 if unused
. n0    - The number of `f0` boundary pointwise functions to set
. f0    - The array of `f0` boundary residual pointwise functions, or `NULL` to clear the key
. n1    - The number of `f1` boundary pointwise functions to set
- f1    - The array of `f1` boundary residual pointwise functions, or `NULL` to clear the key

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormGetBdResidual()`, `PetscWeakFormAddBdResidual()`, `PetscWeakFormSetResidual()`
@*/
PetscErrorCode PetscWeakFormSetBdResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt n0, void (**f0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n1, void (**f1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDF0], label, val, f, part, n0, (PetscVoidFn **)f0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDF1], label, val, f, part, n1, (PetscVoidFn **)f1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetIndexBdResidual - Set the boundary residual pointwise functions `f0` and `f1` at the given indices for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the boundary region, or `NULL` for the entire boundary
. val   - The label value selecting the boundary region
. f     - The field number
. part  - The equation part, or 0 if unused
. i0    - The index at which to store `f0` in the `f0` list
. f0    - The `f0` boundary residual pointwise function; a `NULL` is ignored
. i1    - The index at which to store `f1` in the `f1` list
- f1    - The `f1` boundary residual pointwise function; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetBdResidual()`, `PetscWeakFormAddBdResidual()`, `PetscWeakFormGetBdResidual()`, `PetscWeakFormClearIndex()`
@*/
PetscErrorCode PetscWeakFormSetIndexBdResidual(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt i0, void (*f0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i1, void (*f1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDF0], label, val, f, part, i0, (PetscVoidFn *)f0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDF1], label, val, f, part, i1, (PetscVoidFn *)f1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormHasJacobian - Returns whether the `PetscWeakForm` has any Jacobian (`g0`, `g1`, `g2`, or `g3`) pointwise functions registered

  Not Collective

  Input Parameter:
. wf - The `PetscWeakForm`

  Output Parameter:
. hasJac - `PETSC_TRUE` if any Jacobian pointwise functions are registered, `PETSC_FALSE` otherwise

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetJacobian()`, `PetscWeakFormGetJacobian()`, `PetscWeakFormHasJacobianPreconditioner()`, `PetscWeakFormHasBdJacobian()`
@*/
PetscErrorCode PetscWeakFormHasJacobian(PetscWeakForm wf, PetscBool *hasJac)
{
  PetscInt n0, n1, n2, n3;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscAssertPointer(hasJac, 2);
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_G0], &n0));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_G1], &n1));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_G2], &n2));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_G3], &n3));
  *hasJac = n0 + n1 + n2 + n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormGetJacobian - Retrieve the lists of Jacobian pointwise functions `g0`, `g1`, `g2`, and `g3` for a given key from a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The test field number
. g     - The trial field number
- part  - The equation part, or 0 if unused

  Output Parameters:
+ n0 - The number of `g0` pointwise functions registered for this key
. g0 - The array of `g0` Jacobian pointwise functions
. n1 - The number of `g1` pointwise functions registered for this key
. g1 - The array of `g1` Jacobian pointwise functions
. n2 - The number of `g2` pointwise functions registered for this key
. g2 - The array of `g2` Jacobian pointwise functions
. n3 - The number of `g3` pointwise functions registered for this key
- g3 - The array of `g3` Jacobian pointwise functions

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetJacobian()`, `PetscWeakFormAddJacobian()`, `PetscWeakFormHasJacobian()`, `PetscWeakFormGetJacobianPreconditioner()`
@*/
PetscErrorCode PetscWeakFormGetJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt *n0, void (***g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n1, void (***g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n2, void (***g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n3, void (***g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_G0], label, val, find, part, n0, (void (***)(void))g0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_G1], label, val, find, part, n1, (void (***)(void))g1));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_G2], label, val, find, part, n2, (void (***)(void))g2));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_G3], label, val, find, part, n3, (void (***)(void))g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormAddJacobian - Append Jacobian pointwise functions `g0`, `g1`, `g2`, and `g3` to the lists for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. g0    - The `g0` Jacobian pointwise function to append; a `NULL` is ignored
. g1    - The `g1` Jacobian pointwise function to append; a `NULL` is ignored
. g2    - The `g2` Jacobian pointwise function to append; a `NULL` is ignored
- g3    - The `g3` Jacobian pointwise function to append; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetJacobian()`, `PetscWeakFormGetJacobian()`, `PetscWeakFormSetIndexJacobian()`
@*/
PetscErrorCode PetscWeakFormAddJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, void (*g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_G0], label, val, find, part, (PetscVoidFn *)g0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_G1], label, val, find, part, (PetscVoidFn *)g1));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_G2], label, val, find, part, (PetscVoidFn *)g2));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_G3], label, val, find, part, (PetscVoidFn *)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetJacobian - Set the lists of Jacobian pointwise functions `g0`, `g1`, `g2`, and `g3` for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. n0    - The number of `g0` pointwise functions to set
. g0    - The array of `g0` Jacobian pointwise functions, or `NULL` to clear the key
. n1    - The number of `g1` pointwise functions to set
. g1    - The array of `g1` Jacobian pointwise functions, or `NULL` to clear the key
. n2    - The number of `g2` pointwise functions to set
. g2    - The array of `g2` Jacobian pointwise functions, or `NULL` to clear the key
. n3    - The number of `g3` pointwise functions to set
- g3    - The array of `g3` Jacobian pointwise functions, or `NULL` to clear the key

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormGetJacobian()`, `PetscWeakFormAddJacobian()`, `PetscWeakFormSetIndexJacobian()`, `PetscWeakFormSetJacobianPreconditioner()`
@*/
PetscErrorCode PetscWeakFormSetJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt n0, void (**g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n1, void (**g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n2, void (**g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n3, void (**g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_G0], label, val, find, part, n0, (PetscVoidFn **)g0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_G1], label, val, find, part, n1, (PetscVoidFn **)g1));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_G2], label, val, find, part, n2, (PetscVoidFn **)g2));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_G3], label, val, find, part, n3, (PetscVoidFn **)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetIndexJacobian - Set the Jacobian pointwise functions `g0`, `g1`, `g2`, and `g3` at the given indices for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. i0    - The index at which to store `g0` in the `g0` list
. g0    - The `g0` Jacobian pointwise function; a `NULL` is ignored
. i1    - The index at which to store `g1` in the `g1` list
. g1    - The `g1` Jacobian pointwise function; a `NULL` is ignored
. i2    - The index at which to store `g2` in the `g2` list
. g2    - The `g2` Jacobian pointwise function; a `NULL` is ignored
. i3    - The index at which to store `g3` in the `g3` list
- g3    - The `g3` Jacobian pointwise function; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetJacobian()`, `PetscWeakFormAddJacobian()`, `PetscWeakFormGetJacobian()`, `PetscWeakFormClearIndex()`
@*/
PetscErrorCode PetscWeakFormSetIndexJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt i0, void (*g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i1, void (*g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i2, void (*g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i3, void (*g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_G0], label, val, find, part, i0, (PetscVoidFn *)g0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_G1], label, val, find, part, i1, (PetscVoidFn *)g1));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_G2], label, val, find, part, i2, (PetscVoidFn *)g2));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_G3], label, val, find, part, i3, (PetscVoidFn *)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormHasJacobianPreconditioner - Returns whether the `PetscWeakForm` has any Jacobian preconditioner (`g0`, `g1`, `g2`, or `g3`) pointwise functions registered

  Not Collective

  Input Parameter:
. wf - The `PetscWeakForm`

  Output Parameter:
. hasJacPre - `PETSC_TRUE` if any Jacobian preconditioner pointwise functions are registered, `PETSC_FALSE` otherwise

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetJacobianPreconditioner()`, `PetscWeakFormGetJacobianPreconditioner()`, `PetscWeakFormHasJacobian()`, `PetscWeakFormHasBdJacobianPreconditioner()`
@*/
PetscErrorCode PetscWeakFormHasJacobianPreconditioner(PetscWeakForm wf, PetscBool *hasJacPre)
{
  PetscInt n0, n1, n2, n3;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscAssertPointer(hasJacPre, 2);
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GP0], &n0));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GP1], &n1));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GP2], &n2));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GP3], &n3));
  *hasJacPre = n0 + n1 + n2 + n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormGetJacobianPreconditioner - Retrieve the lists of Jacobian preconditioner pointwise functions `g0`, `g1`, `g2`, and `g3` for a given key from a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The test field number
. g     - The trial field number
- part  - The equation part, or 0 if unused

  Output Parameters:
+ n0 - The number of `g0` pointwise functions registered for this key
. g0 - The array of `g0` Jacobian preconditioner pointwise functions
. n1 - The number of `g1` pointwise functions registered for this key
. g1 - The array of `g1` Jacobian preconditioner pointwise functions
. n2 - The number of `g2` pointwise functions registered for this key
. g2 - The array of `g2` Jacobian preconditioner pointwise functions
. n3 - The number of `g3` pointwise functions registered for this key
- g3 - The array of `g3` Jacobian preconditioner pointwise functions

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetJacobianPreconditioner()`, `PetscWeakFormAddJacobianPreconditioner()`, `PetscWeakFormHasJacobianPreconditioner()`, `PetscWeakFormGetJacobian()`
@*/
PetscErrorCode PetscWeakFormGetJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt *n0, void (***g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n1, void (***g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n2, void (***g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n3, void (***g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GP0], label, val, find, part, n0, (void (***)(void))g0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GP1], label, val, find, part, n1, (void (***)(void))g1));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GP2], label, val, find, part, n2, (void (***)(void))g2));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GP3], label, val, find, part, n3, (void (***)(void))g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormAddJacobianPreconditioner - Append Jacobian preconditioner pointwise functions `g0`, `g1`, `g2`, and `g3` to the lists for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. g0    - The `g0` Jacobian preconditioner pointwise function to append; a `NULL` is ignored
. g1    - The `g1` Jacobian preconditioner pointwise function to append; a `NULL` is ignored
. g2    - The `g2` Jacobian preconditioner pointwise function to append; a `NULL` is ignored
- g3    - The `g3` Jacobian preconditioner pointwise function to append; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetJacobianPreconditioner()`, `PetscWeakFormGetJacobianPreconditioner()`, `PetscWeakFormAddJacobian()`
@*/
PetscErrorCode PetscWeakFormAddJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, void (*g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GP0], label, val, find, part, (PetscVoidFn *)g0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GP1], label, val, find, part, (PetscVoidFn *)g1));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GP2], label, val, find, part, (PetscVoidFn *)g2));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GP3], label, val, find, part, (PetscVoidFn *)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetJacobianPreconditioner - Set the lists of Jacobian preconditioner pointwise functions `g0`, `g1`, `g2`, and `g3` for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. n0    - The number of `g0` pointwise functions to set
. g0    - The array of `g0` Jacobian preconditioner pointwise functions, or `NULL` to clear the key
. n1    - The number of `g1` pointwise functions to set
. g1    - The array of `g1` Jacobian preconditioner pointwise functions, or `NULL` to clear the key
. n2    - The number of `g2` pointwise functions to set
. g2    - The array of `g2` Jacobian preconditioner pointwise functions, or `NULL` to clear the key
. n3    - The number of `g3` pointwise functions to set
- g3    - The array of `g3` Jacobian preconditioner pointwise functions, or `NULL` to clear the key

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormGetJacobianPreconditioner()`, `PetscWeakFormAddJacobianPreconditioner()`, `PetscWeakFormSetJacobian()`
@*/
PetscErrorCode PetscWeakFormSetJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt n0, void (**g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n1, void (**g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n2, void (**g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n3, void (**g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GP0], label, val, find, part, n0, (PetscVoidFn **)g0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GP1], label, val, find, part, n1, (PetscVoidFn **)g1));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GP2], label, val, find, part, n2, (PetscVoidFn **)g2));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GP3], label, val, find, part, n3, (PetscVoidFn **)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetIndexJacobianPreconditioner - Set the Jacobian preconditioner pointwise functions `g0`, `g1`, `g2`, and `g3` at the given indices for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. i0    - The index at which to store `g0` in the `g0` list
. g0    - The `g0` Jacobian preconditioner pointwise function; a `NULL` is ignored
. i1    - The index at which to store `g1` in the `g1` list
. g1    - The `g1` Jacobian preconditioner pointwise function; a `NULL` is ignored
. i2    - The index at which to store `g2` in the `g2` list
. g2    - The `g2` Jacobian preconditioner pointwise function; a `NULL` is ignored
. i3    - The index at which to store `g3` in the `g3` list
- g3    - The `g3` Jacobian preconditioner pointwise function; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetJacobianPreconditioner()`, `PetscWeakFormAddJacobianPreconditioner()`, `PetscWeakFormGetJacobianPreconditioner()`, `PetscWeakFormClearIndex()`
@*/
PetscErrorCode PetscWeakFormSetIndexJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt i0, void (*g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i1, void (*g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i2, void (*g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i3, void (*g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GP0], label, val, find, part, i0, (PetscVoidFn *)g0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GP1], label, val, find, part, i1, (PetscVoidFn *)g1));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GP2], label, val, find, part, i2, (PetscVoidFn *)g2));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GP3], label, val, find, part, i3, (PetscVoidFn *)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormHasBdJacobian - Returns whether the `PetscWeakForm` has any boundary Jacobian (`g0`, `g1`, `g2`, or `g3`) pointwise functions registered

  Not Collective

  Input Parameter:
. wf - The `PetscWeakForm`

  Output Parameter:
. hasJac - `PETSC_TRUE` if any boundary Jacobian pointwise functions are registered, `PETSC_FALSE` otherwise

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetBdJacobian()`, `PetscWeakFormGetBdJacobian()`, `PetscWeakFormHasJacobian()`, `PetscWeakFormHasBdJacobianPreconditioner()`
@*/
PetscErrorCode PetscWeakFormHasBdJacobian(PetscWeakForm wf, PetscBool *hasJac)
{
  PetscInt n0, n1, n2, n3;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscAssertPointer(hasJac, 2);
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDG0], &n0));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDG1], &n1));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDG2], &n2));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDG3], &n3));
  *hasJac = n0 + n1 + n2 + n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormGetBdJacobian - Retrieve the lists of boundary Jacobian pointwise functions `g0`, `g1`, `g2`, and `g3` for a given key from a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the boundary region, or `NULL` for the entire boundary
. val   - The label value selecting the boundary region
. f     - The test field number
. g     - The trial field number
- part  - The equation part, or 0 if unused

  Output Parameters:
+ n0 - The number of `g0` boundary pointwise functions registered for this key
. g0 - The array of `g0` boundary Jacobian pointwise functions
. n1 - The number of `g1` boundary pointwise functions registered for this key
. g1 - The array of `g1` boundary Jacobian pointwise functions
. n2 - The number of `g2` boundary pointwise functions registered for this key
. g2 - The array of `g2` boundary Jacobian pointwise functions
. n3 - The number of `g3` boundary pointwise functions registered for this key
- g3 - The array of `g3` boundary Jacobian pointwise functions

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetBdJacobian()`, `PetscWeakFormAddBdJacobian()`, `PetscWeakFormHasBdJacobian()`, `PetscWeakFormGetJacobian()`
@*/
PetscErrorCode PetscWeakFormGetBdJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt *n0, void (***g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n1, void (***g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n2, void (***g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n3, void (***g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDG0], label, val, find, part, n0, (void (***)(void))g0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDG1], label, val, find, part, n1, (void (***)(void))g1));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDG2], label, val, find, part, n2, (void (***)(void))g2));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDG3], label, val, find, part, n3, (void (***)(void))g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormAddBdJacobian - Append boundary Jacobian pointwise functions `g0`, `g1`, `g2`, and `g3` to the lists for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the boundary region, or `NULL` for the entire boundary
. val   - The label value selecting the boundary region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. g0    - The `g0` boundary Jacobian pointwise function to append; a `NULL` is ignored
. g1    - The `g1` boundary Jacobian pointwise function to append; a `NULL` is ignored
. g2    - The `g2` boundary Jacobian pointwise function to append; a `NULL` is ignored
- g3    - The `g3` boundary Jacobian pointwise function to append; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetBdJacobian()`, `PetscWeakFormGetBdJacobian()`, `PetscWeakFormAddJacobian()`
@*/
PetscErrorCode PetscWeakFormAddBdJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, void (*g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDG0], label, val, find, part, (PetscVoidFn *)g0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDG1], label, val, find, part, (PetscVoidFn *)g1));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDG2], label, val, find, part, (PetscVoidFn *)g2));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDG3], label, val, find, part, (PetscVoidFn *)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetBdJacobian - Set the lists of boundary Jacobian pointwise functions `g0`, `g1`, `g2`, and `g3` for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the boundary region, or `NULL` for the entire boundary
. val   - The label value selecting the boundary region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. n0    - The number of `g0` boundary pointwise functions to set
. g0    - The array of `g0` boundary Jacobian pointwise functions, or `NULL` to clear the key
. n1    - The number of `g1` boundary pointwise functions to set
. g1    - The array of `g1` boundary Jacobian pointwise functions, or `NULL` to clear the key
. n2    - The number of `g2` boundary pointwise functions to set
. g2    - The array of `g2` boundary Jacobian pointwise functions, or `NULL` to clear the key
. n3    - The number of `g3` boundary pointwise functions to set
- g3    - The array of `g3` boundary Jacobian pointwise functions, or `NULL` to clear the key

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormGetBdJacobian()`, `PetscWeakFormAddBdJacobian()`, `PetscWeakFormSetJacobian()`
@*/
PetscErrorCode PetscWeakFormSetBdJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt n0, void (**g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n1, void (**g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n2, void (**g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n3, void (**g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDG0], label, val, find, part, n0, (PetscVoidFn **)g0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDG1], label, val, find, part, n1, (PetscVoidFn **)g1));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDG2], label, val, find, part, n2, (PetscVoidFn **)g2));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDG3], label, val, find, part, n3, (PetscVoidFn **)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetIndexBdJacobian - Set the boundary Jacobian pointwise functions `g0`, `g1`, `g2`, and `g3` at the given indices for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the boundary region, or `NULL` for the entire boundary
. val   - The label value selecting the boundary region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. i0    - The index at which to store `g0` in the `g0` list
. g0    - The `g0` boundary Jacobian pointwise function; a `NULL` is ignored
. i1    - The index at which to store `g1` in the `g1` list
. g1    - The `g1` boundary Jacobian pointwise function; a `NULL` is ignored
. i2    - The index at which to store `g2` in the `g2` list
. g2    - The `g2` boundary Jacobian pointwise function; a `NULL` is ignored
. i3    - The index at which to store `g3` in the `g3` list
- g3    - The `g3` boundary Jacobian pointwise function; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetBdJacobian()`, `PetscWeakFormAddBdJacobian()`, `PetscWeakFormGetBdJacobian()`, `PetscWeakFormClearIndex()`
@*/
PetscErrorCode PetscWeakFormSetIndexBdJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt i0, void (*g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i1, void (*g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i2, void (*g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i3, void (*g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDG0], label, val, find, part, i0, (PetscVoidFn *)g0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDG1], label, val, find, part, i1, (PetscVoidFn *)g1));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDG2], label, val, find, part, i2, (PetscVoidFn *)g2));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDG3], label, val, find, part, i3, (PetscVoidFn *)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormHasBdJacobianPreconditioner - Returns whether the `PetscWeakForm` has any boundary Jacobian preconditioner (`g0`, `g1`, `g2`, or `g3`) pointwise functions registered

  Not Collective

  Input Parameter:
. wf - The `PetscWeakForm`

  Output Parameter:
. hasJacPre - `PETSC_TRUE` if any boundary Jacobian preconditioner pointwise functions are registered, `PETSC_FALSE` otherwise

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetBdJacobianPreconditioner()`, `PetscWeakFormGetBdJacobianPreconditioner()`, `PetscWeakFormHasBdJacobian()`, `PetscWeakFormHasJacobianPreconditioner()`
@*/
PetscErrorCode PetscWeakFormHasBdJacobianPreconditioner(PetscWeakForm wf, PetscBool *hasJacPre)
{
  PetscInt n0, n1, n2, n3;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscAssertPointer(hasJacPre, 2);
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDGP0], &n0));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDGP1], &n1));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDGP2], &n2));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_BDGP3], &n3));
  *hasJacPre = n0 + n1 + n2 + n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormGetBdJacobianPreconditioner - Retrieve the lists of boundary Jacobian preconditioner pointwise functions `g0`, `g1`, `g2`, and `g3` for a given key from a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the boundary region, or `NULL` for the entire boundary
. val   - The label value selecting the boundary region
. f     - The test field number
. g     - The trial field number
- part  - The equation part, or 0 if unused

  Output Parameters:
+ n0 - The number of `g0` boundary pointwise functions registered for this key
. g0 - The array of `g0` boundary Jacobian preconditioner pointwise functions
. n1 - The number of `g1` boundary pointwise functions registered for this key
. g1 - The array of `g1` boundary Jacobian preconditioner pointwise functions
. n2 - The number of `g2` boundary pointwise functions registered for this key
. g2 - The array of `g2` boundary Jacobian preconditioner pointwise functions
. n3 - The number of `g3` boundary pointwise functions registered for this key
- g3 - The array of `g3` boundary Jacobian preconditioner pointwise functions

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetBdJacobianPreconditioner()`, `PetscWeakFormAddBdJacobianPreconditioner()`, `PetscWeakFormHasBdJacobianPreconditioner()`, `PetscWeakFormGetBdJacobian()`
@*/
PetscErrorCode PetscWeakFormGetBdJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt *n0, void (***g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n1, void (***g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n2, void (***g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n3, void (***g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDGP0], label, val, find, part, n0, (void (***)(void))g0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDGP1], label, val, find, part, n1, (void (***)(void))g1));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDGP2], label, val, find, part, n2, (void (***)(void))g2));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_BDGP3], label, val, find, part, n3, (void (***)(void))g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormAddBdJacobianPreconditioner - Append boundary Jacobian preconditioner pointwise functions `g0`, `g1`, `g2`, and `g3` to the lists for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the boundary region, or `NULL` for the entire boundary
. val   - The label value selecting the boundary region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. g0    - The `g0` boundary Jacobian preconditioner pointwise function to append; a `NULL` is ignored
. g1    - The `g1` boundary Jacobian preconditioner pointwise function to append; a `NULL` is ignored
. g2    - The `g2` boundary Jacobian preconditioner pointwise function to append; a `NULL` is ignored
- g3    - The `g3` boundary Jacobian preconditioner pointwise function to append; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetBdJacobianPreconditioner()`, `PetscWeakFormGetBdJacobianPreconditioner()`, `PetscWeakFormAddBdJacobian()`
@*/
PetscErrorCode PetscWeakFormAddBdJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, void (*g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDGP0], label, val, find, part, (PetscVoidFn *)g0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDGP1], label, val, find, part, (PetscVoidFn *)g1));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDGP2], label, val, find, part, (PetscVoidFn *)g2));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_BDGP3], label, val, find, part, (PetscVoidFn *)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetBdJacobianPreconditioner - Set the lists of boundary Jacobian preconditioner pointwise functions `g0`, `g1`, `g2`, and `g3` for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the boundary region, or `NULL` for the entire boundary
. val   - The label value selecting the boundary region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. n0    - The number of `g0` boundary pointwise functions to set
. g0    - The array of `g0` boundary Jacobian preconditioner pointwise functions, or `NULL` to clear the key
. n1    - The number of `g1` boundary pointwise functions to set
. g1    - The array of `g1` boundary Jacobian preconditioner pointwise functions, or `NULL` to clear the key
. n2    - The number of `g2` boundary pointwise functions to set
. g2    - The array of `g2` boundary Jacobian preconditioner pointwise functions, or `NULL` to clear the key
. n3    - The number of `g3` boundary pointwise functions to set
- g3    - The array of `g3` boundary Jacobian preconditioner pointwise functions, or `NULL` to clear the key

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormGetBdJacobianPreconditioner()`, `PetscWeakFormAddBdJacobianPreconditioner()`, `PetscWeakFormSetBdJacobian()`
@*/
PetscErrorCode PetscWeakFormSetBdJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt n0, void (**g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n1, void (**g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n2, void (**g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n3, void (**g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDGP0], label, val, find, part, n0, (PetscVoidFn **)g0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDGP1], label, val, find, part, n1, (PetscVoidFn **)g1));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDGP2], label, val, find, part, n2, (PetscVoidFn **)g2));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_BDGP3], label, val, find, part, n3, (PetscVoidFn **)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetIndexBdJacobianPreconditioner - Set the boundary Jacobian preconditioner pointwise functions `g0`, `g1`, `g2`, and `g3` at the given indices for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the boundary region, or `NULL` for the entire boundary
. val   - The label value selecting the boundary region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. i0    - The index at which to store `g0` in the `g0` list
. g0    - The `g0` boundary Jacobian preconditioner pointwise function; a `NULL` is ignored
. i1    - The index at which to store `g1` in the `g1` list
. g1    - The `g1` boundary Jacobian preconditioner pointwise function; a `NULL` is ignored
. i2    - The index at which to store `g2` in the `g2` list
. g2    - The `g2` boundary Jacobian preconditioner pointwise function; a `NULL` is ignored
. i3    - The index at which to store `g3` in the `g3` list
- g3    - The `g3` boundary Jacobian preconditioner pointwise function; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetBdJacobianPreconditioner()`, `PetscWeakFormAddBdJacobianPreconditioner()`, `PetscWeakFormGetBdJacobianPreconditioner()`, `PetscWeakFormClearIndex()`
@*/
PetscErrorCode PetscWeakFormSetIndexBdJacobianPreconditioner(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt i0, void (*g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i1, void (*g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i2, void (*g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i3, void (*g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDGP0], label, val, find, part, i0, (PetscVoidFn *)g0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDGP1], label, val, find, part, i1, (PetscVoidFn *)g1));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDGP2], label, val, find, part, i2, (PetscVoidFn *)g2));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_BDGP3], label, val, find, part, i3, (PetscVoidFn *)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormHasDynamicJacobian - Returns whether the `PetscWeakForm` has any dynamic Jacobian (`g0`, `g1`, `g2`, or `g3`) pointwise functions registered

  Not Collective

  Input Parameter:
. wf - The `PetscWeakForm`

  Output Parameter:
. hasDynJac - `PETSC_TRUE` if any dynamic Jacobian pointwise functions are registered, `PETSC_FALSE` otherwise

  Level: intermediate

  Note:
  The dynamic Jacobian is the Jacobian of the time-derivative term for transient problems.

.seealso: `PetscWeakForm`, `PetscWeakFormSetDynamicJacobian()`, `PetscWeakFormGetDynamicJacobian()`, `PetscWeakFormHasJacobian()`
@*/
PetscErrorCode PetscWeakFormHasDynamicJacobian(PetscWeakForm wf, PetscBool *hasDynJac)
{
  PetscInt n0, n1, n2, n3;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscAssertPointer(hasDynJac, 2);
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GT0], &n0));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GT1], &n1));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GT2], &n2));
  PetscCall(PetscHMapFormGetSize(wf->form[PETSC_WF_GT3], &n3));
  *hasDynJac = n0 + n1 + n2 + n3 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormGetDynamicJacobian - Retrieve the lists of dynamic Jacobian pointwise functions `g0`, `g1`, `g2`, and `g3` for a given key from a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The test field number
. g     - The trial field number
- part  - The equation part, or 0 if unused

  Output Parameters:
+ n0 - The number of `g0` pointwise functions registered for this key
. g0 - The array of `g0` dynamic Jacobian pointwise functions
. n1 - The number of `g1` pointwise functions registered for this key
. g1 - The array of `g1` dynamic Jacobian pointwise functions
. n2 - The number of `g2` pointwise functions registered for this key
. g2 - The array of `g2` dynamic Jacobian pointwise functions
. n3 - The number of `g3` pointwise functions registered for this key
- g3 - The array of `g3` dynamic Jacobian pointwise functions

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetDynamicJacobian()`, `PetscWeakFormAddDynamicJacobian()`, `PetscWeakFormHasDynamicJacobian()`, `PetscWeakFormGetJacobian()`
@*/
PetscErrorCode PetscWeakFormGetDynamicJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt *n0, void (***g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n1, void (***g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n2, void (***g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt *n3, void (***g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GT0], label, val, find, part, n0, (void (***)(void))g0));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GT1], label, val, find, part, n1, (void (***)(void))g1));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GT2], label, val, find, part, n2, (void (***)(void))g2));
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_GT3], label, val, find, part, n3, (void (***)(void))g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormAddDynamicJacobian - Append dynamic Jacobian pointwise functions `g0`, `g1`, `g2`, and `g3` to the lists for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. g0    - The `g0` dynamic Jacobian pointwise function to append; a `NULL` is ignored
. g1    - The `g1` dynamic Jacobian pointwise function to append; a `NULL` is ignored
. g2    - The `g2` dynamic Jacobian pointwise function to append; a `NULL` is ignored
- g3    - The `g3` dynamic Jacobian pointwise function to append; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetDynamicJacobian()`, `PetscWeakFormGetDynamicJacobian()`, `PetscWeakFormAddJacobian()`
@*/
PetscErrorCode PetscWeakFormAddDynamicJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, void (*g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void (*g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GT0], label, val, find, part, (PetscVoidFn *)g0));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GT1], label, val, find, part, (PetscVoidFn *)g1));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GT2], label, val, find, part, (PetscVoidFn *)g2));
  PetscCall(PetscWeakFormAddFunction_Private(wf, wf->form[PETSC_WF_GT3], label, val, find, part, (PetscVoidFn *)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetDynamicJacobian - Set the lists of dynamic Jacobian pointwise functions `g0`, `g1`, `g2`, and `g3` for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. n0    - The number of `g0` pointwise functions to set
. g0    - The array of `g0` dynamic Jacobian pointwise functions, or `NULL` to clear the key
. n1    - The number of `g1` pointwise functions to set
. g1    - The array of `g1` dynamic Jacobian pointwise functions, or `NULL` to clear the key
. n2    - The number of `g2` pointwise functions to set
. g2    - The array of `g2` dynamic Jacobian pointwise functions, or `NULL` to clear the key
. n3    - The number of `g3` pointwise functions to set
- g3    - The array of `g3` dynamic Jacobian pointwise functions, or `NULL` to clear the key

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormGetDynamicJacobian()`, `PetscWeakFormAddDynamicJacobian()`, `PetscWeakFormSetJacobian()`
@*/
PetscErrorCode PetscWeakFormSetDynamicJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt n0, void (**g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n1, void (**g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n2, void (**g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt n3, void (**g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GT0], label, val, find, part, n0, (PetscVoidFn **)g0));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GT1], label, val, find, part, n1, (PetscVoidFn **)g1));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GT2], label, val, find, part, n2, (PetscVoidFn **)g2));
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_GT3], label, val, find, part, n3, (PetscVoidFn **)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetIndexDynamicJacobian - Set the dynamic Jacobian pointwise functions `g0`, `g1`, `g2`, and `g3` at the given indices for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The test field number
. g     - The trial field number
. part  - The equation part, or 0 if unused
. i0    - The index at which to store `g0` in the `g0` list
. g0    - The `g0` dynamic Jacobian pointwise function; a `NULL` is ignored
. i1    - The index at which to store `g1` in the `g1` list
. g1    - The `g1` dynamic Jacobian pointwise function; a `NULL` is ignored
. i2    - The index at which to store `g2` in the `g2` list
. g2    - The `g2` dynamic Jacobian pointwise function; a `NULL` is ignored
. i3    - The index at which to store `g3` in the `g3` list
- g3    - The `g3` dynamic Jacobian pointwise function; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetDynamicJacobian()`, `PetscWeakFormAddDynamicJacobian()`, `PetscWeakFormGetDynamicJacobian()`, `PetscWeakFormClearIndex()`
@*/
PetscErrorCode PetscWeakFormSetIndexDynamicJacobian(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt g, PetscInt part, PetscInt i0, void (*g0)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i1, void (*g1)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i2, void (*g2)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt i3, void (*g3)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscInt find = f * wf->Nf + g;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GT0], label, val, find, part, i0, (PetscVoidFn *)g0));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GT1], label, val, find, part, i1, (PetscVoidFn *)g1));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GT2], label, val, find, part, i2, (PetscVoidFn *)g2));
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_GT3], label, val, find, part, i3, (PetscVoidFn *)g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormGetRiemannSolver - Retrieve the list of Riemann solver pointwise functions for a given key from a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
- part  - The equation part, or 0 if unused

  Output Parameters:
+ n - The number of Riemann solver pointwise functions registered for this key
- r - The array of Riemann solver pointwise functions

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetRiemannSolver()`, `PetscWeakFormSetIndexRiemannSolver()`
@*/
PetscErrorCode PetscWeakFormGetRiemannSolver(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt *n, void (***r)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetFunction_Private(wf, wf->form[PETSC_WF_R], label, val, f, part, n, (void (***)(void))r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetRiemannSolver - Set the list of Riemann solver pointwise functions for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
. part  - The equation part, or 0 if unused
. n     - The number of Riemann solver pointwise functions to set
- r     - The array of Riemann solver pointwise functions, or `NULL` to clear the key

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormGetRiemannSolver()`, `PetscWeakFormSetIndexRiemannSolver()`
@*/
PetscErrorCode PetscWeakFormSetRiemannSolver(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt n, void (**r)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetFunction_Private(wf, wf->form[PETSC_WF_R], label, val, f, part, n, (PetscVoidFn **)r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscWeakFormSetIndexRiemannSolver - Set a single Riemann solver pointwise function at the given index for a given key in a `PetscWeakForm`

  Not Collective

  Input Parameters:
+ wf    - The `PetscWeakForm`
. label - The label selecting the mesh region, or `NULL` for the entire domain
. val   - The label value selecting the mesh region
. f     - The field number
. part  - The equation part, or 0 if unused
. i     - The index into the list of Riemann solver pointwise functions for this key
- r     - The Riemann solver pointwise function to store at position `i`; a `NULL` is ignored

  Level: intermediate

.seealso: `PetscWeakForm`, `PetscWeakFormSetRiemannSolver()`, `PetscWeakFormGetRiemannSolver()`, `PetscWeakFormClearIndex()`
@*/
PetscErrorCode PetscWeakFormSetIndexRiemannSolver(PetscWeakForm wf, DMLabel label, PetscInt val, PetscInt f, PetscInt part, PetscInt i, void (*r)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *))
{
  PetscFunctionBegin;
  PetscCall(PetscWeakFormSetIndexFunction_Private(wf, wf->form[PETSC_WF_R], label, val, f, part, i, (PetscVoidFn *)r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormGetNumFields - Returns the number of fields in a `PetscWeakForm`

  Not Collective

  Input Parameter:
. wf - The `PetscWeakForm` object

  Output Parameter:
. Nf - The number of fields

  Level: beginner

.seealso: `PetscWeakForm`, `PetscWeakFormSetNumFields()`, `PetscWeakFormCreate()`
@*/
PetscErrorCode PetscWeakFormGetNumFields(PetscWeakForm wf, PetscInt *Nf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscAssertPointer(Nf, 2);
  *Nf = wf->Nf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormSetNumFields - Sets the number of fields

  Not Collective

  Input Parameters:
+ wf - The `PetscWeakForm` object
- Nf - The number of fields

  Level: beginner

.seealso: `PetscWeakForm`, `PetscWeakFormGetNumFields()`, `PetscWeakFormCreate()`
@*/
PetscErrorCode PetscWeakFormSetNumFields(PetscWeakForm wf, PetscInt Nf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  wf->Nf = Nf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormDestroy - Destroys a `PetscWeakForm` object

  Collective

  Input Parameter:
. wf - the `PetscWeakForm` object to destroy

  Level: developer

.seealso: `PetscWeakForm`, `PetscWeakFormCreate()`, `PetscWeakFormView()`
@*/
PetscErrorCode PetscWeakFormDestroy(PetscWeakForm *wf)
{
  PetscInt f;

  PetscFunctionBegin;
  if (!*wf) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*wf, PETSCWEAKFORM_CLASSID, 1);

  if (--((PetscObject)*wf)->refct > 0) {
    *wf = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  ((PetscObject)*wf)->refct = 0;
  PetscCall(PetscChunkBufferDestroy(&(*wf)->funcs));
  for (f = 0; f < PETSC_NUM_WF; ++f) PetscCall(PetscHMapFormDestroy(&(*wf)->form[f]));
  PetscCall(PetscFree((*wf)->form));
  PetscCall(PetscHeaderDestroy(wf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscWeakFormViewTable_Ascii(PetscWeakForm wf, PetscViewer viewer, PetscBool splitField, const char tableName[], PetscHMapForm map)
{
  PetscInt Nf = wf->Nf, Nk;

  PetscFunctionBegin;
  PetscCall(PetscHMapFormGetSize(map, &Nk));
  if (Nk) {
    PetscFormKey *keys;
    PetscVoidFn **funcs = NULL;
    const char  **names;
    PetscInt     *fields, *values, *idx1, *idx2, *idx3, *idx;
    PetscBool     showPart = PETSC_FALSE, showPointer = PETSC_FALSE;
    PetscInt      off = 0;

    PetscCall(PetscMalloc4(Nk, &keys, Nk, &fields, Nk, &names, Nk, &values));
    PetscCall(PetscMalloc4(Nk, &idx1, Nk, &idx2, Nk, &idx3, Nk, &idx));
    PetscCall(PetscHMapFormGetKeys(map, &off, keys));
    // Sort keys by field, and label name and value
    {
      /* First sort values */
      for (PetscInt k = 0; k < Nk; ++k) {
        values[k] = keys[k].value;
        idx1[k]   = k;
      }
      PetscCall(PetscSortIntWithPermutation(Nk, values, idx1));
      // If the string sort is stable, it will be sorted correctly overall
      for (PetscInt k = 0; k < Nk; ++k) {
        if (keys[idx1[k]].label) PetscCall(PetscObjectGetName((PetscObject)keys[idx1[k]].label, &names[k]));
        else names[k] = "";
        idx2[k] = k;
      }
      PetscCall(PetscSortStrWithPermutation(Nk, names, idx2));
      // If the field sort is stable, it will be sorted correctly overall
      for (PetscInt k = 0; k < Nk; ++k) {
        fields[k] = keys[idx1[idx2[k]]].field;
        idx3[k]   = k;
      }
      PetscCall(PetscSortIntWithPermutation(Nk, fields, idx3));
      for (PetscInt k = 0; k < Nk; ++k) {
        if (keys[k].label) PetscCall(PetscObjectGetName((PetscObject)keys[k].label, &names[k]));
        else names[k] = "";
        idx[k] = idx1[idx2[idx3[k]]];
      }
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s\n", tableName));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    for (PetscInt k = 0; k < Nk; ++k) {
      if (keys[k].part != 0) showPart = PETSC_TRUE;
    }
    for (PetscInt k = 0; k < Nk; ++k) {
      const PetscInt i = idx[k];
      PetscInt       n;

      if (keys[i].label) {
        if (showPointer) PetscCall(PetscViewerASCIIPrintf(viewer, "(%s:%p, %" PetscInt_FMT ") ", names[i], (void *)keys[i].label, keys[i].value));
        else PetscCall(PetscViewerASCIIPrintf(viewer, "(%s, %" PetscInt_FMT ") ", names[i], keys[i].value));
      }
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
      if (splitField) PetscCall(PetscViewerASCIIPrintf(viewer, "(%" PetscInt_FMT ", %" PetscInt_FMT ") ", keys[i].field / Nf, keys[i].field % Nf));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "(%" PetscInt_FMT ") ", keys[i].field));
      if (showPart) PetscCall(PetscViewerASCIIPrintf(viewer, "(%" PetscInt_FMT ") ", keys[i].part));
      PetscCall(PetscWeakFormGetFunction_Private(wf, map, keys[i].label, keys[i].value, keys[i].field, keys[i].part, &n, &funcs));
      for (PetscInt f = 0; f < n; ++f) {
        char  *fname;
        size_t len;

        if (f > 0) PetscCall(PetscViewerASCIIPrintf(viewer, ", "));
        PetscCall(PetscDLAddr(funcs[f], &fname));
        if (fname) {
          /* Eliminate argument types */
          PetscCall(PetscStrlen(fname, &len));
          for (PetscInt l = 0; l < (PetscInt)len; ++l)
            if (fname[l] == '(') {
              fname[l] = '\0';
              break;
            }
          PetscCall(PetscViewerASCIIPrintf(viewer, "%s", fname));
        } else if (showPointer) {
#if defined(__clang__)
          PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN("-Wformat-pedantic")
#elif defined(__GNUC__) || defined(__GNUG__)
          PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN("-Wformat")
#endif
          PetscCall(PetscViewerASCIIPrintf(viewer, "%p", funcs[f]));
          PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END()
        }
        PetscCall(PetscFree(fname));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscFree4(keys, fields, names, values));
    PetscCall(PetscFree4(idx1, idx2, idx3, idx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscWeakFormView_Ascii(PetscWeakForm wf, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Weak Form System with %" PetscInt_FMT " fields\n", wf->Nf));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  for (PetscInt f = 0; f < PETSC_NUM_WF; ++f) PetscCall(PetscWeakFormViewTable_Ascii(wf, viewer, PETSC_TRUE, PetscWeakFormKinds[f], wf->form[f]));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormView - Views a `PetscWeakForm`

  Collective

  Input Parameters:
+ wf - the `PetscWeakForm` object to view
- v  - the viewer

  Level: developer

.seealso: `PetscViewer`, `PetscWeakForm`, `PetscWeakFormDestroy()`, `PetscWeakFormCreate()`
@*/
PetscErrorCode PetscWeakFormView(PetscWeakForm wf, PetscViewer v)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  if (!v) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)wf), &v));
  else PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscWeakFormView_Ascii(wf, v));
  PetscTryTypeMethod(wf, view, v);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormGetKeys - Return an array of `PetscFormKey` of a certain `PetscWeakFormKind`

  Not Collective

  Input Parameters:
+ wf - the `PetscWeakForm` object
- k  - the `PetscWeakFormKind` to get keys for

  Output Parameters:
+ Nk   - the number of keys returned
- keys - the array of `PetscFormKey` objects

  Level: developer

  Note:
  The caller must `PetscFree()` the `keys` array when it is no longer needed.

.seealso: `PetscWeakForm`, `PetscWeakFormDestroy()`, `PetscWeakFormCreate()`
@*/
PetscErrorCode PetscWeakFormGetKeys(PetscWeakForm wf, PetscWeakFormKind k, PetscInt *Nk, PetscFormKey *keys[])
{
  PetscHMapForm map;
  PetscInt      off = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(wf, PETSCWEAKFORM_CLASSID, 1);
  PetscAssertPointer(Nk, 3);
  PetscAssertPointer(keys, 4);
  map = wf->form[k];
  PetscCall(PetscHMapFormGetSize(map, Nk));
  PetscCall(PetscMalloc1(*Nk, keys));
  PetscCall(PetscHMapFormGetKeys(map, &off, *keys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscWeakFormCreate - Creates an empty `PetscWeakForm` object.

  Collective

  Input Parameter:
. comm - The communicator for the `PetscWeakForm` object

  Output Parameter:
. wf - The `PetscWeakForm` object

  Level: beginner

.seealso: `PetscWeakForm`, `PetscDS`, `PetscWeakFormDestroy()`
@*/
PetscErrorCode PetscWeakFormCreate(MPI_Comm comm, PetscWeakForm *wf)
{
  PetscWeakForm p;

  PetscFunctionBegin;
  PetscAssertPointer(wf, 2);
  PetscCall(PetscDSInitializePackage());

  PetscCall(PetscHeaderCreate(p, PETSCWEAKFORM_CLASSID, "PetscWeakForm", "Weak Form System", "PetscWeakForm", comm, PetscWeakFormDestroy, PetscWeakFormView));
  p->Nf = 0;
  PetscCall(PetscChunkBufferCreate(sizeof(&PetscWeakFormCreate), 2, &p->funcs));
  PetscCall(PetscMalloc1(PETSC_NUM_WF, &p->form));
  for (PetscInt f = 0; f < PETSC_NUM_WF; ++f) PetscCall(PetscHMapFormCreate(&p->form[f]));
  *wf = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}
