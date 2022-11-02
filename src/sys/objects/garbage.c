#include <petsc/private/garbagecollector.h>

/* Fetches garbage hashmap from communicator */
static PetscErrorCode GarbageGetHMap_Private(MPI_Comm comm, PetscGarbage *garbage)
{
  PetscMPIInt  flag;
  PetscHMapObj garbage_map;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_get_attr(comm, Petsc_Garbage_HMap_keyval, garbage, &flag));
  if (!flag) {
    /* No garbage,create one */
    PetscCall(PetscHMapObjCreate(&garbage_map));
    garbage->map = garbage_map;
    PetscCallMPI(MPI_Comm_set_attr(comm, Petsc_Garbage_HMap_keyval, garbage->ptr));
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscObjectDelayedDestroy - Adds an object to a data structure for
    later destruction.

    Not Collective

    Input Parameters:
.   obj - object to be destroyed

    Notes:
    Analogue to `PetscObjectDestroy()` for use in managed languages.

    A PETSc object is given a creation index at initialisation based on
    the communicator it was created on and the order in which it is
    created. When this function is passed a PETSc object, a pointer to
    the object is stashed on a garbage dictionary (PetscHMapObj) which is
    keyed by its creation index.

    Objects stashed on this garbage dictionary can later be destroyed
    with a call to `PetscGarbageCleanup()`.

    This function is intended for use with managed languages such as
    Python or Julia, which may not destroy objects in a deterministic
    order.

    Level: developer

.seealso: `PetscGarbageCleanup()`
@*/
PetscErrorCode PetscObjectDelayedDestroy(PetscObject *obj)
{
  MPI_Comm     petsc_comm;
  PetscInt     count;
  PetscGarbage garbage;

  PetscFunctionBegin;
  PetscValidPointer(obj, 1);
  /* Don't stash NULL pointers */
  if (*obj != NULL) {
    /* Elaborate check for getting non-cyclic reference counts */
    if (!(*obj)->non_cyclic_references) {
      count = --(*obj)->refct;
    } else {
      PetscCall((*obj)->non_cyclic_references(*obj, &count));
      --count;
      --(*obj)->refct;
    }
    /* Only stash if the (non-cyclic) reference count hits 0 */
    if (count == 0) {
      (*obj)->refct = 1;
      PetscCall(PetscObjectGetComm(*obj, &petsc_comm));
      PetscCall(GarbageGetHMap_Private(petsc_comm, &garbage));
      PetscCall(PetscHMapObjSet(garbage.map, (*obj)->cidx, *obj));
    }
  }
  *obj = NULL;
  PetscFunctionReturn(0);
}

/* Performs the intersection of 2 sorted arrays seta and setb of lengths
   lena and lenb respectively,returning the result in seta and lena
   This is an O(n) operation */
static PetscErrorCode GarbageKeySortedIntersect_Private(PetscInt64 seta[], PetscInt *lena, PetscInt64 setb[], PetscInt lenb)
{
  /* The arrays seta and setb MUST be sorted! */
  PetscInt ii, jj = 0, counter = 0;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscBool sorted = PETSC_FALSE;
    /* In debug mode check whether the array are sorted */
    PetscCall(PetscSortedInt64(*lena, seta, &sorted));
    PetscCheck(sorted, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Provided array in argument 1 is not sorted");
    PetscCall(PetscSortedInt64(lenb, setb, &sorted));
    PetscCheck(sorted, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Provided array in argument 3 is not sorted");
  }
  for (ii = 0; ii < *lena; ii++) {
    while (jj < lenb && seta[ii] > setb[jj]) { jj++; }
    if (jj >= lenb) break;
    if (seta[ii] == setb[jj]) {
      seta[counter] = seta[ii];
      counter++;
    }
  }

  *lena = counter;
  PetscFunctionReturn(0);
}

/* Wrapper to create MPI reduce operator for set intersection */
void PetscGarbageKeySortedIntersect(void *inset, void *inoutset, PetscMPIInt *length, MPI_Datatype *dtype)
{
  PetscInt64 *seta, *setb;

  seta = (PetscInt64 *)inoutset;
  setb = (PetscInt64 *)inset;

  GarbageKeySortedIntersect_Private(&seta[1], (PetscInt *)&seta[0], &setb[1], (PetscInt)setb[0]);
}

/* Performs a collective allreduce intersection of one array per rank */
PetscErrorCode GarbageKeyAllReduceIntersect_Private(MPI_Comm comm, PetscInt64 *set, PetscInt *entries)
{
  PetscInt     ii, max_entries;
  PetscInt64  *sendset, *recvset;
  MPI_Datatype keyset_type;

  PetscFunctionBegin;
  /* Sort keys first for use with `GarbageKeySortedIntersect_Private()`*/
  PetscCall(PetscSortInt64(*entries, set));

  /* Get the maximum size of all key sets */
  PetscCallMPI(MPI_Allreduce(entries, &max_entries, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(PetscMalloc1(max_entries + 1, &sendset));
  PetscCall(PetscMalloc1(max_entries + 1, &recvset));
  sendset[0] = (PetscInt64)*entries;
  for (ii = 1; ii < *entries + 1; ii++) sendset[ii] = set[ii - 1];

  /* Create a custom data type to hold the set */
  PetscCallMPI(MPI_Type_contiguous(max_entries + 1, MPIU_INT64, &keyset_type));
  /* PetscCallMPI(MPI_Type_set_name(keyset_type,"PETSc garbage key set type")); */
  PetscCallMPI(MPI_Type_commit(&keyset_type));

  /* Perform custom intersect reduce operation over sets */
  PetscCallMPI(MPI_Allreduce(sendset, recvset, 1, keyset_type, Petsc_Garbage_SetIntersectOp, comm));

  PetscCallMPI(MPI_Type_free(&keyset_type));

  *entries = (PetscInt)recvset[0];
  for (ii = 0; ii < *entries; ii++) set[ii] = recvset[ii + 1];

  PetscCall(PetscFree(sendset));
  PetscCall(PetscFree(recvset));
  PetscFunctionReturn(0);
}

/*@C
    PetscGarbageCleanup - Destroys objects placed in the garbage by
    PetscObjectDelayedDestroy().

    Collective

    Input Parameters:
.   comm      - communicator over which to perform collective cleanup

    Notes:
    Implements a collective garbage collection.
    A per- MPI communicator garbage dictionary is created to store
    references to objects destroyed using PetscObjectDelayedDestroy().
    Objects that appear in this dictionary on all ranks can be destroyed
    by calling PetscGarbageCleanup().

    This is done as follows:
    1.  Keys of the garbage dictionary, which correspond to the creation
        indices of the objects stashed, are sorted.
    2.  A collective intersection of dictionary keys is performed by all
        ranks in the communicator.
    3.  The intersection is broadcast back to all ranks in the
        communicator.
    4.  The objects on the dictionary are collectively destroyed in
        creation index order using a call to PetscObjectDestroy().

    This function is intended for use with managed languages such as
    Python or Julia, which may not destroy objects in a deterministic
    order.

    Level: developer

.seealso: PetscObjectDelayedDestroy()
@*/
PetscErrorCode PetscGarbageCleanup(MPI_Comm comm)
{
  PetscInt     ii, entries, offset;
  PetscInt64  *keys;
  PetscObject  obj;
  PetscGarbage garbage;

  PetscFunctionBegin;
  /* Duplicate comm to prevent it being cleaned up by PetscObjectDestroy() */
  PetscCall(PetscCommDuplicate(comm, &comm, NULL));

  /* Grab garbage from comm and remove it
   this avoids calling PetscCommDestroy() and endlessly recursing */
  PetscCall(GarbageGetHMap_Private(comm, &garbage));
  PetscCallMPI(MPI_Comm_delete_attr(comm, Petsc_Garbage_HMap_keyval));

  /* Get keys from garbage hash map */
  PetscCall(PetscHMapObjGetSize(garbage.map, &entries));
  PetscCall(PetscMalloc1(entries, &keys));
  offset = 0;
  PetscCall(PetscHMapObjGetKeys(garbage.map, &offset, keys));

  /* Gather and intersect */
  PetscCall(GarbageKeyAllReduceIntersect_Private(comm, keys, &entries));

  /* Collectively destroy objects objects that appear in garbage in
     creation index order */
  for (ii = 0; ii < entries; ii++) {
    PetscCall(PetscHMapObjGet(garbage.map, keys[ii], &obj));
    PetscCall(PetscObjectDestroy(&obj));
    PetscCall(PetscFree(obj));
    PetscCall(PetscHMapObjDel(garbage.map, keys[ii]));
  }
  PetscCall(PetscFree(keys));

  /* Put garbage back */
  PetscCallMPI(MPI_Comm_set_attr(comm, Petsc_Garbage_HMap_keyval, garbage.ptr));
  PetscCall(PetscCommDestroy(&comm));
  PetscFunctionReturn(0);
}

/* Utility function for printing the contents of the garbage on a given comm */
PetscErrorCode PetscGarbageView(MPI_Comm comm, PetscViewer viewer)
{
  char         text[64];
  PetscInt     ii, entries, offset;
  PetscInt64  *keys;
  PetscObject  obj;
  PetscGarbage garbage;
  PetscMPIInt  rank;

  PetscFunctionBegin;
  PetscCall(PetscPrintf(comm, "PETSc garbage on "));
  if (comm == PETSC_COMM_WORLD) {
    PetscCall(PetscPrintf(comm, "PETSC_COMM_WORLD\n"));
  } else if (comm == PETSC_COMM_SELF) {
    PetscCall(PetscPrintf(comm, "PETSC_COMM_SELF\n"));
  } else {
    PetscCall(PetscPrintf(comm, "UNKNOWN_COMM\n"));
  }
  PetscCall(PetscCommDuplicate(comm, &comm, NULL));
  PetscCall(GarbageGetHMap_Private(comm, &garbage));

  /* Get keys from garbage hash map and sort */
  PetscCall(PetscHMapObjGetSize(garbage.map, &entries));
  PetscCall(PetscMalloc1(entries, &keys));
  offset = 0;
  PetscCall(PetscHMapObjGetKeys(garbage.map, &offset, keys));

  /* Pretty print entries in a table */
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSynchronizedPrintf(comm, "Rank %i:: ", rank));
  PetscCall(PetscFormatConvert("Total entries: %D\n", text));
  PetscCall(PetscSynchronizedPrintf(comm, text, entries));
  if (entries) {
    PetscCall(PetscSynchronizedPrintf(comm, "| Key   | Type                   | Name                             | Object ID |\n"));
    PetscCall(PetscSynchronizedPrintf(comm, "|-------|------------------------|----------------------------------|-----------|\n"));
  }
  for (ii = 0; ii < entries; ii++) {
    PetscCall(PetscHMapObjGet(garbage.map, keys[ii], &obj));
    PetscCall(PetscFormatConvert("| %5" PetscInt64_FMT " | %-22s | %-32s | %6D    |\n", text));
    PetscCall(PetscSynchronizedPrintf(comm, text, keys[ii], obj->class_name, obj->description, obj->id));
  }
  PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));

  PetscCall(PetscFree(keys));
  PetscCall(PetscCommDestroy(&comm));
  PetscFunctionReturn(0);
}
