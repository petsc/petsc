
#include <../src/dm/impls/composite/packimpl.h> /*I  "petscdmcomposite.h"  I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/glvisviewerimpl.h>
#include <petscds.h>

/*@C
    DMCompositeSetCoupling - Sets user provided routines that compute the coupling between the
      separate components `DM` in a `DMCOMPOSITE` to build the correct matrix nonzero structure.

    Logically Collective

    Input Parameters:
+   dm - the composite object
-   formcouplelocations - routine to set the nonzero locations in the matrix

    Level: advanced

    Note:
    See `DMSetApplicationContext()` and `DMGetApplicationContext()` for how to get user information into
    this routine

    Fortran Note:
    Not available from Fortran

.seealso: `DMCOMPOSITE`, `DM`
@*/
PetscErrorCode DMCompositeSetCoupling(DM dm, PetscErrorCode (*FormCoupleLocations)(DM, Mat, PetscInt *, PetscInt *, PetscInt, PetscInt, PetscInt, PetscInt))
{
  DM_Composite *com = (DM_Composite *)dm->data;
  PetscBool     flg;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  com->FormCoupleLocations = FormCoupleLocations;
  PetscFunctionReturn(0);
}

PetscErrorCode DMDestroy_Composite(DM dm)
{
  struct DMCompositeLink *next, *prev;
  DM_Composite           *com = (DM_Composite *)dm->data;

  PetscFunctionBegin;
  next = com->next;
  while (next) {
    prev = next;
    next = next->next;
    PetscCall(DMDestroy(&prev->dm));
    PetscCall(PetscFree(prev->grstarts));
    PetscCall(PetscFree(prev));
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMSetUpGLVisViewer_C", NULL));
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  PetscCall(PetscFree(com));
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_Composite(DM dm, PetscViewer v)
{
  PetscBool     iascii;
  DM_Composite *com = (DM_Composite *)dm->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    struct DMCompositeLink *lnk = com->next;
    PetscInt                i;

    PetscCall(PetscViewerASCIIPrintf(v, "DM (%s)\n", ((PetscObject)dm)->prefix ? ((PetscObject)dm)->prefix : "no prefix"));
    PetscCall(PetscViewerASCIIPrintf(v, "  contains %" PetscInt_FMT " DMs\n", com->nDM));
    PetscCall(PetscViewerASCIIPushTab(v));
    for (i = 0; lnk; lnk = lnk->next, i++) {
      PetscCall(PetscViewerASCIIPrintf(v, "Link %" PetscInt_FMT ": DM of type %s\n", i, ((PetscObject)lnk->dm)->type_name));
      PetscCall(PetscViewerASCIIPushTab(v));
      PetscCall(DMView(lnk->dm, v));
      PetscCall(PetscViewerASCIIPopTab(v));
    }
    PetscCall(PetscViewerASCIIPopTab(v));
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
PetscErrorCode DMSetUp_Composite(DM dm)
{
  PetscInt                nprev = 0;
  PetscMPIInt             rank, size;
  DM_Composite           *com  = (DM_Composite *)dm->data;
  struct DMCompositeLink *next = com->next;
  PetscLayout             map;

  PetscFunctionBegin;
  PetscCheck(!com->setup, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Packer has already been setup");
  PetscCall(PetscLayoutCreate(PetscObjectComm((PetscObject)dm), &map));
  PetscCall(PetscLayoutSetLocalSize(map, com->n));
  PetscCall(PetscLayoutSetSize(map, PETSC_DETERMINE));
  PetscCall(PetscLayoutSetBlockSize(map, 1));
  PetscCall(PetscLayoutSetUp(map));
  PetscCall(PetscLayoutGetSize(map, &com->N));
  PetscCall(PetscLayoutGetRange(map, &com->rstart, NULL));
  PetscCall(PetscLayoutDestroy(&map));

  /* now set the rstart for each linked vector */
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  while (next) {
    next->rstart = nprev;
    nprev += next->n;
    next->grstart = com->rstart + next->rstart;
    PetscCall(PetscMalloc1(size, &next->grstarts));
    PetscCallMPI(MPI_Allgather(&next->grstart, 1, MPIU_INT, next->grstarts, 1, MPIU_INT, PetscObjectComm((PetscObject)dm)));
    next = next->next;
  }
  com->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------*/

/*@
    DMCompositeGetNumberDM - Get's the number of `DM` objects in the `DMCOMPOSITE`
       representation.

    Not Collective

    Input Parameter:
.    dm - the `DMCOMPOSITE` object

    Output Parameter:
.     nDM - the number of `DM`

    Level: beginner

.seealso: `DMCOMPOSITE`, `DM`
@*/
PetscErrorCode DMCompositeGetNumberDM(DM dm, PetscInt *nDM)
{
  DM_Composite *com = (DM_Composite *)dm->data;
  PetscBool     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  *nDM = com->nDM;
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGetAccess - Allows one to access the individual packed vectors in their global
       representation.

    Collective on dm

    Input Parameters:
+    dm - the `DMCOMPOSITE` object
-    gvec - the global vector

    Output Parameters:
.    Vec* ... - the packed parallel vectors, NULL for those that are not needed

    Level: advanced

    Note:
    Use `DMCompositeRestoreAccess()` to return the vectors when you no longer need them

    Fortran Note:
    Fortran callers must use numbered versions of this routine, e.g., DMCompositeGetAccess4(dm,gvec,vec1,vec2,vec3,vec4)
    or use the alternative interface `DMCompositeGetAccessArray()`.

.seealso: `DMCOMPOSITE`, `DM`, `DMCompositeGetEntries()`, `DMCompositeScatter()`
@*/
PetscErrorCode DMCompositeGetAccess(DM dm, Vec gvec, ...)
{
  va_list                 Argp;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscInt                readonly;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(gvec, VEC_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  next = com->next;
  if (!com->setup) PetscCall(DMSetUp(dm));

  PetscCall(VecLockGet(gvec, &readonly));
  /* loop over packed objects, handling one at at time */
  va_start(Argp, gvec);
  while (next) {
    Vec *vec;
    vec = va_arg(Argp, Vec *);
    if (vec) {
      PetscCall(DMGetGlobalVector(next->dm, vec));
      if (readonly) {
        const PetscScalar *array;
        PetscCall(VecGetArrayRead(gvec, &array));
        PetscCall(VecPlaceArray(*vec, array + next->rstart));
        PetscCall(VecLockReadPush(*vec));
        PetscCall(VecRestoreArrayRead(gvec, &array));
      } else {
        PetscScalar *array;
        PetscCall(VecGetArray(gvec, &array));
        PetscCall(VecPlaceArray(*vec, array + next->rstart));
        PetscCall(VecRestoreArray(gvec, &array));
      }
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGetAccessArray - Allows one to access the individual packed vectors in their global
       representation.

    Collective on dm

    Input Parameters:
+    dm - the `DMCOMPOSITE`
.    pvec - packed vector
.    nwanted - number of vectors wanted
-    wanted - sorted array of vectors wanted, or NULL to get all vectors

    Output Parameters:
.    vecs - array of requested global vectors (must be allocated)

    Level: advanced

    Note:
    Use `DMCompositeRestoreAccessArray()` to return the vectors when you no longer need them

.seealso: `DMCOMPOSITE`, `DM`, `DMCompositeGetAccess()`, `DMCompositeGetEntries()`, `DMCompositeScatter()`, `DMCompositeGather()`
@*/
PetscErrorCode DMCompositeGetAccessArray(DM dm, Vec pvec, PetscInt nwanted, const PetscInt *wanted, Vec *vecs)
{
  struct DMCompositeLink *link;
  PetscInt                i, wnum;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscInt                readonly;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(pvec, VEC_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  if (!com->setup) PetscCall(DMSetUp(dm));

  PetscCall(VecLockGet(pvec, &readonly));
  for (i = 0, wnum = 0, link = com->next; link && wnum < nwanted; i++, link = link->next) {
    if (!wanted || i == wanted[wnum]) {
      Vec v;
      PetscCall(DMGetGlobalVector(link->dm, &v));
      if (readonly) {
        const PetscScalar *array;
        PetscCall(VecGetArrayRead(pvec, &array));
        PetscCall(VecPlaceArray(v, array + link->rstart));
        PetscCall(VecLockReadPush(v));
        PetscCall(VecRestoreArrayRead(pvec, &array));
      } else {
        PetscScalar *array;
        PetscCall(VecGetArray(pvec, &array));
        PetscCall(VecPlaceArray(v, array + link->rstart));
        PetscCall(VecRestoreArray(pvec, &array));
      }
      vecs[wnum++] = v;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGetLocalAccessArray - Allows one to access the individual
    packed vectors in their local representation.

    Collective on dm.

    Input Parameters:
+    dm - the `DMCOMPOSITE`
.    pvec - packed vector
.    nwanted - number of vectors wanted
-    wanted - sorted array of vectors wanted, or NULL to get all vectors

    Output Parameters:
.    vecs - array of requested local vectors (must be allocated)

    Level: advanced

    Note:
    Use `DMCompositeRestoreLocalAccessArray()` to return the vectors
    when you no longer need them.

.seealso: `DMCOMPOSITE`, `DM`, `DMCompositeRestoreLocalAccessArray()`, `DMCompositeGetAccess()`,
          `DMCompositeGetEntries()`, `DMCompositeScatter()`, `DMCompositeGather()`
@*/
PetscErrorCode DMCompositeGetLocalAccessArray(DM dm, Vec pvec, PetscInt nwanted, const PetscInt *wanted, Vec *vecs)
{
  struct DMCompositeLink *link;
  PetscInt                i, wnum;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscInt                readonly;
  PetscInt                nlocal = 0;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(pvec, VEC_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  if (!com->setup) PetscCall(DMSetUp(dm));

  PetscCall(VecLockGet(pvec, &readonly));
  for (i = 0, wnum = 0, link = com->next; link && wnum < nwanted; i++, link = link->next) {
    if (!wanted || i == wanted[wnum]) {
      Vec v;
      PetscCall(DMGetLocalVector(link->dm, &v));
      if (readonly) {
        const PetscScalar *array;
        PetscCall(VecGetArrayRead(pvec, &array));
        PetscCall(VecPlaceArray(v, array + nlocal));
        // this method does not make sense. The local vectors are not updated with a global-to-local and the user can not do it because it is locked
        PetscCall(VecLockReadPush(v));
        PetscCall(VecRestoreArrayRead(pvec, &array));
      } else {
        PetscScalar *array;
        PetscCall(VecGetArray(pvec, &array));
        PetscCall(VecPlaceArray(v, array + nlocal));
        PetscCall(VecRestoreArray(pvec, &array));
      }
      vecs[wnum++] = v;
    }

    nlocal += link->nlocal;
  }

  PetscFunctionReturn(0);
}

/*@C
    DMCompositeRestoreAccess - Returns the vectors obtained with `DMCompositeGetAccess()`
       representation.

    Collective on dm

    Input Parameters:
+    dm - the `DMCOMPOSITE` object
.    gvec - the global vector
-    Vec* ... - the individual parallel vectors, NULL for those that are not needed

    Level: advanced

.seealso: `DMCOMPOSITE`, `DM`, `DMCompositeAddDM()`, `DMCreateGlobalVector()`,
         `DMCompositeGather()`, `DMCompositeCreate()`, `DMCompositeGetISLocalToGlobalMappings()`, `DMCompositeScatter()`,
         `DMCompositeRestoreAccess()`, `DMCompositeGetAccess()`
@*/
PetscErrorCode DMCompositeRestoreAccess(DM dm, Vec gvec, ...)
{
  va_list                 Argp;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscInt                readonly;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(gvec, VEC_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  next = com->next;
  if (!com->setup) PetscCall(DMSetUp(dm));

  PetscCall(VecLockGet(gvec, &readonly));
  /* loop over packed objects, handling one at at time */
  va_start(Argp, gvec);
  while (next) {
    Vec *vec;
    vec = va_arg(Argp, Vec *);
    if (vec) {
      PetscCall(VecResetArray(*vec));
      if (readonly) PetscCall(VecLockReadPop(*vec));
      PetscCall(DMRestoreGlobalVector(next->dm, vec));
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeRestoreAccessArray - Returns the vectors obtained with `DMCompositeGetAccessArray()`

    Collective on dm

    Input Parameters:
+    dm - the `DMCOMPOSITE` object
.    pvec - packed vector
.    nwanted - number of vectors wanted
.    wanted - sorted array of vectors wanted, or NULL to get all vectors
-    vecs - array of global vectors to return

    Level: advanced

.seealso: `DMCOMPOSITE`, `DM`, `DMCompositeRestoreAccess()`, `DMCompositeRestoreEntries()`, `DMCompositeScatter()`, `DMCompositeGather()`
@*/
PetscErrorCode DMCompositeRestoreAccessArray(DM dm, Vec pvec, PetscInt nwanted, const PetscInt *wanted, Vec *vecs)
{
  struct DMCompositeLink *link;
  PetscInt                i, wnum;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscInt                readonly;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(pvec, VEC_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  if (!com->setup) PetscCall(DMSetUp(dm));

  PetscCall(VecLockGet(pvec, &readonly));
  for (i = 0, wnum = 0, link = com->next; link && wnum < nwanted; i++, link = link->next) {
    if (!wanted || i == wanted[wnum]) {
      PetscCall(VecResetArray(vecs[wnum]));
      if (readonly) PetscCall(VecLockReadPop(vecs[wnum]));
      PetscCall(DMRestoreGlobalVector(link->dm, &vecs[wnum]));
      wnum++;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeRestoreLocalAccessArray - Returns the vectors obtained with `DMCompositeGetLocalAccessArray()`.

    Collective on dm.

    Input Parameters:
+    dm - the `DMCOMPOSITE` object
.    pvec - packed vector
.    nwanted - number of vectors wanted
.    wanted - sorted array of vectors wanted, or NULL to restore all vectors
-    vecs - array of local vectors to return

    Level: advanced

    Note:
    nwanted and wanted must match the values given to `DMCompositeGetLocalAccessArray()`
    otherwise the call will fail.

.seealso: `DMCOMPOSITE`, `DM`, `DMCompositeGetLocalAccessArray()`, `DMCompositeRestoreAccessArray()`,
          `DMCompositeRestoreAccess()`, `DMCompositeRestoreEntries()`,
          `DMCompositeScatter()`, `DMCompositeGather()`
@*/
PetscErrorCode DMCompositeRestoreLocalAccessArray(DM dm, Vec pvec, PetscInt nwanted, const PetscInt *wanted, Vec *vecs)
{
  struct DMCompositeLink *link;
  PetscInt                i, wnum;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscInt                readonly;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(pvec, VEC_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  if (!com->setup) PetscCall(DMSetUp(dm));

  PetscCall(VecLockGet(pvec, &readonly));
  for (i = 0, wnum = 0, link = com->next; link && wnum < nwanted; i++, link = link->next) {
    if (!wanted || i == wanted[wnum]) {
      PetscCall(VecResetArray(vecs[wnum]));
      if (readonly) PetscCall(VecLockReadPop(vecs[wnum]));
      PetscCall(DMRestoreLocalVector(link->dm, &vecs[wnum]));
      wnum++;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeScatter - Scatters from a global packed vector into its individual local vectors

    Collective on dm

    Input Parameters:
+    dm - the `DMCOMPOSITE` object
.    gvec - the global vector
-    Vec ... - the individual sequential vectors, NULL for those that are not needed

    Level: advanced

    Note:
    `DMCompositeScatterArray()` is a non-variadic alternative that is often more convenient for library callers and is
    accessible from Fortran.

.seealso: `DMCOMPOSITE`, `DM`, `DMDestroy()`, `DMCompositeAddDM()`, `DMCreateGlobalVector()`,
         `DMCompositeGather()`, `DMCompositeCreate()`, `DMCompositeGetISLocalToGlobalMappings()`, `DMCompositeGetAccess()`,
         `DMCompositeGetLocalVectors()`, `DMCompositeRestoreLocalVectors()`, `DMCompositeGetEntries()`
         `DMCompositeScatterArray()`
@*/
PetscErrorCode DMCompositeScatter(DM dm, Vec gvec, ...)
{
  va_list                 Argp;
  struct DMCompositeLink *next;
  PETSC_UNUSED PetscInt   cnt;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(gvec, VEC_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  if (!com->setup) PetscCall(DMSetUp(dm));

  /* loop over packed objects, handling one at at time */
  va_start(Argp, gvec);
  for (cnt = 3, next = com->next; next; cnt++, next = next->next) {
    Vec local;
    local = va_arg(Argp, Vec);
    if (local) {
      Vec                global;
      const PetscScalar *array;
      PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidHeaderSpecific(local, VEC_CLASSID, (int)cnt));
      PetscCall(DMGetGlobalVector(next->dm, &global));
      PetscCall(VecGetArrayRead(gvec, &array));
      PetscCall(VecPlaceArray(global, array + next->rstart));
      PetscCall(DMGlobalToLocalBegin(next->dm, global, INSERT_VALUES, local));
      PetscCall(DMGlobalToLocalEnd(next->dm, global, INSERT_VALUES, local));
      PetscCall(VecRestoreArrayRead(gvec, &array));
      PetscCall(VecResetArray(global));
      PetscCall(DMRestoreGlobalVector(next->dm, &global));
    }
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@
    DMCompositeScatterArray - Scatters from a global packed vector into its individual local vectors

    Collective on dm

    Input Parameters:
+    dm - the `DMCOMPOSITE` object
.    gvec - the global vector
-    lvecs - array of local vectors, NULL for any that are not needed

    Level: advanced

    Note:
    This is a non-variadic alternative to `DMCompositeScatter()`

.seealso: `DMCOMPOSITE`, `DM`, `DMDestroy()`, `DMCompositeAddDM()`, `DMCreateGlobalVector()`
         `DMCompositeGather()`, `DMCompositeCreate()`, `DMCompositeGetISLocalToGlobalMappings()`, `DMCompositeGetAccess()`,
         `DMCompositeGetLocalVectors()`, `DMCompositeRestoreLocalVectors()`, `DMCompositeGetEntries()`
@*/
PetscErrorCode DMCompositeScatterArray(DM dm, Vec gvec, Vec *lvecs)
{
  struct DMCompositeLink *next;
  PetscInt                i;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(gvec, VEC_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  if (!com->setup) PetscCall(DMSetUp(dm));

  /* loop over packed objects, handling one at at time */
  for (i = 0, next = com->next; next; next = next->next, i++) {
    if (lvecs[i]) {
      Vec                global;
      const PetscScalar *array;
      PetscValidHeaderSpecific(lvecs[i], VEC_CLASSID, 3);
      PetscCall(DMGetGlobalVector(next->dm, &global));
      PetscCall(VecGetArrayRead(gvec, &array));
      PetscCall(VecPlaceArray(global, (PetscScalar *)array + next->rstart));
      PetscCall(DMGlobalToLocalBegin(next->dm, global, INSERT_VALUES, lvecs[i]));
      PetscCall(DMGlobalToLocalEnd(next->dm, global, INSERT_VALUES, lvecs[i]));
      PetscCall(VecRestoreArrayRead(gvec, &array));
      PetscCall(VecResetArray(global));
      PetscCall(DMRestoreGlobalVector(next->dm, &global));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGather - Gathers into a global packed vector from its individual local vectors

    Collective on dm

    Input Parameters:
+    dm - the `DMCOMPOSITE` object
.    gvec - the global vector
.    imode - `INSERT_VALUES` or `ADD_VALUES`
-    Vec ... - the individual sequential vectors, NULL for any that are not needed

    Level: advanced

    Fortran Note:
    Fortran users should use `DMCompositeGatherArray()`

.seealso: `DMCOMPOSITE`, `DM`, `DMDestroy()`, `DMCompositeAddDM()`, `DMCreateGlobalVector()`,
         `DMCompositeScatter()`, `DMCompositeCreate()`, `DMCompositeGetISLocalToGlobalMappings()`, `DMCompositeGetAccess()`,
         `DMCompositeGetLocalVectors()`, `DMCompositeRestoreLocalVectors()`, `DMCompositeGetEntries()`
@*/
PetscErrorCode DMCompositeGather(DM dm, InsertMode imode, Vec gvec, ...)
{
  va_list                 Argp;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PETSC_UNUSED PetscInt   cnt;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(gvec, VEC_CLASSID, 3);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  if (!com->setup) PetscCall(DMSetUp(dm));

  /* loop over packed objects, handling one at at time */
  va_start(Argp, gvec);
  for (cnt = 3, next = com->next; next; cnt++, next = next->next) {
    Vec local;
    local = va_arg(Argp, Vec);
    if (local) {
      PetscScalar *array;
      Vec          global;
      PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidHeaderSpecific(local, VEC_CLASSID, (int)cnt));
      PetscCall(DMGetGlobalVector(next->dm, &global));
      PetscCall(VecGetArray(gvec, &array));
      PetscCall(VecPlaceArray(global, array + next->rstart));
      PetscCall(DMLocalToGlobalBegin(next->dm, local, imode, global));
      PetscCall(DMLocalToGlobalEnd(next->dm, local, imode, global));
      PetscCall(VecRestoreArray(gvec, &array));
      PetscCall(VecResetArray(global));
      PetscCall(DMRestoreGlobalVector(next->dm, &global));
    }
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@
    DMCompositeGatherArray - Gathers into a global packed vector from its individual local vectors

    Collective on dm

    Input Parameters:
+    dm - the `DMCOMPOSITE` object
.    gvec - the global vector
.    imode - `INSERT_VALUES` or `ADD_VALUES`
-    lvecs - the individual sequential vectors, NULL for any that are not needed

    Level: advanced

    Note:
    This is a non-variadic alternative to `DMCompositeGather()`.

.seealso: `DMCOMPOSITE`, `DM`, `DMDestroy()`, `DMCompositeAddDM()`, `DMCreateGlobalVector()`,
         `DMCompositeScatter()`, `DMCompositeCreate()`, `DMCompositeGetISLocalToGlobalMappings()`, `DMCompositeGetAccess()`,
         `DMCompositeGetLocalVectors()`, `DMCompositeRestoreLocalVectors()`, `DMCompositeGetEntries()`,
@*/
PetscErrorCode DMCompositeGatherArray(DM dm, InsertMode imode, Vec gvec, Vec *lvecs)
{
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscInt                i;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(gvec, VEC_CLASSID, 3);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  if (!com->setup) PetscCall(DMSetUp(dm));

  /* loop over packed objects, handling one at at time */
  for (next = com->next, i = 0; next; next = next->next, i++) {
    if (lvecs[i]) {
      PetscScalar *array;
      Vec          global;
      PetscValidHeaderSpecific(lvecs[i], VEC_CLASSID, 4);
      PetscCall(DMGetGlobalVector(next->dm, &global));
      PetscCall(VecGetArray(gvec, &array));
      PetscCall(VecPlaceArray(global, array + next->rstart));
      PetscCall(DMLocalToGlobalBegin(next->dm, lvecs[i], imode, global));
      PetscCall(DMLocalToGlobalEnd(next->dm, lvecs[i], imode, global));
      PetscCall(VecRestoreArray(gvec, &array));
      PetscCall(VecResetArray(global));
      PetscCall(DMRestoreGlobalVector(next->dm, &global));
    }
  }
  PetscFunctionReturn(0);
}

/*@
    DMCompositeAddDM - adds a `DM` vector to a `DMCOMPOSITE`

    Collective on dm

    Input Parameters:
+    dmc - the  `DMCOMPOSITE` object
-    dm - the `DM` object

    Level: advanced

.seealso: `DMCOMPOSITE`, `DM`, `DMDestroy()`, `DMCompositeGather()`, `DMCompositeAddDM()`, `DMCreateGlobalVector()`,
         `DMCompositeScatter()`, `DMCompositeCreate()`, `DMCompositeGetISLocalToGlobalMappings()`, `DMCompositeGetAccess()`,
         `DMCompositeGetLocalVectors()`, `DMCompositeRestoreLocalVectors()`, `DMCompositeGetEntries()`
@*/
PetscErrorCode DMCompositeAddDM(DM dmc, DM dm)
{
  PetscInt                n, nlocal;
  struct DMCompositeLink *mine, *next;
  Vec                     global, local;
  DM_Composite           *com = (DM_Composite *)dmc->data;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)dmc, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  next = com->next;
  PetscCheck(!com->setup, PetscObjectComm((PetscObject)dmc), PETSC_ERR_ARG_WRONGSTATE, "Cannot add a DM once you have used the DMComposite");

  /* create new link */
  PetscCall(PetscNew(&mine));
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscCall(DMGetGlobalVector(dm, &global));
  PetscCall(VecGetLocalSize(global, &n));
  PetscCall(DMRestoreGlobalVector(dm, &global));
  PetscCall(DMGetLocalVector(dm, &local));
  PetscCall(VecGetSize(local, &nlocal));
  PetscCall(DMRestoreLocalVector(dm, &local));

  mine->n      = n;
  mine->nlocal = nlocal;
  mine->dm     = dm;
  mine->next   = NULL;
  com->n += n;
  com->nghost += nlocal;

  /* add to end of list */
  if (!next) com->next = mine;
  else {
    while (next->next) next = next->next;
    next->next = mine;
  }
  com->nDM++;
  com->nmine++;
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
PETSC_EXTERN PetscErrorCode VecView_MPI(Vec, PetscViewer);
PetscErrorCode              VecView_DMComposite(Vec gvec, PetscViewer viewer)
{
  DM                      dm;
  struct DMCompositeLink *next;
  PetscBool               isdraw;
  DM_Composite           *com;

  PetscFunctionBegin;
  PetscCall(VecGetDM(gvec, &dm));
  PetscCheck(dm, PetscObjectComm((PetscObject)gvec), PETSC_ERR_ARG_WRONG, "Vector not generated from a DMComposite");
  com  = (DM_Composite *)dm->data;
  next = com->next;

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
  if (!isdraw) {
    /* do I really want to call this? */
    PetscCall(VecView_MPI(gvec, viewer));
  } else {
    PetscInt cnt = 0;

    /* loop over packed objects, handling one at at time */
    while (next) {
      Vec                vec;
      const PetscScalar *array;
      PetscInt           bs;

      /* Should use VecGetSubVector() eventually, but would need to forward the DM for that to work */
      PetscCall(DMGetGlobalVector(next->dm, &vec));
      PetscCall(VecGetArrayRead(gvec, &array));
      PetscCall(VecPlaceArray(vec, (PetscScalar *)array + next->rstart));
      PetscCall(VecRestoreArrayRead(gvec, &array));
      PetscCall(VecView(vec, viewer));
      PetscCall(VecResetArray(vec));
      PetscCall(VecGetBlockSize(vec, &bs));
      PetscCall(DMRestoreGlobalVector(next->dm, &vec));
      PetscCall(PetscViewerDrawBaseAdd(viewer, bs));
      cnt += bs;
      next = next->next;
    }
    PetscCall(PetscViewerDrawBaseAdd(viewer, -cnt));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateGlobalVector_Composite(DM dm, Vec *gvec)
{
  DM_Composite *com = (DM_Composite *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)dm), gvec));
  PetscCall(VecSetType(*gvec, dm->vectype));
  PetscCall(VecSetSizes(*gvec, com->n, com->N));
  PetscCall(VecSetDM(*gvec, dm));
  PetscCall(VecSetOperation(*gvec, VECOP_VIEW, (void (*)(void))VecView_DMComposite));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateLocalVector_Composite(DM dm, Vec *lvec)
{
  DM_Composite *com = (DM_Composite *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!com->setup) {
    PetscCall(DMSetFromOptions(dm));
    PetscCall(DMSetUp(dm));
  }
  PetscCall(VecCreate(PETSC_COMM_SELF, lvec));
  PetscCall(VecSetType(*lvec, dm->vectype));
  PetscCall(VecSetSizes(*lvec, com->nghost, PETSC_DECIDE));
  PetscCall(VecSetDM(*lvec, dm));
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGetISLocalToGlobalMappings - gets an `ISLocalToGlobalMapping` for each `DM` in the `DMCOMPOSITE`, maps to the composite global space

    Collective on dm

    Input Parameter:
.    dm - the `DMCOMPOSITE` object

    Output Parameters:
.    ltogs - the individual mappings for each packed vector. Note that this includes
           all the ghost points that individual ghosted `DMDA` may have.

    Level: advanced

    Note:
    Each entry of ltogs should be destroyed with `ISLocalToGlobalMappingDestroy()`, the ltogs array should be freed with `PetscFree()`.

    Fortran Note:
    Not available from Fortran

.seealso: `DMCOMPOSITE`, `DM`, `DMDestroy()`, `DMCompositeAddDM()`, `DMCreateGlobalVector()`,
         `DMCompositeGather()`, `DMCompositeCreate()`, `DMCompositeGetAccess()`, `DMCompositeScatter()`,
         `DMCompositeGetLocalVectors()`, `DMCompositeRestoreLocalVectors()`, `DMCompositeGetEntries()`
@*/
PetscErrorCode DMCompositeGetISLocalToGlobalMappings(DM dm, ISLocalToGlobalMapping **ltogs)
{
  PetscInt                i, *idx, n, cnt;
  struct DMCompositeLink *next;
  PetscMPIInt             rank;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  PetscCall(DMSetUp(dm));
  PetscCall(PetscMalloc1(com->nDM, ltogs));
  next = com->next;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));

  /* loop over packed objects, handling one at at time */
  cnt = 0;
  while (next) {
    ISLocalToGlobalMapping ltog;
    PetscMPIInt            size;
    const PetscInt        *suboff, *indices;
    Vec                    global;

    /* Get sub-DM global indices for each local dof */
    PetscCall(DMGetLocalToGlobalMapping(next->dm, &ltog));
    PetscCall(ISLocalToGlobalMappingGetSize(ltog, &n));
    PetscCall(ISLocalToGlobalMappingGetIndices(ltog, &indices));
    PetscCall(PetscMalloc1(n, &idx));

    /* Get the offsets for the sub-DM global vector */
    PetscCall(DMGetGlobalVector(next->dm, &global));
    PetscCall(VecGetOwnershipRanges(global, &suboff));
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)global), &size));

    /* Shift the sub-DM definition of the global space to the composite global space */
    for (i = 0; i < n; i++) {
      PetscInt subi = indices[i], lo = 0, hi = size, t;
      /* There's no consensus on what a negative index means,
         except for skipping when setting the values in vectors and matrices */
      if (subi < 0) {
        idx[i] = subi - next->grstarts[rank];
        continue;
      }
      /* Binary search to find which rank owns subi */
      while (hi - lo > 1) {
        t = lo + (hi - lo) / 2;
        if (suboff[t] > subi) hi = t;
        else lo = t;
      }
      idx[i] = subi - suboff[lo] + next->grstarts[lo];
    }
    PetscCall(ISLocalToGlobalMappingRestoreIndices(ltog, &indices));
    PetscCall(ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)dm), 1, n, idx, PETSC_OWN_POINTER, &(*ltogs)[cnt]));
    PetscCall(DMRestoreGlobalVector(next->dm, &global));
    next = next->next;
    cnt++;
  }
  PetscFunctionReturn(0);
}

/*@C
   DMCompositeGetLocalISs - Gets index sets for each component of a composite local vector

   Not Collective

   Input Parameter:
. dm - the `DMCOMPOSITE`

   Output Parameter:
. is - array of serial index sets for each each component of the `DMCOMPOSITE`

   Level: intermediate

   Notes:
   At present, a composite local vector does not normally exist.  This function is used to provide index sets for
   `MatGetLocalSubMatrix()`.  In the future, the scatters for each entry in the `DMCOMPOSITE` may be be merged into a single
   scatter to a composite local vector.  The user should not typically need to know which is being done.

   To get the composite global indices at all local points (including ghosts), use `DMCompositeGetISLocalToGlobalMappings()`.

   To get index sets for pieces of the composite global vector, use `DMCompositeGetGlobalISs()`.

   Each returned `IS` should be destroyed with `ISDestroy()`, the array should be freed with `PetscFree()`.

   Fortran Note:
   Not available from Fortran

.seealso: `DMCOMPOSITE`, `DM`, `DMCompositeGetGlobalISs()`, `DMCompositeGetISLocalToGlobalMappings()`, `MatGetLocalSubMatrix()`, `MatCreateLocalRef()`
@*/
PetscErrorCode DMCompositeGetLocalISs(DM dm, IS **is)
{
  DM_Composite           *com = (DM_Composite *)dm->data;
  struct DMCompositeLink *link;
  PetscInt                cnt, start;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(is, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  PetscCall(PetscMalloc1(com->nmine, is));
  for (cnt = 0, start = 0, link = com->next; link; start += link->nlocal, cnt++, link = link->next) {
    PetscInt bs;
    PetscCall(ISCreateStride(PETSC_COMM_SELF, link->nlocal, start, 1, &(*is)[cnt]));
    PetscCall(DMGetBlockSize(link->dm, &bs));
    PetscCall(ISSetBlockSize((*is)[cnt], bs));
  }
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGetGlobalISs - Gets the index sets for each composed object in a `DMCOMPOSITE`

    Collective on dm

    Input Parameter:
.    dm - the `DMCOMPOSITE` object

    Output Parameters:
.    is - the array of index sets

    Level: advanced

    Notes:
       The is entries should be destroyed with `ISDestroy()`, the is array should be freed with `PetscFree()`

       These could be used to extract a subset of vector entries for a "multi-physics" preconditioner

       Use `DMCompositeGetLocalISs()` for index sets in the packed local numbering, and
       `DMCompositeGetISLocalToGlobalMappings()` for to map local sub-`DM` (including ghost) indices to packed global
       indices.

    Fortran Note:
    The output argument 'is' must be an allocated array of sufficient length, which can be learned using `DMCompositeGetNumberDM()`.

.seealso: `DMCOMPOSITE`, `DM`, `DMDestroy()`, `DMCompositeAddDM()`, `DMCreateGlobalVector()`,
         `DMCompositeGather()`, `DMCompositeCreate()`, `DMCompositeGetAccess()`, `DMCompositeScatter()`,
         `DMCompositeGetLocalVectors()`, `DMCompositeRestoreLocalVectors()`, `DMCompositeGetEntries()`
@*/
PetscErrorCode DMCompositeGetGlobalISs(DM dm, IS *is[])
{
  PetscInt                cnt = 0;
  struct DMCompositeLink *next;
  PetscMPIInt             rank;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  PetscCheck(dm->setupcalled, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Must call DMSetUp() before");
  PetscCall(PetscMalloc1(com->nDM, is));
  next = com->next;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));

  /* loop over packed objects, handling one at at time */
  while (next) {
    PetscDS prob;

    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)dm), next->n, next->grstart, 1, &(*is)[cnt]));
    PetscCall(DMGetDS(dm, &prob));
    if (prob) {
      MatNullSpace space;
      Mat          pmat;
      PetscObject  disc;
      PetscInt     Nf;

      PetscCall(PetscDSGetNumFields(prob, &Nf));
      if (cnt < Nf) {
        PetscCall(PetscDSGetDiscretization(prob, cnt, &disc));
        PetscCall(PetscObjectQuery(disc, "nullspace", (PetscObject *)&space));
        if (space) PetscCall(PetscObjectCompose((PetscObject)(*is)[cnt], "nullspace", (PetscObject)space));
        PetscCall(PetscObjectQuery(disc, "nearnullspace", (PetscObject *)&space));
        if (space) PetscCall(PetscObjectCompose((PetscObject)(*is)[cnt], "nearnullspace", (PetscObject)space));
        PetscCall(PetscObjectQuery(disc, "pmat", (PetscObject *)&pmat));
        if (pmat) PetscCall(PetscObjectCompose((PetscObject)(*is)[cnt], "pmat", (PetscObject)pmat));
      }
    }
    cnt++;
    next = next->next;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateFieldIS_Composite(DM dm, PetscInt *numFields, char ***fieldNames, IS **fields)
{
  PetscInt nDM;
  DM      *dms;
  PetscInt i;

  PetscFunctionBegin;
  PetscCall(DMCompositeGetNumberDM(dm, &nDM));
  if (numFields) *numFields = nDM;
  PetscCall(DMCompositeGetGlobalISs(dm, fields));
  if (fieldNames) {
    PetscCall(PetscMalloc1(nDM, &dms));
    PetscCall(PetscMalloc1(nDM, fieldNames));
    PetscCall(DMCompositeGetEntriesArray(dm, dms));
    for (i = 0; i < nDM; i++) {
      char        buf[256];
      const char *splitname;

      /* Split naming precedence: object name, prefix, number */
      splitname = ((PetscObject)dm)->name;
      if (!splitname) {
        PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dms[i], &splitname));
        if (splitname) {
          size_t len;
          PetscCall(PetscStrncpy(buf, splitname, sizeof(buf)));
          buf[sizeof(buf) - 1] = 0;
          PetscCall(PetscStrlen(buf, &len));
          if (buf[len - 1] == '_') buf[len - 1] = 0; /* Remove trailing underscore if it was used */
          splitname = buf;
        }
      }
      if (!splitname) {
        PetscCall(PetscSNPrintf(buf, sizeof(buf), "%" PetscInt_FMT, i));
        splitname = buf;
      }
      PetscCall(PetscStrallocpy(splitname, &(*fieldNames)[i]));
    }
    PetscCall(PetscFree(dms));
  }
  PetscFunctionReturn(0);
}

/*
 This could take over from DMCreateFieldIS(), as it is more general,
 making DMCreateFieldIS() a special case -- calling with dmlist == NULL;
 At this point it's probably best to be less intrusive, however.
 */
PetscErrorCode DMCreateFieldDecomposition_Composite(DM dm, PetscInt *len, char ***namelist, IS **islist, DM **dmlist)
{
  PetscInt nDM;
  PetscInt i;

  PetscFunctionBegin;
  PetscCall(DMCreateFieldIS_Composite(dm, len, namelist, islist));
  if (dmlist) {
    PetscCall(DMCompositeGetNumberDM(dm, &nDM));
    PetscCall(PetscMalloc1(nDM, dmlist));
    PetscCall(DMCompositeGetEntriesArray(dm, *dmlist));
    for (i = 0; i < nDM; i++) PetscCall(PetscObjectReference((PetscObject)((*dmlist)[i])));
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*@C
    DMCompositeGetLocalVectors - Gets local vectors for each part of a `DMCOMPOSITE`
       Use `DMCompositeRestoreLocalVectors()` to return them.

    Not Collective

    Input Parameter:
.    dm - the `DMCOMPOSITE` object

    Output Parameter:
.   Vec ... - the individual sequential Vecs

    Level: advanced

    Fortran Note:
    Not available from Fortran

.seealso: `DMCOMPOSITE`, `DM`, `DMDestroy()`, `DMCompositeAddDM()`, `DMCreateGlobalVector()`,
         `DMCompositeGather()`, `DMCompositeCreate()`, `DMCompositeGetISLocalToGlobalMappings()`, `DMCompositeGetAccess()`,
         `DMCompositeRestoreLocalVectors()`, `DMCompositeScatter()`, `DMCompositeGetEntries()`
@*/
PetscErrorCode DMCompositeGetLocalVectors(DM dm, ...)
{
  va_list                 Argp;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  next = com->next;
  /* loop over packed objects, handling one at at time */
  va_start(Argp, dm);
  while (next) {
    Vec *vec;
    vec = va_arg(Argp, Vec *);
    if (vec) PetscCall(DMGetLocalVector(next->dm, vec));
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeRestoreLocalVectors - Restores local vectors for each part of a `DMCOMPOSITE`

    Not Collective

    Input Parameter:
.    dm - the `DMCOMPOSITE` object

    Output Parameter:
.   Vec ... - the individual sequential `Vec`

    Level: advanced

    Fortran Note:
    Not available from Fortran

.seealso: `DMCOMPOSITE`, `DM`, `DMDestroy()`, `DMCompositeAddDM()`, `DMCreateGlobalVector()`,
         `DMCompositeGather()`, `DMCompositeCreate()`, `DMCompositeGetISLocalToGlobalMappings()`, `DMCompositeGetAccess()`,
         `DMCompositeGetLocalVectors()`, `DMCompositeScatter()`, `DMCompositeGetEntries()`
@*/
PetscErrorCode DMCompositeRestoreLocalVectors(DM dm, ...)
{
  va_list                 Argp;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  next = com->next;
  /* loop over packed objects, handling one at at time */
  va_start(Argp, dm);
  while (next) {
    Vec *vec;
    vec = va_arg(Argp, Vec *);
    if (vec) PetscCall(DMRestoreLocalVector(next->dm, vec));
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*@C
    DMCompositeGetEntries - Gets the `DM` for each entry in a `DMCOMPOSITE`.

    Not Collective

    Input Parameter:
.    dm - the `DMCOMPOSITE` object

    Output Parameter:
.   DM ... - the individual entries `DM`

    Level: advanced

    Fortran Note:
    Available as `DMCompositeGetEntries()` for one output `DM`, DMCompositeGetEntries2() for 2, etc

.seealso: `DMCOMPOSITE`, `DM`, `DMDestroy()`, `DMCompositeAddDM()`, `DMCreateGlobalVector()`, `DMCompositeGetEntriesArray()`
         `DMCompositeGather()`, `DMCompositeCreate()`, `DMCompositeGetISLocalToGlobalMappings()`, `DMCompositeGetAccess()`,
         `DMCompositeRestoreLocalVectors()`, `DMCompositeGetLocalVectors()`, `DMCompositeScatter()`,
         `DMCompositeGetLocalVectors()`, `DMCompositeRestoreLocalVectors()`
@*/
PetscErrorCode DMCompositeGetEntries(DM dm, ...)
{
  va_list                 Argp;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  next = com->next;
  /* loop over packed objects, handling one at at time */
  va_start(Argp, dm);
  while (next) {
    DM *dmn;
    dmn = va_arg(Argp, DM *);
    if (dmn) *dmn = next->dm;
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGetEntriesArray - Gets the DM for each entry in a `DMCOMPOSITE`

    Not Collective

    Input Parameter:
.    dm - the `DMCOMPOSITE` object

    Output Parameter:
.    dms - array of sufficient length (see `DMCompositeGetNumberDM()`) to hold the individual `DM`

    Level: advanced

.seealso: `DMCOMPOSITE`, `DM`, `DMDestroy()`, `DMCompositeAddDM()`, `DMCreateGlobalVector()`, `DMCompositeGetEntries()`
         `DMCompositeGather()`, `DMCompositeCreate()`, `DMCompositeGetISLocalToGlobalMappings()`, `DMCompositeGetAccess()`,
         `DMCompositeRestoreLocalVectors()`, `DMCompositeGetLocalVectors()`, `DMCompositeScatter()`,
         `DMCompositeGetLocalVectors()`, `DMCompositeRestoreLocalVectors()`
@*/
PetscErrorCode DMCompositeGetEntriesArray(DM dm, DM dms[])
{
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite *)dm->data;
  PetscInt                i;
  PetscBool               flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMCOMPOSITE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Not for type %s", ((PetscObject)dm)->type_name);
  /* loop over packed objects, handling one at at time */
  for (next = com->next, i = 0; next; next = next->next, i++) dms[i] = next->dm;
  PetscFunctionReturn(0);
}

typedef struct {
  DM           dm;
  PetscViewer *subv;
  Vec         *vecs;
} GLVisViewerCtx;

static PetscErrorCode DestroyGLVisViewerCtx_Private(void *vctx)
{
  GLVisViewerCtx *ctx = (GLVisViewerCtx *)vctx;
  PetscInt        i, n;

  PetscFunctionBegin;
  PetscCall(DMCompositeGetNumberDM(ctx->dm, &n));
  for (i = 0; i < n; i++) PetscCall(PetscViewerDestroy(&ctx->subv[i]));
  PetscCall(PetscFree2(ctx->subv, ctx->vecs));
  PetscCall(DMDestroy(&ctx->dm));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCompositeSampleGLVisFields_Private(PetscObject oX, PetscInt nf, PetscObject oXfield[], void *vctx)
{
  Vec             X   = (Vec)oX;
  GLVisViewerCtx *ctx = (GLVisViewerCtx *)vctx;
  PetscInt        i, n, cumf;

  PetscFunctionBegin;
  PetscCall(DMCompositeGetNumberDM(ctx->dm, &n));
  PetscCall(DMCompositeGetAccessArray(ctx->dm, X, n, NULL, ctx->vecs));
  for (i = 0, cumf = 0; i < n; i++) {
    PetscErrorCode (*g2l)(PetscObject, PetscInt, PetscObject[], void *);
    void    *fctx;
    PetscInt nfi;

    PetscCall(PetscViewerGLVisGetFields_Private(ctx->subv[i], &nfi, NULL, NULL, &g2l, NULL, &fctx));
    if (!nfi) continue;
    if (g2l) PetscCall((*g2l)((PetscObject)ctx->vecs[i], nfi, oXfield + cumf, fctx));
    else PetscCall(VecCopy(ctx->vecs[i], (Vec)(oXfield[cumf])));
    cumf += nfi;
  }
  PetscCall(DMCompositeRestoreAccessArray(ctx->dm, X, n, NULL, ctx->vecs));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetUpGLVisViewer_Composite(PetscObject odm, PetscViewer viewer)
{
  DM              dm = (DM)odm, *dms;
  Vec            *Ufds;
  GLVisViewerCtx *ctx;
  PetscInt        i, n, tnf, *sdim;
  char          **fecs;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  PetscCall(PetscObjectReference((PetscObject)dm));
  ctx->dm = dm;
  PetscCall(DMCompositeGetNumberDM(dm, &n));
  PetscCall(PetscMalloc1(n, &dms));
  PetscCall(DMCompositeGetEntriesArray(dm, dms));
  PetscCall(PetscMalloc2(n, &ctx->subv, n, &ctx->vecs));
  for (i = 0, tnf = 0; i < n; i++) {
    PetscInt nf;

    PetscCall(PetscViewerCreate(PetscObjectComm(odm), &ctx->subv[i]));
    PetscCall(PetscViewerSetType(ctx->subv[i], PETSCVIEWERGLVIS));
    PetscCall(PetscViewerGLVisSetDM_Private(ctx->subv[i], (PetscObject)dms[i]));
    PetscCall(PetscViewerGLVisGetFields_Private(ctx->subv[i], &nf, NULL, NULL, NULL, NULL, NULL));
    tnf += nf;
  }
  PetscCall(PetscFree(dms));
  PetscCall(PetscMalloc3(tnf, &fecs, tnf, &sdim, tnf, &Ufds));
  for (i = 0, tnf = 0; i < n; i++) {
    PetscInt    *sd, nf, f;
    const char **fec;
    Vec         *Uf;

    PetscCall(PetscViewerGLVisGetFields_Private(ctx->subv[i], &nf, &fec, &sd, NULL, (PetscObject **)&Uf, NULL));
    for (f = 0; f < nf; f++) {
      PetscCall(PetscStrallocpy(fec[f], &fecs[tnf + f]));
      Ufds[tnf + f] = Uf[f];
      sdim[tnf + f] = sd[f];
    }
    tnf += nf;
  }
  PetscCall(PetscViewerGLVisSetFields(viewer, tnf, (const char **)fecs, sdim, DMCompositeSampleGLVisFields_Private, (PetscObject *)Ufds, ctx, DestroyGLVisViewerCtx_Private));
  for (i = 0; i < tnf; i++) PetscCall(PetscFree(fecs[i]));
  PetscCall(PetscFree3(fecs, sdim, Ufds));
  PetscFunctionReturn(0);
}

PetscErrorCode DMRefine_Composite(DM dmi, MPI_Comm comm, DM *fine)
{
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite *)dmi->data;
  DM                      dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmi, DM_CLASSID, 1);
  if (comm == MPI_COMM_NULL) PetscCall(PetscObjectGetComm((PetscObject)dmi, &comm));
  PetscCall(DMSetUp(dmi));
  next = com->next;
  PetscCall(DMCompositeCreate(comm, fine));

  /* loop over packed objects, handling one at at time */
  while (next) {
    PetscCall(DMRefine(next->dm, comm, &dm));
    PetscCall(DMCompositeAddDM(*fine, dm));
    PetscCall(PetscObjectDereference((PetscObject)dm));
    next = next->next;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCoarsen_Composite(DM dmi, MPI_Comm comm, DM *fine)
{
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite *)dmi->data;
  DM                      dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmi, DM_CLASSID, 1);
  PetscCall(DMSetUp(dmi));
  if (comm == MPI_COMM_NULL) PetscCall(PetscObjectGetComm((PetscObject)dmi, &comm));
  next = com->next;
  PetscCall(DMCompositeCreate(comm, fine));

  /* loop over packed objects, handling one at at time */
  while (next) {
    PetscCall(DMCoarsen(next->dm, comm, &dm));
    PetscCall(DMCompositeAddDM(*fine, dm));
    PetscCall(PetscObjectDereference((PetscObject)dm));
    next = next->next;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateInterpolation_Composite(DM coarse, DM fine, Mat *A, Vec *v)
{
  PetscInt                m, n, M, N, nDM, i;
  struct DMCompositeLink *nextc;
  struct DMCompositeLink *nextf;
  Vec                     gcoarse, gfine, *vecs;
  DM_Composite           *comcoarse = (DM_Composite *)coarse->data;
  DM_Composite           *comfine   = (DM_Composite *)fine->data;
  Mat                    *mats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse, DM_CLASSID, 1);
  PetscValidHeaderSpecific(fine, DM_CLASSID, 2);
  PetscCall(DMSetUp(coarse));
  PetscCall(DMSetUp(fine));
  /* use global vectors only for determining matrix layout */
  PetscCall(DMGetGlobalVector(coarse, &gcoarse));
  PetscCall(DMGetGlobalVector(fine, &gfine));
  PetscCall(VecGetLocalSize(gcoarse, &n));
  PetscCall(VecGetLocalSize(gfine, &m));
  PetscCall(VecGetSize(gcoarse, &N));
  PetscCall(VecGetSize(gfine, &M));
  PetscCall(DMRestoreGlobalVector(coarse, &gcoarse));
  PetscCall(DMRestoreGlobalVector(fine, &gfine));

  nDM = comfine->nDM;
  PetscCheck(nDM == comcoarse->nDM, PetscObjectComm((PetscObject)fine), PETSC_ERR_ARG_INCOMP, "Fine DMComposite has %" PetscInt_FMT " entries, but coarse has %" PetscInt_FMT, nDM, comcoarse->nDM);
  PetscCall(PetscCalloc1(nDM * nDM, &mats));
  if (v) PetscCall(PetscCalloc1(nDM, &vecs));

  /* loop over packed objects, handling one at at time */
  for (nextc = comcoarse->next, nextf = comfine->next, i = 0; nextc; nextc = nextc->next, nextf = nextf->next, i++) {
    if (!v) PetscCall(DMCreateInterpolation(nextc->dm, nextf->dm, &mats[i * nDM + i], NULL));
    else PetscCall(DMCreateInterpolation(nextc->dm, nextf->dm, &mats[i * nDM + i], &vecs[i]));
  }
  PetscCall(MatCreateNest(PetscObjectComm((PetscObject)fine), nDM, NULL, nDM, NULL, mats, A));
  if (v) PetscCall(VecCreateNest(PetscObjectComm((PetscObject)fine), nDM, NULL, vecs, v));
  for (i = 0; i < nDM * nDM; i++) PetscCall(MatDestroy(&mats[i]));
  PetscCall(PetscFree(mats));
  if (v) {
    for (i = 0; i < nDM; i++) PetscCall(VecDestroy(&vecs[i]));
    PetscCall(PetscFree(vecs));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetLocalToGlobalMapping_Composite(DM dm)
{
  DM_Composite           *com = (DM_Composite *)dm->data;
  ISLocalToGlobalMapping *ltogs;
  PetscInt                i;

  PetscFunctionBegin;
  /* Set the ISLocalToGlobalMapping on the new matrix */
  PetscCall(DMCompositeGetISLocalToGlobalMappings(dm, &ltogs));
  PetscCall(ISLocalToGlobalMappingConcatenate(PetscObjectComm((PetscObject)dm), com->nDM, ltogs, &dm->ltogmap));
  for (i = 0; i < com->nDM; i++) PetscCall(ISLocalToGlobalMappingDestroy(&ltogs[i]));
  PetscCall(PetscFree(ltogs));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateColoring_Composite(DM dm, ISColoringType ctype, ISColoring *coloring)
{
  PetscInt         n, i, cnt;
  ISColoringValue *colors;
  PetscBool        dense  = PETSC_FALSE;
  ISColoringValue  maxcol = 0;
  DM_Composite    *com    = (DM_Composite *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheck(ctype != IS_COLORING_LOCAL, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only global coloring supported");
  if (ctype == IS_COLORING_GLOBAL) {
    n = com->n;
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Unknown ISColoringType");
  PetscCall(PetscMalloc1(n, &colors)); /* freed in ISColoringDestroy() */

  PetscCall(PetscOptionsGetBool(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-dmcomposite_dense_jacobian", &dense, NULL));
  if (dense) {
    for (i = 0; i < n; i++) colors[i] = (ISColoringValue)(com->rstart + i);
    maxcol = com->N;
  } else {
    struct DMCompositeLink *next = com->next;
    PetscMPIInt             rank;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
    cnt = 0;
    while (next) {
      ISColoring lcoloring;

      PetscCall(DMCreateColoring(next->dm, IS_COLORING_GLOBAL, &lcoloring));
      for (i = 0; i < lcoloring->N; i++) colors[cnt++] = maxcol + lcoloring->colors[i];
      maxcol += lcoloring->n;
      PetscCall(ISColoringDestroy(&lcoloring));
      next = next->next;
    }
  }
  PetscCall(ISColoringCreate(PetscObjectComm((PetscObject)dm), maxcol, n, colors, PETSC_OWN_POINTER, coloring));
  PetscFunctionReturn(0);
}

PetscErrorCode DMGlobalToLocalBegin_Composite(DM dm, Vec gvec, InsertMode mode, Vec lvec)
{
  struct DMCompositeLink *next;
  PetscScalar            *garray, *larray;
  DM_Composite           *com = (DM_Composite *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(gvec, VEC_CLASSID, 2);

  if (!com->setup) PetscCall(DMSetUp(dm));

  PetscCall(VecGetArray(gvec, &garray));
  PetscCall(VecGetArray(lvec, &larray));

  /* loop over packed objects, handling one at at time */
  next = com->next;
  while (next) {
    Vec      local, global;
    PetscInt N;

    PetscCall(DMGetGlobalVector(next->dm, &global));
    PetscCall(VecGetLocalSize(global, &N));
    PetscCall(VecPlaceArray(global, garray));
    PetscCall(DMGetLocalVector(next->dm, &local));
    PetscCall(VecPlaceArray(local, larray));
    PetscCall(DMGlobalToLocalBegin(next->dm, global, mode, local));
    PetscCall(DMGlobalToLocalEnd(next->dm, global, mode, local));
    PetscCall(VecResetArray(global));
    PetscCall(VecResetArray(local));
    PetscCall(DMRestoreGlobalVector(next->dm, &global));
    PetscCall(DMRestoreLocalVector(next->dm, &local));

    larray += next->nlocal;
    garray += next->n;
    next = next->next;
  }

  PetscCall(VecRestoreArray(gvec, NULL));
  PetscCall(VecRestoreArray(lvec, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode DMGlobalToLocalEnd_Composite(DM dm, Vec gvec, InsertMode mode, Vec lvec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(gvec, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(lvec, VEC_CLASSID, 4);
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocalToGlobalBegin_Composite(DM dm, Vec lvec, InsertMode mode, Vec gvec)
{
  struct DMCompositeLink *next;
  PetscScalar            *larray, *garray;
  DM_Composite           *com = (DM_Composite *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(lvec, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(gvec, VEC_CLASSID, 4);

  if (!com->setup) PetscCall(DMSetUp(dm));

  PetscCall(VecGetArray(lvec, &larray));
  PetscCall(VecGetArray(gvec, &garray));

  /* loop over packed objects, handling one at at time */
  next = com->next;
  while (next) {
    Vec global, local;

    PetscCall(DMGetLocalVector(next->dm, &local));
    PetscCall(VecPlaceArray(local, larray));
    PetscCall(DMGetGlobalVector(next->dm, &global));
    PetscCall(VecPlaceArray(global, garray));
    PetscCall(DMLocalToGlobalBegin(next->dm, local, mode, global));
    PetscCall(DMLocalToGlobalEnd(next->dm, local, mode, global));
    PetscCall(VecResetArray(local));
    PetscCall(VecResetArray(global));
    PetscCall(DMRestoreGlobalVector(next->dm, &global));
    PetscCall(DMRestoreLocalVector(next->dm, &local));

    garray += next->n;
    larray += next->nlocal;
    next = next->next;
  }

  PetscCall(VecRestoreArray(gvec, NULL));
  PetscCall(VecRestoreArray(lvec, NULL));

  PetscFunctionReturn(0);
}

PetscErrorCode DMLocalToGlobalEnd_Composite(DM dm, Vec lvec, InsertMode mode, Vec gvec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(lvec, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(gvec, VEC_CLASSID, 4);
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocalToLocalBegin_Composite(DM dm, Vec vec1, InsertMode mode, Vec vec2)
{
  struct DMCompositeLink *next;
  PetscScalar            *array1, *array2;
  DM_Composite           *com = (DM_Composite *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec1, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(vec2, VEC_CLASSID, 4);

  if (!com->setup) PetscCall(DMSetUp(dm));

  PetscCall(VecGetArray(vec1, &array1));
  PetscCall(VecGetArray(vec2, &array2));

  /* loop over packed objects, handling one at at time */
  next = com->next;
  while (next) {
    Vec local1, local2;

    PetscCall(DMGetLocalVector(next->dm, &local1));
    PetscCall(VecPlaceArray(local1, array1));
    PetscCall(DMGetLocalVector(next->dm, &local2));
    PetscCall(VecPlaceArray(local2, array2));
    PetscCall(DMLocalToLocalBegin(next->dm, local1, mode, local2));
    PetscCall(DMLocalToLocalEnd(next->dm, local1, mode, local2));
    PetscCall(VecResetArray(local2));
    PetscCall(DMRestoreLocalVector(next->dm, &local2));
    PetscCall(VecResetArray(local1));
    PetscCall(DMRestoreLocalVector(next->dm, &local1));

    array1 += next->nlocal;
    array2 += next->nlocal;
    next = next->next;
  }

  PetscCall(VecRestoreArray(vec1, NULL));
  PetscCall(VecRestoreArray(vec2, NULL));

  PetscFunctionReturn(0);
}

PetscErrorCode DMLocalToLocalEnd_Composite(DM dm, Vec lvec, InsertMode mode, Vec gvec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(lvec, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(gvec, VEC_CLASSID, 4);
  PetscFunctionReturn(0);
}

/*MC
   DMCOMPOSITE = "composite" - A `DM` object that is used to manage data for a collection of `DM`

  Level: intermediate

.seealso: `DMType`, `DM`, `DMDACreate()`, `DMCreate()`, `DMSetType()`, `DMCompositeCreate()`
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Composite(DM p)
{
  DM_Composite *com;

  PetscFunctionBegin;
  PetscCall(PetscNew(&com));
  p->data     = com;
  com->n      = 0;
  com->nghost = 0;
  com->next   = NULL;
  com->nDM    = 0;

  p->ops->createglobalvector       = DMCreateGlobalVector_Composite;
  p->ops->createlocalvector        = DMCreateLocalVector_Composite;
  p->ops->getlocaltoglobalmapping  = DMGetLocalToGlobalMapping_Composite;
  p->ops->createfieldis            = DMCreateFieldIS_Composite;
  p->ops->createfielddecomposition = DMCreateFieldDecomposition_Composite;
  p->ops->refine                   = DMRefine_Composite;
  p->ops->coarsen                  = DMCoarsen_Composite;
  p->ops->createinterpolation      = DMCreateInterpolation_Composite;
  p->ops->creatematrix             = DMCreateMatrix_Composite;
  p->ops->getcoloring              = DMCreateColoring_Composite;
  p->ops->globaltolocalbegin       = DMGlobalToLocalBegin_Composite;
  p->ops->globaltolocalend         = DMGlobalToLocalEnd_Composite;
  p->ops->localtoglobalbegin       = DMLocalToGlobalBegin_Composite;
  p->ops->localtoglobalend         = DMLocalToGlobalEnd_Composite;
  p->ops->localtolocalbegin        = DMLocalToLocalBegin_Composite;
  p->ops->localtolocalend          = DMLocalToLocalEnd_Composite;
  p->ops->destroy                  = DMDestroy_Composite;
  p->ops->view                     = DMView_Composite;
  p->ops->setup                    = DMSetUp_Composite;

  PetscCall(PetscObjectComposeFunction((PetscObject)p, "DMSetUpGLVisViewer_C", DMSetUpGLVisViewer_Composite));
  PetscFunctionReturn(0);
}

/*@
    DMCompositeCreate - Creates a `DMCOMPOSITE`, used to generate "composite"
      vectors made up of several subvectors.

    Collective

    Input Parameter:
.   comm - the processors that will share the global vector

    Output Parameters:
.   packer - the `DMCOMPOSITE` object

    Level: advanced

.seealso: `DMCOMPOSITE`, `DM`, `DMDestroy()`, `DMCompositeAddDM()`, `DMCompositeScatter()`, `DMCOMPOSITE`, `DMCreate()`
          `DMCompositeGather()`, `DMCreateGlobalVector()`, `DMCompositeGetISLocalToGlobalMappings()`, `DMCompositeGetAccess()`
          `DMCompositeGetLocalVectors()`, `DMCompositeRestoreLocalVectors()`, `DMCompositeGetEntries()`
@*/
PetscErrorCode DMCompositeCreate(MPI_Comm comm, DM *packer)
{
  PetscFunctionBegin;
  PetscValidPointer(packer, 2);
  PetscCall(DMCreate(comm, packer));
  PetscCall(DMSetType(*packer, DMCOMPOSITE));
  PetscFunctionReturn(0);
}
