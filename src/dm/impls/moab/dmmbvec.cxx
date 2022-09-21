#include <petsc/private/dmmbimpl.h> /*I  "petscdmmoab.h"   I*/
#include <petsc/private/vecimpl.h>

#include <petscdmmoab.h>
#include <MBTagConventions.hpp>

#define USE_NATIVE_PETSCVEC

/* declare some private DMMoab specific overrides */
static PetscErrorCode DMCreateVector_Moab_Private(DM dm, moab::Tag tag, const moab::Range *userrange, PetscBool is_global_vec, PetscBool destroy_tag, Vec *vec);
static PetscErrorCode DMVecUserDestroy_Moab(void *user);
static PetscErrorCode DMVecDuplicate_Moab(Vec x, Vec *y);
#ifdef MOAB_HAVE_MPI
static PetscErrorCode DMVecCreateTagName_Moab_Private(moab::Interface *mbiface, moab::ParallelComm *pcomm, char **tag_name);
#else
static PetscErrorCode DMVecCreateTagName_Moab_Private(moab::Interface *mbiface, char **tag_name);
#endif

/*@C
  DMMoabCreateVector - Create a Vec from either an existing tag, or a specified tag size, and a range of entities

  Collective

  Input Parameters:
+ dm              - The DMMoab object being set
. tag             - If non-zero, block size will be taken from the tag size
. range           - If non-empty, Vec corresponds to these entities, otherwise to the entities set on the DMMoab
. is_global_vec   - If true, this is a local representation of the Vec (including ghosts in parallel), otherwise a truly parallel one
- destroy_tag     - If true, MOAB tag is destroyed with Vec, otherwise it is left on MOAB

  Output Parameter:
. vec             - The created vector

  Level: beginner

.seealso: `VecCreate()`
@*/
PetscErrorCode DMMoabCreateVector(DM dm, moab::Tag tag, const moab::Range *range, PetscBool is_global_vec, PetscBool destroy_tag, Vec *vec)
{
  PetscFunctionBegin;
  PetscCheck(tag || (range && !range->empty()), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Both tag and range cannot be null.");

  PetscCall(DMCreateVector_Moab_Private(dm, tag, range, is_global_vec, destroy_tag, vec));
  PetscFunctionReturn(0);
}

/*@C
  DMMoabGetVecTag - Get the MOAB tag associated with this Vec

  Input Parameter:
. vec - Vec being queried

  Output Parameter:
. tag - Tag associated with this Vec. NULL if vec is a native PETSc Vec.

  Level: beginner

.seealso: `DMMoabCreateVector()`, `DMMoabGetVecRange()`
@*/
PetscErrorCode DMMoabGetVecTag(Vec vec, moab::Tag *tag)
{
  PetscContainer moabdata;
  Vec_MOAB      *vmoab;

  PetscFunctionBegin;
  PetscValidPointer(tag, 2);

  /* Get the MOAB private data */
  PetscCall(PetscObjectQuery((PetscObject)vec, "MOABData", (PetscObject *)&moabdata));
  PetscCall(PetscContainerGetPointer(moabdata, (void **)&vmoab));

  *tag = vmoab->tag;
  PetscFunctionReturn(0);
}

/*@C
  DMMoabGetVecRange - Get the MOAB entities associated with this Vec

  Input Parameter:
. vec   - Vec being queried

  Output Parameter:
. range - Entities associated with this Vec. NULL if vec is a native PETSc Vec.

  Level: beginner

.seealso: `DMMoabCreateVector()`, `DMMoabGetVecTag()`
@*/
PetscErrorCode DMMoabGetVecRange(Vec vec, moab::Range *range)
{
  PetscContainer moabdata;
  Vec_MOAB      *vmoab;

  PetscFunctionBegin;
  PetscValidPointer(range, 2);

  /* Get the MOAB private data handle */
  PetscCall(PetscObjectQuery((PetscObject)vec, "MOABData", (PetscObject *)&moabdata));
  PetscCall(PetscContainerGetPointer(moabdata, (void **)&vmoab));

  *range = *vmoab->tag_range;
  PetscFunctionReturn(0);
}

/*@C
  DMMoabVecGetArray - Returns the writable direct access array to the local representation of MOAB tag data for the underlying vector using locally owned+ghosted range of entities

  Collective

  Input Parameters:
+ dm              - The DMMoab object being set
- vec             - The Vector whose underlying data is requested

  Output Parameter:
. array           - The local data array

  Level: intermediate

.seealso: `DMMoabVecRestoreArray()`, `DMMoabVecGetArrayRead()`, `DMMoabVecRestoreArrayRead()`
@*/
PetscErrorCode DMMoabVecGetArray(DM dm, Vec vec, void *array)
{
  DM_Moab        *dmmoab;
  moab::ErrorCode merr;
  PetscInt        count, i, f;
  moab::Tag       vtag;
  PetscScalar   **varray;
  PetscScalar    *marray;
  PetscContainer  moabdata;
  Vec_MOAB       *vmoab, *xmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscValidPointer(array, 3);
  dmmoab = (DM_Moab *)dm->data;

  /* Get the Vec_MOAB struct for the original vector */
  PetscCall(PetscObjectQuery((PetscObject)vec, "MOABData", (PetscObject *)&moabdata));
  PetscCall(PetscContainerGetPointer(moabdata, (void **)&vmoab));

  /* Get the real scalar array handle */
  varray = reinterpret_cast<PetscScalar **>(array);

  if (vmoab->is_native_vec) {
    /* get the local representation of the arrays from Vectors */
    PetscCall(DMGetLocalVector(dm, &vmoab->local));
    PetscCall(DMGlobalToLocalBegin(dm, vec, INSERT_VALUES, vmoab->local));
    PetscCall(DMGlobalToLocalEnd(dm, vec, INSERT_VALUES, vmoab->local));

    /* Get the Vec_MOAB struct for the original vector */
    PetscCall(PetscObjectQuery((PetscObject)vmoab->local, "MOABData", (PetscObject *)&moabdata));
    PetscCall(PetscContainerGetPointer(moabdata, (void **)&xmoab));

    /* get the local representation of the arrays from Vectors */
    PetscCall(VecGhostGetLocalForm(vmoab->local, &xmoab->local));
    PetscCall(VecGhostUpdateBegin(vmoab->local, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGhostUpdateEnd(vmoab->local, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(VecGetArray(xmoab->local, varray));
  } else {
    /* Get the MOAB private data */
    PetscCall(DMMoabGetVecTag(vec, &vtag));

#ifdef MOAB_HAVE_MPI
    /* exchange the data into ghost cells first */
    merr = dmmoab->pcomm->exchange_tags(vtag, *dmmoab->vlocal);
    MBERRNM(merr);
#endif

    PetscCall(PetscMalloc1((dmmoab->nloc + dmmoab->nghost) * dmmoab->numFields, varray));

    /* Get the array data for local entities */
    merr = dmmoab->mbiface->tag_iterate(vtag, dmmoab->vlocal->begin(), dmmoab->vlocal->end(), count, reinterpret_cast<void *&>(marray), false);
    MBERRNM(merr);
    PetscCheck(count == dmmoab->nloc + dmmoab->nghost, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mismatch between local vertices and tag partition for Vec. %" PetscInt_FMT " != %" PetscInt_FMT ".", count, dmmoab->nloc + dmmoab->nghost);

    i = 0;
    for (moab::Range::iterator iter = dmmoab->vlocal->begin(); iter != dmmoab->vlocal->end(); iter++) {
      for (f = 0; f < dmmoab->numFields; f++, i++) (*varray)[dmmoab->lidmap[(PetscInt)*iter - dmmoab->seqstart] * dmmoab->numFields + f] = marray[i];
      //(*varray)[dmmoab->llmap[dmmoab->lidmap[((PetscInt)*iter-dmmoab->seqstart)]*dmmoab->numFields+f]]=marray[i];
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMMoabVecRestoreArray - Restores the writable direct access array obtained via DMMoabVecGetArray

  Collective

  Input Parameters:
+ dm              - The DMMoab object being set
. vec             - The Vector whose underlying data is requested
- array           - The local data array

  Level: intermediate

.seealso: `DMMoabVecGetArray()`, `DMMoabVecGetArrayRead()`, `DMMoabVecRestoreArrayRead()`
@*/
PetscErrorCode DMMoabVecRestoreArray(DM dm, Vec vec, void *array)
{
  DM_Moab        *dmmoab;
  moab::ErrorCode merr;
  moab::Tag       vtag;
  PetscInt        count, i, f;
  PetscScalar   **varray;
  PetscScalar    *marray;
  PetscContainer  moabdata;
  Vec_MOAB       *vmoab, *xmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscValidPointer(array, 3);
  dmmoab = (DM_Moab *)dm->data;

  /* Get the Vec_MOAB struct for the original vector */
  PetscCall(PetscObjectQuery((PetscObject)vec, "MOABData", (PetscObject *)&moabdata));
  PetscCall(PetscContainerGetPointer(moabdata, (void **)&vmoab));

  /* Get the real scalar array handle */
  varray = reinterpret_cast<PetscScalar **>(array);

  if (vmoab->is_native_vec) {
    /* Get the Vec_MOAB struct for the original vector */
    PetscCall(PetscObjectQuery((PetscObject)vmoab->local, "MOABData", (PetscObject *)&moabdata));
    PetscCall(PetscContainerGetPointer(moabdata, (void **)&xmoab));

    /* get the local representation of the arrays from Vectors */
    PetscCall(VecRestoreArray(xmoab->local, varray));
    PetscCall(VecGhostUpdateBegin(vmoab->local, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecGhostUpdateEnd(vmoab->local, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecGhostRestoreLocalForm(vmoab->local, &xmoab->local));

    /* restore local pieces */
    PetscCall(DMLocalToGlobalBegin(dm, vmoab->local, INSERT_VALUES, vec));
    PetscCall(DMLocalToGlobalEnd(dm, vmoab->local, INSERT_VALUES, vec));
    PetscCall(DMRestoreLocalVector(dm, &vmoab->local));
  } else {
    /* Get the MOAB private data */
    PetscCall(DMMoabGetVecTag(vec, &vtag));

    /* Get the array data for local entities */
    merr = dmmoab->mbiface->tag_iterate(vtag, dmmoab->vlocal->begin(), dmmoab->vlocal->end(), count, reinterpret_cast<void *&>(marray), false);
    MBERRNM(merr);
    PetscCheck(count == dmmoab->nloc + dmmoab->nghost, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mismatch between local vertices and tag partition for Vec. %" PetscInt_FMT " != %" PetscInt_FMT ".", count, dmmoab->nloc + dmmoab->nghost);

    i = 0;
    for (moab::Range::iterator iter = dmmoab->vlocal->begin(); iter != dmmoab->vlocal->end(); iter++) {
      for (f = 0; f < dmmoab->numFields; f++, i++) marray[i] = (*varray)[dmmoab->lidmap[(PetscInt)*iter - dmmoab->seqstart] * dmmoab->numFields + f];
      //marray[i] = (*varray)[dmmoab->llmap[dmmoab->lidmap[((PetscInt)*iter-dmmoab->seqstart)]*dmmoab->numFields+f]];
    }

#ifdef MOAB_HAVE_MPI
    /* reduce the tags correctly -> should probably let the user choose how to reduce in the future
      For all FEM residual based assembly calculations, MPI_SUM should serve well */
    merr = dmmoab->pcomm->reduce_tags(vtag, MPI_SUM, *dmmoab->vlocal);
    MBERRV(dmmoab->mbiface, merr);
#endif
    PetscCall(PetscFree(*varray));
  }
  PetscFunctionReturn(0);
}

/*@C
  DMMoabVecGetArrayRead - Returns the read-only direct access array to the local representation of MOAB tag data for the underlying vector using locally owned+ghosted range of entities

  Collective

  Input Parameters:
+ dm              - The DMMoab object being set
- vec             - The Vector whose underlying data is requested

  Output Parameter:
. array           - The local data array

  Level: intermediate

.seealso: `DMMoabVecRestoreArrayRead()`, `DMMoabVecGetArray()`, `DMMoabVecRestoreArray()`
@*/
PetscErrorCode DMMoabVecGetArrayRead(DM dm, Vec vec, void *array)
{
  DM_Moab        *dmmoab;
  moab::ErrorCode merr;
  PetscInt        count, i, f;
  moab::Tag       vtag;
  PetscScalar   **varray;
  PetscScalar    *marray;
  PetscContainer  moabdata;
  Vec_MOAB       *vmoab, *xmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscValidPointer(array, 3);
  dmmoab = (DM_Moab *)dm->data;

  /* Get the Vec_MOAB struct for the original vector */
  PetscCall(PetscObjectQuery((PetscObject)vec, "MOABData", (PetscObject *)&moabdata));
  PetscCall(PetscContainerGetPointer(moabdata, (void **)&vmoab));

  /* Get the real scalar array handle */
  varray = reinterpret_cast<PetscScalar **>(array);

  if (vmoab->is_native_vec) {
    /* get the local representation of the arrays from Vectors */
    PetscCall(DMGetLocalVector(dm, &vmoab->local));
    PetscCall(DMGlobalToLocalBegin(dm, vec, INSERT_VALUES, vmoab->local));
    PetscCall(DMGlobalToLocalEnd(dm, vec, INSERT_VALUES, vmoab->local));

    /* Get the Vec_MOAB struct for the original vector */
    PetscCall(PetscObjectQuery((PetscObject)vmoab->local, "MOABData", (PetscObject *)&moabdata));
    PetscCall(PetscContainerGetPointer(moabdata, (void **)&xmoab));

    /* get the local representation of the arrays from Vectors */
    PetscCall(VecGhostGetLocalForm(vmoab->local, &xmoab->local));
    PetscCall(VecGhostUpdateBegin(vmoab->local, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGhostUpdateEnd(vmoab->local, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArray(xmoab->local, varray));
  } else {
    /* Get the MOAB private data */
    PetscCall(DMMoabGetVecTag(vec, &vtag));

#ifdef MOAB_HAVE_MPI
    /* exchange the data into ghost cells first */
    merr = dmmoab->pcomm->exchange_tags(vtag, *dmmoab->vlocal);
    MBERRNM(merr);
#endif
    PetscCall(PetscMalloc1((dmmoab->nloc + dmmoab->nghost) * dmmoab->numFields, varray));

    /* Get the array data for local entities */
    merr = dmmoab->mbiface->tag_iterate(vtag, dmmoab->vlocal->begin(), dmmoab->vlocal->end(), count, reinterpret_cast<void *&>(marray), false);
    MBERRNM(merr);
    PetscCheck(count == dmmoab->nloc + dmmoab->nghost, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mismatch between local vertices and tag partition for Vec. %" PetscInt_FMT " != %" PetscInt_FMT ".", count, dmmoab->nloc + dmmoab->nghost);

    i = 0;
    for (moab::Range::iterator iter = dmmoab->vlocal->begin(); iter != dmmoab->vlocal->end(); iter++) {
      for (f = 0; f < dmmoab->numFields; f++, i++) (*varray)[dmmoab->lidmap[(PetscInt)*iter - dmmoab->seqstart] * dmmoab->numFields + f] = marray[i];
      //(*varray)[dmmoab->llmap[dmmoab->lidmap[((PetscInt)*iter-dmmoab->seqstart)]*dmmoab->numFields+f]]=marray[i];
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMMoabVecRestoreArrayRead - Restores the read-only direct access array obtained via DMMoabVecGetArray

  Collective

  Input Parameters:
+ dm              - The DMMoab object being set
. vec             - The Vector whose underlying data is requested
- array           - The local data array

  Level: intermediate

.seealso: `DMMoabVecGetArrayRead()`, `DMMoabVecGetArray()`, `DMMoabVecRestoreArray()`
@*/
PetscErrorCode DMMoabVecRestoreArrayRead(DM dm, Vec vec, void *array)
{
  PetscScalar  **varray;
  PetscContainer moabdata;
  Vec_MOAB      *vmoab, *xmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscValidPointer(array, 3);

  /* Get the Vec_MOAB struct for the original vector */
  PetscCall(PetscObjectQuery((PetscObject)vec, "MOABData", (PetscObject *)&moabdata));
  PetscCall(PetscContainerGetPointer(moabdata, (void **)&vmoab));

  /* Get the real scalar array handle */
  varray = reinterpret_cast<PetscScalar **>(array);

  if (vmoab->is_native_vec) {
    /* Get the Vec_MOAB struct for the original vector */
    PetscCall(PetscObjectQuery((PetscObject)vmoab->local, "MOABData", (PetscObject *)&moabdata));
    PetscCall(PetscContainerGetPointer(moabdata, (void **)&xmoab));

    /* restore the local representation of the arrays from Vectors */
    PetscCall(VecRestoreArray(xmoab->local, varray));
    PetscCall(VecGhostRestoreLocalForm(vmoab->local, &xmoab->local));

    /* restore local pieces */
    PetscCall(DMRestoreLocalVector(dm, &vmoab->local));
  } else {
    /* Nothing to do but just free the memory allocated before */
    PetscCall(PetscFree(*varray));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateVector_Moab_Private(DM dm, moab::Tag tag, const moab::Range *userrange, PetscBool is_global_vec, PetscBool destroy_tag, Vec *vec)
{
  moab::ErrorCode    merr;
  PetscBool          is_newtag;
  const moab::Range *range;
  PetscInt           count, lnative_vec, gnative_vec;
  std::string        ttname;
  PetscScalar       *data_ptr, *defaultvals;

  Vec_MOAB *vmoab;
  DM_Moab  *dmmoab = (DM_Moab *)dm->data;
#ifdef MOAB_HAVE_MPI
  moab::ParallelComm *pcomm = dmmoab->pcomm;
#endif
  moab::Interface *mbiface = dmmoab->mbiface;

  PetscFunctionBegin;
  PetscCheck(sizeof(PetscReal) == sizeof(PetscScalar), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "MOAB tags only support Real types (Complex-type unsupported)");
  if (!userrange) range = dmmoab->vowned;
  else range = userrange;
  PetscCheck(range, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Input range cannot be empty or call DMSetUp first.");

#ifndef USE_NATIVE_PETSCVEC
  /* If the tag data is in a single sequence, use PETSc native vector since tag_iterate isn't useful anymore */
  lnative_vec = (range->psize() - 1);
#else
  lnative_vec = 1; /* NOTE: Testing PETSc vector: will force to create native vector all the time */
                   //  lnative_vec=0; /* NOTE: Testing MOAB vector: will force to create MOAB tag_iterate based vector all the time */
#endif

#ifdef MOAB_HAVE_MPI
  PetscCall(MPIU_Allreduce(&lnative_vec, &gnative_vec, 1, MPI_INT, MPI_MAX, (((PetscObject)dm)->comm)));
#else
  gnative_vec = lnative_vec;
#endif

  /* Create the MOAB internal data object */
  PetscCall(PetscNew(&vmoab));
  vmoab->is_native_vec = (gnative_vec > 0 ? PETSC_TRUE : PETSC_FALSE);

  if (!vmoab->is_native_vec) {
    merr = moab::MB_SUCCESS;
    if (tag != 0) merr = mbiface->tag_get_name(tag, ttname);
    if (!ttname.length() || merr != moab::MB_SUCCESS) {
      /* get the new name for the anonymous MOABVec -> the tag_name will be destroyed along with Tag */
      char *tag_name = NULL;
#ifdef MOAB_HAVE_MPI
      PetscCall(DMVecCreateTagName_Moab_Private(mbiface, pcomm, &tag_name));
#else
      PetscCall(DMVecCreateTagName_Moab_Private(mbiface, &tag_name));
#endif
      is_newtag = PETSC_TRUE;

      /* Create the default value for the tag (all zeros) */
      PetscCall(PetscCalloc1(dmmoab->numFields, &defaultvals));

      /* Create the tag */
      merr = mbiface->tag_get_handle(tag_name, dmmoab->numFields, moab::MB_TYPE_DOUBLE, tag, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, defaultvals);
      MBERRNM(merr);
      PetscCall(PetscFree(tag_name));
      PetscCall(PetscFree(defaultvals));
    } else {
      /* Make sure the tag data is of type "double" */
      moab::DataType tag_type;
      merr = mbiface->tag_get_data_type(tag, tag_type);
      MBERRNM(merr);
      PetscCheck(tag_type == moab::MB_TYPE_DOUBLE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Tag data type must be MB_TYPE_DOUBLE");
      is_newtag = destroy_tag;
    }

    vmoab->tag     = tag;
    vmoab->new_tag = is_newtag;
  }
  vmoab->mbiface = mbiface;
#ifdef MOAB_HAVE_MPI
  vmoab->pcomm = pcomm;
#endif
  vmoab->is_global_vec = is_global_vec;
  vmoab->tag_size      = dmmoab->bs;

  if (vmoab->is_native_vec) {
    /* Create the PETSc Vector directly and attach our functions accordingly */
    if (!is_global_vec) {
      /* This is an MPI Vector with ghosted padding */
      PetscCall(VecCreateGhostBlock((((PetscObject)dm)->comm), dmmoab->bs, dmmoab->numFields * dmmoab->nloc, dmmoab->numFields * dmmoab->n, dmmoab->nghost, &dmmoab->gsindices[dmmoab->nloc], vec));
    } else {
      /* This is an MPI/SEQ Vector */
      PetscCall(VecCreate((((PetscObject)dm)->comm), vec));
      PetscCall(VecSetSizes(*vec, dmmoab->numFields * dmmoab->nloc, PETSC_DECIDE));
      PetscCall(VecSetBlockSize(*vec, dmmoab->bs));
      PetscCall(VecSetType(*vec, VECMPI));
    }
  } else {
    /* Call tag_iterate. This will cause MOAB to allocate memory for the
       tag data if it hasn't already happened */
    merr = mbiface->tag_iterate(tag, range->begin(), range->end(), count, (void *&)data_ptr);
    MBERRNM(merr);

    /* set the reference for vector range */
    vmoab->tag_range = new moab::Range(*range);
    merr             = mbiface->tag_get_length(tag, dmmoab->numFields);
    MBERRNM(merr);

    /* Create the PETSc Vector
      Query MOAB mesh to check if there are any ghosted entities
        -> if we do, create a ghosted vector to map correctly to the same layout
        -> else, create a non-ghosted parallel vector */
    if (!is_global_vec) {
      /* This is an MPI Vector with ghosted padding */
      PetscCall(VecCreateGhostBlockWithArray((((PetscObject)dm)->comm), dmmoab->bs, dmmoab->numFields * dmmoab->nloc, dmmoab->numFields * dmmoab->n, dmmoab->nghost, &dmmoab->gsindices[dmmoab->nloc], data_ptr, vec));
    } else {
      /* This is an MPI Vector without ghosted padding */
      PetscCall(VecCreateMPIWithArray((((PetscObject)dm)->comm), dmmoab->bs, dmmoab->numFields * range->size(), PETSC_DECIDE, data_ptr, vec));
    }
  }
  PetscCall(VecSetFromOptions(*vec));

  /* create a container and store the internal MOAB data for faster access based on Entities etc */
  PetscContainer moabdata;
  PetscCall(PetscContainerCreate(PETSC_COMM_WORLD, &moabdata));
  PetscCall(PetscContainerSetPointer(moabdata, vmoab));
  PetscCall(PetscContainerSetUserDestroy(moabdata, DMVecUserDestroy_Moab));
  PetscCall(PetscObjectCompose((PetscObject)*vec, "MOABData", (PetscObject)moabdata));
  (*vec)->ops->duplicate = DMVecDuplicate_Moab;
  PetscCall(PetscContainerDestroy(&moabdata));

  /* Vector created, manually set local to global mapping */
  if (dmmoab->ltog_map) PetscCall(VecSetLocalToGlobalMapping(*vec, dmmoab->ltog_map));

  /* set the DM reference to the vector */
  PetscCall(VecSetDM(*vec, dm));
  PetscFunctionReturn(0);
}

/*  DMVecCreateTagName_Moab_Private
 *
 *  Creates a unique tag name that will be shared across processes. If
 *  pcomm is NULL, then this is a serial vector. A unique tag name
 *  will be returned in tag_name in either case.
 *
 *  The tag names have the format _PETSC_VEC_N where N is some integer.
 *
 *  NOTE: The tag_name is allocated in this routine; The user needs to free
 *        the character array.
 */
#ifdef MOAB_HAVE_MPI
PetscErrorCode DMVecCreateTagName_Moab_Private(moab::Interface *mbiface, moab::ParallelComm *pcomm, char **tag_name)
#else
PetscErrorCode DMVecCreateTagName_Moab_Private(moab::Interface *mbiface, char **tag_name)
#endif
{
  moab::ErrorCode mberr;
  PetscInt        n, global_n;
  moab::Tag       indexTag;

  PetscFunctionBegin;
  const char *PVEC_PREFIX = "__PETSC_VEC_";
  PetscCall(PetscMalloc1(PETSC_MAX_PATH_LEN, tag_name));

  moab::EntityHandle rootset = mbiface->get_root_set();

  /* Check to see if there are any PETSc vectors defined */
  /* Create a tag in MOAB mesh to index and keep track of number of Petsc vec tags */
  mberr = mbiface->tag_get_handle("__PETSC_VECS__", 1, moab::MB_TYPE_INTEGER, indexTag, moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT, 0);
  MBERRNM(mberr);
  mberr = mbiface->tag_get_data(indexTag, &rootset, 1, &n);
  if (mberr == moab::MB_TAG_NOT_FOUND) n = 0; /* this is the first temporary vector */
  else MBERRNM(mberr);

  /* increment the new value of n */
  ++n;

#ifdef MOAB_HAVE_MPI
  /* Make sure that n is consistent across all processes */
  PetscCall(MPIU_Allreduce(&n, &global_n, 1, MPI_INT, MPI_MAX, pcomm->comm()));
#else
  global_n = n;
#endif

  /* Set the new name accordingly and return */
  PetscCall(PetscSNPrintf(*tag_name, PETSC_MAX_PATH_LEN - 1, "%s_%" PetscInt_FMT, PVEC_PREFIX, global_n));
  mberr = mbiface->tag_set_data(indexTag, &rootset, 1, (const void *)&global_n);
  MBERRNM(mberr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMCreateGlobalVector_Moab(DM dm, Vec *gvec)
{
  DM_Moab *dmmoab = (DM_Moab *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(gvec, 2);
  PetscCall(DMCreateVector_Moab_Private(dm, NULL, dmmoab->vowned, PETSC_TRUE, PETSC_TRUE, gvec));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMCreateLocalVector_Moab(DM dm, Vec *lvec)
{
  DM_Moab *dmmoab = (DM_Moab *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(lvec, 2);
  PetscCall(DMCreateVector_Moab_Private(dm, NULL, dmmoab->vlocal, PETSC_FALSE, PETSC_TRUE, lvec));
  PetscFunctionReturn(0);
}

PetscErrorCode DMVecDuplicate_Moab(Vec x, Vec *y)
{
  DM             dm;
  PetscContainer moabdata;
  Vec_MOAB      *vmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(y, 2);

  /* Get the Vec_MOAB struct for the original vector */
  PetscCall(PetscObjectQuery((PetscObject)x, "MOABData", (PetscObject *)&moabdata));
  PetscCall(PetscContainerGetPointer(moabdata, (void **)&vmoab));

  PetscCall(VecGetDM(x, &dm));
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);

  PetscCall(DMCreateVector_Moab_Private(dm, NULL, vmoab->tag_range, vmoab->is_global_vec, PETSC_TRUE, y));
  PetscCall(VecSetDM(*y, dm));
  PetscFunctionReturn(0);
}

PetscErrorCode DMVecUserDestroy_Moab(void *user)
{
  Vec_MOAB       *vmoab = (Vec_MOAB *)user;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  if (vmoab->new_tag && vmoab->tag) {
    /* Tag was created via a call to VecDuplicate, delete the underlying tag in MOAB */
    merr = vmoab->mbiface->tag_delete(vmoab->tag);
    MBERRNM(merr);
  }
  delete vmoab->tag_range;
  vmoab->tag     = NULL;
  vmoab->mbiface = NULL;
#ifdef MOAB_HAVE_MPI
  vmoab->pcomm = NULL;
#endif
  PetscCall(PetscFree(vmoab));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMGlobalToLocalBegin_Moab(DM dm, Vec g, InsertMode mode, Vec l)
{
  DM_Moab *dmmoab = (DM_Moab *)dm->data;

  PetscFunctionBegin;
  PetscCall(VecScatterBegin(dmmoab->ltog_sendrecv, g, l, mode, SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMGlobalToLocalEnd_Moab(DM dm, Vec g, InsertMode mode, Vec l)
{
  DM_Moab *dmmoab = (DM_Moab *)dm->data;

  PetscFunctionBegin;
  PetscCall(VecScatterEnd(dmmoab->ltog_sendrecv, g, l, mode, SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMLocalToGlobalBegin_Moab(DM dm, Vec l, InsertMode mode, Vec g)
{
  DM_Moab *dmmoab = (DM_Moab *)dm->data;

  PetscFunctionBegin;
  PetscCall(VecScatterBegin(dmmoab->ltog_sendrecv, l, g, mode, SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMLocalToGlobalEnd_Moab(DM dm, Vec l, InsertMode mode, Vec g)
{
  DM_Moab *dmmoab = (DM_Moab *)dm->data;

  PetscFunctionBegin;
  PetscCall(VecScatterEnd(dmmoab->ltog_sendrecv, l, g, mode, SCATTER_FORWARD));
  PetscFunctionReturn(0);
}
