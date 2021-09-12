#include <petsc/private/dmmbimpl.h> /*I  "petscdmmoab.h"   I*/
#include <petsc/private/vecimpl.h>

#include <petscdmmoab.h>
#include <MBTagConventions.hpp>

#define USE_NATIVE_PETSCVEC

/* declare some private DMMoab specific overrides */
static PetscErrorCode DMCreateVector_Moab_Private(DM dm,moab::Tag tag,const moab::Range* userrange,PetscBool is_global_vec,PetscBool destroy_tag,Vec *vec);
static PetscErrorCode DMVecUserDestroy_Moab(void *user);
static PetscErrorCode DMVecDuplicate_Moab(Vec x,Vec *y);
#ifdef MOAB_HAVE_MPI
static PetscErrorCode DMVecCreateTagName_Moab_Private(moab::Interface *mbiface,moab::ParallelComm *pcomm,char** tag_name);
#else
static PetscErrorCode DMVecCreateTagName_Moab_Private(moab::Interface *mbiface,char** tag_name);
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

.seealso: VecCreate()
@*/
PetscErrorCode DMMoabCreateVector(DM dm, moab::Tag tag, const moab::Range* range, PetscBool is_global_vec, PetscBool destroy_tag, Vec *vec)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!tag && (!range || range->empty())) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Both tag and range cannot be null.");

  ierr = DMCreateVector_Moab_Private(dm, tag, range, is_global_vec, destroy_tag, vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMMoabGetVecTag - Get the MOAB tag associated with this Vec

  Input Parameter:
. vec - Vec being queried

  Output Parameter:
. tag - Tag associated with this Vec. NULL if vec is a native PETSc Vec.

  Level: beginner

.seealso: DMMoabCreateVector(), DMMoabGetVecRange()
@*/
PetscErrorCode DMMoabGetVecTag(Vec vec, moab::Tag *tag)
{
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(tag, 2);

  /* Get the MOAB private data */
  ierr = PetscObjectQuery((PetscObject)vec, "MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**) &vmoab);CHKERRQ(ierr);

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

.seealso: DMMoabCreateVector(), DMMoabGetVecTag()
@*/
PetscErrorCode DMMoabGetVecRange(Vec vec, moab::Range *range)
{
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(range, 2);

  /* Get the MOAB private data handle */
  ierr = PetscObjectQuery((PetscObject)vec, "MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**) &vmoab);CHKERRQ(ierr);

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

.seealso: DMMoabVecRestoreArray(), DMMoabVecGetArrayRead(), DMMoabVecRestoreArrayRead()
@*/
PetscErrorCode  DMMoabVecGetArray(DM dm, Vec vec, void* array)
{
  DM_Moab        *dmmoab;
  moab::ErrorCode merr;
  PetscErrorCode  ierr;
  PetscInt        count, i, f;
  moab::Tag       vtag;
  PetscScalar     **varray;
  PetscScalar     *marray;
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab, *xmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscValidPointer(array, 3);
  dmmoab = (DM_Moab*)dm->data;

  /* Get the Vec_MOAB struct for the original vector */
  ierr = PetscObjectQuery((PetscObject)vec, "MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**)&vmoab);CHKERRQ(ierr);

  /* Get the real scalar array handle */
  varray = reinterpret_cast<PetscScalar**>(array);

  if (vmoab->is_native_vec) {

    /* get the local representation of the arrays from Vectors */
    ierr = DMGetLocalVector(dm, &vmoab->local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, vec, INSERT_VALUES, vmoab->local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, vec, INSERT_VALUES, vmoab->local);CHKERRQ(ierr);

    /* Get the Vec_MOAB struct for the original vector */
    ierr = PetscObjectQuery((PetscObject)vmoab->local, "MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
    ierr = PetscContainerGetPointer(moabdata, (void**)&xmoab);CHKERRQ(ierr);

    /* get the local representation of the arrays from Vectors */
    ierr = VecGhostGetLocalForm(vmoab->local, &xmoab->local);CHKERRQ(ierr);
    ierr = VecGhostUpdateBegin(vmoab->local, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(vmoab->local, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecGetArray(xmoab->local, varray);CHKERRQ(ierr);
  }
  else {

    /* Get the MOAB private data */
    ierr = DMMoabGetVecTag(vec, &vtag);CHKERRQ(ierr);

#ifdef MOAB_HAVE_MPI
    /* exchange the data into ghost cells first */
    merr = dmmoab->pcomm->exchange_tags(vtag, *dmmoab->vlocal); MBERRNM(merr);
#endif

    ierr = PetscMalloc1((dmmoab->nloc + dmmoab->nghost) * dmmoab->numFields, varray);CHKERRQ(ierr);

    /* Get the array data for local entities */
    merr = dmmoab->mbiface->tag_iterate(vtag, dmmoab->vlocal->begin(), dmmoab->vlocal->end(), count, reinterpret_cast<void*&>(marray), false); MBERRNM(merr);
    if (count != dmmoab->nloc + dmmoab->nghost) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mismatch between local vertices and tag partition for Vec. %D != %D.", count, dmmoab->nloc + dmmoab->nghost);

    i = 0;
    for (moab::Range::iterator iter = dmmoab->vlocal->begin(); iter != dmmoab->vlocal->end(); iter++) {
      for (f = 0; f < dmmoab->numFields; f++, i++)
        (*varray)[dmmoab->lidmap[(PetscInt)*iter - dmmoab->seqstart]*dmmoab->numFields + f] = marray[i];
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

.seealso: DMMoabVecGetArray(), DMMoabVecGetArrayRead(), DMMoabVecRestoreArrayRead()
@*/
PetscErrorCode  DMMoabVecRestoreArray(DM dm, Vec vec, void* array)
{
  DM_Moab        *dmmoab;
  moab::ErrorCode merr;
  PetscErrorCode  ierr;
  moab::Tag       vtag;
  PetscInt        count, i, f;
  PetscScalar     **varray;
  PetscScalar     *marray;
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab, *xmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscValidPointer(array, 3);
  dmmoab = (DM_Moab*)dm->data;

  /* Get the Vec_MOAB struct for the original vector */
  ierr = PetscObjectQuery((PetscObject)vec, "MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**)&vmoab);CHKERRQ(ierr);

  /* Get the real scalar array handle */
  varray = reinterpret_cast<PetscScalar**>(array);

  if (vmoab->is_native_vec) {

    /* Get the Vec_MOAB struct for the original vector */
    ierr = PetscObjectQuery((PetscObject)vmoab->local, "MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
    ierr = PetscContainerGetPointer(moabdata, (void**)&xmoab);CHKERRQ(ierr);

    /* get the local representation of the arrays from Vectors */
    ierr = VecRestoreArray(xmoab->local, varray);CHKERRQ(ierr);
    ierr = VecGhostUpdateBegin(vmoab->local, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(vmoab->local, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecGhostRestoreLocalForm(vmoab->local, &xmoab->local);CHKERRQ(ierr);

    /* restore local pieces */
    ierr = DMLocalToGlobalBegin(dm, vmoab->local, INSERT_VALUES, vec);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, vmoab->local, INSERT_VALUES, vec);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &vmoab->local);CHKERRQ(ierr);
  }
  else {

    /* Get the MOAB private data */
    ierr = DMMoabGetVecTag(vec, &vtag);CHKERRQ(ierr);

    /* Get the array data for local entities */
    merr = dmmoab->mbiface->tag_iterate(vtag, dmmoab->vlocal->begin(), dmmoab->vlocal->end(), count, reinterpret_cast<void*&>(marray), false); MBERRNM(merr);
    if (count != dmmoab->nloc + dmmoab->nghost) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mismatch between local vertices and tag partition for Vec. %D != %D.", count, dmmoab->nloc + dmmoab->nghost);

    i = 0;
    for (moab::Range::iterator iter = dmmoab->vlocal->begin(); iter != dmmoab->vlocal->end(); iter++) {
      for (f = 0; f < dmmoab->numFields; f++, i++)
        marray[i] = (*varray)[dmmoab->lidmap[(PetscInt) * iter - dmmoab->seqstart] * dmmoab->numFields + f];
      //marray[i] = (*varray)[dmmoab->llmap[dmmoab->lidmap[((PetscInt)*iter-dmmoab->seqstart)]*dmmoab->numFields+f]];
    }

#ifdef MOAB_HAVE_MPI
    /* reduce the tags correctly -> should probably let the user choose how to reduce in the future
      For all FEM residual based assembly calculations, MPI_SUM should serve well */
    merr = dmmoab->pcomm->reduce_tags(vtag, MPI_SUM, *dmmoab->vlocal); MBERRV(dmmoab->mbiface, merr);
#endif
    ierr = PetscFree(*varray);CHKERRQ(ierr);
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

.seealso: DMMoabVecRestoreArrayRead(), DMMoabVecGetArray(), DMMoabVecRestoreArray()
@*/
PetscErrorCode  DMMoabVecGetArrayRead(DM dm, Vec vec, void* array)
{
  DM_Moab        *dmmoab;
  moab::ErrorCode merr;
  PetscErrorCode  ierr;
  PetscInt        count, i, f;
  moab::Tag       vtag;
  PetscScalar     **varray;
  PetscScalar     *marray;
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab, *xmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscValidPointer(array, 3);
  dmmoab = (DM_Moab*)dm->data;

  /* Get the Vec_MOAB struct for the original vector */
  ierr = PetscObjectQuery((PetscObject)vec, "MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**)&vmoab);CHKERRQ(ierr);

  /* Get the real scalar array handle */
  varray = reinterpret_cast<PetscScalar**>(array);

  if (vmoab->is_native_vec) {
    /* get the local representation of the arrays from Vectors */
    ierr = DMGetLocalVector(dm, &vmoab->local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, vec, INSERT_VALUES, vmoab->local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, vec, INSERT_VALUES, vmoab->local);CHKERRQ(ierr);

    /* Get the Vec_MOAB struct for the original vector */
    ierr = PetscObjectQuery((PetscObject)vmoab->local, "MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
    ierr = PetscContainerGetPointer(moabdata, (void**)&xmoab);CHKERRQ(ierr);

    /* get the local representation of the arrays from Vectors */
    ierr = VecGhostGetLocalForm(vmoab->local, &xmoab->local);CHKERRQ(ierr);
    ierr = VecGhostUpdateBegin(vmoab->local, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(vmoab->local, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(xmoab->local, varray);CHKERRQ(ierr);
  }
  else {
    /* Get the MOAB private data */
    ierr = DMMoabGetVecTag(vec, &vtag);CHKERRQ(ierr);

#ifdef MOAB_HAVE_MPI
    /* exchange the data into ghost cells first */
    merr = dmmoab->pcomm->exchange_tags(vtag, *dmmoab->vlocal); MBERRNM(merr);
#endif
    ierr = PetscMalloc1((dmmoab->nloc + dmmoab->nghost) * dmmoab->numFields, varray);CHKERRQ(ierr);

    /* Get the array data for local entities */
    merr = dmmoab->mbiface->tag_iterate(vtag, dmmoab->vlocal->begin(), dmmoab->vlocal->end(), count, reinterpret_cast<void*&>(marray), false); MBERRNM(merr);
    if (count != dmmoab->nloc + dmmoab->nghost) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mismatch between local vertices and tag partition for Vec. %D != %D.", count, dmmoab->nloc + dmmoab->nghost);

    i = 0;
    for (moab::Range::iterator iter = dmmoab->vlocal->begin(); iter != dmmoab->vlocal->end(); iter++) {
      for (f = 0; f < dmmoab->numFields; f++, i++)
        (*varray)[dmmoab->lidmap[(PetscInt)*iter - dmmoab->seqstart]*dmmoab->numFields + f] = marray[i];
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

.seealso: DMMoabVecGetArrayRead(), DMMoabVecGetArray(), DMMoabVecRestoreArray()
@*/
PetscErrorCode  DMMoabVecRestoreArrayRead(DM dm, Vec vec, void* array)
{
  PetscErrorCode  ierr;
  PetscScalar     **varray;
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab, *xmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscValidPointer(array, 3);

  /* Get the Vec_MOAB struct for the original vector */
  ierr = PetscObjectQuery((PetscObject)vec, "MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**)&vmoab);CHKERRQ(ierr);

  /* Get the real scalar array handle */
  varray = reinterpret_cast<PetscScalar**>(array);

  if (vmoab->is_native_vec) {
    /* Get the Vec_MOAB struct for the original vector */
    ierr = PetscObjectQuery((PetscObject)vmoab->local, "MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
    ierr = PetscContainerGetPointer(moabdata, (void**)&xmoab);CHKERRQ(ierr);

    /* restore the local representation of the arrays from Vectors */
    ierr = VecRestoreArray(xmoab->local, varray);CHKERRQ(ierr);
    ierr = VecGhostRestoreLocalForm(vmoab->local, &xmoab->local);CHKERRQ(ierr);

    /* restore local pieces */
    ierr = DMRestoreLocalVector(dm, &vmoab->local);CHKERRQ(ierr);
  }
  else {
    /* Nothing to do but just free the memory allocated before */
    ierr = PetscFree(*varray);CHKERRQ(ierr);

  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateVector_Moab_Private(DM dm, moab::Tag tag, const moab::Range* userrange, PetscBool is_global_vec, PetscBool destroy_tag, Vec *vec)
{
  PetscErrorCode    ierr;
  moab::ErrorCode   merr;
  PetscBool         is_newtag;
  const moab::Range *range;
  PetscInt          count, lnative_vec, gnative_vec;
  std::string       ttname;
  PetscScalar       *data_ptr, *defaultvals;

  Vec_MOAB *vmoab;
  DM_Moab *dmmoab = (DM_Moab*)dm->data;
#ifdef MOAB_HAVE_MPI
  moab::ParallelComm *pcomm = dmmoab->pcomm;
#endif
  moab::Interface *mbiface = dmmoab->mbiface;

  PetscFunctionBegin;
  if (sizeof(PetscReal) != sizeof(PetscScalar)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "MOAB tags only support Real types (Complex-type unsupported)");
  if (!userrange) range = dmmoab->vowned;
  else range = userrange;
  if (!range) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Input range cannot be empty or call DMSetUp first.");

#ifndef USE_NATIVE_PETSCVEC
  /* If the tag data is in a single sequence, use PETSc native vector since tag_iterate isn't useful anymore */
  lnative_vec = (range->psize() - 1);
#else
  lnative_vec = 1; /* NOTE: Testing PETSc vector: will force to create native vector all the time */
//  lnative_vec=0; /* NOTE: Testing MOAB vector: will force to create MOAB tag_iterate based vector all the time */
#endif

#ifdef MOAB_HAVE_MPI
  ierr = MPIU_Allreduce(&lnative_vec, &gnative_vec, 1, MPI_INT, MPI_MAX, (((PetscObject)dm)->comm));CHKERRMPI(ierr);
#else
  gnative_vec = lnative_vec;
#endif

  /* Create the MOAB internal data object */
  ierr = PetscNew(&vmoab);CHKERRQ(ierr);
  vmoab->is_native_vec = (gnative_vec > 0 ? PETSC_TRUE : PETSC_FALSE);

  if (!vmoab->is_native_vec) {
    merr = moab::MB_SUCCESS;
    if (tag != 0) merr = mbiface->tag_get_name(tag, ttname);
    if (!ttname.length() || merr != moab::MB_SUCCESS) {
      /* get the new name for the anonymous MOABVec -> the tag_name will be destroyed along with Tag */
      char *tag_name = NULL;
#ifdef MOAB_HAVE_MPI
      ierr = DMVecCreateTagName_Moab_Private(mbiface,pcomm,&tag_name);CHKERRQ(ierr);
#else
      ierr = DMVecCreateTagName_Moab_Private(mbiface,&tag_name);CHKERRQ(ierr);
#endif
      is_newtag = PETSC_TRUE;

      /* Create the default value for the tag (all zeros) */
      ierr = PetscCalloc1(dmmoab->numFields, &defaultvals);CHKERRQ(ierr);

      /* Create the tag */
      merr = mbiface->tag_get_handle(tag_name, dmmoab->numFields, moab::MB_TYPE_DOUBLE, tag,
                                     moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, defaultvals); MBERRNM(merr);
      ierr = PetscFree(tag_name);CHKERRQ(ierr);
      ierr = PetscFree(defaultvals);CHKERRQ(ierr);
    }
    else {
      /* Make sure the tag data is of type "double" */
      moab::DataType tag_type;
      merr = mbiface->tag_get_data_type(tag, tag_type); MBERRNM(merr);
      if (tag_type != moab::MB_TYPE_DOUBLE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Tag data type must be MB_TYPE_DOUBLE");
      is_newtag = destroy_tag;
    }

    vmoab->tag = tag;
    vmoab->new_tag = is_newtag;
  }
  vmoab->mbiface = mbiface;
#ifdef MOAB_HAVE_MPI
  vmoab->pcomm = pcomm;
#endif
  vmoab->is_global_vec = is_global_vec;
  vmoab->tag_size = dmmoab->bs;

  if (vmoab->is_native_vec) {

    /* Create the PETSc Vector directly and attach our functions accordingly */
    if (!is_global_vec) {
      /* This is an MPI Vector with ghosted padding */
      ierr = VecCreateGhostBlock((((PetscObject)dm)->comm), dmmoab->bs, dmmoab->numFields * dmmoab->nloc,
                                 dmmoab->numFields * dmmoab->n, dmmoab->nghost, &dmmoab->gsindices[dmmoab->nloc], vec);CHKERRQ(ierr);
    }
    else {
      /* This is an MPI/SEQ Vector */
      ierr = VecCreate((((PetscObject)dm)->comm), vec);CHKERRQ(ierr);
      ierr = VecSetSizes(*vec, dmmoab->numFields * dmmoab->nloc, PETSC_DECIDE);CHKERRQ(ierr);
      ierr = VecSetBlockSize(*vec, dmmoab->bs);CHKERRQ(ierr);
      ierr = VecSetType(*vec, VECMPI);CHKERRQ(ierr);
    }
  }
  else {
    /* Call tag_iterate. This will cause MOAB to allocate memory for the
       tag data if it hasn't already happened */
    merr = mbiface->tag_iterate(tag, range->begin(), range->end(), count, (void*&)data_ptr); MBERRNM(merr);

    /* set the reference for vector range */
    vmoab->tag_range = new moab::Range(*range);
    merr = mbiface->tag_get_length(tag, dmmoab->numFields); MBERRNM(merr);

    /* Create the PETSc Vector
      Query MOAB mesh to check if there are any ghosted entities
        -> if we do, create a ghosted vector to map correctly to the same layout
        -> else, create a non-ghosted parallel vector */
    if (!is_global_vec) {
      /* This is an MPI Vector with ghosted padding */
      ierr = VecCreateGhostBlockWithArray((((PetscObject)dm)->comm), dmmoab->bs, dmmoab->numFields * dmmoab->nloc,
                                          dmmoab->numFields * dmmoab->n, dmmoab->nghost, &dmmoab->gsindices[dmmoab->nloc], data_ptr, vec);CHKERRQ(ierr);
    }
    else {
      /* This is an MPI Vector without ghosted padding */
      ierr = VecCreateMPIWithArray((((PetscObject)dm)->comm), dmmoab->bs, dmmoab->numFields * range->size(),
                                   PETSC_DECIDE, data_ptr, vec);CHKERRQ(ierr);
    }
  }
  ierr = VecSetFromOptions(*vec);CHKERRQ(ierr);

  /* create a container and store the internal MOAB data for faster access based on Entities etc */
  PetscContainer moabdata;
  ierr = PetscContainerCreate(PETSC_COMM_WORLD, &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(moabdata, vmoab);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(moabdata, DMVecUserDestroy_Moab);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) * vec, "MOABData", (PetscObject)moabdata);CHKERRQ(ierr);
  (*vec)->ops->duplicate = DMVecDuplicate_Moab;
  ierr = PetscContainerDestroy(&moabdata);CHKERRQ(ierr);

  /* Vector created, manually set local to global mapping */
  if (dmmoab->ltog_map) {
    ierr = VecSetLocalToGlobalMapping(*vec, dmmoab->ltog_map);CHKERRQ(ierr);
  }

  /* set the DM reference to the vector */
  ierr = VecSetDM(*vec, dm);CHKERRQ(ierr);
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
PetscErrorCode DMVecCreateTagName_Moab_Private(moab::Interface *mbiface, moab::ParallelComm *pcomm, char** tag_name)
#else
PetscErrorCode DMVecCreateTagName_Moab_Private(moab::Interface *mbiface, char** tag_name)
#endif
{
  moab::ErrorCode mberr;
  PetscErrorCode  ierr;
  PetscInt        n, global_n;
  moab::Tag indexTag;

  PetscFunctionBegin;
  const char*       PVEC_PREFIX      = "__PETSC_VEC_";
  ierr = PetscMalloc1(PETSC_MAX_PATH_LEN, tag_name);CHKERRQ(ierr);

  moab::EntityHandle rootset = mbiface->get_root_set();

  /* Check to see if there are any PETSc vectors defined */
  /* Create a tag in MOAB mesh to index and keep track of number of Petsc vec tags */
  mberr = mbiface->tag_get_handle("__PETSC_VECS__", 1, moab::MB_TYPE_INTEGER, indexTag,
                                  moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT, 0); MBERRNM(mberr);
  mberr = mbiface->tag_get_data(indexTag, &rootset, 1, &n);
  if (mberr == moab::MB_TAG_NOT_FOUND) n = 0; /* this is the first temporary vector */
  else MBERRNM(mberr);

  /* increment the new value of n */
  ++n;

#ifdef MOAB_HAVE_MPI
  /* Make sure that n is consistent across all processes */
  ierr = MPIU_Allreduce(&n, &global_n, 1, MPI_INT, MPI_MAX, pcomm->comm());CHKERRMPI(ierr);
#else
  global_n = n;
#endif

  /* Set the new name accordingly and return */
  ierr = PetscSNPrintf(*tag_name, PETSC_MAX_PATH_LEN - 1, "%s_%D", PVEC_PREFIX, global_n);CHKERRQ(ierr);
  mberr = mbiface->tag_set_data(indexTag, &rootset, 1, (const void*)&global_n); MBERRNM(mberr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMCreateGlobalVector_Moab(DM dm, Vec *gvec)
{
  PetscErrorCode  ierr;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gvec,2);
  ierr = DMCreateVector_Moab_Private(dm,NULL,dmmoab->vowned,PETSC_TRUE,PETSC_TRUE,gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMCreateLocalVector_Moab(DM dm, Vec *lvec)
{
  PetscErrorCode  ierr;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(lvec,2);
  ierr = DMCreateVector_Moab_Private(dm,NULL,dmmoab->vlocal,PETSC_FALSE,PETSC_TRUE,lvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMVecDuplicate_Moab(Vec x, Vec *y)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(y, 2);

  /* Get the Vec_MOAB struct for the original vector */
  ierr = PetscObjectQuery((PetscObject)x, "MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**)&vmoab);CHKERRQ(ierr);

  ierr = VecGetDM(x, &dm);CHKERRQ(ierr);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);

  ierr = DMCreateVector_Moab_Private(dm,NULL,vmoab->tag_range,vmoab->is_global_vec,PETSC_TRUE,y);CHKERRQ(ierr);
  ierr = VecSetDM(*y, dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMVecUserDestroy_Moab(void *user)
{
  Vec_MOAB        *vmoab = (Vec_MOAB*)user;
  PetscErrorCode  ierr;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  if (vmoab->new_tag && vmoab->tag) {
    /* Tag was created via a call to VecDuplicate, delete the underlying tag in MOAB */
    merr = vmoab->mbiface->tag_delete(vmoab->tag); MBERRNM(merr);
  }
  delete vmoab->tag_range;
  vmoab->tag = NULL;
  vmoab->mbiface = NULL;
#ifdef MOAB_HAVE_MPI
  vmoab->pcomm = NULL;
#endif
  ierr = PetscFree(vmoab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode  DMGlobalToLocalBegin_Moab(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscErrorCode    ierr;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  ierr = VecScatterBegin(dmmoab->ltog_sendrecv, g, l, mode, SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode  DMGlobalToLocalEnd_Moab(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscErrorCode    ierr;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  ierr = VecScatterEnd(dmmoab->ltog_sendrecv, g, l, mode, SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode  DMLocalToGlobalBegin_Moab(DM dm, Vec l, InsertMode mode, Vec g)
{
  PetscErrorCode    ierr;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  ierr = VecScatterBegin(dmmoab->ltog_sendrecv, l, g, mode, SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode  DMLocalToGlobalEnd_Moab(DM dm, Vec l, InsertMode mode, Vec g)
{
  PetscErrorCode    ierr;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  ierr = VecScatterEnd(dmmoab->ltog_sendrecv, l, g, mode, SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

