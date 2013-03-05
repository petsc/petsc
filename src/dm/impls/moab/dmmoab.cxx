#include <petsc-private/dmimpl.h> /*I  "petscdm.h"   I*/
#include "../src/vec/vec/impls/moab/vecmoabimpl.h" /*I  "petscvec.h"   I*/

#include <petscdmmoab.h>
#include "MBTagConventions.hpp"

typedef struct {
  PetscInt bs; /* Number of degrees of freedom on each entity, aka tag size in moab */
  PetscBool icreatedinstance; /* true if DM created moab instance internally, will destroy instance in DMDestroy */
  moab::ParallelComm *pcomm;
  moab::Interface *mbint;
  moab::Tag ltog_tag; /* moab supports "global id" tags, which are usually local to global numbering */
  moab::Range range;
} DM_Moab;

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Moab"
PetscErrorCode DMCreateGlobalVector_Moab(DM dm,Vec *gvec)
{
  PetscErrorCode  ierr;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gvec,2);
  PetscInt block_size = ((DM_Moab*)dm->data)->bs;
  ierr = VecMoabCreate(dmmoab->mbint,dmmoab->pcomm,block_size,dmmoab->ltog_tag,dmmoab->range,PETSC_FALSE,PETSC_TRUE,gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Moab"
PetscErrorCode DMCreateLocalVector_Moab(DM dm,Vec *gvec)
{
  PetscErrorCode  ierr;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  PetscInt bs = 1;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gvec,2);
  ierr = VecMoabCreate(dmmoab->mbint,dmmoab->pcomm,bs,dmmoab->ltog_tag,dmmoab->range,PETSC_TRUE,PETSC_TRUE,gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Moab"
PetscErrorCode DMDestroy_Moab(DM dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);

  // Delete the DM_Moab:
  if(dm->data) {
    if (((DM_Moab*)dm->data)->icreatedinstance) {
      delete ((DM_Moab*)dm->data)->mbint;
      ((DM_Moab*)dm->data)->mbint = NULL;
      ((DM_Moab*)dm->data)->pcomm = NULL;
    }
    delete (DM_Moab*)dm->data;
    dm->data = NULL;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMInitialize_Moab"
PetscErrorCode DMInitialize_Moab(DM dm)
{
  PetscFunctionBegin;

  // Create the DM_Moab and set dm->data
  DM_Moab *dmmoab = new DM_Moab;
  dmmoab->bs       = 0;
  dmmoab->pcomm    = NULL;
  dmmoab->mbint    = NULL;
  dmmoab->ltog_tag = (moab::Tag)0;
  dmmoab->icreatedinstance = PETSC_FALSE;
  dm->data      = dmmoab;

    // initialize various functions
  dm->ops->createglobalvector              = DMCreateGlobalVector_Moab;
  dm->ops->createlocalvector               = DMCreateLocalVector_Moab;
  dm->ops->destroy                         = DMDestroy_Moab;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabCreate"
/*@
  DMMoabCreate - Creates a DMMoab object, which encapsulates a moab instance

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMMoab object

  Output Parameter:
. moab  - The DMMoab object

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabCreate(MPI_Comm comm, DM *moab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(moab,2);
  ierr = DMCreate(comm, moab);CHKERRQ(ierr);
  ierr = DMSetType(*moab, DMMOAB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMCreate_Moab"
PetscErrorCode DMCreate_Moab(DM dm)
{
  DM_Moab        *moab;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr     = PetscNewLog(dm, DM_Moab, &moab);CHKERRQ(ierr);
  dm->data = moab;

  PetscFunctionReturn(0);
}
EXTERN_C_END

/*@
  DMMoabCreateFromInstance - Creates a DMMoab object from an instance and (optionally) other data

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMMoab object
. moab - The MOAB Instance
. pcomm - A ParallelComm; if NULL, creates one internally for the whole communicator
. ltog_tag - A tag to use to retrieve global id for an entity; if 0, will use GLOBAL_ID_TAG_NAME/tag
. range - If non-NULL, contains range of entities to which DOFs will be assigned

  Output Parameter:
. moab  - The DMMoab object

  Level: beginner

.keywords: DMMoab, create
@*/
#undef __FUNCT__
#define __FUNCT__ "DMMoabCreateFromInstance"
PetscErrorCode DMMoabCreateFromInstance(MPI_Comm comm, moab::Interface *iface, moab::ParallelComm *pcomm, moab::Tag ltog_tag, moab::Range *range, DM *moab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(moab,2);
  ierr = DMCreate(comm, moab);CHKERRQ(ierr);
  ierr = DMSetType(*moab, DMMOAB);CHKERRQ(ierr);
  ierr = DMInitialize_Moab(*moab);CHKERRQ(ierr);
  ierr = DMMoabSetInterface(*moab, iface);CHKERRQ(ierr);
  if (!pcomm) pcomm = new moab::ParallelComm(iface, comm);
  ierr = DMMoabSetParallelComm(*moab, pcomm);CHKERRQ(ierr);
  if (!ltog_tag) {
    moab::ErrorCode merr = iface->tag_get_handle(GLOBAL_ID_TAG_NAME, ltog_tag);MBERRNM(merr);
  }
  ierr = DMMoabSetLocalToGlobalTag(*moab, ltog_tag);CHKERRQ(ierr);
  if (range) {
    ierr = DMMoabSetRange(*moab, *range);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMMoabCreateDMAndInstance - Creates a DMMoab object and a MOAB instance

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMMoab object

  Output Parameter:
. moab  - The DMMoab object

  Level: beginner

.keywords: DMMoab, create
@*/
#undef __FUNCT__
#define __FUNCT__ "DMMoabCreateDMAndInstance"
PetscErrorCode DMMoabCreateDMAndInstance(MPI_Comm comm, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm,2);
  PetscInt rank, nprocs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);
  moab::Interface *iface = new moab::Core();
  moab::ParallelComm *pcomm = new moab::ParallelComm(iface, comm);
  ierr = DMMoabCreateFromInstance(comm, iface, pcomm, 0, NULL, dm);CHKERRQ(ierr);
  ((DM_Moab*)(*dm)->data)->icreatedinstance = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabSetParallelComm"
PetscErrorCode DMMoabSetParallelComm(DM dm,moab::ParallelComm *pcomm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Moab*)dm->data)->pcomm = pcomm;
  ((DM_Moab*)dm->data)->mbint = pcomm->get_moab();
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetParallelComm"
PetscErrorCode DMMoabGetParallelComm(DM dm,moab::ParallelComm **pcomm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *pcomm = ((DM_Moab*)dm->data)->pcomm;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetInterface"
PetscErrorCode DMMoabSetInterface(DM dm,moab::Interface *iface)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Moab*)dm->data)->pcomm = NULL;
  ((DM_Moab*)dm->data)->mbint = iface;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetInterface"
PetscErrorCode DMMoabGetInterface(DM dm,moab::Interface **mbint)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *mbint = ((DM_Moab*)dm->data)->mbint;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetRange"
PetscErrorCode DMMoabSetRange(DM dm,moab::Range range)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Moab*)dm->data)->range = range;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetRange"
PetscErrorCode DMMoabGetRange(DM dm,moab::Range *range)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *range = ((DM_Moab*)dm->data)->range;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabSetLocalToGlobalTag"
PetscErrorCode DMMoabSetLocalToGlobalTag(DM dm,moab::Tag ltogtag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Moab*)dm->data)->ltog_tag = ltogtag;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetLocalToGlobalTag"
PetscErrorCode DMMoabGetLocalToGlobalTag(DM dm,moab::Tag *ltog_tag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *ltog_tag = ((DM_Moab*)dm->data)->ltog_tag;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetBlockSize"
PetscErrorCode DMMoabSetBlockSize(DM dm,PetscInt bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Moab*)dm->data)->bs = bs;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetBlockSize"
PetscErrorCode DMMoabGetBlockSize(DM dm,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *bs = ((DM_Moab*)dm->data)->bs;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabCreateVectorFromTag"
PetscErrorCode DMMoabCreateVectorFromTag(DM dm,moab::Tag tag,moab::Range range,PetscBool serial, PetscBool destroy_tag,Vec *X)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;
  ierr = VecMoabCreateFromTag(dmmoab->mbint, dmmoab->pcomm, tag, dmmoab->ltog_tag,range, serial, destroy_tag, X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabCreateVector"
PetscErrorCode DMMoabCreateVector(DM dm,PetscInt tag_size,moab::Range range,PetscBool serial,PetscBool destroy_tag,Vec *vec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;
  ierr = VecMoabCreate(dmmoab->mbint, dmmoab->pcomm, tag_size, dmmoab->ltog_tag, range, serial, destroy_tag, vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetVecTag"
PetscErrorCode DMMoabGetVecTag(Vec vec,moab::Tag *tag)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecMoabGetTag(vec,tag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetVecRange"
PetscErrorCode DMMoabGetVecRange(Vec vec,moab::Range *range)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecMoabGetRange(vec,range);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


