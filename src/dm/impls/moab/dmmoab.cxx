#include <petsc-private/dmimpl.h> /*I  "petscdm.h"   I*/
#include <petsc-private/vecimpl.h> /*I  "petscdm.h"   I*/

#include <petscdmmoab.h>
#include <MBTagConventions.hpp>
#include <sstream>

typedef struct {
  PetscInt bs; /* Number of degrees of freedom on each entity, aka tag size in moab */
  PetscBool icreatedinstance; /* true if DM created moab instance internally, will destroy instance in DMDestroy */
  moab::ParallelComm *pcomm;
  moab::Interface *mbiface;
  moab::Tag ltog_tag; /* moab supports "global id" tags, which are usually local to global numbering */
  moab::Range range;
} DM_Moab;

typedef struct {
  moab::Interface    *mbiface;
  moab::ParallelComm *pcomm;
  moab::Range         tag_range; /* entities to which this tag applies */
  moab::Tag           tag;
  moab::Tag           ltog_tag;
  PetscInt            tag_size;
  PetscBool           new_tag;
  PetscBool           serial;

} Vec_MOAB;

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
  moab::Tag tag = 0;
  ierr = DMMoabCreateVector(dm,tag,block_size,dmmoab->range,PETSC_FALSE,PETSC_TRUE,gvec);CHKERRQ(ierr);
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
  moab::Tag tag = 0;
  ierr = DMMoabCreateVector(dm,tag,bs,dmmoab->range,PETSC_TRUE,PETSC_TRUE,gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Moab"
PetscErrorCode DMDestroy_Moab(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (((DM_Moab*)dm->data)->icreatedinstance) {
    delete ((DM_Moab*)dm->data)->mbiface;
    ((DM_Moab*)dm->data)->mbiface = NULL;
    ((DM_Moab*)dm->data)->pcomm = NULL;
    ((DM_Moab*)dm->data)->range.~Range();
  }
  ierr = PetscFree(dm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreate_Moab"
PETSC_EXTERN PetscErrorCode DMCreate_Moab(DM dm)
{
  DM_Moab        *moab;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscNewLog(dm,DM_Moab,&moab);CHKERRQ(ierr);
  dm->data = moab;
  new (moab) DM_Moab();

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

#undef __FUNCT__
#define __FUNCT__ "DMMoabCreateMoab"
/*@
  DMMoabCreate - Creates a DMMoab object, optionally from an instance and other data

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMMoab object
. moab - (ptr to) the MOAB Instance; if passed in NULL, MOAB instance is created inside PETSc, and destroyed
         along with the DMMoab
. pcomm - (ptr to) a ParallelComm; if NULL, creates one internally for the whole communicator
. ltog_tag - A tag to use to retrieve global id for an entity; if 0, will use GLOBAL_ID_TAG_NAME/tag
. range - If non-NULL, contains range of entities to which DOFs will be assigned

  Output Parameter:
. moab  - The DMMoab object

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabCreateMoab(MPI_Comm comm, moab::Interface *mbiface, moab::ParallelComm *pcomm, moab::Tag ltog_tag, moab::Range *range, DM *moab)
{
  PetscErrorCode ierr;
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidPointer(moab,2);
  ierr = DMMoabCreate(comm, moab);CHKERRQ(ierr);
  dmmoab = (DM_Moab*)(*moab)->data;

  if (!mbiface) {
    mbiface = new moab::Core();
    dmmoab->icreatedinstance = PETSC_TRUE;
  }
  else
    dmmoab->icreatedinstance = PETSC_FALSE;

  if (!pcomm) {
    PetscInt rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    pcomm = new moab::ParallelComm(mbiface, comm);
  }

    // do the initialization of the DM
  dmmoab->bs       = 0;
  dmmoab->pcomm    = pcomm;
  dmmoab->mbiface    = mbiface;
  dmmoab->ltog_tag = ltog_tag;

  ierr = DMMoabSetInterface(*moab, mbiface);CHKERRQ(ierr);
  if (!pcomm) pcomm = new moab::ParallelComm(mbiface, comm);
  ierr = DMMoabSetParallelComm(*moab, pcomm);CHKERRQ(ierr);
  if (!ltog_tag) {
    moab::ErrorCode merr = mbiface->tag_get_handle(GLOBAL_ID_TAG_NAME, ltog_tag);MBERRNM(merr);
  }
  if (ltog_tag) {
    ierr = DMMoabSetLocalToGlobalTag(*moab, ltog_tag);CHKERRQ(ierr);
  }
  if (range) {
    ierr = DMMoabSetRange(*moab, *range);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabSetParallelComm"
/*@
  DMMoabSetParallelComm - Set the ParallelComm used with this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm    - The DMMoab object being set
. pcomm - The ParallelComm being set on the DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabSetParallelComm(DM dm,moab::ParallelComm *pcomm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Moab*)dm->data)->pcomm = pcomm;
  ((DM_Moab*)dm->data)->mbiface = pcomm->get_moab();
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetParallelComm"
/*@
  DMMoabGetParallelComm - Get the ParallelComm used with this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm    - The DMMoab object being set

  Output Parameter:
. pcomm - The ParallelComm for the DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetParallelComm(DM dm,moab::ParallelComm **pcomm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *pcomm = ((DM_Moab*)dm->data)->pcomm;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetInterface"
/*@
  DMMoabSetInterface - Set the MOAB instance used with this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm      - The DMMoab object being set
. mbiface - The MOAB instance being set on this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabSetInterface(DM dm,moab::Interface *mbiface)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Moab*)dm->data)->pcomm = NULL;
  ((DM_Moab*)dm->data)->mbiface = mbiface;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetInterface"
/*@
  DMMoabGetInterface - Get the MOAB instance used with this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm      - The DMMoab object being set

  Output Parameter:
. mbiface - The MOAB instance set on this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetInterface(DM dm,moab::Interface **mbiface)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *mbiface = ((DM_Moab*)dm->data)->mbiface;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetRange"
/*@
  DMMoabSetRange - Set the entities having DOFs on this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm    - The DMMoab object being set
. range - The entities treated by this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabSetRange(DM dm,moab::Range range)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Moab*)dm->data)->range = range;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetRange"
/*@
  DMMoabGetRange - Get the entities having DOFs on this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm    - The DMMoab object being set

  Output Parameter:
. range - The entities treated by this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetRange(DM dm,moab::Range *range)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *range = ((DM_Moab*)dm->data)->range;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabSetLocalToGlobalTag"
/*@
  DMMoabSetLocalToGlobalTag - Set the tag used for local to global numbering

  Collective on MPI_Comm

  Input Parameter:
. dm      - The DMMoab object being set
. ltogtag - The MOAB tag used for local to global ids

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabSetLocalToGlobalTag(DM dm,moab::Tag ltogtag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Moab*)dm->data)->ltog_tag = ltogtag;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetLocalToGlobalTag"
/*@
  DMMoabGetLocalToGlobalTag - Get the tag used for local to global numbering

  Collective on MPI_Comm

  Input Parameter:
. dm      - The DMMoab object being set

  Output Parameter:
. ltogtag - The MOAB tag used for local to global ids

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetLocalToGlobalTag(DM dm,moab::Tag *ltog_tag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *ltog_tag = ((DM_Moab*)dm->data)->ltog_tag;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetBlockSize"
/*@
  DMMoabSetBlockSize - Set the block size used with this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm - The DMMoab object being set
. bs - The block size used with this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabSetBlockSize(DM dm,PetscInt bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Moab*)dm->data)->bs = bs;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetBlockSize"
/*@
  DMMoabGetBlockSize - Get the block size used with this DMMoab

  Collective on MPI_Comm

  Input Parameter:
. dm - The DMMoab object being set

  Output Parameter:
. bs - The block size used with this DMMoab

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetBlockSize(DM dm,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *bs = ((DM_Moab*)dm->data)->bs;
  PetscFunctionReturn(0);
}


// declare for use later but before they're defined
PetscErrorCode DMMoab_VecUserDestroy(void *user);
PetscErrorCode DMMoab_VecDuplicate(Vec x,Vec *y);
PetscErrorCode DMMoab_CreateTagName(const moab::ParallelComm *pcomm,std::string& tag_name);
PetscErrorCode DMMoab_CreateVector(moab::Interface *iface,moab::ParallelComm *pcomm,moab::Tag tag,PetscInt tag_size,moab::Tag ltog_tag,moab::Range range,PetscBool serial, PetscBool destroy_tag,Vec *vec);

#undef __FUNCT__
#define __FUNCT__ "DMMoabCreateVector"
/*@
  DMMoabCreateVector - Create a Vec from either an existing tag, or a specified tag size, and a range of entities

  Collective on MPI_Comm

  Input Parameter:
. dm          - The DMMoab object being set
. tag         - If non-zero, block size will be taken from the tag size
. tag_size    - If tag was zero, this parameter specifies the block size; unique tag name will be generated automatically
. range       - If non-empty, Vec corresponds to these entities, otherwise to the entities set on the DMMoab
. serial      - If true, this is a serial Vec, otherwise a parallel one
. destroy_tag - If true, MOAB tag is destroyed with Vec, otherwise it is left on MOAB

  Output Parameter:
. vec         - The created vector

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabCreateVector(DM dm,moab::Tag tag,PetscInt tag_size,moab::Range range,PetscBool serial, PetscBool destroy_tag,Vec *vec)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;

  DM_Moab *dmmoab = (DM_Moab*)dm->data;
  moab::ParallelComm *pcomm = dmmoab->pcomm;
  moab::Interface *mbiface = dmmoab->mbiface;
  moab::Tag ltog_tag = dmmoab->ltog_tag;

  if (!tag && !tag_size) {
    PetscFunctionReturn(PETSC_ERR_ARG_WRONG);
  }
  else {
    ierr = DMMoab_CreateVector(mbiface,pcomm,tag,tag_size,ltog_tag,range,serial,destroy_tag,vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoab_CreateVector"
PetscErrorCode DMMoab_CreateVector(moab::Interface *mbiface,moab::ParallelComm *pcomm,moab::Tag tag,PetscInt tag_size,moab::Tag ltog_tag,moab::Range range,PetscBool serial, PetscBool destroy_tag,Vec *vec)
{
  PetscErrorCode     ierr;
  moab::ErrorCode    merr;

  PetscFunctionBegin;

  if (!tag) {
    std::string tag_name;
    ierr = DMMoab_CreateTagName(pcomm,tag_name);CHKERRQ(ierr);

      // Create the default value for the tag (all zeros):
    std::vector<PetscScalar> default_value(tag_size, 0.0);

      // Create the tag:
    merr = mbiface->tag_get_handle(tag_name.c_str(),tag_size,moab::MB_TYPE_DOUBLE,tag,
                                   moab::MB_TAG_DENSE | moab::MB_TAG_CREAT,default_value.data());MBERRNM(merr);
  }
  else {

      // Make sure the tag data is of type "double":
    moab::DataType tag_type;
    merr = mbiface->tag_get_data_type(tag, tag_type);MBERRNM(merr);
    if(tag_type != moab::MB_TYPE_DOUBLE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Tag data type must be MB_TYPE_DOUBLE");
  }

    // Create the MOAB internal data object
  Vec_MOAB *vmoab;
  ierr = PetscMalloc(sizeof(Vec_MOAB),&vmoab);CHKERRQ(ierr);
  new (vmoab) Vec_MOAB();
  vmoab->tag = tag;
  vmoab->ltog_tag = ltog_tag;
  vmoab->mbiface = mbiface;
  vmoab->pcomm = pcomm;
  vmoab->tag_range = range;
  vmoab->new_tag = destroy_tag;
  vmoab->serial = serial;
  merr = mbiface->tag_get_length(tag,vmoab->tag_size);MBERR("tag_get_size", merr);

    // Call tag_iterate. This will cause MOAB to allocate memory for the
    // tag data if it hasn't already happened:
  int  count;
  void *void_ptr;
  merr = mbiface->tag_iterate(tag,range.begin(),range.end(),count,void_ptr);MBERRNM(merr);

    // Check to make sure the tag data is in a single sequence:
  if ((unsigned)count != range.size()) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Can only create MOAB Vector for single sequence");
  PetscScalar *data_ptr = (PetscScalar*)void_ptr;

    // Create the PETSc Vector:
  if(!serial) {
      // This is an MPI Vector:
    ierr = VecCreateMPIWithArray(vmoab->pcomm->comm(),vmoab->tag_size,vmoab->tag_size*range.size(),
                                 PETSC_DECIDE,data_ptr,vec);CHKERRQ(ierr);

      // Vector created, manually set local to global mapping:
    ISLocalToGlobalMapping ltog;
    PetscInt               *gindices = new PetscInt[range.size()];
    PetscInt               count = 0;
    moab::Range::iterator  iter;
    for(iter = range.begin(); iter != range.end(); iter++) {
      int dof;
      merr = mbiface->tag_get_data(ltog_tag,&(*iter),1,&dof);MBERRNM(merr);
      gindices[count] = dof;
      count++;
    }

    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,range.size(),gindices,
                                        PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMappingBlock(*vec,ltog);CHKERRQ(ierr);

      // Clean up:
    ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
    delete [] gindices;
  } else {
      // This is a serial vector:
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,vmoab->tag_size,vmoab->tag_size*range.size(),data_ptr,vec);CHKERRQ(ierr);
  }


  PetscContainer moabdata;
  ierr = PetscContainerCreate(PETSC_COMM_SELF,&moabdata);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(moabdata,vmoab);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(moabdata,DMMoab_VecUserDestroy);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*vec,"MOABData",(PetscObject)moabdata);CHKERRQ(ierr);
  (*vec)->ops->duplicate = DMMoab_VecDuplicate;

  ierr = PetscContainerDestroy(&moabdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabGetVecTag"
/*@
  DMMoabGetVecTag - Get the MOAB tag associated with this Vec

  Collective on MPI_Comm

  Input Parameter:
. vec - Vec being queried

  Output Parameter:
. tag - Tag associated with this Vec

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetVecTag(Vec vec,moab::Tag *tag)
{
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  // Get the MOAB private data:
  ierr = PetscObjectQuery((PetscObject)vec,"MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**) &vmoab);CHKERRQ(ierr);

  *tag = vmoab->tag;

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetVecRange"
/*@
  DMMoabGetVecRange - Get the MOAB entities associated with this Vec

  Collective on MPI_Comm

  Input Parameter:
. vec   - Vec being queried

  Output Parameter:
. range - Entities associated with this Vec

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetVecRange(Vec vec,moab::Range *range)
{
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  // Get the MOAB private data:
  ierr = PetscObjectQuery((PetscObject)vec,"MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**) &vmoab);CHKERRQ(ierr);

  *range = vmoab->tag_range;

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoab_VecDuplicate"
PetscErrorCode DMMoab_VecDuplicate(Vec x,Vec *y)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(y,2);

  // Get the Vec_MOAB struct for the original vector:
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab;
  ierr = PetscObjectQuery((PetscObject)x,"MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**)&vmoab);CHKERRQ(ierr);

  ierr = DMMoab_CreateVector(vmoab->mbiface,vmoab->pcomm,0,vmoab->tag_size,vmoab->ltog_tag,vmoab->tag_range,vmoab->serial,PETSC_TRUE,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoab_CreateTagName"
/*  DMMoab_CreateTagName
 *
 *  Creates a unique tag name that will be shared across processes. If
 *  pcomm is NULL, then this is a serial vector. A unique tag name
 *  will be returned in tag_name in either case.
 *
 *  The tag names have the format _PETSC_VEC_N where N is some integer.
 */
PetscErrorCode DMMoab_CreateTagName(const moab::ParallelComm *pcomm,std::string& tag_name)
{
  moab::ErrorCode mberr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  const std::string PVEC_PREFIX      = "_PETSC_VEC_";
  const PetscInt    PVEC_PREFIX_SIZE = PVEC_PREFIX.size();

  // Check to see if there are any PETSc vectors defined:
  const moab::Interface  *mbiface = pcomm->get_moab();
  std::vector<moab::Tag> tags;
  PetscInt               n = 0;
  mberr = mbiface->tag_get_tags(tags);MBERRNM(mberr);
  for(unsigned i = 0; i < tags.size(); i++) {
    std::string s;
    mberr = mbiface->tag_get_name(tags[i],s);MBERRNM(mberr);
    if(s.find(PVEC_PREFIX) != std::string::npos){
      // This tag represents a PETSc vector. Find the vector number:
      PetscInt m;
      std::istringstream(s.substr(PVEC_PREFIX_SIZE)) >> m;
      if(m >= n) n = m+1;
    }
  }

  // Make sure that n is consistent across all processes:
  PetscInt global_n;
  MPI_Comm comm = PETSC_COMM_SELF;
  if(pcomm) comm = pcomm->comm();
  ierr = MPI_Allreduce(&n,&global_n,1,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);

  // Set the answer and return:
  std::ostringstream ss;
  ss << PVEC_PREFIX << global_n;
  tag_name = ss.str();
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoab_VecUserDestroy"
PetscErrorCode DMMoab_VecUserDestroy(void *user)
{
  Vec_MOAB        *vmoab;
  PetscErrorCode  ierr;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  vmoab = (Vec_MOAB*)user;
  vmoab->tag_range.~Range();
  if(vmoab->new_tag) {
    // Tag created via a call to VecDuplicate, delete the underlying tag in MOAB...
    merr = vmoab->mbiface->tag_delete(vmoab->tag);MBERRNM(merr);
  }

  ierr = PetscFree(vmoab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

