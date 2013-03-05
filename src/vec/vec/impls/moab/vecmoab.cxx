#include "../src/vec/vec/impls/moab/vecmoabimpl.h" /*I  "petscvec.h"   I*/

#include <MBParallelConventions.h>

#include <sstream>

/* The MBERR macro is used to save typing. It checks a MOAB error code
 * (rval) and calls SETERRQ if not MB_SUCCESS. A message (msg) can
 * also be passed in. */
#define MBERR(msg,rval) do{if(rval != moab::MB_SUCCESS) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"MOAB ERROR (%i): %s",rval,msg);} while(0)
#define MBERRNM(rval) do{if(rval != moab::MB_SUCCESS) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"MOAB ERROR (%i)",rval);} while(0)

#undef __FUNCT__
#define __FUNCT__ "VecMoabCreateFromTag"
PetscErrorCode VecMoabCreateFromTag(moab::Interface *mbint, moab::ParallelComm *pcomm, moab::Tag tag,moab::Tag ltog_tag,moab::Range range,PetscBool serial, PetscBool destroy_tag,Vec *X)
{
  PetscErrorCode     ierr;
  moab::ErrorCode    merr;

  PetscFunctionBegin;

  if(!tag) {
    MBERRNM(moab::MB_FAILURE);;
  }

    // Make sure the tag data is of type "double":
  moab::DataType tag_type;
  merr = mbint->tag_get_data_type(tag, tag_type);MBERRNM(merr);
  if(tag_type != moab::MB_TYPE_DOUBLE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Tag data type must be MB_TYPE_DOUBLE");

  // Create the MOAB internal data object
  Vec_MOAB *vmoab;
  ierr = PetscMalloc(sizeof(Vec_MOAB),&vmoab);CHKERRQ(ierr);
  new (vmoab) Vec_MOAB();
  vmoab->tag = tag;
  vmoab->ltog_tag = ltog_tag;
  vmoab->mbint = mbint;
  vmoab->pcomm = pcomm;
  vmoab->tag_range = range;
  vmoab->new_tag = destroy_tag;
  vmoab->serial = serial;
  merr = mbint->tag_get_length(tag,vmoab->tag_size);MBERR("tag_get_size", merr);

  // Call tag_iterate. This will cause MOAB to allocate memory for the
  // tag data if it hasn't already happened:
  int  count;
  void *void_ptr;
  merr = mbint->tag_iterate(tag,range.begin(),range.end(),count,void_ptr);MBERRNM(merr);

  // Check to make sure the tag data is in a single sequence:
  if ((unsigned)count != range.size()) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Can only create MOAB Vector for single sequence");
  PetscScalar *data_ptr = (PetscScalar*)void_ptr;

  // Create the PETSc Vector:
  if(!serial) {
    // This is an MPI Vector:
    ierr = VecCreateMPIWithArray(vmoab->pcomm->comm(),vmoab->tag_size,vmoab->tag_size*range.size(),
				 PETSC_DECIDE,data_ptr,X);CHKERRXX(ierr);

    // Vector created, manually set local to global mapping:
    ISLocalToGlobalMapping ltog;
    PetscInt               *gindices = new PetscInt[range.size()];
    PetscInt               count = 0;
    moab::Range::iterator  iter;
    for(iter = range.begin(); iter != range.end(); iter++) {
      int dof;
      merr = mbint->tag_get_data(ltog_tag,&(*iter),1,&dof);MBERRNM(merr);
      gindices[count] = dof;
      count++;
    }

    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,range.size(),gindices,
					PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMappingBlock(*X,ltog);CHKERRQ(ierr);

    // Clean up:
    ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
    delete [] gindices;
  } else {
    // This is a serial vector:
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,vmoab->tag_size,vmoab->tag_size*range.size(),data_ptr,X);CHKERRXX(ierr);
  }


  PetscContainer moabdata;
  ierr = PetscContainerCreate(PETSC_COMM_SELF,&moabdata);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(moabdata,vmoab);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(moabdata,VecMoabDestroy_Private);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*X,"MOABData",(PetscObject)moabdata);CHKERRQ(ierr);
  (*X)->ops->duplicate = VecMoab_Duplicate;

  ierr = PetscContainerDestroy(&moabdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMoabCreate"
PetscErrorCode VecMoabCreate(moab::Interface *mbint,moab::ParallelComm *pcomm,PetscInt tag_size,moab::Tag ltog_tag,moab::Range range,PetscBool serial,PetscBool destroy_tag,Vec *vec)
{
  moab::ErrorCode    merr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;

  // Create the tag name;
  std::string tag_name;
  ierr = VecMoabGetTagName_Private(pcomm,tag_name);CHKERRQ(ierr);

  // Create the default value for the tag (all zeros):
  std::vector<PetscScalar> default_value(tag_size, 0.0);

  // Create the tag:
  moab::Tag tag;
  merr = mbint->tag_get_handle(tag_name.c_str(),tag_size,moab::MB_TYPE_DOUBLE,tag,
			       moab::MB_TAG_DENSE | moab::MB_TAG_CREAT,default_value.data());MBERRNM(merr);

  return VecMoabCreateFromTag(mbint, pcomm, tag, ltog_tag, range, serial, destroy_tag, vec);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecMoab_Duplicate"
PetscErrorCode VecMoab_Duplicate(Vec x,Vec *y)
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
  PetscInt tag_size;
  ierr = VecGetBlockSize(x,&tag_size);

  ierr = VecMoabCreate(vmoab->mbint,vmoab->pcomm,tag_size,vmoab->ltog_tag,vmoab->tag_range,vmoab->serial,PETSC_TRUE,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@C
  Get the underlying Tag from a MOAB vector

   Input Parameter:
.  X - A MOAB vector

   Output Parameter:
.  tag_handle - the tag handle

@*/
#undef __FUNCT__
#define __FUNCT__ "VecMoabGetTag"
PetscErrorCode VecMoabGetTag(Vec X, moab::Tag *tag_handle)
{
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  // Get the MOAB private data:
  ierr = PetscObjectQuery((PetscObject)X,"MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**) &vmoab);CHKERRQ(ierr);

  *tag_handle = vmoab->tag;

  PetscFunctionReturn(0);
}

/*@C
  Get the underlying MOAB Range from a MOAB vector

   Input Parameter:
.  X - A MOAB vector

   Output Parameter:
.  range - the MOAB range (entities) to which this vec applies

@*/
#undef __FUNCT__
#define __FUNCT__ "VecMoabGetRange"
PetscErrorCode VecMoabGetRange(Vec X, moab::Range *range)
{
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  // Get the MOAB private data:
  ierr = PetscObjectQuery((PetscObject)X,"MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**) &vmoab);CHKERRQ(ierr);

  *range = vmoab->tag_range;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMoabGetTagName_Private"
/*  VecMoabGetTagName_Private
 *
 *  Creates a unique tag name that will be shared across processes. If
 *  pcomm is NULL, then this is a serial vector. A unique tag name
 *  will be returned in tag_name in either case.
 *
 *  The tag names have the format _PETSC_VEC_N where N is some integer.
 */
PetscErrorCode VecMoabGetTagName_Private(const moab::ParallelComm *pcomm,std::string& tag_name)
{
  moab::ErrorCode mberr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  const std::string PVEC_PREFIX      = "_PETSC_VEC_";
  const PetscInt    PVEC_PREFIX_SIZE = PVEC_PREFIX.size();

  // Check to see if there are any PETSc vectors defined:
  const moab::Interface  *mbint = pcomm->get_moab();
  std::vector<moab::Tag> tags;
  PetscInt               n = 0;
  mberr = mbint->tag_get_tags(tags);MBERRNM(mberr);
  for(unsigned i = 0; i < tags.size(); i++) {
    std::string s;
    mberr = mbint->tag_get_name(tags[i],s);MBERRNM(mberr);
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
  std::stringstream ss;
  ss << PVEC_PREFIX << global_n;
  tag_name = ss.str();
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecMoabDestroy_Private"
PetscErrorCode VecMoabDestroy_Private(void *ctx)
{
  Vec_MOAB        *vmoab = (Vec_MOAB*) ctx;
  PetscErrorCode  ierr;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  if(vmoab->new_tag == PETSC_TRUE) {
    // Tag created via a call to VecDuplicate, delete the underlying tag in MOAB...
    merr = vmoab->mbint->tag_delete(vmoab->tag);MBERRNM(merr);
  }

  ierr = PetscFree(vmoab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

