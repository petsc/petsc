/*
      Some PETSc utilites
*/
#include "petscsys.h"             /*I    "petscsys.h"   I*/
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

/* ---------------------------------------------------------------- */
/*
   A simple way to manage tags inside a private 
   communicator.  It uses the attribute to determine if a new communicator
   is needed.

   Notes on the implementation

   The tagvalues to use are stored in a two element array.  The first element
   is the first free tag value.  The second is used to indicate how
   many "copies" of the communicator there are used in destroying.
*/

static PetscMPIInt Petsc_Tag_keyval = MPI_KEYVAL_INVALID;

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "Petsc_DelTag" 
/*
   Private routine to delete internal storage when a communicator is freed.
  This is called by MPI, not by users.

  The binding for the first argument changed from MPI 1.0 to 1.1; in 1.0
  it was MPI_Comm *comm.  

    Note: this is declared extern "C" because it is passed to the system routine signal()
          which is an extern "C" routine. 
*/
PetscMPIInt Petsc_DelTag(MPI_Comm comm,PetscMPIInt keyval,void* attr_val,void* extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogInfo((0,"Petsc_DelTag:Deleting tag data in an MPI_Comm %ld\n",(long)comm));CHKERRQ(ierr);
  ierr = PetscFree(attr_val);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectGetNewTag" 
/*@C
    PetscObjectGetNewTag - Gets a unique new tag from a PETSc object. All 
    processors that share the object MUST call this routine EXACTLY the same
    number of times.  This tag should only be used with the current objects
    communicator; do NOT use it with any other MPI communicator.

    Collective on PetscObject

    Input Parameter:
.   obj - the PETSc object; this must be cast with a (PetscObject), for example, 
         PetscObjectGetNewTag((PetscObject)mat,&tag);

    Output Parameter:
.   tag - the new tag

    Level: developer

    Concepts: tag^getting
    Concepts: message tag^getting
    Concepts: MPI message tag^getting

.seealso: PetscCommGetNewTag()
@*/
PetscErrorCode PetscObjectGetNewTag(PetscObject obj,PetscMPIInt *tag)
{
  PetscErrorCode ierr;
  PetscMPIInt    *tagvalp = 0,*maxval;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidIntPointer(tag,2);

  ierr = MPI_Attr_get(obj->comm,Petsc_Tag_keyval,(void**)&tagvalp,(PetscMPIInt*)&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Bad MPI communicator in PETSc object, likely memory corruption");

  if (tagvalp[0] < 1) {
    ierr = PetscLogInfo((0,"PetscObjectGetNewTag:Out of tags for object, starting to recycle. Number tags issued %d\n",tagvalp[1]));CHKERRQ(ierr);
    ierr       = MPI_Attr_get(MPI_COMM_WORLD,MPI_TAG_UB,(void**)&maxval,(PetscMPIInt*)&flg);CHKERRQ(ierr);
    if (!flg) {
      SETERRQ(PETSC_ERR_LIB,"MPI error: MPI_Attr_get() is not returning a MPI_TAG_UB");
    }
    tagvalp[0] = *maxval - 128; /* hope that any still active tags were issued right at the beginning of the run */
  }
  *tag = tagvalp[0]--;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscCommGetNewTag" 
/*@C
    PetscCommGetNewTag - Gets a unique new tag from a PETSc communicator. All 
    processors that share the communicator MUST call this routine EXACTLY the same
    number of times.  This tag should only be used with the current objects
    communicator; do NOT use it with any other MPI communicator.

    Collective on comm

    Input Parameter:
.   comm - the PETSc communicator

    Output Parameter:
.   tag - the new tag

    Level: developer

    Concepts: tag^getting
    Concepts: message tag^getting
    Concepts: MPI message tag^getting

.seealso: PetscObjectGetNewTag(), PetscCommDuplicate()
@*/
PetscErrorCode PetscCommGetNewTag(MPI_Comm comm,PetscMPIInt *tag)
{
  PetscErrorCode ierr;
  PetscMPIInt    *tagvalp=0,*maxval;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidIntPointer(tag,2);

  ierr = MPI_Attr_get(comm,Petsc_Tag_keyval,(void**)&tagvalp,(PetscMPIInt*)&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Bad MPI communicator supplied; must be a PETSc communicator");


  if (tagvalp[0] < 1) {
    ierr = PetscLogInfo((0,"PetscCommGetNewTag:Out of tags for object, starting to recycle. Number tags issued %d\n",tagvalp[1]));CHKERRQ(ierr);
    ierr       = MPI_Attr_get(MPI_COMM_WORLD,MPI_TAG_UB,(void**)&maxval,(PetscMPIInt*)&flg);CHKERRQ(ierr);
    if (!flg) {
      SETERRQ(PETSC_ERR_LIB,"MPI error: MPI_Attr_get() is not returning a MPI_TAG_UB");
    }
    tagvalp[0] = *maxval - 128; /* hope that any still active tags were issued right at the beginning of the run */
  }

  *tag = tagvalp[0]--;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscCommDuplicate" 
/*@C
  PetscCommDuplicate - Duplicates the communicator only if it is not already a PETSc 
                         communicator.

  Collective on MPI_Comm

  Input Parameters:
. comm_in - Input communicator

  Output Parameters:
+ comm_out - Output communicator.  May be comm_in.
- first_tag - Tag available that has not already been used with this communicator (you may
              pass in PETSC_NULL if you do not need a tag)

  PETSc communicators are just regular MPI communicators that keep track of which 
  tags have been used to prevent tag conflict. If you pass a non-PETSc communicator into
  a PETSc creation routine it will be duplicated for use in the object.

  Level: developer

  Concepts: communicator^duplicate

.seealso: PetscObjectGetNewTag(), PetscCommGetNewTag()
@*/
PetscErrorCode PetscCommDuplicate(MPI_Comm comm_in,MPI_Comm *comm_out,PetscMPIInt* first_tag)
{
  PetscErrorCode ierr;
  PetscMPIInt    *tagvalp,*maxval;
  PetscTruth     flg;

  PetscFunctionBegin;
  if (Petsc_Tag_keyval == MPI_KEYVAL_INVALID) {
    /* 
       The calling sequence of the 2nd argument to this function changed
       between MPI Standard 1.0 and the revisions 1.1 Here we match the 
       new standard, if you are using an MPI implementation that uses 
       the older version you will get a warning message about the next line;
       it is only a warning message and should do no harm.
    */
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelTag,&Petsc_Tag_keyval,(void*)0);CHKERRQ(ierr);
  }

  ierr = MPI_Attr_get(comm_in,Petsc_Tag_keyval,(void**)&tagvalp,(PetscMPIInt*)&flg);CHKERRQ(ierr);

  if (!flg) {
    /* This communicator is not yet known to this system, so we duplicate it and set its value */
    ierr       = MPI_Comm_dup(comm_in,comm_out);CHKERRQ(ierr);
    ierr       = MPI_Attr_get(MPI_COMM_WORLD,MPI_TAG_UB,(void**)&maxval,(PetscMPIInt*)&flg);CHKERRQ(ierr);
    if (!flg) {
      SETERRQ(PETSC_ERR_LIB,"MPI error: MPI_Attr_get() is not returning a MPI_TAG_UB");
    }
    ierr = PetscMalloc(2*sizeof(PetscMPIInt),&tagvalp);CHKERRQ(ierr);
    tagvalp[0] = *maxval;
    tagvalp[1] = 0;
    ierr       = MPI_Attr_put(*comm_out,Petsc_Tag_keyval,tagvalp);CHKERRQ(ierr);
    ierr = PetscLogInfo((0,"PetscCommDuplicate: Duplicating a communicator %ld %ld max tags = %d\n",(long)comm_in,(long)*comm_out,*maxval));CHKERRQ(ierr);
  } else {
#if defined(PETSC_USE_DEBUG)
    PetscMPIInt tag;
    ierr = MPI_Allreduce(tagvalp,&tag,1,MPI_INT,MPI_BOR,comm_in);CHKERRQ(ierr);
    if (tag != tagvalp[0]) {
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Communicator was used on subset of processors.");
    }
#endif
    *comm_out = comm_in;
  }

  if (tagvalp[0] < 1) {
    ierr = PetscLogInfo((0,"PetscCommDuplicate:Out of tags for object, starting to recycle. Number tags issued %d\n",tagvalp[1]));CHKERRQ(ierr);
    ierr       = MPI_Attr_get(MPI_COMM_WORLD,MPI_TAG_UB,(void**)&maxval,(PetscMPIInt*)&flg);CHKERRQ(ierr);
    if (!flg) {
      SETERRQ(PETSC_ERR_LIB,"MPI error: MPI_Attr_get() is not returning a MPI_TAG_UB");
    }
    tagvalp[0] = *maxval - 128; /* hope that any still active tags were issued right at the beginning of the run */
  }

  if (first_tag) {
    *first_tag = tagvalp[0]--;
    tagvalp[1]++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscCommDestroy" 
/*@C
   PetscCommDestroy - Frees communicator.  Use in conjunction with PetscCommDuplicate().

   Collective on MPI_Comm

   Input Parameter:
.  comm - the communicator to free

   Level: developer

   Concepts: communicator^destroy

@*/
PetscErrorCode PetscCommDestroy(MPI_Comm *comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    *tagvalp;
  PetscTruth     flg;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(*comm,Petsc_Tag_keyval,(void**)&tagvalp,(PetscMPIInt*)&flg);CHKERRQ(ierr);
  if (!flg) {
    SETERRQ(PETSC_ERR_ARG_CORRUPT,"Error freeing MPI_Comm, problem with corrupted memory");
  }
  tagvalp[1]--;
  if (!tagvalp[1]) {
    ierr = PetscLogInfo((0,"PetscCommDestroy:Deleting MPI_Comm %ld\n",(long)*comm));CHKERRQ(ierr);
    ierr = MPI_Comm_free(comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

