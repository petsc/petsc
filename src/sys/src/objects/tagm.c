/*$Id: tagm.c,v 1.20 2000/04/09 03:09:13 bsmith Exp bsmith $*/
/*
      Some PETSc utilites
*/
#include "sys.h"             /*I    "sys.h"   I*/
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

static int Petsc_Tag_keyval = MPI_KEYVAL_INVALID;

EXTERN_C_BEGIN
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"Petsc_DelTag" 
/*
   Private routine to delete internal storage when a communicator is freed.
  This is called by MPI, not by users.

  The binding for the first argument changed from MPI 1.0 to 1.1; in 1.0
  it was MPI_Comm *comm.  

    Note: this is declared extern "C" because it is passed to the system routine signal()
          which is an extern "C" routine. The Solaris 2.7 OS compilers require that this be
          extern "C".
*/
int Petsc_DelTag(MPI_Comm comm,int keyval,void* attr_val,void* extra_state)
{
  int ierr;

  PetscFunctionBegin;
  PLogInfo(0,"Petsc_DelTag:Deleting tag data in an MPI_Comm %d\n",(int)comm);
  ierr = PetscFree(attr_val);CHKERRQ(ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}
EXTERN_C_END

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscObjectGetNewTag" 
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

.keywords: object, get, new, tag

.seealso: PetscObjectRestoreNewTag(), PetscCommGetNewTag(), PetscObjectRestoreNewTag()
@*/
int PetscObjectGetNewTag(PetscObject obj,int *tag)
{
  int        ierr,*tagvalp=0;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeader(obj);
  PetscValidIntPointer(tag);

  ierr = MPI_Attr_get(obj->comm,Petsc_Tag_keyval,(void**)&tagvalp,(int*)&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Bad MPI communicator in PETSc object, likely memory corruption");

  if (*tagvalp < 1) SETERRQ(PETSC_ERR_PLIB,0,"Out of tags for object");
  *tag = tagvalp[0]--;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscObjectRestoreNewTag" 
/*@C
    PetscObjectRestoreNewTag - Restores a new tag from a PETSc object. All 
    processors that share the object MUST call this routine EXACTLY the same
    number of times. 

    Collective on PetscObject

    Input Parameter:
.   obj - the PETSc object; this must be cast with a (PetscObject), for example, 
          PetscObjectRestoreNewTag((PetscObject)mat,&tag);

    Output Parameter:
.   tag - the new tag

    Level: developer

.keywords: object, restore, new, tag

.seealso: PetscObjectGetNewTag()
@*/
int PetscObjectRestoreNewTag(PetscObject obj,int *tag)
{
  int        ierr,*tagvalp=0;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeader(obj);
  PetscValidIntPointer(tag);

  ierr = MPI_Attr_get(obj->comm,Petsc_Tag_keyval,(void**)&tagvalp,(int*)&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Bad MPI communicator in PETSc object; likely memory corruption");

  if (*tagvalp == *tag - 1) {
    tagvalp[0]++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscCommGetNewTag" 
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

.keywords: comm, get, new, tag

.seealso: PetscCommRestoreNewTag(),PetscObjectGetNewTag(),PetscObjectRestoreNewTag()
@*/
int PetscCommGetNewTag(MPI_Comm comm,int *tag)
{
  int        ierr,*tagvalp=0;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidIntPointer(tag);

  ierr = MPI_Attr_get(comm,Petsc_Tag_keyval,(void**)&tagvalp,(int*)&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Bad MPI communicator supplied; must be a PETSc communicator");

  if (*tagvalp < 1) SETERRQ(PETSC_ERR_PLIB,0,"Out of tags for communicator");
  *tag = tagvalp[0]--;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscCommRestoreNewTag" 
/*@C
    PetscCommRestoreNewTag - Restores a new tag from a PETSc comm. All 
    processors that share the communicator MUST call this routine EXACTLY 
    the same number of times. 

    Collective on MPI_Comm

    Input Parameters:
+   comm - the PETSc communicator
-   tag - the new tag

    Level: developer

.keywords: comm, restore, new, tag

.seealso:  PetscCommGetNewTag(),PetscObjectGetNewTag(),PetscObjectRestoreNewTag()
@*/
int PetscCommRestoreNewTag(MPI_Comm comm,int *tag)
{
  int        ierr,*tagvalp=0;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidIntPointer(tag);

  ierr = MPI_Attr_get(comm,Petsc_Tag_keyval,(void**)&tagvalp,(int*)&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Bad communicator supplied; must be a PETSc communicator");

  if (*tagvalp == *tag - 1) {
    tagvalp[0]++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscCommDuplicate_Private" 
/*
  PetscCommDuplicate_Private - Duplicates the communicator only if it is not already a PETSc 
                         communicator.

  Input Parameters:
. comm_in - Input communicator

  Output Parameters:
+ comm_out - Output communicator.  May be comm_in.
- first_tag - First tag available

  Notes:
  This routine returns one tag number.

*/
int PetscCommDuplicate_Private(MPI_Comm comm_in,MPI_Comm *comm_out,int* first_tag)
{
  int        ierr = MPI_SUCCESS,*tagvalp,*maxval;
  PetscTruth flg;

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

  ierr = MPI_Attr_get(comm_in,Petsc_Tag_keyval,(void**)&tagvalp,(int*)&flg);CHKERRQ(ierr);

  if (!flg) {
    /* This communicator is not yet known to this system, so we duplicate it and set its value */
    ierr       = MPI_Comm_dup(comm_in,comm_out);CHKERRQ(ierr);
    ierr       = MPI_Attr_get(MPI_COMM_WORLD,MPI_TAG_UB,(void**)&maxval,(int*)&flg);CHKERRQ(ierr);
    if (!flg) {
      SETERRQ(1,1,"MPI error: MPI_Attr_get() is not returning a MPI_TAG_UB");
    }
    tagvalp    = (int*)PetscMalloc(2*sizeof(int));CHKPTRQ(tagvalp);
    tagvalp[0] = *maxval;
    tagvalp[1] = 0;
    ierr       = MPI_Attr_put(*comm_out,Petsc_Tag_keyval,tagvalp);CHKERRQ(ierr);
    PLogInfo(0,"PetscCommDuplicate_Private: Duplicating a communicator %d %d max tags = %d\n",(int)comm_in,(int)*comm_out,*maxval);
  } else {
#if defined(PETSC_USE_BOPT_g)
    int tag;
    ierr = MPI_Allreduce(tagvalp,&tag,1,MPI_INT,MPI_BOR,comm_in);CHKERRQ(ierr);
    if (tag != tagvalp[0]) {
      SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Communicator was used on subset of processors.");
    }
#endif
    *comm_out = comm_in;
  }

  if (tagvalp[0] < 1) {
    PLogInfo(0,"Out of tags for object, starting to recycle. Number tags issued %d",tagvalp[1]);
    ierr       = MPI_Attr_get(MPI_COMM_WORLD,MPI_TAG_UB,(void**)&maxval,(int*)&flg);CHKERRQ(ierr);
    if (!flg) {
      SETERRQ(1,1,"MPI error: MPI_Attr_get() is not returning a MPI_TAG_UB");
    }
    tagvalp[0] = *maxval - 128; /* hope that any still active tags were issued right at the beginning of the run */
  }

  *first_tag = tagvalp[0]--;
  tagvalp[1]++;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscCommDestroy_Private" 
/*
  PetscCommDestroy_Private - Frees communicator.  Use in conjunction with PetscCommDuplicate_Private().
*/
int PetscCommDestroy_Private(MPI_Comm *comm)
{
  int        ierr,*tagvalp;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(*comm,Petsc_Tag_keyval,(void**)&tagvalp,(int*)&flg);CHKERRQ(ierr);
  if (!flg) {
    SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Error freeing MPI_Comm, problem with corrupted memory");
  }
  tagvalp[1]--;
  if (!tagvalp[1]) {
    PLogInfo(0,"PetscCommDestroy_Private:Deleting MPI_Comm %d\n",(int)*comm);
    ierr = MPI_Comm_free(comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}




