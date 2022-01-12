#include <petsc/private/petscimpl.h>             /*I    "petscsys.h"   I*/
/* ---------------------------------------------------------------- */
/*
   A simple way to manage tags inside a communicator.

   It uses the attributes to determine if a new communicator
      is needed and to store the available tags.

*/

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

.seealso: PetscCommGetNewTag()
@*/
PetscErrorCode  PetscObjectGetNewTag(PetscObject obj,PetscMPIInt *tag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCommGetNewTag(obj->comm,tag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PetscCommGetNewTag - Gets a unique new tag from a PETSc communicator. All
    processors that share the communicator MUST call this routine EXACTLY the same
    number of times.  This tag should only be used with the current objects
    communicator; do NOT use it with any other MPI communicator.

    Collective

    Input Parameter:
.   comm - the MPI communicator

    Output Parameter:
.   tag - the new tag

    Level: developer

.seealso: PetscObjectGetNewTag(), PetscCommDuplicate()
@*/
PetscErrorCode  PetscCommGetNewTag(MPI_Comm comm,PetscMPIInt *tag)
{
  PetscErrorCode   ierr;
  PetscCommCounter *counter;
  PetscMPIInt      *maxval,flg;

  PetscFunctionBegin;
  PetscValidIntPointer(tag,2);

  ierr = MPI_Comm_get_attr(comm,Petsc_Counter_keyval,&counter,&flg);CHKERRMPI(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Bad MPI communicator supplied; must be a PETSc communicator");

  if (counter->tag < 1) {

    ierr = PetscInfo1(NULL,"Out of tags for object, starting to recycle. Comm reference count %" PetscInt_FMT "\n",counter->refcount);CHKERRQ(ierr);
    ierr = MPI_Comm_get_attr(MPI_COMM_WORLD,MPI_TAG_UB,&maxval,&flg);CHKERRMPI(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"MPI error: MPI_Comm_get_attr() is not returning a MPI_TAG_UB");
    counter->tag = *maxval - 128; /* hope that any still active tags were issued right at the beginning of the run */
  }

  *tag = counter->tag--;
  if (PetscDefined(USE_DEBUG)) {
    /*
     Hanging here means that some processes have called PetscCommGetNewTag() and others have not.
     */
    ierr = MPI_Barrier(comm);CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscCommGetComm - get an MPI communicator from a PETSc communicator that can be passed off to another package

  Collective

  Input Parameter:
. comm_in - Input communicator

  Output Parameters:
. comm_out - Output communicator

  Notes:
    Use PetscCommRestoreComm() to return the communicator when the external package no longer needs it

    Certain MPI implementations have MPI_Comm_free() that do not work, thus one can run out of available MPI communicators causing
    mysterious crashes in the code after running a long time. This routine allows reusing previously obtained MPI communicators that
    are no longer needed.

Level: developer

.seealso: PetscObjectGetNewTag(), PetscCommGetNewTag(), PetscCommDestroy(), PetscCommRestoreComm()
@*/
PetscErrorCode  PetscCommGetComm(MPI_Comm comm_in,MPI_Comm *comm_out)
{
  PetscErrorCode   ierr;
  PetscCommCounter *counter;
  PetscMPIInt      flg;

  PetscFunctionBegin;
  ierr = PetscSpinlockLock(&PetscCommSpinLock);CHKERRQ(ierr);
  ierr = MPI_Comm_get_attr(comm_in,Petsc_Counter_keyval,&counter,&flg);CHKERRMPI(ierr);
  if (!flg) SETERRQ(comm_in,PETSC_ERR_ARG_WRONGSTATE,"Requires a PETSc communicator as input, do not use something like MPI_COMM_WORLD");

  if (counter->comms) {
    struct PetscCommStash *pcomms = counter->comms;

    *comm_out = pcomms->comm;
    counter->comms = pcomms->next;
    ierr = PetscFree(pcomms);CHKERRQ(ierr);
    ierr = PetscInfo2(NULL,"Reusing a communicator %ld %ld\n",(long)comm_in,(long)*comm_out);CHKERRQ(ierr);
  } else {
    ierr = MPI_Comm_dup(comm_in,comm_out);CHKERRMPI(ierr);
  }
  ierr = PetscSpinlockUnlock(&PetscCommSpinLock);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscCommRestoreComm - restores an MPI communicator that was obtained with PetscCommGetComm()

  Collective

  Input Parameters:
+  comm_in - Input communicator
-  comm_out - returned communicator

Level: developer

.seealso: PetscObjectGetNewTag(), PetscCommGetNewTag(), PetscCommDestroy(), PetscCommRestoreComm()
@*/
PetscErrorCode PetscCommRestoreComm(MPI_Comm comm_in,MPI_Comm *comm_out)
{
  PetscErrorCode        ierr;
  PetscCommCounter      *counter;
  PetscMPIInt           flg;
  struct PetscCommStash *pcomms,*ncomm;

  PetscFunctionBegin;
  ierr = PetscSpinlockLock(&PetscCommSpinLock);CHKERRQ(ierr);
  ierr = MPI_Comm_get_attr(comm_in,Petsc_Counter_keyval,&counter,&flg);CHKERRMPI(ierr);
  if (!flg) SETERRQ(comm_in,PETSC_ERR_ARG_WRONGSTATE,"Requires a PETSc communicator as input, do not use something like MPI_COMM_WORLD");

  ierr = PetscMalloc(sizeof(struct PetscCommStash),&ncomm);CHKERRQ(ierr);
  ncomm->comm = *comm_out;
  ncomm->next = NULL;
  pcomms = counter->comms;
  while (pcomms && pcomms->next) pcomms = pcomms->next;
  if (pcomms) {
    pcomms->next   = ncomm;
  } else {
    counter->comms = ncomm;
  }
  *comm_out = 0;
  ierr = PetscSpinlockUnlock(&PetscCommSpinLock);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscCommDuplicate - Duplicates the communicator only if it is not already a PETSc communicator.

  Collective

  Input Parameter:
  . comm_in - Input communicator

  Output Parameters:
  + comm_out - Output communicator.  May be comm_in.
  - first_tag - Tag available that has not already been used with this communicator (you may
  pass in NULL if you do not need a tag)

  PETSc communicators are just regular MPI communicators that keep track of which
  tags have been used to prevent tag conflict. If you pass a non-PETSc communicator into
  a PETSc creation routine it will attach a private communicator for use in the objects communications.
  The internal MPI_Comm is used to perform all the MPI calls for PETSc, the outer MPI_Comm is a user
  level MPI_Comm that may be performing communication for the user or other library and so IS NOT used by PETSc.

Level: developer

.seealso: PetscObjectGetNewTag(), PetscCommGetNewTag(), PetscCommDestroy()
@*/
PetscErrorCode  PetscCommDuplicate(MPI_Comm comm_in,MPI_Comm *comm_out,PetscMPIInt *first_tag)
{
  PetscErrorCode   ierr;
  PetscCommCounter *counter;
  PetscMPIInt      *maxval,flg;

  PetscFunctionBegin;
  ierr = PetscSpinlockLock(&PetscCommSpinLock);CHKERRQ(ierr);
  ierr = MPI_Comm_get_attr(comm_in,Petsc_Counter_keyval,&counter,&flg);CHKERRMPI(ierr);

  if (!flg) {  /* this is NOT a PETSc comm */
    union {MPI_Comm comm; void *ptr;} ucomm;
    /* check if this communicator has a PETSc communicator imbedded in it */
    ierr = MPI_Comm_get_attr(comm_in,Petsc_InnerComm_keyval,&ucomm,&flg);CHKERRMPI(ierr);
    if (!flg) {
      /* This communicator is not yet known to this system, so we duplicate it and make an internal communicator */
      ierr = MPI_Comm_dup(comm_in,comm_out);CHKERRMPI(ierr);
      ierr = MPI_Comm_get_attr(MPI_COMM_WORLD,MPI_TAG_UB,&maxval,&flg);CHKERRMPI(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"MPI error: MPI_Comm_get_attr() is not returning a MPI_TAG_UB");
      ierr = PetscNew(&counter);CHKERRQ(ierr); /* all fields of counter are zero'ed */
      counter->tag = *maxval;
      ierr = MPI_Comm_set_attr(*comm_out,Petsc_Counter_keyval,counter);CHKERRMPI(ierr);
      ierr = PetscInfo3(NULL,"Duplicating a communicator %ld %ld max tags = %d\n",(long)comm_in,(long)*comm_out,*maxval);CHKERRQ(ierr);

      /* save PETSc communicator inside user communicator, so we can get it next time */
      ucomm.comm = *comm_out;   /* ONLY the comm part of the union is significant. */
      ierr = MPI_Comm_set_attr(comm_in,Petsc_InnerComm_keyval,ucomm.ptr);CHKERRMPI(ierr);
      ucomm.comm = comm_in;
      ierr = MPI_Comm_set_attr(*comm_out,Petsc_OuterComm_keyval,ucomm.ptr);CHKERRMPI(ierr);
    } else {
      *comm_out = ucomm.comm;
      /* pull out the inner MPI_Comm and hand it back to the caller */
      ierr = MPI_Comm_get_attr(*comm_out,Petsc_Counter_keyval,&counter,&flg);CHKERRMPI(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inner PETSc communicator does not have its tag/name counter attribute set");
      ierr = PetscInfo2(NULL,"Using internal PETSc communicator %ld %ld\n",(long)comm_in,(long)*comm_out);CHKERRQ(ierr);
    }
  } else *comm_out = comm_in;

  if (PetscDefined(USE_DEBUG)) {
    /*
     Hanging here means that some processes have called PetscCommDuplicate() and others have not.
     This likley means that a subset of processes in a MPI_Comm have attempted to create a PetscObject!
     ALL processes that share a communicator MUST shared objects created from that communicator.
     */
    ierr = MPI_Barrier(comm_in);CHKERRMPI(ierr);
  }

  if (counter->tag < 1) {
    ierr = PetscInfo1(NULL,"Out of tags for object, starting to recycle. Comm reference count %" PetscInt_FMT "\n",counter->refcount);CHKERRQ(ierr);
    ierr = MPI_Comm_get_attr(MPI_COMM_WORLD,MPI_TAG_UB,&maxval,&flg);CHKERRMPI(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"MPI error: MPI_Comm_get_attr() is not returning a MPI_TAG_UB");
    counter->tag = *maxval - 128; /* hope that any still active tags were issued right at the beginning of the run */
  }

  if (first_tag) *first_tag = counter->tag--;

  counter->refcount++; /* number of references to this comm */
  ierr = PetscSpinlockUnlock(&PetscCommSpinLock);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscCommDestroy - Frees communicator.  Use in conjunction with PetscCommDuplicate().

   Collective

   Input Parameter:
.  comm - the communicator to free

   Level: developer

.seealso:   PetscCommDuplicate()
@*/
PetscErrorCode  PetscCommDestroy(MPI_Comm *comm)
{
  PetscErrorCode   ierr;
  PetscCommCounter *counter;
  PetscMPIInt      flg;
  MPI_Comm         icomm = *comm,ocomm;
  union {MPI_Comm comm; void *ptr;} ucomm;

  PetscFunctionBegin;
  if (*comm == MPI_COMM_NULL) PetscFunctionReturn(0);
  ierr = PetscSpinlockLock(&PetscCommSpinLock);CHKERRQ(ierr);
  ierr = MPI_Comm_get_attr(icomm,Petsc_Counter_keyval,&counter,&flg);CHKERRMPI(ierr);
  if (!flg) { /* not a PETSc comm, check if it has an inner comm */
    ierr = MPI_Comm_get_attr(icomm,Petsc_InnerComm_keyval,&ucomm,&flg);CHKERRMPI(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"MPI_Comm does not have tag/name counter nor does it have inner MPI_Comm");
    icomm = ucomm.comm;
    ierr = MPI_Comm_get_attr(icomm,Petsc_Counter_keyval,&counter,&flg);CHKERRMPI(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Inner MPI_Comm does not have expected tag/name counter, problem with corrupted memory");
  }

  counter->refcount--;

  if (!counter->refcount) {
    /* if MPI_Comm has outer comm then remove reference to inner MPI_Comm from outer MPI_Comm */
    ierr = MPI_Comm_get_attr(icomm,Petsc_OuterComm_keyval,&ucomm,&flg);CHKERRMPI(ierr);
    if (flg) {
      ocomm = ucomm.comm;
      ierr = MPI_Comm_get_attr(ocomm,Petsc_InnerComm_keyval,&ucomm,&flg);CHKERRMPI(ierr);
      if (flg) {
        ierr = MPI_Comm_delete_attr(ocomm,Petsc_InnerComm_keyval);CHKERRMPI(ierr);
      } else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Outer MPI_Comm %ld does not have expected reference to inner comm %ld, problem with corrupted memory",(long int)ocomm,(long int)icomm);
    }

    ierr = PetscInfo1(NULL,"Deleting PETSc MPI_Comm %ld\n",(long)icomm);CHKERRQ(ierr);
    ierr = MPI_Comm_free(&icomm);CHKERRMPI(ierr);
  }
  *comm = MPI_COMM_NULL;
  ierr = PetscSpinlockUnlock(&PetscCommSpinLock);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    PetscObjectsListGetGlobalNumbering - computes a global numbering
    of PetscObjects living on subcommunicators of a given communicator.

    Collective.

    Input Parameters:
+   comm    - MPI_Comm
.   len     - local length of objlist
-   objlist - a list of PETSc objects living on subcomms of comm and containing this comm rank
              (subcomm ordering is assumed to be deadlock-free)

    Output Parameters:
+   count      - global number of distinct subcommunicators on objlist (may be > len)
-   numbering  - global numbers of objlist entries (allocated by user)

    Level: developer

@*/
PetscErrorCode  PetscObjectsListGetGlobalNumbering(MPI_Comm comm, PetscInt len, PetscObject *objlist, PetscInt *count, PetscInt *numbering)
{
  PetscErrorCode ierr;
  PetscInt       i, roots, offset;
  PetscMPIInt    size, rank;

  PetscFunctionBegin;
  PetscValidPointer(objlist,3);
  if (!count && !numbering) PetscFunctionReturn(0);

  ierr  = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr  = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  roots = 0;
  for (i = 0; i < len; ++i) {
    PetscMPIInt srank;
    ierr = MPI_Comm_rank(objlist[i]->comm, &srank);CHKERRMPI(ierr);
    /* Am I the root of the i-th subcomm? */
    if (!srank) ++roots;
  }
  if (count) {
    /* Obtain the sum of all roots -- the global number of distinct subcomms. */
    ierr = MPIU_Allreduce(&roots,count,1,MPIU_INT,MPI_SUM,comm);CHKERRMPI(ierr);
  }
  if (numbering) {
    /* Introduce a global numbering for subcomms, initially known only by subcomm roots. */
    /*
      At each subcomm root number all of the subcomms it owns locally
      and make it global by calculating the shift among all of the roots.
      The roots are ordered using the comm ordering.
    */
    ierr    = MPI_Scan(&roots,&offset,1,MPIU_INT,MPI_SUM,comm);CHKERRMPI(ierr);
    offset -= roots;
    /* Now we are ready to broadcast global subcomm numbers within each subcomm.*/
    /*
      This is where the assumption of a deadlock-free ordering of the subcomms is assumed:
      broadcast is collective on the subcomm.
    */
    roots = 0;
    for (i = 0; i < len; ++i) {
      PetscMPIInt srank;
      numbering[i] = offset + roots; /* only meaningful if !srank. */

      ierr = MPI_Comm_rank(objlist[i]->comm, &srank);CHKERRMPI(ierr);
      ierr = MPI_Bcast(numbering+i,1,MPIU_INT,0,objlist[i]->comm);CHKERRMPI(ierr);
      if (!srank) ++roots;
    }
  }
  PetscFunctionReturn(0);
}

