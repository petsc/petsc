#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mpiu.c,v 1.79 1997/11/28 16:19:03 bsmith Exp bsmith $";
#endif
/*
      Some PETSc utilites
*/
#include "petsc.h"        
#include "sys.h"             /*I    "sys.h"   I*/
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"


/*
     PetscSetDisplay - Tries to set the X windows display variable for all processors.
                       The variable PetscDisplay contains the X windows display variable.

*/
static char PetscDisplay[128]; 

#undef __FUNC__  
#define __FUNC__ "PetscSetDisplay" 
int PetscSetDisplay()
{
  int  size,rank,len,ierr,flag;
  char *str;

  PetscFunctionBegin;
  ierr = OptionsGetString(0,"-display",PetscDisplay,128,&flag);CHKERRQ(ierr);
  if (flag) PetscFunctionReturn(0);

  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);  
  if (!rank) {
    str = getenv("DISPLAY");
    if (!str || (str[0] == ':' && size > 1)) {
      ierr = PetscGetHostName(PetscDisplay,124); CHKERRQ(ierr);
      PetscStrcat(PetscDisplay,":0.0");
    } else {
      PetscStrncpy(PetscDisplay,str,128);
    }
    len  = PetscStrlen(PetscDisplay);
    ierr = MPI_Bcast(&len,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = MPI_Bcast(PetscDisplay,len,MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  } else {
    ierr = MPI_Bcast(&len,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = MPI_Bcast(PetscDisplay,len,MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    PetscDisplay[len] = 0;
  }
  PetscFunctionReturn(0);  
}

#undef __FUNC__  
#define __FUNC__ "PetscGetDisplay" 
/*
     PetscGetDisplay - Gets the display variable for all processors.

  Input Parameters:
.   n - length of string display

  Output Parameters:
.   display - the display string, may (and should) be freed.

*/
int PetscGetDisplay(char *display,int n)
{
  PetscFunctionBegin;
  PetscStrncpy(display,PetscDisplay,n);
  PetscFunctionReturn(0);  
}

#undef __FUNC__  
#define __FUNC__ "PetscSequentialPhaseBegin_Private" 
int PetscSequentialPhaseBegin_Private(MPI_Comm comm,int ng )
{
  int        lidx, np, tag = 0,ierr;
  MPI_Status status;

  PetscFunctionBegin;
  MPI_Comm_size( comm, &np );
  if (np == 1) PetscFunctionReturn(0);
  MPI_Comm_rank( comm, &lidx );
  if (lidx != 0) {
    ierr = MPI_Recv( 0, 0, MPI_INT, lidx-1, tag, comm, &status );CHKERRQ(ierr);
  }
  /* Send to the next process in the group unless we are the last process */ 
  if ((lidx % ng) < ng - 1 && lidx != np - 1) {
    ierr = MPI_Send( 0, 0, MPI_INT, lidx + 1, tag, comm );CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscSequentialPhaseEnd_Private" 
int PetscSequentialPhaseEnd_Private(MPI_Comm comm,int ng )
{
  int        lidx, np, tag = 0,ierr;
  MPI_Status status;

  PetscFunctionBegin;
  MPI_Comm_rank( comm, &lidx );
  MPI_Comm_size( comm, &np );
  if (np == 1) PetscFunctionReturn(0);

  /* Send to the first process in the next group */
  if ((lidx % ng) == ng - 1 || lidx == np - 1) {
    ierr = MPI_Send( 0, 0, MPI_INT, (lidx + 1) % np, tag, comm );CHKERRQ(ierr);
  }
  if (lidx == 0) {
    ierr = MPI_Recv( 0, 0, MPI_INT, np-1, tag, comm, &status );CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Seq_keyval is used to indicate an MPI attribute that
  is attached to a communicator that manages the sequential phase code below.
*/
static int Petsc_Seq_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ "PetscSequentialPhaseBegin" 
/*@C
   PetscSequentialPhaseBegin - Begins a sequential section of code.  

   Input Parameters:
.  comm - Communicator to sequentialize.  
.  ng   - Number in processor group.  This many processes are allowed to execute
   at the same time (usually 1)

   Notes:
   PetscSequentialPhaseBegin() and PetscSequentialPhaseEnd() provide a
   way to force a section of code to be executed by the processes in
   rank order.  Typically, this is done with
$  PetscSequentialPhaseBegin( comm, 1 );
$  <code to be executed sequentially>
$  PetscSequentialPhaseEnd( comm, 1 );
$
   Often, the sequential code contains output statements (e.g., printf) to
   be executed.  Note that you may need to flush the I/O buffers before
   calling PetscSequentialPhaseEnd().  Also, note that some systems do
   not propagate I/O in any order to the controling terminal (in other words, 
   even if you flush the output, you may not get the data in the order
   that you want).

.seealso: PetscSequentialPhaseEnd()

.keywords: sequential, phase, begin
@*/
int PetscSequentialPhaseBegin(MPI_Comm comm,int ng )
{
  int        ierr, np;
  MPI_Comm   local_comm,*addr_local_comm;

  PetscFunctionBegin;
  MPI_Comm_size( comm, &np );
  if (np == 1) PetscFunctionReturn(0);

  /* Get the private communicator for the sequential operations */
  if (Petsc_Seq_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Seq_keyval,0);CHKERRQ(ierr);
  }

  ierr = MPI_Comm_dup( comm, &local_comm );CHKERRQ(ierr);
  addr_local_comm  = (MPI_Comm *) PetscMalloc(sizeof(MPI_Comm));CHKPTRQ(addr_local_comm);
  *addr_local_comm = local_comm;
  ierr = MPI_Attr_put( comm, Petsc_Seq_keyval, (void *) addr_local_comm );CHKERRQ(ierr);
  ierr = PetscSequentialPhaseBegin_Private(local_comm,ng); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscSequentialPhaseEnd" 
/*@C
   PetscSequentialPhaseEnd - Ends a sequential section of code.

   Input Parameters:
.  comm - Communicator to sequentialize.  
.  ng   - Number in processor group.  This many processes are allowed to execute
   at the same time (usually 1)


   Notes:
   See PetscSequentialPhaseBegin() for more details.

.seealso: PetscSequentialPhaseBegin()

.keywords: sequential, phase, end
@*/
int PetscSequentialPhaseEnd(MPI_Comm comm,int ng )
{
  int        ierr, np, flag;
  MPI_Comm   local_comm,*addr_local_comm;

  PetscFunctionBegin;
  MPI_Comm_size( comm, &np );
  if (np == 1) PetscFunctionReturn(0);

  ierr = MPI_Attr_get( comm, Petsc_Seq_keyval, (void **)&addr_local_comm, &flag );CHKERRQ(ierr);
  if (!flag) MPI_Abort( comm, MPI_ERR_UNKNOWN );
  local_comm = *addr_local_comm;

  ierr = PetscSequentialPhaseEnd_Private(local_comm,ng); CHKERRQ(ierr);

  PetscFree(addr_local_comm); 
  ierr = MPI_Comm_free(&local_comm);CHKERRQ(ierr);
  ierr = MPI_Attr_delete(comm,Petsc_Seq_keyval);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
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

#undef __FUNC__  
#define __FUNC__ "Petsc_DelTag" 
/*
   Private routine to delete internal storage when a communicator is freed.
  This is called by MPI, not by users.

  The binding for the first argument changed from MPI 1.0 to 1.1; in 1.0
  it was MPI_Comm *comm.  
*/
static int Petsc_DelTag(MPI_Comm comm,int keyval,void* attr_val,void* extra_state )
{
  PetscFunctionBegin;
  PetscFree( attr_val );
  PetscFunctionReturn(MPI_SUCCESS);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetNewTag" 
/*@
    PetscObjectGetNewTag - Gets a unique new tag from a PETSc object. All 
    processors that share the object MUST call this routine EXACTLY the same
    number of times.  This tag should only be used with the current object's
    communicator; do NOT use it with any other MPI communicator.

    Input Parameter:
.   obj - the PETSc object

    Output Parameter:
.   tag - the new tag

.keywords: object, get, new, tag

.seealso: PetscObjectRestoreNewTag()
@*/
int PetscObjectGetNewTag(PetscObject obj,int *tag)
{
  int ierr,*tagvalp=0,flag;

  PetscFunctionBegin;
  PetscValidHeader(obj);
  PetscValidIntPointer(tag);

  ierr = MPI_Attr_get(obj->comm,Petsc_Tag_keyval,(void**)&tagvalp,&flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Bad comm in PETSc object");

  if (*tagvalp < 1) SETERRQ(PETSC_ERR_PLIB,0,"Out of tags for object");
  *tag = tagvalp[0]--;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectRestoreNewTag" 
/*@
    PetscObjectRestoreNewTag - Restores a new tag from a PETSc object. All 
    processors that share the object MUST call this routine EXACTLY the same
    number of times. 

    Input Parameter:
.   obj - the PETSc object

    Output Parameter:
.   tag - the new tag

.keywords: object, restore, new, tag

.seealso: PetscObjectGetNewTag()
@*/
int PetscObjectRestoreNewTag(PetscObject obj,int *tag)
{
  int ierr,*tagvalp=0,flag;

  PetscFunctionBegin;
  PetscValidHeader(obj);
  PetscValidIntPointer(tag);

  ierr = MPI_Attr_get(obj->comm,Petsc_Tag_keyval,(void**)&tagvalp,&flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Bad comm in PETSc object");

  if (*tagvalp == *tag - 1) {
    tagvalp[0]++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscCommDup_Private" 
/*
  PetscCommDup_Private - Duplicates the communicator only if it is not already PETSc 
                         communicator.

  Input Parameters:
. comm_in - Input communicator

  Output Parameters:
. comm_out - Output communicator.  May be 'comm_in'.
. first_tag - First tag available

  Notes:
  This routine returns one tag number.
  Call Petsc_Comm_free() when finished with the communicator.
*/
int PetscCommDup_Private(MPI_Comm comm_in,MPI_Comm *comm_out,int* first_tag)
{
  int ierr = MPI_SUCCESS, *tagvalp, *maxval, flag;

  PetscFunctionBegin;
  if (Petsc_Tag_keyval == MPI_KEYVAL_INVALID) {
    /* 
       The calling sequence of the 2nd argument to this function changed
       between MPI Standard 1.0 and the revisions 1.1 Here we match the 
       new standard, if you are using an MPI implementation that uses 
       the older version you will get a warning message about the next line;
       it is only a warning message and should do no harm.
    */
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN, Petsc_DelTag,&Petsc_Tag_keyval,(void*)0);CHKERRQ(ierr);
  }

  ierr = MPI_Attr_get(comm_in,Petsc_Tag_keyval,(void**)&tagvalp,&flag);CHKERRQ(ierr);

  if (!flag) {
    /* This communicator is not yet known to this system, so we dup it and set its value */
    ierr       = MPI_Comm_dup( comm_in, comm_out );CHKERRQ(ierr);
    ierr       = MPI_Attr_get( MPI_COMM_WORLD, MPI_TAG_UB, (void**)&maxval, &flag );CHKERRQ(ierr);
    tagvalp    = (int *) PetscMalloc( 2*sizeof(int) ); CHKPTRQ(tagvalp);
    tagvalp[0] = *maxval;
    tagvalp[1] = 0;
    ierr       = MPI_Attr_put(*comm_out,Petsc_Tag_keyval, tagvalp);CHKERRQ(ierr);
  } else {
    *comm_out = comm_in;
  }

  if (*tagvalp < 1) SETERRQ(PETSC_ERR_PLIB,0,"Out of tags for object");
  *first_tag = tagvalp[0]--;
  tagvalp[1]++;
#if defined(USE_PETSC_BOPT_g)
  if (*comm_out == comm_in) {
    int size;
    MPI_Comm_size(*comm_out,&size);
    if (size > 1) {
      int tag1 = *first_tag, tag2;
      ierr = MPI_Allreduce(&tag1,&tag2,1,MPI_INT,MPI_BOR,*comm_out);CHKERRQ(ierr);
      if (tag2 != tag1) {
        SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Communicator was used on subset\n of processors.");
      }
    }
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscCommFree_Private" 
/*
  PetscCommFree_Private - Frees communicator.  Use in conjunction with PetscCommDup_Private().
*/
int PetscCommFree_Private(MPI_Comm *comm)
{
  int ierr,*tagvalp,flag;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(*comm,Petsc_Tag_keyval,(void**)&tagvalp,&flag);CHKERRQ(ierr);
  if (!flag) {
    SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Error freeing PETSc object, problem with corrupted memory");
  }
  tagvalp[1]--;
  if (!tagvalp[1]) {ierr = MPI_Comm_free(comm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}




