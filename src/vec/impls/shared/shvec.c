

#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: shvec.c,v 1.4 1997/11/29 15:35:56 bsmith Exp bsmith $";
#endif

/*
   This file contains routines for Parallel vector operations that use shared memory
 */

#include "petsc.h"
#include <math.h>
#include "src/vec/impls/mpi/pvecimpl.h"   /*I  "vec.h"   I*/

/*
     Could not get the include files to work properly on the SGI with 
  the C++ compiler.
*/
#if defined(USE_SHARED_MEMORY) && !defined(__cplusplus)

extern void *PetscSharedMalloc(int,int,MPI_Comm);

#undef __FUNC__  
#define __FUNC__ "VecDuplicate_Shared"
int VecDuplicate_Shared( Vec win, Vec *v)
{
  int     ierr,rank;
  Vec_MPI *vw, *w = (Vec_MPI *)win->data;
  Scalar  *array;

  PetscFunctionBegin;
  MPI_Comm_rank(win->comm,&rank);

  /* first processor allocates entire array and sends it's address to the others */
  array = (Scalar *) PetscSharedMalloc(w->n*sizeof(Scalar),w->N*sizeof(Scalar),win->comm);CHKPTRQ(array);

  ierr = VecCreateMPI_Private(win->comm,w->n,w->N,w->nghost,w->size,rank,w->ownership,array,v);CHKERRQ(ierr);
  vw   = (Vec_MPI *)(*v)->data;

  /* New vector should inherit stashing property of parent */
  vw->stash.donotstash = w->stash.donotstash;
  
  (*v)->childcopy    = win->childcopy;
  (*v)->childdestroy = win->childdestroy;
  if (win->mapping) {
    (*v)->mapping = win->mapping;
    PetscObjectReference((PetscObject)win->mapping);
  }
  if (win->child) {
    ierr = (*win->childcopy)(win->child,&(*v)->child);CHKERRQ(ierr);
  }
  (*v)->ops.duplicate = VecDuplicate_Shared;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecCreateShared"
/*@C
   VecCreateShared - Creates a parallel vector.

   Input Parameters:
.  comm - the MPI communicator to use
.  n - local vector length (or PETSC_DECIDE to have calculated if N is given)
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)

   Output Parameter:
.  vv - the vector
 
   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

.keywords: vector, create, MPI

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPIWithArray(), VecCreateGhostWithArray()

@*/ 
int VecCreateShared(MPI_Comm comm,int n,int N,Vec *vv)
{
  int     sum, work = n, size, rank,ierr,*rowners,i;
  Scalar  *array;

  PetscFunctionBegin;
  *vv = 0;

  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank); 
  if (N == PETSC_DECIDE) { 
    ierr = MPI_Allreduce( &work, &sum,1,MPI_INT,MPI_SUM,comm );CHKERRQ(ierr);
    N = sum;
  }
  if (n == PETSC_DECIDE) { 
    n = N/size + ((N % size) > rank);
  }
  /*  Determine ownership range for each processor */
  rowners = (int *) PetscMalloc((size+1)*sizeof(int));CHKPTRQ(rowners);
  ierr = MPI_Allgather(&n,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++ ) {
    rowners[i] += rowners[i-1];
  }

  array = (Scalar *) PetscSharedMalloc(n*sizeof(Scalar),N*sizeof(Scalar),comm);CHKPTRQ(array); 

  ierr = VecCreateMPI_Private(comm,n,N,0,size,rank,rowners,array,vv);CHKERRQ(ierr);
  PetscFree(rowners);
  (*vv)->ops.duplicate = VecDuplicate_Shared;
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------
     Code to manage shared memory allocation under the SGI with MPI

  We associate with a communicator a shared memory "areana" from which memory may be shmalloced.
*/
#include "src/sys/src/files.h"
static int Petsc_Shared_keyval = MPI_KEYVAL_INVALID;
static int Petsc_Shared_size   = 100000000;

#undef __FUNC__  
#define __FUNC__ "Petsc_DeleteShared" 
/*
   Private routine to delete internal storage when a communicator is freed.
  This is called by MPI, not by users.

  The binding for the first argument changed from MPI 1.0 to 1.1; in 1.0
  it was MPI_Comm *comm.  
*/
static int Petsc_DeleteShared(MPI_Comm comm,int keyval,void* attr_val,void* extra_state )
{
  PetscFunctionBegin;
  PetscFree( attr_val );
  PetscFunctionReturn(MPI_SUCCESS);
}

#undef __FUNC__  
#define __FUNC__ "PetscSharedMemorySetSize"
int PetscSharedMemorySetSize(int s)
{
  PetscFunctionBegin;
  Petsc_Shared_size = s;
  PetscFunctionReturn(0);
}

#include "pinclude/petscfix.h"

#include <ulocks.h>

#undef __FUNC__  
#define __FUNC__ "PetscSharedInitialize"
int PetscSharedInitialize(MPI_Comm comm)
{
  int     rank,len,ierr,flag;
  char    filename[256];
  usptr_t **arena;

  PetscFunctionBegin;

  if (Petsc_Shared_keyval == MPI_KEYVAL_INVALID) {
    /* 
       The calling sequence of the 2nd argument to this function changed
       between MPI Standard 1.0 and the revisions 1.1 Here we match the 
       new standard, if you are using an MPI implementation that uses 
       the older version you will get a warning message about the next line;
       it is only a warning message and should do no harm.
    */
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DeleteShared,&Petsc_Shared_keyval,0);CHKERRQ(ierr);
  }

  ierr = MPI_Attr_get(comm,Petsc_Shared_keyval,(void**)&arena,&flag);CHKERRQ(ierr);

  if (!flag) {
    /* This communicator does not yet have a shared memory areana */
    arena    = (usptr_t**) PetscMalloc( sizeof(usptr_t*) ); CHKPTRQ(arena);

    MPI_Comm_rank(comm,&rank);
    if (!rank) {
      PetscStrcpy(filename,"/tmp/PETScArenaXXXXXX");
      mktemp(filename);
      len      = PetscStrlen(filename);
    } 
    ierr     = MPI_Bcast(&len,1,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr     = MPI_Bcast(filename,len+1,MPI_CHAR,0,comm);CHKERRQ(ierr);
    ierr     = OptionsGetInt(PETSC_NULL,"-shared_size",&Petsc_Shared_size,&flag);CHKERRQ(ierr);
    usconfig(CONF_INITSIZE,Petsc_Shared_size);
    *arena   = usinit(filename); 
    ierr     = MPI_Attr_put(comm,Petsc_Shared_keyval, arena);CHKERRQ(ierr);
  } 

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscSharedMalloc"
void *PetscSharedMalloc(int llen,int len,MPI_Comm comm)
{
  char    *value;
  int     ierr,shift,rank,flag;
  usptr_t **arena;

  PetscFunctionBegin;
  if (Petsc_Shared_keyval == MPI_KEYVAL_INVALID) {
    ierr = PetscSharedInitialize(comm);
  }
  ierr = MPI_Attr_get(comm,Petsc_Shared_keyval,(void**)&arena,&flag);
  if (ierr) PetscFunctionReturn(0);
  if (!flag) { 
    ierr = PetscSharedInitialize(comm);
    if (ierr) {PetscFunctionReturn(0);}
    ierr = MPI_Attr_get(comm,Petsc_Shared_keyval,(void**)&arena,&flag);
    if (ierr) PetscFunctionReturn(0);
  } 

  ierr   = MPI_Scan(&llen,&shift,1,MPI_INT,MPI_SUM,comm); if (ierr) PetscFunctionReturn(0);
  shift -= llen;

  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    value = (char *) usmalloc((size_t) len, *arena);
    if (!value) {
      PetscErrorPrintf("PETSC ERROR: Unable to allocate shared memory location\n");
      PetscErrorPrintf("PETSC ERROR: Run with option -shared_size <size> \n");
      PetscErrorPrintf("PETSC_ERROR: with size > %d \n",(int)(1.2*(Petsc_Shared_size+len)));
      PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,1,1,"Unable to malloc shared memory");
      PetscFunctionReturn(0);
    }
  }
  ierr = MPI_Bcast(&value,8,MPI_BYTE,0,comm); if (ierr) PetscFunctionReturn(0);
  value += shift; 

  PetscFunctionReturn((void *)value);
}

#else

int VecCreateShared(MPI_Comm comm,int n,int N,Vec *vv)
{
  int ierr;

  PetscFunctionBegin;
  ierr = VecCreateMPI(comm,n,N,vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif



