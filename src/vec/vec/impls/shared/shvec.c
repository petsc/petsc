#define PETSCVEC_DLL
/*
   This file contains routines for Parallel vector operations that use shared memory
 */
#include "src/vec/vec/impls/mpi/pvecimpl.h"   /*I  "petscvec.h"   I*/

/*
     Could not get the include files to work properly on the SGI with 
  the C++ compiler.
*/
#if defined(PETSC_USE_SHARED_MEMORY) && !defined(__cplusplus)

EXTERN PetscErrorCode PetscSharedMalloc(MPI_Comm,PetscInt,PetscInt,void**);

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_Shared"
PetscErrorCode VecDuplicate_Shared(Vec win,Vec *v)
{
  PetscErrorCode ierr;
  Vec_MPI        *w = (Vec_MPI *)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;

  /* first processor allocates entire array and sends it's address to the others */
  ierr = PetscSharedMalloc(win->comm,win->map.n*sizeof(PetscScalar),win->map.N*sizeof(PetscScalar),(void**)&array);CHKERRQ(ierr);

  ierr = VecCreate(win->comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,win->map.n,win->map.N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*v,w->nghost,array,win->map);CHKERRQ(ierr);

  /* New vector should inherit stashing property of parent */
  (*v)->stash.donotstash = win->stash.donotstash;
  
  ierr = PetscOListDuplicate(win->olist,&(*v)->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(win->qlist,&(*v)->qlist);CHKERRQ(ierr);

  if (win->mapping) {
    (*v)->mapping = win->mapping;
    ierr = PetscObjectReference((PetscObject)win->mapping);CHKERRQ(ierr);
  }
  (*v)->ops->duplicate = VecDuplicate_Shared;
  (*v)->map.bs    = win->map.bs;
  (*v)->bstash.bs = win->bstash.bs;
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_Shared"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Shared(Vec vv)
{
  PetscErrorCode ierr;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = PetscSplitOwnership(vv->comm,&vv->map.n,&vv->map.N);CHKERRQ(ierr);
  ierr = PetscSharedMalloc(vv->comm,vv->map.n*sizeof(PetscScalar),vv->map.N*sizeof(PetscScalar),(void**)&array);CHKERRQ(ierr); 

  ierr = VecCreate_MPI_Private(vv,0,array,PETSC_NULL);CHKERRQ(ierr);
  vv->ops->duplicate = VecDuplicate_Shared;

  PetscFunctionReturn(0);
}
EXTERN_C_END


/* ----------------------------------------------------------------------------------------
     Code to manage shared memory allocation under the SGI with MPI

  We associate with a communicator a shared memory "areana" from which memory may be shmalloced.
*/
#include "petscsys.h"
#include "petscfix.h"
#if defined(PETSC_HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_SYS_PARAM_H)
#include <sys/param.h>
#endif
#if defined(PETSC_HAVE_SYS_UTSNAME_H)
#include <sys/utsname.h>
#endif
#include <fcntl.h>
#include <time.h>  
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "petscfix.h"

static PetscMPIInt Petsc_Shared_keyval = MPI_KEYVAL_INVALID;
static PetscInt Petsc_Shared_size   = 100000000;

#undef __FUNCT__  
#define __FUNCT__ "Petsc_DeleteShared" 
/*
   Private routine to delete internal storage when a communicator is freed.
  This is called by MPI, not by users.

  The binding for the first argument changed from MPI 1.0 to 1.1; in 1.0
  it was MPI_Comm *comm.  
*/
static PetscErrorCode Petsc_DeleteShared(MPI_Comm comm,PetscInt keyval,void* attr_val,void* extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(attr_val);CHKERRQ(ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSharedMemorySetSize"
PetscErrorCode PetscSharedMemorySetSize(PetscInt s)
{
  PetscFunctionBegin;
  Petsc_Shared_size = s;
  PetscFunctionReturn(0);
}

#include "petscfix.h"

#include <ulocks.h>

#undef __FUNCT__  
#define __FUNCT__ "PetscSharedInitialize"
PetscErrorCode PetscSharedInitialize(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,flag;
  char           filename[PETSC_MAX_PATH_LEN];
  usptr_t        **arena;

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
    ierr = PetscMalloc(sizeof(usptr_t*),&arena);CHKERRQ(ierr);

    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscStrcpy(filename,"/tmp/PETScArenaXXXXXX");CHKERRQ(ierr);
#ifdef PETSC_HAVE_MKSTEMP
      if (mkstemp(filename) < 0) {
        SETERRQ1(PETSC_ERR_FILE_OPEN, "Unable to open temporary file %s", filename);
      }
#else
      if (!mktemp(filename)) {
        SETERRQ1(PETSC_ERR_FILE_OPEN, "Unable to open temporary file %s", filename);
      }
#endif
    } 
    ierr     = MPI_Bcast(filename,PETSC_MAX_PATH_LEN,MPI_CHAR,0,comm);CHKERRQ(ierr);
    ierr     = PetscOptionsGetInt(PETSC_NULL,"-shared_size",&Petsc_Shared_size,&flag);CHKERRQ(ierr);
    usconfig(CONF_INITSIZE,Petsc_Shared_size);
    *arena   = usinit(filename); 
    ierr     = MPI_Attr_put(comm,Petsc_Shared_keyval,arena);CHKERRQ(ierr);
  } 

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSharedMalloc"
PetscErrorCode PetscSharedMalloc(MPI_Comm comm,PetscInt llen,PetscInt len,void **result)
{
  char           *value;
  PetscErrorCode ierr;
  PetscInt       shift;
  PetscMPIInt    rank,flag;
  usptr_t        **arena;

  PetscFunctionBegin;
  *result = 0;
  if (Petsc_Shared_keyval == MPI_KEYVAL_INVALID) {
    ierr = PetscSharedInitialize(comm);CHKERRQ(ierr);
  }
  ierr = MPI_Attr_get(comm,Petsc_Shared_keyval,(void**)&arena,&flag);CHKERRQ(ierr);
  if (!flag) { 
    ierr = PetscSharedInitialize(comm);CHKERRQ(ierr);
    ierr = MPI_Attr_get(comm,Petsc_Shared_keyval,(void**)&arena,&flag);CHKERRQ(ierr);
    if (!flag) SETERRQ(PETSC_ERR_LIB,"Unable to initialize shared memory");
  } 

  ierr   = MPI_Scan(&llen,&shift,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  shift -= llen;

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    value = (char*)usmalloc((size_t) len,*arena);
    if (!value) {
      (*PetscErrorPrintf)("Unable to allocate shared memory location\n");
      (*PetscErrorPrintf)("Run with option -shared_size <size> \n");
      (*PetscErrorPrintf)("with size > %d \n",(int)(1.2*(Petsc_Shared_size+len)));
      SETERRQ(PETSC_ERR_LIB,"Unable to malloc shared memory");
    }
  }
  ierr = MPI_Bcast(&value,8,MPI_BYTE,0,comm);CHKERRQ(ierr);
  value += shift; 

  PetscFunctionReturn(0);
}

#else

EXTERN_C_BEGIN
extern PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Seq(Vec);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_Shared"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Shared(Vec vv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(vv->comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    SETERRQ(PETSC_ERR_SUP_SYS,"No supported for shared memory vector objects on this machine");
  }
  ierr = VecCreate_Seq(vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif

#undef __FUNCT__  
#define __FUNCT__ "VecCreateShared"
/*@
   VecCreateShared - Creates a parallel vector that uses shared memory.

   Input Parameters:
.  comm - the MPI communicator to use
.  n - local vector length (or PETSC_DECIDE to have calculated if N is given)
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)

   Output Parameter:
.  vv - the vector

   Collective on MPI_Comm
 
   Notes:
   Currently VecCreateShared() is available only on the SGI; otherwise,
   this routine is the same as VecCreateMPI().

   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   Level: advanced

   Concepts: vectors^creating with shared memory

.seealso: VecCreateSeq(), VecCreate(), VecCreateMPI(), VecDuplicate(), VecDuplicateVecs(), 
          VecCreateGhost(), VecCreateMPIWithArray(), VecCreateGhostWithArray()

@*/ 
PetscErrorCode PETSCVEC_DLLEXPORT VecCreateShared(MPI_Comm comm,PetscInt n,PetscInt N,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,N);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSHARED);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





