#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gcreatev.c,v 1.40 1998/04/13 17:26:10 bsmith Exp bsmith $";
#endif


#include "sys.h"
#include "petsc.h"
#include "is.h"
#include "vec.h"    /*I "vec.h" I*/

#undef __FUNC__  
#define __FUNC__ "VecCreate"
/*@C
    VecCreate - Creates a vector, where the vector type is determined 
    from the options database.  Generates a parallel MPI vector if the 
    communicator has more than one processor.

    Input Parameters:
.   comm - MPI communicator
.   n - local vector length (or PETSC_DECIDE)
.   N - global vector length (or PETSC_DETERMINE)
 
    Output Parameter:
.   V - location to stash resulting vector

    Collective on MPI_Comm

    Options Database Key:
$   -vec_mpi - use MPI vectors, even for the uniprocessor case
$   -vec_shared - used shared memory parallel vectors

    Notes:
    Use VecDuplicate() or VecDuplicateVecs() to form additional vectors
    of the same type as an existing vector.

.keywords: vector, create, initial

.seealso: VecCreateSeq(), VecCreateMPI(), VecDuplicate(), VecDuplicateVecs()
@*/
int VecCreate(MPI_Comm comm,int n,int N,Vec *V)
{
  int ierr,size,flg,flgs;

  PetscFunctionBegin;
  MPI_Comm_size(comm,&size);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg) {
    (*PetscHelpPrintf)(comm,"VecCreate() option: -vec_mpi\n");
    (*PetscHelpPrintf)(comm,"                    -vec_shared\n");
  }
  ierr = OptionsHasName(PETSC_NULL,"-vec_mpi",&flg); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-vec_shared",&flgs); CHKERRQ(ierr);
  if (flgs) {
    ierr = VecCreateShared(comm,n,N,V); CHKERRQ(ierr);
  } else if (size > 1 || flg) {
    ierr = VecCreateMPI(comm,n,N,V); CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeq(comm,PetscMax(n,N),V);CHKERRQ(ierr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include "src/vec/vecimpl.h"
#undef __FUNC__  
#define __FUNC__ "VecGetType"
/*@C
   VecGetType - Gets the vector type and name (as a string) from the vector.

   Input Parameter:
.  mat - the vector

   Output Parameter:
.  type - the vector type (or use PETSC_NULL)
.  name - name of vector type (or use PETSC_NULL)

   Not Collective

.keywords: vector, get, type, name
@*/
int VecGetType(Vec vec,VecType *type,char **name)
{
  int  itype = (int)vec->type;
  char *vecname[10];

  PetscFunctionBegin;
  if (type) *type = (VecType) vec->type;
  if (name) {
    /* Note:  Be sure that this list corresponds to the enum in vec.h */
    vecname[0] = "VECSEQ";
    vecname[1] = "VECMPI";
    if (itype < 0 || itype > 1) *name = "Unknown vector type";
    else                        *name = vecname[itype];
  }
  PetscFunctionReturn(0);
}

 
