#ifndef lint
static char vcid[] = "$Id: gcreatev.c,v 1.28 1996/01/29 23:59:26 curfman Exp bsmith $";
#endif


#include "sys.h"
#include "petsc.h"
#include "is.h"
#include "vec.h"    /*I "vec.h" I*/

/*@C
    VecCreate - Creates a vector, where the vector type is determined 
    from the options database.  Generates a parallel MPI vector if the 
    communicator has more than one processor.

    Input Parameters:
.   comm - MPI communicator
.   n - global vector length
 
    Output Parameter:
.   V - location to stash resulting vector

    Options Database Key:
$   -vec_mpi : use MPI vectors, even for the uniprocessor case

    Notes:
    Use VecDuplicate() or VecDuplicateVecs() to form additional vectors
    of the same type as an existing vector.

.keywords: vector, create, initial

.seealso: VecCreateSeq(), VecCreateMPI(), VecDuplicate(), VecDuplicateVecs()
@*/
int VecCreate(MPI_Comm comm,int n,Vec *V)
{
  int size,flg,ierr;

  MPI_Comm_size(comm,&size);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg) {
    PetscPrintf(comm,"VecCreate() option: -vec_mpi\n");
  }
  ierr = OptionsHasName(PETSC_NULL,"-vec_mpi",&flg); CHKERRQ(ierr);
  if (size > 1 || flg) {
    return VecCreateMPI(comm,PETSC_DECIDE,n,V);
  }
  return VecCreateSeq(comm,n,V);
}

#include "vecimpl.h"
/*@C
   VecGetType - Gets the vector type and name (as a string) from the vector.

   Input Parameter:
.  mat - the vector

   Output Parameter:
.  type - the vector type (or use PETSC_NULL)
.  name - name of vector type (or use PETSC_NULL)

.keywords: vector, get, type, name
@*/
int VecGetType(Vec vec,VecType *type,char **name)
{
  int  itype = (int)vec->type;
  char *vecname[10];

  if (type) *type = (VecType) vec->type;
  if (name) {
    /* Note:  Be sure that this list corresponds to the enum in vec.h */
    vecname[0] = "VECSEQ";
    vecname[1] = "VECMPI";
    if (itype < 0 || itype > 1) *name = "Unknown vector type";
    else                        *name = vecname[itype];
  }
  return 0;
}

 
