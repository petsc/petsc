#ifndef lint
static char vcid[] = "$Id: isltog.c,v 1.1 1996/11/17 23:48:28 bsmith Exp bsmith $";
#endif

#include "sys.h"   /*I "sys.h" I*/
#include "is.h"    /*I "is.h"  I*/

/*@
    ISLocalToGlobalMappingCreate - Creates a mapping between a local (0 to n)
      ordering and a global parallel ordering.

   Input Parameters:
.    n - the number of local elements
.    indices - the global index for each local element

   Output Parameters:
.    mapping - new mapping data structure

.keywords: IS, local to global mapping

.seealso: ISLocalToGlobalMappingDestroy(), 
@*/
int ISLocalToGlobalMappingCreate(int n, int *indices,ISLocalToGlobalMapping *mapping)
{
  PetscValidIntPointer(indices);
  PetscValidPointer(mapping);

  *mapping = PetscNew(struct _ISLocalToGlobalMapping); CHKPTRQ(*mapping);
  (*mapping)->refcnt  = 1;
  (*mapping)->indices = (int *) PetscMalloc((n+1)*sizeof(int));CHKPTRQ((*mapping)->indices);
  PetscMemcpy((*mapping)->indices,indices,n*sizeof(int));
  return 0;
}
  
/*@
    ISLocalToGlobalMappingDestroy - Destroys a mapping between a local (0 to n)
      ordering and a global parallel ordering.

   Input Parameters:
.    mapping - mapping data structure

.keywords: IS, local to global mapping

.seealso: ISLocalToGlobalMappingCreate(), 
@*/
int ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping mapping)
{
  PetscValidPointer(mapping);
  if (--mapping->refcnt) return 0;

  PetscFree(mapping->indices);
  PetscFree(mapping);
  return 0;
}
  
/*@
    ISLocalToGlobalMappingApplyIS - Creates a new IS using the global numbering
      defined in an ISLocalToGlobalMapping from an IS in the local numbering.

   Input Parameters:
.   ISLocalToGlobalMapping - mapping between local and global numbering
.   is - index set in local numbering

   Output Parameters:
.   newis - index set in global numbering

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()

@*/
int ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping mapping, IS is, IS *newis)
{
  int ierr,n,i,*idxin,*idxmap,*idxout;
  PetscValidPointer(mapping);
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidPointer(newis);

  ierr   = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr   = ISGetIndices(is,&idxin); CHKERRQ(ierr);
  idxmap = mapping->indices;
  
  idxout = (int *) PetscMalloc((n+1)*sizeof(int));CHKPTRQ(idxout);
  for ( i=0; i<n; i++ ) {
    idxout[i] = idxmap[idxin[i]];
  }
  ierr = ISCreateGeneral(MPI_COMM_SELF,n,idxout,newis); CHKERRQ(ierr);
  PetscFree(idxout);
  return 0;
}



