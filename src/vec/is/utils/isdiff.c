
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: isdiff.c,v 1.1 1997/10/01 20:08:39 bsmith Exp bsmith $";
#endif

#include "is.h"    /*I "is.h"  I*/
#include "petscmath.h"
#include "src/inline/bitarray.h"

#undef __FUNC__  
#define __FUNC__ "ISColoringDestroy"
/*@
     ISDifference - Computes the difference between two index sets.

  Input Parameter:
.   is1 - first index, to have items removed from it
.   is2 - index values to be removed

  Output Parameters:
.   isout - is1 - is2

  Notes: Negative values are removed from the lists. is2 may have values
         that are not in is1. This requires O(imax-imin) memory and 
         O(imax-imin) work, where imin and imax are the bounds on the 
         indices in is1.

.seealso: ISDestroy(), ISView()

.keywords: Index set difference
@*/
int ISDifference(IS is1,IS is2, IS *isout)
{
  int      i,ierr,*i1,*i2,n1,n2,imin,imax,nout,*iout;
  BT       mask;
  MPI_Comm comm;

  PetscValidHeaderSpecific(is1,IS_COOKIE);
  PetscValidHeaderSpecific(is2,IS_COOKIE);
  PetscValidPointer(isout);

  ierr = ISGetIndices(is1,&i1); CHKERRQ(ierr);
  ierr = ISGetSize(is1,&n1); CHKERRQ(ierr);

  /* Create a bit mask array to contain required values */
  if (n1) {
    imin = PETSC_MAX_INT;
    imax = 0;  
    for ( i=0; i<n1; i++ ) {
      if (i1[i] < 0) continue;
      imin = PetscMin(imin,i1[i]);
      imax = PetscMax(imax,i1[i]);
    }
  } else {
    imin = imax = 0;
  }
  ierr = BTCreate(imax-imin,mask); CHKERRQ(ierr);
  /* Put the values from is1 */
  for ( i=0; i<n1; i++ ) {
    if (i1[i] < 0) continue;
    BTSet(mask,i1[i] - imin);
  }
  ierr = ISRestoreIndices(is1,&i1); CHKERRQ(ierr);
  /* Remove the values from is2 */
  ierr = ISGetIndices(is2,&i2); CHKERRQ(ierr);
  ierr = ISGetSize(is2,&n2); CHKERRQ(ierr);
  for ( i=0; i<n2; i++ ) {
    if (i2[i] < imin || i2[i] > imax) continue;
    BTClear(mask,i2[i] - imin);
  }
  ierr = ISRestoreIndices(is2,&i2); CHKERRQ(ierr);
  
  /* Count the number in the difference */
  nout = 0;
  for ( i=0; i<imax-imin; i++ ) {
    if (BTLookup(mask,i)) nout++;
  }

  /* create the new IS containing the difference */
  iout = (int *) PetscMalloc((nout+1)*sizeof(int));CHKPTRQ(iout);
  nout = 0;
  for ( i=0; i<imax-imin; i++ ) {
    if (BTLookup(mask,i)) iout[nout++] = i + imin;
  }
  ierr = PetscObjectGetComm((PetscObject)is1,&comm); CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,nout,iout,isout);CHKERRQ(ierr);
  PetscFree(iout);

  BTDestroy(mask);
  return 0;
}

