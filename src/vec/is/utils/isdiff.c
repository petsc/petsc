/*$Id: isdiff.c,v 1.24 2001/03/23 23:21:16 balay Exp $*/

#include "petscis.h"                    /*I "petscis.h"  I*/
#include "petscbt.h"

#undef __FUNCT__  
#define __FUNCT__ "ISDifference"
/*@
   ISDifference - Computes the difference between two index sets.

   Collective on IS

   Input Parameter:
+  is1 - first index, to have items removed from it
-  is2 - index values to be removed

   Output Parameters:
.  isout - is1 - is2

   Notes:
   Negative values are removed from the lists. is2 may have values
   that are not in is1. This requires O(imax-imin) memory and O(imax-imin)
   work, where imin and imax are the bounds on the indices in is1.

   Level: intermediate

   Concepts: index sets^difference
   Concepts: IS^difference

.seealso: ISDestroy(), ISView(), ISSum()

@*/
int ISDifference(IS is1,IS is2,IS *isout)
{
  int      i,ierr,*i1,*i2,n1,n2,imin,imax,nout,*iout;
  PetscBT  mask;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_COOKIE);
  PetscValidHeaderSpecific(is2,IS_COOKIE);
  PetscValidPointer(isout);

  ierr = ISGetIndices(is1,&i1);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is1,&n1);CHKERRQ(ierr);

  /* Create a bit mask array to contain required values */
  if (n1) {
    imin = PETSC_MAX_INT;
    imax = 0;  
    for (i=0; i<n1; i++) {
      if (i1[i] < 0) continue;
      imin = PetscMin(imin,i1[i]);
      imax = PetscMax(imax,i1[i]);
    }
  } else {
    imin = imax = 0;
  }
  ierr = PetscBTCreate(imax-imin,mask);CHKERRQ(ierr);
  /* Put the values from is1 */
  for (i=0; i<n1; i++) {
    if (i1[i] < 0) continue;
    PetscBTSet(mask,i1[i] - imin);
  }
  ierr = ISRestoreIndices(is1,&i1);CHKERRQ(ierr);
  /* Remove the values from is2 */
  ierr = ISGetIndices(is2,&i2);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is2,&n2);CHKERRQ(ierr);
  for (i=0; i<n2; i++) {
    if (i2[i] < imin || i2[i] > imax) continue;
    PetscBTClear(mask,i2[i] - imin);
  }
  ierr = ISRestoreIndices(is2,&i2);CHKERRQ(ierr);
  
  /* Count the number in the difference */
  nout = 0;
  for (i=0; i<imax-imin+1; i++) {
    if (PetscBTLookup(mask,i)) nout++;
  }

  /* create the new IS containing the difference */
  ierr = PetscMalloc((nout+1)*sizeof(int),&iout);CHKERRQ(ierr);
  nout = 0;
  for (i=0; i<imax-imin+1; i++) {
    if (PetscBTLookup(mask,i)) iout[nout++] = i + imin;
  }
  ierr = PetscObjectGetComm((PetscObject)is1,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,nout,iout,isout);CHKERRQ(ierr);
  ierr = PetscFree(iout);CHKERRQ(ierr);

  ierr = PetscBTDestroy(mask);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSum"
/*@
   ISSum - Computes the sum (union) of two index sets.

   Collective on IS

   Input Parameter:
+  is1 - first index set
-  is2 - index values to be added

   Output Parameters:
.  isout - is1 + is2 The index set is2 is appended to is1 removing duplicates

   Notes:
   Negative values are removed from the lists. This requires O(imax-imin) 
   memory and O(imax-imin) work, where imin and imax are the bounds on the 
   indices in is1 and is2.

   Level: intermediate

.seealso: ISDestroy(), ISView(), ISDifference()

   Concepts: index sets^difference
   Concepts: IS^difference

@*/
int ISSum(IS is1,IS is2,IS *isout)
{
  int      i,ierr,*i1,*i2,n1,n2,imin,imax,nout,*iout;
  PetscBT  mask;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_COOKIE);
  PetscValidHeaderSpecific(is2,IS_COOKIE);
  PetscValidPointer(isout);

  ierr = ISGetIndices(is1,&i1);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is1,&n1);CHKERRQ(ierr);
  ierr = ISGetIndices(is2,&i2);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is2,&n2);CHKERRQ(ierr);

  /* Create a bit mask array to contain required values */
  if (n1 || n2) {
    imin = PETSC_MAX_INT;
    imax = 0;  
    for (i=0; i<n1; i++) {
      if (i1[i] < 0) continue;
      imin = PetscMin(imin,i1[i]);
      imax = PetscMax(imax,i1[i]);
    }
    for (i=0; i<n2; i++) {
      if (i2[i] < 0) continue;
      imin = PetscMin(imin,i2[i]);
      imax = PetscMax(imax,i2[i]);
    }
  } else {
    imin = imax = 0;
  }
  ierr = PetscMalloc((n1+n2+1)*sizeof(int),&iout);CHKERRQ(ierr);
  nout = 0;
  ierr = PetscBTCreate(imax-imin,mask);CHKERRQ(ierr);
  /* Put the values from is1 */
  for (i=0; i<n1; i++) {
    if (i1[i] < 0) continue;
    if (!PetscBTLookupSet(mask,i1[i] - imin)) {
      iout[nout++] = i1[i];
    }
  }
  ierr = ISRestoreIndices(is1,&i1);CHKERRQ(ierr);
  /* Put the values from is2 */
  for (i=0; i<n2; i++) {
    if (i2[i] < 0) continue;
    if (!PetscBTLookupSet(mask,i2[i] - imin)) {
      iout[nout++] = i2[i];
    }
  }
  ierr = ISRestoreIndices(is2,&i2);CHKERRQ(ierr);

  /* create the new IS containing the sum */
  ierr = PetscObjectGetComm((PetscObject)is1,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,nout,iout,isout);CHKERRQ(ierr);
  ierr = PetscFree(iout);CHKERRQ(ierr);

  ierr = PetscBTDestroy(mask);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

