
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
PetscErrorCode ISDifference(IS is1,IS is2,IS *isout)
{
  PetscErrorCode ierr;
  PetscInt      i,*i1,*i2,n1,n2,imin,imax,nout,*iout;
  PetscBT  mask;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_COOKIE,1);
  PetscValidHeaderSpecific(is2,IS_COOKIE,2);
  PetscValidPointer(isout,3);

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
  ierr = PetscMalloc(nout*sizeof(PetscInt),&iout);CHKERRQ(ierr);
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
   ISSum - Computes the sum (union) of two index sets in place. Note that
           is1 is an existing IS, not merely a pointer.

   Only sequential version (at the moment)

   Input Parameter:
+  is1 - index set to be extended
-  is2 - index values to be added

   Notes:
   If n1 and n2 are the sizes of the sets, this takes O(n1+n2) time;
   if is2 is a subset of is1, is1 is left unchanged, otherwise is1
   is reallocated.
   Both index sets need to be sorted on input.

   Level: intermediate

.seealso: ISDestroy(), ISView(), ISDifference(), ISSum()

   Concepts: index sets^union
   Concepts: IS^union

@*/
PetscErrorCode ISSum(IS *is1,IS is2)
{
  MPI_Comm       comm;
  PetscTruth     f;
  PetscMPIInt    size;
  PetscInt       *i1,*i2,n1,n2,n3, p1,p2, *iout;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(*is1,IS_COOKIE,1);
  PetscValidHeaderSpecific(is2,IS_COOKIE,2);
  ierr = PetscObjectGetComm((PetscObject)(*is1),&comm); CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size>1) SETERRQ(PETSC_ERR_SUP,"Currently only for uni-processor IS");

  ierr = ISSorted(*is1,&f); CHKERRQ(ierr);
  if (!f) SETERRQ(PETSC_ERR_ARG_INCOMP,"Arg 1 is not sorted");
  ierr = ISSorted(is2,&f); CHKERRQ(ierr);
  if (!f) SETERRQ(PETSC_ERR_ARG_INCOMP,"Arg 2 is not sorted");

  ierr = ISGetLocalSize(*is1,&n1);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is2,&n2);CHKERRQ(ierr);
  if (!n2) PetscFunctionReturn(0);
  ierr = ISGetIndices(*is1,&i1);CHKERRQ(ierr);
  ierr = ISGetIndices(is2,&i2);CHKERRQ(ierr);

  p1 = 0; p2 = 0; n3 = 0;
  do {
    if (p1==n1) { /* cleanup for is2 */ n3 += n2-p2; break;
    } else {
      while (p2<n2 && i2[p2]<i1[p1]) {n3++; p2++;}
      if (p2==n2) { /* cleanup for is1 */ n3 += n1-p1; break;
      } else {
	if (i2[p2]==i1[p1]) {n3++; p1++; p2++;}
      }
    }
    if (p2==n2) { /* cleanup for is1 */ n3 += n1-p1; break;
    } else {
      while (p1<n1 && i1[p1]<i2[p2]) {n3++; p1++;}
      if (p1==n1) { /* clean up for is2 */ n3 += n2-p2; break;
      } else {
	if (i1[p1]==i2[p2]) {n3++; p1++; p2++;}
      }
    }
  } while (p1<n1 || p2<n2);

  if (n3==n1) { /* no new elements to be added */
    ierr = ISRestoreIndices(*is1,&i1); CHKERRQ(ierr);
    ierr = ISRestoreIndices(is2,&i2); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscMalloc(n3*sizeof(PetscInt),&iout);CHKERRQ(ierr);

  p1 = 0; p2 = 0; n3 = 0;
  do {
    if (p1==n1) { /* cleanup for is2 */
      while (p2<n2) iout[n3++] = i2[p2++];
      break;
    } else {
      while (p2<n2 && i2[p2]<i1[p1]) iout[n3++] = i2[p2++];
      if (p2==n2) { /* cleanup for is1 */
	while (p1<n1) iout[n3++] = i1[p1++];
	break;
      } else {
	if (i2[p2]==i1[p1]) {iout[n3++] = i1[p1++]; p2++;}
      }
    }
    if (p2==n2) { /* cleanup for is1 */
      while (p1<n1) iout[n3++] = i1[p1++];
      break;
    } else {
      while (p1<n1 && i1[p1]<i2[p2]) iout[n3++] = i1[p1++];
      if (p1==n1) { /* clean up for is2 */
	while (p2<n2) iout[n3++] = i2[p2++];
	break;
      } else {
	if (i1[p1]==i2[p2]) {iout[n3++] = i1[p1++]; p2++;}
      }
    }
  } while (p1<n1 || p2<n2);

  ierr = ISRestoreIndices(*is1,&i1); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is2,&i2); CHKERRQ(ierr);
  ierr = ISDestroy(*is1); CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n3,iout,is1); CHKERRQ(ierr);
  ierr = PetscFree(iout); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

