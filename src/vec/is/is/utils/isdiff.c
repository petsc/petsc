
#include <petsc/private/isimpl.h>                    /*I "petscis.h"  I*/
#include <petsc/private/sectionimpl.h>
#include <petscbt.h>

/*@
   ISDifference - Computes the difference between two index sets.

   Collective on IS

   Input Parameters:
+  is1 - first index, to have items removed from it
-  is2 - index values to be removed

   Output Parameters:
.  isout - is1 - is2

   Notes:
   Negative values are removed from the lists. is2 may have values
   that are not in is1. This requires O(imax-imin) memory and O(imax-imin)
   work, where imin and imax are the bounds on the indices in is1.

   If is2 is NULL, the result is the same as for an empty IS, i.e., a duplicate of is1.

   Level: intermediate

.seealso: `ISDestroy()`, `ISView()`, `ISSum()`, `ISExpand()`
@*/
PetscErrorCode  ISDifference(IS is1,IS is2,IS *isout)
{
  PetscInt       i,n1,n2,imin,imax,nout,*iout;
  const PetscInt *i1,*i2;
  PetscBT        mask;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_CLASSID,1);
  PetscValidPointer(isout,3);
  if (!is2) {
    PetscCall(ISDuplicate(is1, isout));
    PetscFunctionReturn(0);
  }
  PetscValidHeaderSpecific(is2,IS_CLASSID,2);

  PetscCall(ISGetIndices(is1,&i1));
  PetscCall(ISGetLocalSize(is1,&n1));

  /* Create a bit mask array to contain required values */
  if (n1) {
    imin = PETSC_MAX_INT;
    imax = 0;
    for (i=0; i<n1; i++) {
      if (i1[i] < 0) continue;
      imin = PetscMin(imin,i1[i]);
      imax = PetscMax(imax,i1[i]);
    }
  } else imin = imax = 0;

  PetscCall(PetscBTCreate(imax-imin,&mask));
  /* Put the values from is1 */
  for (i=0; i<n1; i++) {
    if (i1[i] < 0) continue;
    PetscCall(PetscBTSet(mask,i1[i] - imin));
  }
  PetscCall(ISRestoreIndices(is1,&i1));
  /* Remove the values from is2 */
  PetscCall(ISGetIndices(is2,&i2));
  PetscCall(ISGetLocalSize(is2,&n2));
  for (i=0; i<n2; i++) {
    if (i2[i] < imin || i2[i] > imax) continue;
    PetscCall(PetscBTClear(mask,i2[i] - imin));
  }
  PetscCall(ISRestoreIndices(is2,&i2));

  /* Count the number in the difference */
  nout = 0;
  for (i=0; i<imax-imin+1; i++) {
    if (PetscBTLookup(mask,i)) nout++;
  }

  /* create the new IS containing the difference */
  PetscCall(PetscMalloc1(nout,&iout));
  nout = 0;
  for (i=0; i<imax-imin+1; i++) {
    if (PetscBTLookup(mask,i)) iout[nout++] = i + imin;
  }
  PetscCall(PetscObjectGetComm((PetscObject)is1,&comm));
  PetscCall(ISCreateGeneral(comm,nout,iout,PETSC_OWN_POINTER,isout));

  PetscCall(PetscBTDestroy(&mask));
  PetscFunctionReturn(0);
}

/*@
   ISSum - Computes the sum (union) of two index sets.

   Only sequential version (at the moment)

   Input Parameters:
+  is1 - index set to be extended
-  is2 - index values to be added

   Output Parameter:
.   is3 - the sum; this can not be is1 or is2

   Notes:
   If n1 and n2 are the sizes of the sets, this takes O(n1+n2) time;

   Both index sets need to be sorted on input.

   Level: intermediate

.seealso: `ISDestroy()`, `ISView()`, `ISDifference()`, `ISExpand()`

@*/
PetscErrorCode  ISSum(IS is1,IS is2,IS *is3)
{
  MPI_Comm       comm;
  PetscBool      f;
  PetscMPIInt    size;
  const PetscInt *i1,*i2;
  PetscInt       n1,n2,n3, p1,p2, *iout;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_CLASSID,1);
  PetscValidHeaderSpecific(is2,IS_CLASSID,2);
  PetscCall(PetscObjectGetComm((PetscObject)(is1),&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCheck(size<=1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Currently only for uni-processor IS");

  PetscCall(ISSorted(is1,&f));
  PetscCheck(f,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Arg 1 is not sorted");
  PetscCall(ISSorted(is2,&f));
  PetscCheck(f,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Arg 2 is not sorted");

  PetscCall(ISGetLocalSize(is1,&n1));
  PetscCall(ISGetLocalSize(is2,&n2));
  if (!n2) {
    PetscCall(ISDuplicate(is1,is3));
    PetscFunctionReturn(0);
  }
  PetscCall(ISGetIndices(is1,&i1));
  PetscCall(ISGetIndices(is2,&i2));

  p1 = 0; p2 = 0; n3 = 0;
  do {
    if (p1==n1) { /* cleanup for is2 */ n3 += n2-p2; break;
    } else {
      while (p2<n2 && i2[p2]<i1[p1]) {
        n3++; p2++;
      }
      if (p2==n2) {
        /* cleanup for is1 */
        n3 += n1-p1; break;
      } else {
        if (i2[p2]==i1[p1]) { n3++; p1++; p2++; }
      }
    }
    if (p2==n2) {
      /* cleanup for is1 */
      n3 += n1-p1; break;
    } else {
      while (p1<n1 && i1[p1]<i2[p2]) {
        n3++; p1++;
      }
      if (p1==n1) {
        /* clean up for is2 */
        n3 += n2-p2; break;
      } else {
        if (i1[p1]==i2[p2]) { n3++; p1++; p2++; }
      }
    }
  } while (p1<n1 || p2<n2);

  if (n3==n1) { /* no new elements to be added */
    PetscCall(ISRestoreIndices(is1,&i1));
    PetscCall(ISRestoreIndices(is2,&i2));
    PetscCall(ISDuplicate(is1,is3));
    PetscFunctionReturn(0);
  }
  PetscCall(PetscMalloc1(n3,&iout));

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
        if (i2[p2]==i1[p1]) { iout[n3++] = i1[p1++]; p2++; }
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
        if (i1[p1]==i2[p2]) { iout[n3++] = i1[p1++]; p2++; }
      }
    }
  } while (p1<n1 || p2<n2);

  PetscCall(ISRestoreIndices(is1,&i1));
  PetscCall(ISRestoreIndices(is2,&i2));
  PetscCall(ISCreateGeneral(comm,n3,iout,PETSC_OWN_POINTER,is3));
  PetscFunctionReturn(0);
}

/*@
   ISExpand - Computes the union of two index sets, by concatenating 2 lists and
   removing duplicates.

   Collective on IS

   Input Parameters:
+  is1 - first index set
-  is2 - index values to be added

   Output Parameters:
.  isout - is1 + is2 The index set is2 is appended to is1 removing duplicates

   Notes:
   Negative values are removed from the lists. This requires O(imax-imin)
   memory and O(imax-imin) work, where imin and imax are the bounds on the
   indices in is1 and is2.

   The IS's do not need to be sorted.

   Level: intermediate

.seealso: `ISDestroy()`, `ISView()`, `ISDifference()`, `ISSum()`

@*/
PetscErrorCode ISExpand(IS is1,IS is2,IS *isout)
{
  PetscInt       i,n1,n2,imin,imax,nout,*iout;
  const PetscInt *i1,*i2;
  PetscBT        mask;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (is1) PetscValidHeaderSpecific(is1,IS_CLASSID,1);
  if (is2) PetscValidHeaderSpecific(is2,IS_CLASSID,2);
  PetscValidPointer(isout,3);

  PetscCheck(is1 || is2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Both arguments cannot be NULL");
  if (!is1) {PetscCall(ISDuplicate(is2, isout));PetscFunctionReturn(0);}
  if (!is2) {PetscCall(ISDuplicate(is1, isout));PetscFunctionReturn(0);}
  PetscCall(ISGetIndices(is1,&i1));
  PetscCall(ISGetLocalSize(is1,&n1));
  PetscCall(ISGetIndices(is2,&i2));
  PetscCall(ISGetLocalSize(is2,&n2));

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
  } else imin = imax = 0;

  PetscCall(PetscMalloc1(n1+n2,&iout));
  nout = 0;
  PetscCall(PetscBTCreate(imax-imin,&mask));
  /* Put the values from is1 */
  for (i=0; i<n1; i++) {
    if (i1[i] < 0) continue;
    if (!PetscBTLookupSet(mask,i1[i] - imin)) iout[nout++] = i1[i];
  }
  PetscCall(ISRestoreIndices(is1,&i1));
  /* Put the values from is2 */
  for (i=0; i<n2; i++) {
    if (i2[i] < 0) continue;
    if (!PetscBTLookupSet(mask,i2[i] - imin)) iout[nout++] = i2[i];
  }
  PetscCall(ISRestoreIndices(is2,&i2));

  /* create the new IS containing the sum */
  PetscCall(PetscObjectGetComm((PetscObject)is1,&comm));
  PetscCall(ISCreateGeneral(comm,nout,iout,PETSC_OWN_POINTER,isout));

  PetscCall(PetscBTDestroy(&mask));
  PetscFunctionReturn(0);
}

/*@
   ISIntersect - Computes the intersection of two index sets, by sorting and comparing.

   Collective on IS

   Input Parameters:
+  is1 - first index set
-  is2 - second index set

   Output Parameters:
.  isout - the sorted intersection of is1 and is2

   Notes:
   Negative values are removed from the lists. This requires O(min(is1,is2))
   memory and O(max(is1,is2)log(max(is1,is2))) work

   The IS's do not need to be sorted.

   Level: intermediate

.seealso: `ISDestroy()`, `ISView()`, `ISDifference()`, `ISSum()`, `ISExpand()`
@*/
PetscErrorCode ISIntersect(IS is1,IS is2,IS *isout)
{
  PetscInt       i,n1,n2,nout,*iout;
  const PetscInt *i1,*i2;
  IS             is1sorted = NULL, is2sorted = NULL;
  PetscBool      sorted, lsorted;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_CLASSID,1);
  PetscValidHeaderSpecific(is2,IS_CLASSID,2);
  PetscCheckSameComm(is1,1,is2,2);
  PetscValidPointer(isout,3);
  PetscCall(PetscObjectGetComm((PetscObject)is1,&comm));

  PetscCall(ISGetLocalSize(is1,&n1));
  PetscCall(ISGetLocalSize(is2,&n2));
  if (n1 < n2) {
    IS       tempis = is1;
    PetscInt ntemp = n1;

    is1 = is2;
    is2 = tempis;
    n1  = n2;
    n2  = ntemp;
  }
  PetscCall(ISSorted(is1,&lsorted));
  PetscCall(MPIU_Allreduce(&lsorted,&sorted,1,MPIU_BOOL,MPI_LAND,comm));
  if (!sorted) {
    PetscCall(ISDuplicate(is1,&is1sorted));
    PetscCall(ISSort(is1sorted));
    PetscCall(ISGetIndices(is1sorted,&i1));
  } else {
    is1sorted = is1;
    PetscCall(PetscObjectReference((PetscObject)is1));
    PetscCall(ISGetIndices(is1,&i1));
  }
  PetscCall(ISSorted(is2,&lsorted));
  PetscCall(MPIU_Allreduce(&lsorted,&sorted,1,MPIU_BOOL,MPI_LAND,comm));
  if (!sorted) {
    PetscCall(ISDuplicate(is2,&is2sorted));
    PetscCall(ISSort(is2sorted));
    PetscCall(ISGetIndices(is2sorted,&i2));
  } else {
    is2sorted = is2;
    PetscCall(PetscObjectReference((PetscObject)is2));
    PetscCall(ISGetIndices(is2,&i2));
  }

  PetscCall(PetscMalloc1(n2,&iout));

  for (i = 0, nout = 0; i < n2; i++) {
    PetscInt key = i2[i];
    PetscInt loc;

    PetscCall(ISLocate(is1sorted,key,&loc));
    if (loc >= 0) {
      if (!nout || iout[nout-1] < key) {
        iout[nout++] = key;
      }
    }
  }
  PetscCall(PetscRealloc(nout*sizeof(PetscInt),&iout));

  /* create the new IS containing the sum */
  PetscCall(ISCreateGeneral(comm,nout,iout,PETSC_OWN_POINTER,isout));

  PetscCall(ISRestoreIndices(is2sorted,&i2));
  PetscCall(ISDestroy(&is2sorted));
  PetscCall(ISRestoreIndices(is1sorted,&i1));
  PetscCall(ISDestroy(&is1sorted));
  PetscFunctionReturn(0);
}

PetscErrorCode ISIntersect_Caching_Internal(IS is1, IS is2, IS *isect)
{
  PetscFunctionBegin;
  *isect = NULL;
  if (is2 && is1) {
    char           composeStr[33] = {0};
    PetscObjectId  is2id;

    PetscCall(PetscObjectGetId((PetscObject)is2,&is2id));
    PetscCall(PetscSNPrintf(composeStr,32,"ISIntersect_Caching_%" PetscInt64_FMT,is2id));
    PetscCall(PetscObjectQuery((PetscObject) is1, composeStr, (PetscObject *) isect));
    if (*isect == NULL) {
      PetscCall(ISIntersect(is1, is2, isect));
      PetscCall(PetscObjectCompose((PetscObject) is1, composeStr, (PetscObject) *isect));
    } else {
      PetscCall(PetscObjectReference((PetscObject) *isect));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   ISConcatenate - Forms a new IS by locally concatenating the indices from an IS list without reordering.

   Collective.

   Input Parameters:
+  comm    - communicator of the concatenated IS.
.  len     - size of islist array (nonnegative)
-  islist  - array of index sets

   Output Parameters:
.  isout   - The concatenated index set; empty, if len == 0.

   Notes:
    The semantics of calling this on comm imply that the comms of the members if islist also contain this rank.

   Level: intermediate

.seealso: `ISDifference()`, `ISSum()`, `ISExpand()`

@*/
PetscErrorCode ISConcatenate(MPI_Comm comm, PetscInt len, const IS islist[], IS *isout)
{
  PetscInt i,n,N;
  const PetscInt *iidx;
  PetscInt *idx;

  PetscFunctionBegin;
  PetscValidPointer(islist,3);
  if (PetscDefined(USE_DEBUG)) {
    for (i = 0; i < len; ++i) if (islist[i]) PetscValidHeaderSpecific(islist[i], IS_CLASSID, 3);
  }
  PetscValidPointer(isout, 4);
  if (!len) {
    PetscCall(ISCreateStride(comm, 0,0,0, isout));
    PetscFunctionReturn(0);
  }
  PetscCheck(len >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Negative array length: %" PetscInt_FMT, len);
  N = 0;
  for (i = 0; i < len; ++i) {
    if (islist[i]) {
      PetscCall(ISGetLocalSize(islist[i], &n));
      N   += n;
    }
  }
  PetscCall(PetscMalloc1(N, &idx));
  N = 0;
  for (i = 0; i < len; ++i) {
    if (islist[i]) {
      PetscCall(ISGetLocalSize(islist[i], &n));
      PetscCall(ISGetIndices(islist[i], &iidx));
      PetscCall(PetscArraycpy(idx+N,iidx, n));
      PetscCall(ISRestoreIndices(islist[i], &iidx));
      N   += n;
    }
  }
  PetscCall(ISCreateGeneral(comm, N, idx, PETSC_OWN_POINTER, isout));
  PetscFunctionReturn(0);
}

/*@
   ISListToPair     -    convert an IS list to a pair of ISs of equal length defining an equivalent integer multimap.
                        Each IS on the input list is assigned an integer j so that all of the indices of that IS are
                        mapped to j.

  Collective.

  Input arguments:
+ comm    -  MPI_Comm
. listlen -  IS list length
- islist  -  IS list

  Output arguments:
+ xis -  domain IS
- yis -  range  IS

  Level: advanced

  Notes:
  The global integers assigned to the ISs of the local input list might not correspond to the
  local numbers of the ISs on that list, but the two *orderings* are the same: the global
  integers assigned to the ISs on the local list form a strictly increasing sequence.

  The ISs on the input list can belong to subcommunicators of comm, and the subcommunicators
  on the input IS list are assumed to be in a "deadlock-free" order.

  Local lists of PetscObjects (or their subcommes) on a comm are "deadlock-free" if subcomm1
  preceeds subcomm2 on any local list, then it preceeds subcomm2 on all ranks.
  Equivalently, the local numbers of the subcomms on each local list are drawn from some global
  numbering. This is ensured, for example, by ISPairToList().

.seealso `ISPairToList()`
@*/
PetscErrorCode ISListToPair(MPI_Comm comm, PetscInt listlen, IS islist[], IS *xis, IS *yis)
{
  PetscInt       ncolors, *colors,i, leni,len,*xinds, *yinds,k,j;
  const PetscInt *indsi;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(listlen, &colors));
  PetscCall(PetscObjectsListGetGlobalNumbering(comm, listlen, (PetscObject*)islist,&ncolors, colors));
  len  = 0;
  for (i = 0; i < listlen; ++i) {
    PetscCall(ISGetLocalSize(islist[i], &leni));
    len += leni;
  }
  PetscCall(PetscMalloc1(len, &xinds));
  PetscCall(PetscMalloc1(len, &yinds));
  k    = 0;
  for (i = 0; i < listlen; ++i) {
    PetscCall(ISGetLocalSize(islist[i], &leni));
    PetscCall(ISGetIndices(islist[i],&indsi));
    for (j = 0; j < leni; ++j) {
      xinds[k] = indsi[j];
      yinds[k] = colors[i];
      ++k;
    }
  }
  PetscCall(PetscFree(colors));
  PetscCall(ISCreateGeneral(comm,len,xinds,PETSC_OWN_POINTER,xis));
  PetscCall(ISCreateGeneral(comm,len,yinds,PETSC_OWN_POINTER,yis));
  PetscFunctionReturn(0);
}

/*@
   ISPairToList   -   convert an IS pair encoding an integer map to a list of ISs.
                     Each IS on the output list contains the preimage for each index on the second input IS.
                     The ISs on the output list are constructed on the subcommunicators of the input IS pair.
                     Each subcommunicator corresponds to the preimage of some index j -- this subcomm contains
                     exactly the ranks that assign some indices i to j.  This is essentially the inverse of
                     ISListToPair().

  Collective on indis.

  Input arguments:
+ xis -  domain IS
- yis -  range IS

  Output arguments:
+ listlen -  length of islist
- islist  -  list of ISs breaking up indis by color

  Note:
    xis and yis must be of the same length and have congruent communicators.

    The resulting ISs have subcommunicators in a "deadlock-free" order (see ISListToPair()).

  Level: advanced

.seealso `ISListToPair()`
 @*/
PetscErrorCode ISPairToList(IS xis, IS yis, PetscInt *listlen, IS **islist)
{
  IS             indis = xis, coloris = yis;
  PetscInt       *inds, *colors, llen, ilen, lstart, lend, lcount,l;
  PetscMPIInt    rank, size, llow, lhigh, low, high,color,subsize;
  const PetscInt *ccolors, *cinds;
  MPI_Comm       comm, subcomm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(xis, IS_CLASSID, 1);
  PetscValidHeaderSpecific(yis, IS_CLASSID, 2);
  PetscCheckSameComm(xis,1,yis,2);
  PetscValidIntPointer(listlen,3);
  PetscValidPointer(islist,4);
  PetscCall(PetscObjectGetComm((PetscObject)xis,&comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_rank(comm, &size));
  /* Extract, copy and sort the local indices and colors on the color. */
  PetscCall(ISGetLocalSize(coloris, &llen));
  PetscCall(ISGetLocalSize(indis,   &ilen));
  PetscCheck(llen == ilen,comm, PETSC_ERR_ARG_SIZ, "Incompatible IS sizes: %" PetscInt_FMT " and %" PetscInt_FMT, ilen, llen);
  PetscCall(ISGetIndices(coloris, &ccolors));
  PetscCall(ISGetIndices(indis, &cinds));
  PetscCall(PetscMalloc2(ilen,&inds,llen,&colors));
  PetscCall(PetscArraycpy(inds,cinds,ilen));
  PetscCall(PetscArraycpy(colors,ccolors,llen));
  PetscCall(PetscSortIntWithArray(llen, colors, inds));
  /* Determine the global extent of colors. */
  llow   = 0; lhigh  = -1;
  lstart = 0; lcount = 0;
  while (lstart < llen) {
    lend = lstart+1;
    while (lend < llen && colors[lend] == colors[lstart]) ++lend;
    llow  = PetscMin(llow,colors[lstart]);
    lhigh = PetscMax(lhigh,colors[lstart]);
    ++lcount;
  }
  PetscCall(MPIU_Allreduce(&llow,&low,1,MPI_INT,MPI_MIN,comm));
  PetscCall(MPIU_Allreduce(&lhigh,&high,1,MPI_INT,MPI_MAX,comm));
  *listlen = 0;
  if (low <= high) {
    if (lcount > 0) {
      *listlen = lcount;
      if (!*islist) {
        PetscCall(PetscMalloc1(lcount, islist));
      }
    }
    /*
     Traverse all possible global colors, and participate in the subcommunicators
     for the locally-supported colors.
     */
    lcount = 0;
    lstart = 0; lend = 0;
    for (l = low; l <= high; ++l) {
      /*
       Find the range of indices with the same color, which is not smaller than l.
       Observe that, since colors is sorted, and is a subsequence of [low,high],
       as soon as we find a new color, it is >= l.
       */
      if (lstart < llen) {
        /* The start of the next locally-owned color is identified.  Now look for the end. */
        if (lstart == lend) {
          lend = lstart+1;
          while (lend < llen && colors[lend] == colors[lstart]) ++lend;
        }
        /* Now check whether the identified color segment matches l. */
        PetscCheck(colors[lstart] >= l,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Locally owned color %" PetscInt_FMT " at location %" PetscInt_FMT " is < than the next global color %" PetscInt_FMT, colors[lstart], lcount, l);
      }
      color = (PetscMPIInt)(colors[lstart] == l);
      /* Check whether a proper subcommunicator exists. */
      PetscCall(MPIU_Allreduce(&color,&subsize,1,MPI_INT,MPI_SUM,comm));

      if (subsize == 1) subcomm = PETSC_COMM_SELF;
      else if (subsize == size) subcomm = comm;
      else {
        /* a proper communicator is necessary, so we create it. */
        PetscCallMPI(MPI_Comm_split(comm, color, rank, &subcomm));
      }
      if (colors[lstart] == l) {
        /* If we have l among the local colors, we create an IS to hold the corresponding indices. */
        PetscCall(ISCreateGeneral(subcomm, lend-lstart,inds+lstart,PETSC_COPY_VALUES,*islist+lcount));
        /* Position lstart at the beginning of the next local color. */
        lstart = lend;
        /* Increment the counter of the local colors split off into an IS. */
        ++lcount;
      }
      if (subsize > 0 && subsize < size) {
        /*
         Irrespective of color, destroy the split off subcomm:
         a subcomm used in the IS creation above is duplicated
         into a proper PETSc comm.
         */
        PetscCallMPI(MPI_Comm_free(&subcomm));
      }
    } /* for (l = low; l < high; ++l) */
  } /* if (low <= high) */
  PetscCall(PetscFree2(inds,colors));
  PetscFunctionReturn(0);
}

/*@
   ISEmbed   -   embed IS a into IS b by finding the locations in b that have the same indices as in a.
                 If c is the IS of these locations, we have a = b*c, regarded as a composition of the
                 corresponding ISLocalToGlobalMaps.

  Not collective.

  Input arguments:
+ a    -  IS to embed
. b    -  IS to embed into
- drop -  flag indicating whether to drop a's indices that are not in b.

  Output arguments:
. c    -  local embedding indices

  Note:
  If some of a's global indices are not among b's indices the embedding is impossible.  The local indices of a
  corresponding to these global indices are either mapped to -1 (if !drop) or are omitted (if drop).  In the former
  case the size of c is that same as that of a, in the latter case c's size may be smaller.

  The resulting IS is sequential, since the index substition it encodes is purely local.

  Level: advanced

.seealso `ISLocalToGlobalMapping`
 @*/
PetscErrorCode ISEmbed(IS a, IS b, PetscBool drop, IS *c)
{
  ISLocalToGlobalMapping     ltog;
  ISGlobalToLocalMappingMode gtoltype = IS_GTOLM_DROP;
  PetscInt                   alen, clen, *cindices, *cindices2;
  const PetscInt             *aindices;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(a, IS_CLASSID, 1);
  PetscValidHeaderSpecific(b, IS_CLASSID, 2);
  PetscValidPointer(c,4);
  PetscCall(ISLocalToGlobalMappingCreateIS(b, &ltog));
  PetscCall(ISGetLocalSize(a, &alen));
  PetscCall(ISGetIndices(a, &aindices));
  PetscCall(PetscMalloc1(alen, &cindices));
  if (!drop) gtoltype = IS_GTOLM_MASK;
  PetscCall(ISGlobalToLocalMappingApply(ltog,gtoltype,alen,aindices,&clen,cindices));
  PetscCall(ISLocalToGlobalMappingDestroy(&ltog));
  if (clen != alen) {
    cindices2 = cindices;
    PetscCall(PetscMalloc1(clen, &cindices));
    PetscCall(PetscArraycpy(cindices,cindices2,clen));
    PetscCall(PetscFree(cindices2));
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,clen,cindices,PETSC_OWN_POINTER,c));
  PetscFunctionReturn(0);
}

/*@
  ISSortPermutation  -  calculate the permutation of the indices into a nondecreasing order.

  Not collective.

  Input arguments:
+ f      -  IS to sort
- always -  build the permutation even when f's indices are nondecreasing.

  Output argument:
. h    -  permutation or NULL, if f is nondecreasing and always == PETSC_FALSE.

  Note: Indices in f are unchanged. f[h[i]] is the i-th smallest f index.
        If always == PETSC_FALSE, an extra check is peformed to see whether
        the f indices are nondecreasing. h is built on PETSC_COMM_SELF, since
        the permutation has a local meaning only.

  Level: advanced

.seealso `ISLocalToGlobalMapping`, `ISSort()`
 @*/
PetscErrorCode ISSortPermutation(IS f,PetscBool always,IS *h)
{
  const PetscInt  *findices;
  PetscInt        fsize,*hindices,i;
  PetscBool       isincreasing;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(f,IS_CLASSID,1);
  PetscValidPointer(h,3);
  PetscCall(ISGetLocalSize(f,&fsize));
  PetscCall(ISGetIndices(f,&findices));
  *h = NULL;
  if (!always) {
    isincreasing = PETSC_TRUE;
    for (i = 1; i < fsize; ++i) {
      if (findices[i] <= findices[i-1]) {
        isincreasing = PETSC_FALSE;
        break;
      }
    }
    if (isincreasing) {
      PetscCall(ISRestoreIndices(f,&findices));
      PetscFunctionReturn(0);
    }
  }
  PetscCall(PetscMalloc1(fsize,&hindices));
  for (i = 0; i < fsize; ++i) hindices[i] = i;
  PetscCall(PetscSortIntWithPermutation(fsize,findices,hindices));
  PetscCall(ISRestoreIndices(f,&findices));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,fsize,hindices,PETSC_OWN_POINTER,h));
  PetscCall(ISSetInfo(*h,IS_PERMUTATION,IS_LOCAL,PETSC_FALSE,PETSC_TRUE));
  PetscFunctionReturn(0);
}
