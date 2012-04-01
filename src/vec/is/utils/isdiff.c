
#include <petscis.h>                    /*I "petscis.h"  I*/
#include <petscbt.h>

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

.seealso: ISDestroy(), ISView(), ISSum(), ISExpand()

@*/
PetscErrorCode  ISDifference(IS is1,IS is2,IS *isout)
{
  PetscErrorCode ierr;
  PetscInt       i,n1,n2,imin,imax,nout,*iout;
  const PetscInt *i1,*i2;
  PetscBT        mask;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_CLASSID,1);
  PetscValidHeaderSpecific(is2,IS_CLASSID,2);
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
  ierr = PetscBTCreate(imax-imin,&mask);CHKERRQ(ierr);
  /* Put the values from is1 */
  for (i=0; i<n1; i++) {
    if (i1[i] < 0) continue;
    ierr = PetscBTSet(mask,i1[i] - imin);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(is1,&i1);CHKERRQ(ierr);
  /* Remove the values from is2 */
  ierr = ISGetIndices(is2,&i2);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is2,&n2);CHKERRQ(ierr);
  for (i=0; i<n2; i++) {
    if (i2[i] < imin || i2[i] > imax) continue;
    ierr = PetscBTClear(mask,i2[i] - imin);CHKERRQ(ierr);
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
  ierr = ISCreateGeneral(comm,nout,iout,PETSC_OWN_POINTER,isout);CHKERRQ(ierr);

  ierr = PetscBTDestroy(&mask);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSum"
/*@
   ISSum - Computes the sum (union) of two index sets.

   Only sequential version (at the moment)

   Input Parameter:
+  is1 - index set to be extended
-  is2 - index values to be added

   Output Parameter:
.   is3 - the sum; this can not be is1 or is2

   Notes:
   If n1 and n2 are the sizes of the sets, this takes O(n1+n2) time;

   Both index sets need to be sorted on input.

   Level: intermediate

.seealso: ISDestroy(), ISView(), ISDifference(), ISExpand()

   Concepts: index sets^union
   Concepts: IS^union

@*/
PetscErrorCode  ISSum(IS is1,IS is2,IS *is3)
{
  MPI_Comm       comm;
  PetscBool      f;
  PetscMPIInt    size;
  const PetscInt *i1,*i2;
  PetscInt       n1,n2,n3, p1,p2, *iout;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_CLASSID,1);
  PetscValidHeaderSpecific(is2,IS_CLASSID,2);
  ierr = PetscObjectGetComm((PetscObject)(is1),&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size>1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Currently only for uni-processor IS");

  ierr = ISSorted(is1,&f);CHKERRQ(ierr);
  if (!f) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Arg 1 is not sorted");
  ierr = ISSorted(is2,&f);CHKERRQ(ierr);
  if (!f) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Arg 2 is not sorted");

  ierr = ISGetLocalSize(is1,&n1);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is2,&n2);CHKERRQ(ierr);
  if (!n2) PetscFunctionReturn(0);
  ierr = ISGetIndices(is1,&i1);CHKERRQ(ierr);
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
    ierr = ISRestoreIndices(is1,&i1);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is2,&i2);CHKERRQ(ierr);
    ierr = ISDuplicate(is1,is3);CHKERRQ(ierr);
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

  ierr = ISRestoreIndices(is1,&i1);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is2,&i2);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n3,iout,PETSC_OWN_POINTER,is3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISExpand"
/*@
   ISExpand - Computes the union of two index sets, by concatenating 2 lists and
   removing duplicates.

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

   The IS's do not need to be sorted.

   Level: intermediate

.seealso: ISDestroy(), ISView(), ISDifference(), ISSum()

   Concepts: index sets^difference
   Concepts: IS^difference

@*/
PetscErrorCode ISExpand(IS is1,IS is2,IS *isout)
{
  PetscErrorCode ierr;
  PetscInt       i,n1,n2,imin,imax,nout,*iout;
  const PetscInt *i1,*i2;
  PetscBT        mask;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_CLASSID,1);
  PetscValidHeaderSpecific(is2,IS_CLASSID,2);
  PetscValidPointer(isout,3);

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
  ierr = PetscMalloc((n1+n2)*sizeof(PetscInt),&iout);CHKERRQ(ierr);
  nout = 0;
  ierr = PetscBTCreate(imax-imin,&mask);CHKERRQ(ierr);
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
  ierr = ISCreateGeneral(comm,nout,iout,PETSC_OWN_POINTER,isout);CHKERRQ(ierr);

  ierr = PetscBTDestroy(&mask);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISConcatenate"
/*@
   ISConcatenate - Forms a new IS by locally concatenating the indices from an IS list without reordering.
 

   Collective on comm.

   Input Parameter:
+  comm    - communicator of the concatenated IS.
.  len     - size of islist array (nonnegative)
-  islist  - array of index sets



   Output Parameters:
.  isout   - The concatenated index set; empty, if len == 0.

   Notes: The semantics of calling this on comm imply that the comms of the members if islist also contain this rank.

   Level: intermediate

.seealso: ISDifference(), ISSum(), ISExpand()

   Concepts: index sets^concatenation
   Concepts: IS^concatenation

@*/
PetscErrorCode ISConcatenate(MPI_Comm comm, PetscInt len, const IS islist[], IS *isout)
{
  PetscErrorCode ierr;
  PetscInt i,n,N;
  const PetscInt *iidx;
  PetscInt *idx;

  PetscFunctionBegin;
  PetscValidPointer(islist,2);
#if defined(PETSC_USE_DEBUG)
  for(i = 0; i < len; ++i) {
    PetscValidHeaderSpecific(islist[i], IS_CLASSID, 1);
  }
#endif
  PetscValidPointer(isout, 5);
  if(!len) {
    ierr = ISCreateStride(comm, 0,0,0, isout); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if(len < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Negative array length: %D", len);
  N = 0; 
  for(i = 0; i < len; ++i) {
    ierr = ISGetLocalSize(islist[i], &n); CHKERRQ(ierr);
    N += n;
  }
  ierr = PetscMalloc(sizeof(PetscInt)*N, &idx); CHKERRQ(ierr);
  N = 0; 
  for(i = 0; i < len; ++i) {
    ierr = ISGetLocalSize(islist[i], &n); CHKERRQ(ierr);
    ierr = ISGetIndices(islist[i], &iidx); CHKERRQ(ierr);
    ierr = PetscMemcpy(idx+N,iidx, sizeof(PetscInt)*n); CHKERRQ(ierr);
    N += n;
  }
  ierr = ISCreateGeneral(comm, N, idx, PETSC_OWN_POINTER, isout); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   ISListToColoring     -    convert an IS list to a pair of ISs of equal length defining an equivalent  coloring.  
                          Each IS on the input list is assigned a color so that we have i and c in the same position 
                          of output ISs, if and only if i belongs to the input IS assigned to c.

  Collective on comm.

  Input arguments:
+ comm    -  MPI_Comm
. listlen -  IS list length
- islist  -  IS list

  Output arguments:
+ indis   -  IS of the indices found on the IS list
- coloris -  IS of colors

  Level: advanced

  Notes:
  The global colors assigned to the ISs of the local input list might not correspond to the
  local numbers of the ISs on that list, but the two *orderings* are the same: the global 
  colors assigned to the ISs on the local list form a strictly increasing sequence.

  The ISs on the input list can belong to subcommunicators of comm, and the subcommunicators 
  on the input IS list are assumed to be in a "deadlock-free" order.

  Local lists of PetscObjects (or their subcommes) on a comm are "deadlock-free" if subcomm1 
  preceeds subcomm2 on any local list, then it preceeds subcomm2 on all ranks.
  Equivalently, the local numbers of the subcomms on each local list are drawn from some global 
  numbering. This is ensured, for example, by ISColoringToList().

.seealso ISColoringToList()
@*/
#undef  __FUNCT__
#define __FUNCT__ "ISListToColoring"
PetscErrorCode ISListToColoring(MPI_Comm comm, PetscInt listlen, IS islist[], IS *indis, IS *coloris) 
{
  PetscErrorCode ierr;
  PetscMPIInt    rank, rankj,size,l,loffset=0,color;
  PetscInt       len,j,lenj,i,k,*inds,*colors;
  const PetscInt *indsj;
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &size); CHKERRQ(ierr);
  /* Order colors this way: the IS is colored by the IS's root -- the zeroth rank in the IS's communicator.
     Multiple ISs anchored at the same root are ordered consecutively, immediately following the ISs colored 
     by the preceeding ranks in the comm. 
   */
  /* Count the number of ISs anchored here, and add up the IS lengths. */
  l = 0;
  len   = 0;
  for(j = 0; j < listlen; ++j) {
    ierr = MPI_Comm_rank(((PetscObject)islist[j])->comm, &rankj); CHKERRQ(ierr);
    if(!rankj) ++l;
    ierr = ISGetLocalSize(islist[j],&lenj);                       CHKERRQ(ierr);
    len += lenj;
  }
  ierr = MPI_Scan(&l,&loffset,1,MPI_INT, MPI_SUM,comm); CHKERRQ(ierr);
  /* Now we can assign colors to individual ISs, broadcast them within each IS comm, and use them to color the IS indices */
  ierr = PetscMalloc(sizeof(PetscInt)*len, &inds);   CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*len, &colors); CHKERRQ(ierr);
  l = 0;
  i = 0;
  for(j = 0; j < listlen; ++j) {
    color = l + loffset; /* This is only meaningful for the j-th root; the rest get this value via MPI_Bcast below. */
    ierr = MPI_Bcast(&color,1,MPI_INT,0,((PetscObject)islist[j])->comm); CHKERRQ(ierr);
    ierr = ISGetLocalSize(islist[j], &lenj);                             CHKERRQ(ierr);
    ierr = ISGetIndices(islist[j],&indsj);                               CHKERRQ(ierr);
    for(k = 0; k < lenj; ++k) {
      inds[i]   = indsj[k];
      colors[i] = color; 
      ++i;
    }
  }
  /* Now create the indices and colors IS. */
  ierr = ISCreateGeneral(comm,len,inds,PETSC_OWN_POINTER,indis);     CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,len,colors,PETSC_OWN_POINTER,coloris); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@
   ISColoringToList   -   convert an IS pair encoding a coloring of the indices (first IS) by integers 
                          (second IS), to a list of ISs.  Each IS on the output list contains the indices (from 
                          the first input IS) colored by the same integer (from the second input IS).  The ISs 
                          on the output list are constructed on the subcommunicators of the input IS pair.
                          Each subcommunicator corresponds to a color -- contains exactly the ranks that contain the 
                          corresponding color in the second input IS.  This is essentially the inverse of ISListToColoring().

  Collective on indis.

  Input arguments:
+ indis   -  IS of indices
- coloris -  IS of colors

  Output arguments:
+ listlen -  length of islist
- islist  -  list of ISs breaking up indis by color

  Note: 
+ indis and coloris must be of the same length and have congruent communicators.  
- The resulting ISs have subcommunicators in a "deadlock-free" order (see ISListToColoring()).

  Level: advanced

.seealso ISListToColoring()
 @*/
#undef  __FUNCT__
#define __FUNCT__ "ISColoringToList"
PetscErrorCode ISColoringToList(IS indis, IS coloris, PetscInt *listlen, IS **islist) 
{
  PetscErrorCode ierr;
  PetscInt *inds, *colors, llen, ilen, lstart, lend, lcount,l;
  PetscMPIInt rank, size, llow, lhigh, low, high,color,subsize;
  const PetscInt *ccolors, *cinds;
  MPI_Comm comm, subcomm;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(indis, IS_CLASSID, 1);
  PetscValidHeaderSpecific(coloris, IS_CLASSID, 2);
  PetscCheckSameComm(indis,1,coloris,2);
  PetscValidIntPointer(listlen,3);
  PetscValidPointer(islist,4);
  comm = ((PetscObject)indis)->comm;
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &size); CHKERRQ(ierr);
  /* Extract, copy and sort the local indices and colors on the color. */
  ierr = ISGetLocalSize(coloris, &llen);  CHKERRQ(ierr);
  ierr = ISGetLocalSize(indis, &ilen);  CHKERRQ(ierr);
  if(llen != ilen) SETERRQ2(comm, PETSC_ERR_ARG_SIZ, "Incompatible IS sizes: %D and %D", ilen, llen);
  ierr = ISGetIndices(coloris, &ccolors); CHKERRQ(ierr);
  ierr = ISGetIndices(indis, &cinds);     CHKERRQ(ierr);
  ierr = PetscMalloc2(ilen,PetscInt,&inds,llen,PetscInt,&colors); CHKERRQ(ierr);
  ierr = PetscMemcpy(inds,cinds,ilen*sizeof(PetscInt));     CHKERRQ(ierr);
  ierr = PetscMemcpy(colors,ccolors,llen*sizeof(PetscInt)); CHKERRQ(ierr);
  ierr = PetscSortIntWithArray(llen, colors, inds);         CHKERRQ(ierr);
  /* Determine the global extent of colors. */
  llow = 0; lhigh = -1;
  lstart = 0; lcount = 0;
  while(lstart < llen) {
    lend = lstart+1;
    while(lend < llen && colors[lend] == colors[lstart]) ++lend;
    llow = PetscMin(llow,colors[lstart]);
    lhigh = PetscMax(lhigh,colors[lstart]);
    ++lcount;
  }
  ierr = MPI_Allreduce(&llow,&low,1,MPI_INT,MPI_MIN,comm);   CHKERRQ(ierr);
  ierr = MPI_Allreduce(&lhigh,&high,1,MPI_INT,MPI_MAX,comm); CHKERRQ(ierr);
  *listlen = 0; 
  if(low <= high) {
    if(lcount > 0) {
      *listlen = lcount;
      if(!*islist) {
        ierr = PetscMalloc(sizeof(IS)*lcount, islist); CHKERRQ(ierr);
      }
    }
    /* 
     Traverse all possible global colors, and participate in the subcommunicators 
     for the locally-supported colors.
     */
    lcount   = 0;
    lstart   = 0; lend = 0;
    for(l = low; l <= high; ++l) {
      /* 
       Find the range of indices with the same color, which is not smaller than l. 
       Observe that, since colors is sorted, and is a subsequence of [low,high], 
       as soon as we find a new color, it is >= l.
       */
      if(lstart < llen) {
        /* The start of the next locally-owned color is identified.  Now look for the end. */
        if(lstart == lend) {
          lend = lstart+1;
          while(lend < llen && colors[lend] == colors[lstart]) ++lend;
        }
        /* Now check whether the identified color segment matches l. */
        if(colors[lstart] < l) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Locally owned color %D at location %D is < than the next global color %D", colors[lstart], lcount, l);
      }  
      color = (PetscMPIInt)(colors[lstart] == l);
      /* Check whether a proper subcommunicator exists. */
      ierr = MPI_Allreduce(&color,&subsize,1,MPI_INT,MPI_SUM,comm); CHKERRQ(ierr);
      
      if(subsize == 1) subcomm = PETSC_COMM_SELF;
      else if(subsize == size) subcomm = comm;
      else {
        /* a proper communicator is necessary, so we create it. */
        ierr = MPI_Comm_split(comm, color, rank, &subcomm); CHKERRQ(ierr);
      }
      if(colors[lstart] == l) {
        /* If we have l among the local colors, we create an IS to hold the corresponding indices. */
        ierr = ISCreateGeneral(subcomm, lend-lstart,inds+lstart,PETSC_COPY_VALUES,*islist+lcount); CHKERRQ(ierr);
        /* Position lstart at the beginning of the next local color. */
        lstart = lend;
        /* Increment the counter of the local colors split off into an IS. */
        ++lcount;
      }
      if(subsize > 0 && subsize < size) {
        /*  
         Irrespective of color, destroy the split off subcomm: 
         a subcomm used in the IS creation above is duplicated
         into a proper PETSc comm.
         */
        ierr = MPI_Comm_free(&subcomm); CHKERRQ(ierr);
      }
    }/* for(l = low; l < high; ++l) */
  }/* if(low <= high) */
  ierr = PetscFree2(inds,colors); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
