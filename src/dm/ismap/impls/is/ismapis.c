#define PETSCDM_DLL

#include "private/ismapimpl.h"  /*I "petscdm.h"  I*/

typedef struct {
  /* The following is used for mapping. */
  IS ix,iy;
  PetscInt *supp;    /* support        -- global input indices from ownership range with nonempty images */
  PetscInt m;        /* size of suppot -- number of input indices with nonempty images */
  PetscInt *image;   /* image          -- distinct global output indices */
  PetscInt n;        /* size of image -- number of distinct global output indices */
  PetscInt *ij;      /* concatenated local images of ALL local input elements (i.e., all indices from the local ownership range), sorted within each image */
  PetscInt *ijlen;   /* image segment boundaries for each local input index */
  PetscInt maxijlen; /* max image segment size */
  /* The following is used for binning. */
  PetscInt *offset, *count; 
} ISMapping_IS;


#undef  __FUNCT__ 
#define __FUNCT__ "ISMappingIS_LocateIndices"
PetscErrorCode ISMappingIS_LocateIndices(PetscInt tablen, const PetscInt table[], PetscInt inlen, const PetscInt inidx[], PetscInt *outlen, PetscInt outidx[], PetscBool drop)
{
  PetscInt low, high,j,lastj;
  PetscInt ind,last;
  PetscInt i, count;
  PetscFunctionBegin;
  low = 0, high = tablen;
  lastj = 0;
  last = table[0];
  count = 0;
  for(i = 0; i < inlen; ++i) {
    ind = inidx[i];
    if(!drop) outidx[i] = -1;
    if(i > 0) {
      /* last and j have valid previous values, that can be used to take advantage of the already known information about the table. */
      if(ind > last) { /* the lower bound is still valid, but the upper bound might not be. */
        /* 
         table is ordered, hence, is a subsequence of the integers. Thus, the distance between 
         ind and last in table is no greater than the distance between them within the integers: ind - last. 
         Therefore, high raised by ind-last is a valid upper bound on ind.
         */
        high = PetscMin(tablen, high+(ind-last));
        /* j is the largest table index not exceeding last; since ind > last, ind is located above j within the table */
        low = j;
      }
      if(ind < last) { /* The upper bound is still valid, but the lower bound might not be. */
        /* 
         table is ordered, hence, is a subsequence of the integers. Thus, the distance between 
         ind and last in image is no greater than the distance between them within the integers: last - ind. 
         Therefore, low lowered by ind-last is a valid upper bound on ind.
         */
        low = PetscMax(0,low+(ind-last));
        /* j is the latest table index not exceeding last; since ind < last, ind is located no higher than j within the table */
        high = j;
      }
    }/* if(i > 0) */
    last = ind;
    while(high - low > 5) {
      j = (high+low)/2;
      if(ind < table[j]) {
        high = j;
      }
      else {
        low = j;
      }
    }
    j = low;
    while(j < high && table[j] <= ind) {
      if(ind == table[j]) {
        if(drop) {
          outidx[count] = ind;
        }
        else {
          outidx[i] = ind;
        }
        ++count;
        break;
      }
      ++j;
    }
  }
  if(outlen) *outlen = count;
  PetscFunctionReturn(0);
}/* ISMappingIS_LocateIndices() */


/*
     Checks if any indices are within [imin,imax) and generate an error, if they are not and 
     if outOfBoundsError == PETSC_TRUE.  Return the result in flag.
 */
#undef __FUNCT__  
#define __FUNCT__ "ISMappingIS_CheckArrayRange"
static PetscErrorCode ISMappingIS_CheckArrayRange(PetscInt n, const PetscInt *idx, PetscInt imin, PetscInt imax, PetscBool outOfBoundsError, PetscBool *flag)
{
  PetscInt i;
  PetscBool inBounds = PETSC_TRUE;
  PetscFunctionBegin;
  
  for (i=0; i<n; i++) {
    if (idx[i] <  imin) {
      if(outOfBoundsError) {
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D at %D location is less than min %D",idx[i],i,imin);
      }
      inBounds = PETSC_FALSE;
      break;
    }
    if (idx[i] >= imax) {
      if(outOfBoundsError) {
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D at %D location greater than max %D",idx[i],i,imax);
      }
      inBounds = PETSC_FALSE;
      break;
    }
  }
  if(flag) *flag = inBounds;
  PetscFunctionReturn(0);
}

/*
     Checks if any indices are within [imin,imax) and generate an error, if they are not and 
     if outOfBoundsError == PETSC_TRUE.  Return the result in flag.
 */
#undef __FUNCT__  
#define __FUNCT__ "ISMappingIS_CheckISRange"
static PetscErrorCode ISMappingIS_CheckISRange(IS is, PetscInt imin, PetscInt imax, PetscBool outOfBoundsError, PetscBool *flag)
{
  PetscInt n;
  PetscBool inBounds = PETSC_TRUE, isstride;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  ierr = ISGetLocalSize(is, &n); CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)is,ISSTRIDE,&isstride); CHKERRQ(ierr);
  if(isstride) {
    PetscInt first, step, last;
    
    ierr = ISStrideGetInfo(is, &first, &step); CHKERRQ(ierr);
    last = first + step*n;
    if (first < imin || last < imin) {
      if(outOfBoundsError) 
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index smaller than min %D", imin);
      inBounds = PETSC_FALSE;
      goto functionend;
    }
    if (first >= imax || last >= imax) {
      if(outOfBoundsError) 
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index greater than max %D", imax);
      inBounds = PETSC_FALSE;
      goto functionend;
    }
  } else { /* not stride */
    const PetscInt *idx;
    ierr = ISGetIndices(is, &idx); CHKERRQ(ierr);
    ierr = ISMappingIS_CheckArrayRange(n,idx,imin,imax,outOfBoundsError,flag); CHKERRQ(ierr);
    ierr = ISRestoreIndices(is, &idx); CHKERRQ(ierr);
  }
  functionend:
  if(flag) *flag = inBounds;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingMapLocal_IS"
static PetscErrorCode ISMappingMapLocal_IS(ISMapping map, PetscInt insize, const PetscInt inidx[], const PetscScalar inval[], PetscInt *outsize, PetscInt outidx[], PetscScalar outval[], PetscInt offsets[], PetscBool drop) 
{
  ISMapping_IS *mapis = (ISMapping_IS*)map->data;
  PetscInt     i,j,k;
  PetscFunctionBegin;

  if(offsets) offsets[0] = 0;
  j = 0;
  for(i = 0; i < insize; ++i) {
    if((inidx[i] < 0 || inidx[i] > mapis->m)){
      if(!drop) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local index %D at %D out of range [%D,%D)", inidx[i], i, 0, mapis->m);
      if(offsets) offsets[i+1] = offsets[i];
      continue;
    }
    if(outidx || (inval && outval)) {
      for(k = mapis->ijlen[inidx[i]]; k < mapis->ijlen[inidx[i]+1]; ++k) {
        if(outidx)        outidx[j] = mapis->image[mapis->ij[k]];
        if(inval&&outval) outval[j] = inval[i];
        ++j;
      }
    }/* if(doApply) */
    else {
      j += mapis->ijlen[inidx[i]+1]-mapis->ijlen[inidx[i]];
    }
    if(offsets) offsets[i+1] = offsets[i] + (mapis->ijlen[inidx[i]+1]-mapis->ijlen[inidx[i]]);
  }/* for(i = 0; i < insize; ++i) */
  if(outsize) *outsize = j;
  PetscFunctionReturn(0);
}/* ISMappingMapLocal_IS() */

#undef __FUNCT__  
#define __FUNCT__ "ISMappingMap_IS"
static PetscErrorCode ISMappingMap_IS(ISMapping map, PetscInt insize, const PetscInt inidx[], const PetscScalar inval[], PetscInt *outsize, PetscInt outidx[], PetscScalar outval[], PetscInt offsets[], PetscBool drop) 
{
  PetscErrorCode ierr;
  ISMapping_IS *mapis = (ISMapping_IS*)map->data;
  PetscInt *inidx_loc;
  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(PetscInt)*insize, &inidx_loc); CHKERRQ(ierr);
  ierr = ISMappingIS_LocateIndices(mapis->m,mapis->supp,insize,inidx,PETSC_NULL,inidx_loc,PETSC_FALSE); CHKERRQ(ierr);
  ierr = ISMappingMap_IS(map,insize,inidx,inval,outsize,outidx,outval,offsets,drop);                    CHKERRQ(ierr);
  ierr = PetscFree(inidx_loc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingMap_IS() */


#undef __FUNCT__  
#define __FUNCT__ "ISMappingBinLocal_IS"
static PetscErrorCode ISMappingBinLocal_IS(ISMapping map, PetscInt insize, const PetscInt inidx[], const PetscScalar inval[], PetscInt *outsize, PetscInt outidx[], PetscScalar outval[], PetscInt offset[], PetscBool drop) 
{
  PetscErrorCode ierr;
  ISMapping_IS *mapis = (ISMapping_IS*)map->data;
  PetscInt      i,j,k;
  PetscInt      *binoff, *bincount;
  PetscFunctionBegin;
  /* We'll need to count contributions to each "bin" and the offset of each bin in outidx, etc. */
  /* Allocate the bin offset array, if necessary. */
  if(!offset) {
    if(!mapis->offset) {
      ierr = PetscMalloc((mapis->n+1)*sizeof(PetscInt), &(mapis->offset)); CHKERRQ(ierr);
    }
    binoff = mapis->offset;
  }
  else {
    binoff = offset;
  }
  if(!mapis->count) {
    ierr = PetscMalloc(mapis->n*sizeof(PetscInt), &(mapis->count)); CHKERRQ(ierr);
  }
  bincount = mapis->count;
  for(j = 0; j < mapis->n; ++j) {
    binoff[j] = 0;
    bincount[j] = 0;
  }
  binoff[mapis->n] = 0;

  /* Now compute bin offsets */
  for(i = 0; i < insize; ++i) {
    if((inidx[i] < 0 || inidx[i] > mapis->m)){
      if(!drop) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local index %D at %D out of range [%D,%D)", inidx[i],i, 0, mapis->m);
      else continue;
    }
    for(k = mapis->ijlen[inidx[i]]; k < mapis->ijlen[inidx[i]+1]; ++k) {
      ++(binoff[mapis->ij[k]+1]);
    }
  }/* for(i = 0; i < insize; ++i) */
  for(j = 0; j < mapis->n; ++j) {
    binoff[j+1] += binoff[j];
  }
  /* Now bin the input indices and values. */
  if(outidx || (inval && outval)) {
    for(i = 0; i < insize; ++i) {
      if((inidx[i] < 0 || inidx[i] > mapis->m)){
        if(!drop) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local index %D at %D out of range [%D,%D)", inidx[i], i, 0, mapis->m);
        else continue;
      }
      for(k = mapis->ijlen[inidx[i]]; k < mapis->ijlen[inidx[i]+1]; ++k) {
        j = mapis->ij[k];
        if(outidx)          outidx[binoff[j]+bincount[j]] = inidx[i];
        if(outval && inval) outval[binoff[j]+bincount[j]] = inval[i];
        ++bincount[j];
      }
    }/* for(i = 0; i < insize; ++i) */
  }/* if(outidx || (invalud && outval)) */
  if(outsize) *outsize = binoff[mapis->n];
  PetscFunctionReturn(0);
}/* ISMappingBinLocal_IS() */

#undef __FUNCT__  
#define __FUNCT__ "ISMappingBin_IS"
static PetscErrorCode ISMappingBin_IS(ISMapping map, PetscInt insize, const PetscInt inidx[], const PetscScalar inval[], PetscInt *outsize, PetscInt outidx[], PetscScalar outval[], PetscInt offset[], PetscBool drop) 
{
  PetscErrorCode ierr;
  ISMapping_IS *mapis = (ISMapping_IS*)map->data;
  PetscInt *inidx_loc;
  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(PetscInt)*insize, &inidx_loc); CHKERRQ(ierr);
  ierr = ISMappingIS_LocateIndices(mapis->m,mapis->supp,insize,inidx,PETSC_NULL,inidx_loc,PETSC_FALSE); CHKERRQ(ierr);
  ierr = ISMappingBin_IS(map,insize,inidx,inval,outsize,outidx,outval,offset,drop);                     CHKERRQ(ierr);
  ierr = PetscFree(inidx_loc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingBin_IS()*/



#undef __FUNCT__  
#define __FUNCT__ "ISMappingIS_AssembleMPI"
static PetscErrorCode ISMappingIS_AssembleMPI(ISMapping map, PetscInt len, const PetscInt ixidx[], const PetscInt iyidx[], PetscInt *_alen, PetscInt *_aixidx[], PetscInt *_aiyidx[]){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  MPI_Comm comm;
  PetscMPIInt size, rank, tag, imdex, n;
  PetscInt idx, lastidx;
  PetscInt i, j, p, count, slen;
  PetscInt   *owner = PETSC_NULL, *starts = PETSC_NULL;
  PetscInt    nsends, nrecvs, recvtotal;
  PetscMPIInt *procn = PETSC_NULL, *onodes, *olengths;
  PetscInt    *rvalues = PETSC_NULL, *svalues = PETSC_NULL, *rsvalues, *values = PETSC_NULL;
  MPI_Request *recv_waits, *send_waits;
  MPI_Status  recv_status, *send_status;
  PetscBool found;
  PetscInt *aixidx, *aiyidx;
    
  ierr = PetscObjectGetNewTag((PetscObject)map, &tag);CHKERRQ(ierr);
  comm = ((PetscObject)map)->comm;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  /*
   Each processor ships off its inidx[j] and inidy[j] to the appropriate processor.
   */
  /*  first count number of contributors to each processor */
  ierr  = PetscMalloc3(size,PetscMPIInt,&procn,len,PetscInt,&owner,(size+1),PetscInt,&starts);CHKERRQ(ierr);
  ierr  = PetscMemzero(procn,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  lastidx = -1;
  p       = 0;
  for (i=0; i<len; ++i) {
    /* if indices are NOT locally sorted, need to start search for the proc owning inidx[i] at the beginning */
    if (lastidx > (idx = ixidx[i])) p = 0;
    lastidx = idx;
    for (; p<size; ++p) {
      if (idx >= map->xlayout->range[p] && idx < map->xlayout->range[p+1]) {
        procn[p]++; 
        owner[i] = p; 
#if defined(PETSC_USE_DEBUG)
        found = PETSC_TRUE; 
#endif
        break;
      }
    }
#if defined(PETSC_USE_DEBUG)
    if (!found) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D out of range",idx);
    found = PETSC_FALSE;
#endif
  }
  nsends = 0;  for (p=0; p<size; ++p) { nsends += (procn[p] > 0);} 
    
  /* inform other processors of number of messages and max length*/
  ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,procn,&nrecvs);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nsends,nrecvs,procn,&onodes,&olengths);CHKERRQ(ierr);
  ierr = PetscSortMPIIntWithArray(nrecvs,onodes,olengths);CHKERRQ(ierr);
  recvtotal = 0; for (i=0; i<nrecvs; i++) recvtotal += olengths[i];
    
  /* post receives:   */
  ierr = PetscMalloc5(2*recvtotal,PetscInt,&rvalues,2*len,PetscInt,&svalues,nrecvs,MPI_Request,&recv_waits,nsends,MPI_Request,&send_waits,nsends,MPI_Status,&send_status);CHKERRQ(ierr);
    
  count = 0;
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((rvalues+2*count),2*olengths[i],MPIU_INT,onodes[i],tag,comm,recv_waits+i);CHKERRQ(ierr);
    count += olengths[i];
  }
  ierr = PetscFree(onodes);CHKERRQ(ierr);
    
  /* do sends:
   1) starts[p] gives the starting index in svalues for stuff going to 
   the pth processor
   */
  starts[0]= 0; 
  for (p=1; p<size; ++p) { starts[p] = starts[p-1] + procn[p-1];} 
  for (i=0; i<len; ++i) {
    svalues[2*starts[owner[i]]]       = ixidx[i];
    svalues[1 + 2*starts[owner[i]]++] = iyidx[i];
  }
    
  starts[0] = 0;
  for (p=1; p<size+1; ++p) { starts[p] = starts[p-1] + procn[p-1];} 
  count = 0;
  for (p=0; p<size; ++p) {
    if (procn[p]) {
      ierr = MPI_Isend(svalues+2*starts[p],2*procn[p],MPIU_INT,p,tag,comm,send_waits+count);CHKERRQ(ierr);
      count++;
    }
  }
  ierr = PetscFree3(procn,owner,starts);CHKERRQ(ierr);
    
  /*  wait on receives */
  count = nrecvs; 
  slen  = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    slen += n/2;
    count--;
  }
  if (slen != recvtotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Total message lengths %D not as expected %D",slen,recvtotal);
    
  ierr  = PetscMalloc(slen*sizeof(PetscInt), &aixidx); CHKERRQ(ierr);
  ierr  = PetscMalloc(slen*sizeof(PetscInt), &aiyidx); CHKERRQ(ierr);
  count = 0;
  rsvalues = rvalues;
  for (i=0; i<nrecvs; i++) {
    values = rsvalues;
    rsvalues += 2*olengths[i];
    for (j=0; j<olengths[i]; j++) {
      aixidx[count]   = values[2*j];
      aiyidx[count++] = values[2*j+1];
    }
  }
  ierr = PetscFree(olengths);CHKERRQ(ierr);
    
  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);}
  ierr = PetscFree5(rvalues,svalues,recv_waits,send_waits,send_status);CHKERRQ(ierr);

  *_alen = recvtotal;
  *_aixidx = aixidx;
  *_aiyidx = aiyidx;

  PetscFunctionReturn(0);
}/* ISMappingIS_AssembleMPI() */


/*
 Sort ix indices, if necessary.
 If ix duplicates exist, arrange iy indices in segments corresponding 
 to the images of the same input element. Remove iy duplicates from each 
 image segment and record the number of images in ijlen.  Convert the iy
 indices to a local numbering with the corresponding global indices stored
 in globals.
*/
#undef __FUNCT__  
#define __FUNCT__ "ISMappingIS_AssembleLocal"
static PetscErrorCode ISMappingIS_AssembleLocal(ISMapping map, PetscInt len, const PetscInt ixidx_const[], const PetscInt iyidx_const[], ISMapping_IS *mapis){
  PetscErrorCode ierr;
  PetscInt *ixidx, *iyidx;
  PetscInt ind,start, end, i, j, totalnij,maxnij, nij, m,n;
  PetscBool xincreasing;
  PetscInt *ij, *ijlen, *supp, *image;
  PetscFunctionBegin;

  /* Assume ixidx_const and iyidx_const have the same size. */
  if(!len) {
    mapis->m        = 0;
    mapis->supp     = PETSC_NULL;
    mapis->n        = 0;
    mapis->image    = PETSC_NULL;
    mapis->ij       = PETSC_NULL;
    mapis->ijlen    = PETSC_NULL;
    PetscFunctionReturn(0);
  }

  /* Copy ixidx and iyidx so they can be manipulated later. */
  ierr = PetscMalloc(len*sizeof(PetscInt), &ixidx); CHKERRQ(ierr);
  ierr = PetscMemcpy(ixidx, ixidx_const, len*sizeof(PetscInt)); CHKERRQ(ierr);
  ierr = PetscMalloc(len*sizeof(PetscInt), &iyidx); CHKERRQ(ierr);
  ierr = PetscMemcpy(iyidx, iyidx_const, len*sizeof(PetscInt)); CHKERRQ(ierr);



  /* Determine whether ixidx is strictly increasing.  */
  ind = ixidx_const[0];
  xincreasing = PETSC_TRUE;
  for(i = 1; i < len; ++i) {
    if(ixidx_const[i] <= ind) {
      xincreasing = PETSC_FALSE;
      break;
    }
  }
  if(!xincreasing) {
    /* ixidx_onst is not strictly increasing, no rearrangement of ixidx is also necessary. */
    /* sort on ixidx */
    ierr  = PetscSortIntWithArray(len,ixidx,iyidx);CHKERRQ(ierr);
  }
  /* Now ixidx is sorted, so we march over it and record in ijlen the number of iyidx indices corresponding to each ixidx index. */
  m        = 0;
  totalnij = 0;
  maxnij   = 0;
  start = 0;
  while (start < len) {
    end = start+1;
    while (end < len && ixidx[end] == ixidx[start]) ++end;
    ++(m); /* count all of ixidx[start:end-1] as a single occurence of an idx index */
    if (end - 1 > start) { /* found 2 or more of ixidx[start] in a row */
      /* sort the relevant portion of iy */
      ierr = PetscSortInt(end-start,iyidx+start);CHKERRQ(ierr);
      /* count unique elements in iyidx[start,end-1] */
      nij = 0;
      ISMapping_CountUnique(iyidx+start,end-start,nij);
      totalnij += nij;
      maxnij = PetscMax(maxnij, nij);
    }
    start = end;
  }
  /* 
   Now we know the size of the support -- m, and the total size of concatenated image segments -- totalnij. 
   Allocate an array for recording the images of each support index -- ij.
   Allocate an array for counting the number of images for each support index -- ijlen.
   */
  ierr = PetscMalloc(sizeof(PetscInt)*(m+1), &ijlen);  CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*(totalnij),&ij); CHKERRQ(ierr);
  /* If m == len we can use ixidx for supp and iyidx for ij, since this implies that the mapping is single-valued. */
  if(m != len) { 
    ierr = PetscMalloc(sizeof(PetscInt)*m, &supp); CHKERRQ(ierr);
  }
  else {
    supp = ixidx;
  }  
  /* We now record in supp only the unique ixidx indices, and in ij the unique iyidx indices in each of the image segments. */
  i = 0;
  j = 0;
  start = 0;
  ijlen[0] = 0;
  while (start < len) {
    end = start+1;
    while (end < len && ixidx[end] == ixidx[start]) ++end;
    /* the relevant portion of iy is already sorted; copy unique iyind only. */
    nij = 0;
    ISMapping_CopyUnique(iyidx+start,end-start,ij+j,nij);
    supp[i] = ixidx[start]; /* if supp == ixidx this a harmless self-assignment. */
    j += nij;
    ++i;
    ijlen[i] = j;
    start = end;
  }
  if(m != len) {
    ierr = PetscFree(ixidx);   CHKERRQ(ierr);
  }
  /* (A) Construct image -- the set of unique iyidx indices. */
  /* (B) Endow the image with a local numbering.  */
  /* (C) Then convert ij to this local numbering. */

  /* (A) */
  ierr = PetscSortInt(len,iyidx); CHKERRQ(ierr);
  n = 0;
  ISMapping_CountUnique(iyidx,len,n);
  ierr = PetscMalloc(sizeof(PetscInt)*n, &image);   CHKERRQ(ierr);
  n = 0;
  ISMapping_CopyUnique(iyidx,len,image,n);
  ierr = PetscFree(iyidx); CHKERRQ(ierr);
  /* (B) */
  ierr = PetscSortInt(n, image); CHKERRQ(ierr);
  /* (C) */
  /* 
   This is done by going through ij and for each k in it doing a binary search in image. 
   The result of the search is ind(k) -- the position of k in image. 
   Then ind(k) replace k in ij.
   */
  ierr = ISMappingIS_LocateIndices(n,image,totalnij,ij,&totalnij,ij,PETSC_TRUE); CHKERRQ(ierr);
  mapis->supp     = supp;
  mapis->m        = m;
  mapis->image    = image;
  mapis->n        = n;
  mapis->ij       = ij;
  mapis->ijlen    = ijlen;
  mapis->maxijlen = maxnij;
  PetscFunctionReturn(0);
}/* ISMappingIS_AssembleLocal() */

#undef __FUNCT__  
#define __FUNCT__ "ISMappingSetUp_ISMapping"
PetscErrorCode ISMappingSetUp_ISMapping(ISMapping map)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscLayoutSetBlockSize(map->xlayout,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(map->ylayout,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map->xlayout);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map->ylayout);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}/* ISMappingSetUp_ISMapping() */

EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "ISMappingISSetEdges"
PetscErrorCode ISMappingISSetEdges(ISMapping map, IS ix, IS iy) {
  ISMapping_IS *mapis = (ISMapping_IS*)(map->data);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscFunctionBegin;
  
  ISMappingCheckType(map, IS_MAPPING_IS,1);
  if(mapis->ix) {
    ierr = ISDestroy(mapis->ix); CHKERRQ(ierr);
  }
  mapis->ix = ix; 
  if(ix) {ierr = PetscObjectReference((PetscObject)ix); CHKERRQ(ierr);}
  if(mapis->iy) {
    ierr = ISDestroy(mapis->iy); CHKERRQ(ierr);
  }
  mapis->iy = iy; 
  if(iy){ierr = PetscObjectReference((PetscObject)iy); CHKERRQ(ierr);}
  map->assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}/* ISMappingSetIS() */
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "ISMappingAssemblyBegin_IS"
static PetscErrorCode ISMappingAssemblyBegin_IS(ISMapping map)
{
  ISMapping_IS   *mapis  = (ISMapping_IS*)(map->data);
  PetscInt       nix, niy;
  PetscErrorCode ierr;
  PetscMPIInt    xsize;
  PetscFunctionBegin;
  /*
      if input or output vertices are not defined, assume they are the total domain or range.
  */
  if(!mapis->ix) {
    ierr = ISCreateStride(((PetscObject)map)->comm,map->xlayout->n,map->xlayout->rstart,1,&(mapis->ix));CHKERRQ(ierr);
  }
  if(!mapis->iy) {
    ierr = ISCreateStride(((PetscObject)map)->comm,map->ylayout->n,map->ylayout->rstart,1,&(mapis->iy));CHKERRQ(ierr);
  }
#if defined(PETSC_USE_DEBUG)
  /* Consistency checks. */
  /* Make sure the IS sizes are compatible */
  ierr = ISGetLocalSize(mapis->ix,&nix);CHKERRQ(ierr);
  ierr = ISGetLocalSize(mapis->iy,&niy);CHKERRQ(ierr);
  if (nix != niy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local IS sizes don't match");
  ierr = ISMappingIS_CheckISRange(mapis->ix, 0, map->xlayout->N, PETSC_TRUE, PETSC_NULL); CHKERRQ(ierr);
  ierr = ISMappingIS_CheckISRange(mapis->iy, 0, map->ylayout->N, PETSC_TRUE, PETSC_NULL); CHKERRQ(ierr);
#endif

  if(mapis->supp) {
    ierr = PetscFree(mapis->supp); CHKERRQ(ierr);
  }
  if(mapis->image) {
    ierr = PetscFree(mapis->image); CHKERRQ(ierr);
  }
  if(mapis->ij) {
    ierr = PetscFree(mapis->ij); CHKERRQ(ierr);
  }

  ierr = MPI_Comm_size(map->xlayout->comm, &xsize); CHKERRQ(ierr);
  if(xsize > 1) {
    PetscInt len, alen;
    const PetscInt *ixidx, *iyidx;
    PetscInt *aixidx, *aiyidx;
    ierr = ISGetLocalSize(mapis->ix, &len); CHKERRQ(ierr);
    ierr = ISGetIndices(mapis->ix, &ixidx); CHKERRQ(ierr);
    ierr = ISGetIndices(mapis->iy, &iyidx); CHKERRQ(ierr);
    /* Assemble edges in parallel. */
    ierr = ISMappingIS_AssembleMPI(map, len, ixidx, iyidx, &alen, &aixidx, &aiyidx); CHKERRQ(ierr);
    ierr = ISRestoreIndices(mapis->ix, &ixidx); CHKERRQ(ierr);
    ierr = ISRestoreIndices(mapis->iy, &iyidx); CHKERRQ(ierr);
    /* Assemble edges locally. */
    ierr = ISMappingIS_AssembleLocal(map, alen, aixidx, aiyidx, mapis); CHKERRQ(ierr);
  }
  else {
    PetscInt len;
    const PetscInt *ixidx, *iyidx;
    ierr = ISGetLocalSize(mapis->ix, &len); CHKERRQ(ierr);
    ierr = ISGetIndices(mapis->ix, &ixidx); CHKERRQ(ierr);
    ierr = ISGetIndices(mapis->iy, &iyidx); CHKERRQ(ierr);
     /* Assemble edges locally. */
    ierr = ISMappingIS_AssembleLocal(map, len, ixidx, iyidx, mapis); CHKERRQ(ierr);

    ierr = ISRestoreIndices(mapis->ix, &ixidx); CHKERRQ(ierr);
    ierr = ISRestoreIndices(mapis->iy, &iyidx); CHKERRQ(ierr);
  }
  ierr = ISDestroy(mapis->ix); CHKERRQ(ierr);
  mapis->ix = PETSC_NULL;
  ierr = ISDestroy(mapis->iy); CHKERRQ(ierr);
  mapis->iy = PETSC_NULL;
  PetscFunctionReturn(0);
}/* ISMappingAssemblyBegin_IS() */


#undef __FUNCT__
#define __FUNCT__ "ISMappingAssemblyEnd_IS"
static PetscErrorCode ISMappingAssemblyEnd_IS(ISMapping map)
{
  PetscFunctionBegin;
  /* Currently a noop */
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISMappingGetSupportIS_IS"
PetscErrorCode ISMappingGetSupportIS_IS(ISMapping map, IS *supp) {
  PetscErrorCode ierr;
  ISMapping_IS         *mapis = (ISMapping_IS *)(map->data);
  PetscFunctionBegin;
  ierr = ISCreateGeneral(map->xlayout->comm, mapis->m, mapis->supp, PETSC_COPY_VALUES, supp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingGetSupportIS_IS() */


#undef  __FUNCT__
#define __FUNCT__ "ISMappingGetImageIS_IS"
PetscErrorCode ISMappingGetImageIS_IS(ISMapping map, IS *image) {
  PetscErrorCode ierr;
  ISMapping_IS         *mapis = (ISMapping_IS *)(map->data);
  PetscFunctionBegin;
  ierr = ISCreateGeneral(map->ylayout->comm, mapis->n, mapis->image, PETSC_COPY_VALUES, image); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingGetImageIS_IS() */


#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetMaxImageSizeLocal_IS"
PetscErrorCode ISMappingGetMaxImageSizeLocal_IS(ISMapping map, PetscInt *maxsize)
{
  PetscErrorCode ierr;
  ISMapping_IS *mapis = (ISMapping_IS *)(map->data);
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
  *maxsize = mapis->maxijlen;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetImageSizeLocal_IS"
PetscErrorCode ISMappingGetImageSizeLocal_IS(ISMapping map, PetscInt *size)
{
  PetscErrorCode ierr;
  ISMapping_IS *mapis = (ISMapping_IS *)(map->data);
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
  *size = mapis->n;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetSupportSizeLocal_IS"
PetscErrorCode ISMappingGetSupportSizeLocal_IS(ISMapping map, PetscInt *size)
{
  PetscErrorCode ierr;
  ISMapping_IS *mapis = (ISMapping_IS *)(map->data);
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
  *size = mapis->m;
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "ISMappingISGetEdges"
PetscErrorCode ISMappingISGetEdges(ISMapping map, IS *ix, IS *iy) {
  PetscErrorCode ierr;
  ISMapping_IS   *mapis = (ISMapping_IS *)(map->data);
  PetscInt len, *ixidx, *iyidx;
  PetscInt i,j;
  PetscFunctionBegin;
  ISMappingCheckType(map, IS_MAPPING_IS, 1);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  PetscValidPointer(ix,2);
  PetscValidPointer(iy,3);
  len = mapis->ijlen[mapis->m];
  ierr = PetscMalloc(sizeof(PetscInt)*len, &ixidx); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*len, &iyidx); CHKERRQ(ierr);
  for(i = 0; i < mapis->m; ++i) {
    for(j = mapis->ijlen[i]; j < mapis->ijlen[i+1]; ++i) {
      ixidx[j] = mapis->supp[i];
      iyidx[j] = mapis->image[mapis->ij[j]];
    }
  }
  ierr = ISCreateGeneral(map->xlayout->comm, len, ixidx, PETSC_USE_POINTER, ix); CHKERRQ(ierr);
  ierr = ISCreateGeneral(map->ylayout->comm, len, iyidx, PETSC_USE_POINTER, iy); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingISGetEdges() */
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetOperator_IS"
PetscErrorCode ISMappingGetOperator_IS(ISMapping map, Mat *mat)
{
  PetscErrorCode ierr;
  PetscMPIInt xsize, ysize;
  Vec x, y;
  IS ix,iy;
  VecScatter scatter;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
 
  ierr = MPI_Comm_size(map->xlayout->comm, &xsize); CHKERRQ(ierr);
  if(xsize > 1) {
    ierr = VecCreateMPI(((PetscObject)mat)->comm, map->xlayout->n, map->xlayout->N, &x); CHKERRQ(ierr);  

  }
  else {
    ierr = VecCreateSeq(PETSC_COMM_SELF, map->xlayout->n, &x); CHKERRQ(ierr);
  }
  ierr = MPI_Comm_size(map->ylayout->comm, &ysize); CHKERRQ(ierr);
  if(ysize > 1) {
    ierr = VecCreateMPI(((PetscObject)mat)->comm, map->ylayout->n, map->ylayout->N, &y); CHKERRQ(ierr);  

  }
  else {
    ierr = VecCreateSeq(PETSC_COMM_SELF, map->ylayout->n, &y); CHKERRQ(ierr);
  }
  ierr = ISMappingISGetEdges(map, &ix, &iy);                   CHKERRQ(ierr);
  ierr = VecScatterCreate(x,ix, y,iy, &scatter);               CHKERRQ(ierr);
  ierr = MatCreateScatter(((PetscObject)mat)->comm, scatter, mat); CHKERRQ(ierr);
  ierr = ISDestroy(ix); CHKERRQ(ierr);
  ierr = ISDestroy(iy); CHKERRQ(ierr);
  ierr = VecDestroy(x); CHKERRQ(ierr);
  ierr = VecDestroy(y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingGetOperator_IS() */


#undef  __FUNCT__
#define __FUNCT__ "ISMappingView_IS"
static PetscErrorCode ISMappingView_IS(ISMapping map, PetscViewer v) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
  /* FIX: actually implement this */
  PetscFunctionReturn(0);
}/* ISMappingView_IS() */



#undef  __FUNCT__
#define __FUNCT__ "ISMappingInvert_IS"
static PetscErrorCode ISMappingInvert_IS(ISMapping map, ISMapping *imap) 
{
  PetscErrorCode ierr;
  IS ix, iy;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ierr = ISMappingCreate(((PetscObject)map)->comm, imap); CHKERRQ(ierr);
  ierr = ISMappingSetSizes(*imap, map->xlayout->n, map->ylayout->n, map->xlayout->N, map->ylayout->N); CHKERRQ(ierr);
  ierr = ISMappingGetEdges(map, &ix,&iy);    CHKERRQ(ierr);
  ierr = ISMappingISSetEdges(*imap,iy, ix);  CHKERRQ(ierr);
  ierr = ISDestroy(ix);                      CHKERRQ(ierr);
  ierr = ISDestroy(iy);                      CHKERRQ(ierr);
  ierr = ISMappingAssemblyBegin(*imap);      CHKERRQ(ierr);
  ierr = ISMappingAssemblyEnd(*imap);        CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingInvert_IS() */


#undef  __FUNCT__
#define __FUNCT__ "ISMappingPushforward_IS_IS"
PetscErrorCode ISMappingPushforward_IS_IS(ISMapping map1, ISMapping map2, ISMapping *_map3) 
{
  PetscErrorCode ierr;
  IS supp1, supp2;
  ISMapping map3;
  const PetscInt *supp1idx, *supp2idx;
  PetscInt nsupp3;
  PetscInt *offsets1, *offsets2, *image1, *image2;
  PetscInt count, i,j,k;
  PetscInt nsupp1, nsupp2;
  PetscInt *supp3idx, *ixidx, *iyidx;
  IS ix,iy;
  PetscFunctionBegin;
  ISMappingCheckType(map1,IS_MAPPING_IS,1);
  ISMappingCheckType(map2,IS_MAPPING_IS,3);
  PetscCheckSameComm(map1,1,map2,2);
  /*
                                                       map3   _
                                                       ...    |
      |-----|                                |-----|  ------> |
         ^                                      ^             |
         |                    ======>           |             -
   map1  |                                map1  |
   ...   |      map2    _                  ...  |
         |      ...     |                       |
      |-----|  ------>  |                    |-----|
                        |
                        -
   */
  ierr = ISMappingGetSupportIS(map1,  &supp1);  CHKERRQ(ierr);
  ierr = ISMappingGetSupportIS(map2,  &supp2);  CHKERRQ(ierr);
  /* Avoid computing the intersection, which may be unscalable in storage. */
  ierr = ISGetLocalSize(supp1,&nsupp1);          CHKERRQ(ierr);
  ierr = ISGetLocalSize(supp2,&nsupp2);          CHKERRQ(ierr);
  ierr = ISGetIndices(supp1,  &supp1idx);        CHKERRQ(ierr);
  ierr = ISGetIndices(supp2,  &supp2idx);        CHKERRQ(ierr);
  /* 
   Count the number of images of the intersection of supports under the "upward" (1) and "rightward" (2) maps. 
   It is done this way: supp1 is mapped by map2 obtaining offsets2, and supp2 is mapped by map1 obtaining offsets1.
   */
  ierr = PetscMalloc2(nsupp1+1, PetscInt, &offsets2, nsupp2+1, PetscInt, &offsets1);                                CHKERRQ(ierr);
  ierr = ISMappingMapIndices(map1,nsupp2,supp2idx,PETSC_NULL,PETSC_NULL,offsets1,PETSC_TRUE); CHKERRQ(ierr);
  ierr = ISMappingMapIndices(map2,nsupp1,supp1idx,PETSC_NULL,PETSC_NULL,offsets2,PETSC_TRUE); CHKERRQ(ierr);
  /* Count the number of supp1 indices with nonzero images in map2 -- that's the size of the intersection. */
  nsupp3 = 0;
  for(i = 0; i < nsupp1; ++i) nsupp3 += (offsets2[i+1]>offsets2[i]);
#if defined(PETSC_USE_DEBUG)
  /* Now count the number of supp2 indices with nonzero images in map1: should be the same. */
  {
    PetscInt nsupp3_1 = 0;
    for(i = 0; i < nsupp2; ++i) nsupp3_1 += (offsets1[i+1]>offsets1[i]);
    if(nsupp3 != nsupp3_1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Intersections supports different in map1: %D and map2: %D", nsupp3_1, nsupp3);
  }
#endif
  /* Allocate indices for the intersection. */
  ierr = PetscMalloc(sizeof(PetscInt)*nsupp3, &supp3idx); CHKERRQ(ierr);
  nsupp3 = 0;
  for(i = 0; i < nsupp1; ++i) {
    if(offsets2[i+1] > offsets2[i]) {
      supp3idx[nsupp3] = supp1idx[i];
      ++nsupp3;
    }
  }
  ierr = ISRestoreIndices(supp1,&supp1idx);     CHKERRQ(ierr);
  ierr = ISRestoreIndices(supp2,&supp2idx);     CHKERRQ(ierr);
  ierr = ISDestroy(supp1);                      CHKERRQ(ierr);
  ierr = ISDestroy(supp2);                      CHKERRQ(ierr);
       
  /* 
   Now allocate the image storage and map the supp3 to obtain the "up" (1) and "right" (2) images.
   Recall that offsets1 are allocated for supp2, and offsets2 for supp1.
   */
  ierr = PetscMalloc2(offsets1[nsupp2],PetscInt,&image1,offsets2[nsupp1],PetscInt,&image2); CHKERRQ(ierr);
  ierr = ISMappingMapIndices(map1,nsupp3,supp3idx,PETSC_NULL,image1,offsets1,PETSC_FALSE);  CHKERRQ(ierr);
  ierr = ISMappingMapIndices(map2,nsupp3,supp3idx,PETSC_NULL,image2,offsets2,PETSC_FALSE);  CHKERRQ(ierr);
  ierr = PetscFree(supp3idx);                                                               CHKERRQ(ierr);

  /* Count the total number of arrows to add to the pushed forward ISMapping. */
  count = 0;
  for(k = 0; k < nsupp3; ++k) {
    count += (offsets1[k+1]-offsets1[k])*(offsets2[k+1]-offsets2[k]);
  }
  /* Allocate storage for the composed indices. */
  ierr = PetscMalloc(count*sizeof(PetscInt), &ixidx); CHKERRQ(ierr);
  ierr = PetscMalloc(count*sizeof(PetscInt), &iyidx); CHKERRQ(ierr);
  count= 0;
  for(k = 0; k < nsupp3; ++k) {
    for(i = offsets1[k]; i < offsets1[k+1]; ++i) {
      for(j = offsets2[k];  j < offsets2[k+1]; ++j) {
        ixidx[count] = image1[i];
        iyidx[count] = image2[j];
        ++count;
      }
    }
  }
  ierr = PetscFree2(image1,image2);     CHKERRQ(ierr);
  ierr = PetscFree2(offsets1,offsets2); CHKERRQ(ierr);
  /* Now construct the ISs and the ISMapping from them. */
  ierr = ISCreateGeneral(map1->ylayout->comm, count, ixidx, PETSC_OWN_POINTER, &ix);                      CHKERRQ(ierr);
  ierr = ISCreateGeneral(map2->ylayout->comm, count, iyidx, PETSC_OWN_POINTER, &iy);                      CHKERRQ(ierr);
  ierr = ISMappingCreate(((PetscObject)map1)->comm, &map3);                                               CHKERRQ(ierr);
  ierr = ISMappingSetType(map3,IS_MAPPING_IS);                                                            CHKERRQ(ierr);
  ierr = ISMappingSetSizes(map3, map1->ylayout->n, map2->ylayout->n, map1->ylayout->N, map2->ylayout->N); CHKERRQ(ierr);
  ierr = ISMappingISSetEdges(map3,ix,iy); CHKERRQ(ierr);
  ierr = ISMappingAssemblyBegin(map3);    CHKERRQ(ierr);
  ierr = ISMappingAssemblyEnd(map3);      CHKERRQ(ierr);
  ierr = ISDestroy(ix);                   CHKERRQ(ierr);
  ierr = ISDestroy(iy);                   CHKERRQ(ierr);

  *_map3 = map3;
  PetscFunctionReturn(0);
}/* ISMappingPushforward_IS_IS() */


#undef  __FUNCT__
#define __FUNCT__ "ISMappingPullback_IS_IS"
PetscErrorCode ISMappingPullback_IS_IS(ISMapping map1, ISMapping map2, ISMapping *_map3) 
{
  PetscErrorCode ierr;
  ISMapping imap1,map3;
  PetscFunctionBegin;
  ISMappingCheckType(map1,IS_MAPPING_IS,1);
  ISMappingCheckType(map2,IS_MAPPING_IS,3);
  PetscCheckSameComm(map1,1,map2,2);
  /*

                 map2   _                             map2   _ 
                 ...    |                             ...    | 
       |-----|  ------> |                 |-----|    ------> |
          ^             |                    ^               |
          |             -                    |               -
    map1  |                ======>     map1  |
     ...  |                            ...   |        map3   _
          |                                  |        ...    |
       |-----|                            |-----|    ------> |
                                                             |
                                                             -
   Convert this to a pushforward by inverting map1 to imap1 and then pushing map2 forward along imap1 
   (reflect the second diagram with respect to a horizontal axis and then compare with Pushforward,
    of just push forward "downward".)

                 map2   _                            map2   _                           map2   _ 
                 ...    |                            ...    |                           ...    | 
       |-----|  ------> |                  |-----|  ------> |                |-----|   ------> |
          ^             |                     |             |                   |              |
          |             -                     |             -                   |              -
    map1  |                ======>     imap1  |                 ======>  imap1  |
     ...  |                              ...  |                           ...   |       map3   _
          |                                   V                                 V       ...    |
       |-----|                             |-----|                           |-----|   ------> |
                                                                                               |
                                                                                               -
   */
  ierr = ISMappingInvert_IS(map1, &imap1);               CHKERRQ(ierr);
  ierr = ISMappingPushforward_IS_IS(imap1, map2, &map3); CHKERRQ(ierr);
  ierr = ISMappingDestroy(imap1);                        CHKERRQ(ierr);
  *_map3 = map3;
  PetscFunctionReturn(0);
}



#undef  __FUNCT__
#define __FUNCT__ "ISMappingDestroy_IS"
PetscErrorCode ISMappingDestroy_IS(ISMapping map) {
  PetscErrorCode ierr;
  ISMapping_IS          *mapis = (ISMapping_IS *)(map->data);
  
  PetscFunctionBegin;
  if(mapis) {
    if(mapis->ijlen) {
      ierr = PetscFree(mapis->ijlen); CHKERRQ(ierr);
    }
    if(mapis->ij) {
      ierr = PetscFree(mapis->ij); CHKERRQ(ierr);
    }    
    if(mapis->image) {
      ierr = PetscFree(mapis->image); CHKERRQ(ierr);
    }
    if(mapis->supp) {
      ierr = PetscFree(mapis->supp); CHKERRQ(ierr);
    }
    if(mapis->ix) {
      ierr = ISDestroy(mapis->ix);   CHKERRQ(ierr);
    }
    if(mapis->iy) {
      ierr = ISDestroy(mapis->iy);   CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(mapis); CHKERRQ(ierr);
  map->data = PETSC_NULL;
  
  map->setup = PETSC_FALSE;
  map->assembled = PETSC_FALSE;

  ierr = PetscObjectChangeTypeName((PetscObject)map,0); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)map,"ISMappingPullback_is_is_C", "",PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)map,"ISMappingPushforward_is_is_C", "",PETSC_NULL); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingDestroy_IS() */


#undef  __FUNCT__
#define __FUNCT__ "ISMappingCreate_IS"
PetscErrorCode ISMappingCreate_IS(ISMapping map) {
  PetscErrorCode ierr;
  ISMapping_IS  *mapis;
  PetscFunctionBegin;
  ierr = PetscNewLog(map, ISMapping_IS, &mapis); CHKERRQ(ierr);
  map->data = (void*)mapis;

  map->ops->view                 = ISMappingView_IS;
  map->ops->setup                = ISMappingSetUp_ISMapping;
  map->ops->assemblybegin        = ISMappingAssemblyBegin_IS;
  map->ops->assemblyend          = ISMappingAssemblyEnd_IS;
  map->ops->getsupportis         = ISMappingGetSupportIS_IS;
  map->ops->getsupportsizelocal  = ISMappingGetSupportSizeLocal_IS;
  map->ops->getimageis           = ISMappingGetSupportIS_IS;
  map->ops->getimagesizelocal    = ISMappingGetImageSizeLocal_IS;
  map->ops->getmaximagesizelocal = ISMappingGetMaxImageSizeLocal_IS;
  map->ops->maplocal             = ISMappingMapLocal_IS;
  map->ops->map                  = ISMappingMap_IS;
  map->ops->binlocal             = ISMappingBinLocal_IS;
  map->ops->bin                  = ISMappingBin_IS;
  map->ops->invert               = ISMappingInvert_IS;
  map->ops->getoperator          = ISMappingGetOperator_IS;

  map->setup     = PETSC_FALSE;
  map->assembled = PETSC_FALSE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)map, 
                                           "ISMappingPullback_is_is_C", "ISMappingPullback_ismappingis_ismappingis", 
                                           ISMappingPullback_IS_IS); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)map, 
                                           "ISMappingPushforward_is_is_C", "ISMappingPushforward_ismappingis_ismappingis", 
                                           ISMappingPushforward_IS_IS); CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)map, IS_MAPPING_IS); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingCreate_IS() */

