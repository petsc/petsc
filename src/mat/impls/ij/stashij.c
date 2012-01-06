#define PETSCMAT_DLL
#include <../src/mat/impls/ij/stashij.h>
/* Need sorting routines. */
#include <petscsys.h>
/* Need PetscHash */
#include <../src/sys/utils/hash.h>


#undef  __FUNCT__
#define __FUNCT__ "StashSeqIJCreate_Private"
PetscErrorCode StashSeqIJCreate_Private(StashSeqIJ *_stash) {
  PetscErrorCode ierr;
  StashSeqIJ stash;
  PetscFunctionBegin;
  ierr = PetscNew(struct _StashSeqIJ, &stash); CHKERRQ(ierr);
  ierr = PetscHashIJCreate(&(stash->h));       CHKERRQ(ierr);
  *_stash = stash;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "StashSeqIJGetMultivalued_Private"
PetscErrorCode StashSeqIJGetMultivalued_Private(StashSeqIJ stash, PetscBool *_multivalued) {
  PetscFunctionBegin;
  *_multivalued = stash->multivalued;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "StashSeqIJSetMultivalued_Private"
PetscErrorCode StashSeqIJSetMultivalued_Private(StashSeqIJ stash, PetscBool multivalued) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(stash->n) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot change multivaluedness of an non-empty Stash");
  stash->multivalued = multivalued;
  ierr = PetscHashIJSetMultivalued(stash->h,multivalued); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




#undef  __FUNCT__
#define __FUNCT__ "StashSeqIJExtend_Private"
PetscErrorCode StashSeqIJExtend_Private(StashSeqIJ stash, PetscInt len, const PetscInt *ixidx, const PetscInt *iyidx) {
  PetscHashIJKey key;
  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  for(i = 0; i < len; ++i) {
    key.i = ixidx[i];
    key.j = iyidx[i];
    ierr = PetscHashIJAdd(stash->h,key,stash->n); CHKERRQ(ierr);
    ++(stash->n);
  }
  PetscFunctionReturn(0);
}




#undef  __FUNCT__
#define __FUNCT__ "StashSeqIJGetIndices_Private"
PetscErrorCode StashSeqIJGetIndices_Private(StashSeqIJ stash, PetscInt *_len, PetscInt **_ixidx, PetscInt **_iyidx) {
  PetscInt len, *ixidx = PETSC_NULL, *iyidx = PETSC_NULL, *kidx, start, end;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if(!_len && !_ixidx && !_iyidx) PetscFunctionReturn(0);

  len = stash->n;
  if(_len) *_len = len;

  if(!_ixidx && !_iyidx) PetscFunctionReturn(0);

  if(_ixidx) {
    if(!*_ixidx) {
      ierr = PetscMalloc(len*sizeof(PetscInt), _ixidx); CHKERRQ(ierr);
    }
    ixidx = *_ixidx;
  }
  if(_iyidx) {
    if(!*_iyidx) {
      ierr = PetscMalloc(len*sizeof(PetscInt), _iyidx); CHKERRQ(ierr);
    }
    iyidx = *_iyidx;
  }
  ierr = PetscMalloc(len*sizeof(PetscInt), &kidx); CHKERRQ(ierr);
  ierr = PetscHashIJGetIndices(stash->h,ixidx,iyidx,kidx); CHKERRQ(ierr);
  ierr = PetscSortIntWithArrayPair(len,ixidx,iyidx,kidx); CHKERRQ(ierr);
  start = 0;
  while (start < len) {
    end = start+1;
    while (end < len && ixidx[end] == ixidx[start]) ++end;
    if (end - 1 > start) { /* found 2 or more of ixidx[start] in a row */
      /* order the relevant portion of iy by k */
      ierr = PetscSortIntWithArray(end-start,kidx+start,iyidx+start); CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(kidx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "StashSeqIJClear_Private"
PetscErrorCode StashSeqIJClear_Private(StashSeqIJ stash) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscHashIJClear(stash->h); CHKERRQ(ierr);
  stash->n = 0;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "StashSeqIJDestroy_Private"
PetscErrorCode StashSeqIJDestroy_Private(StashSeqIJ *_stash) {
  PetscErrorCode ierr;
  StashSeqIJ stash = *_stash;
  PetscFunctionBegin;
  ierr = StashSeqIJClear_Private(stash);     CHKERRQ(ierr);
  ierr = PetscHashIJDestroy(&(stash->h));    CHKERRQ(ierr);
  ierr = PetscFree(stash);                   CHKERRQ(ierr);
  *_stash = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "StashSeqIJSetPreallocation_Private"
PetscErrorCode StashSeqIJSetPreallocation_Private(StashSeqIJ stash, PetscInt size) {
  PetscInt s;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscHashIJKeySize(stash->h,&s); CHKERRQ(ierr);
  if(size < (PetscInt) s)
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot resize stash of size %D down to %D", s, size);
  ierr = PetscHashIJResize(stash->h,size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "StashMPIIJCreate_Private"
PetscErrorCode StashMPIIJCreate_Private(PetscLayout rmap, StashMPIIJ *_stash) {
  PetscErrorCode ierr;
  StashMPIIJ stash;
  PetscFunctionBegin;
  ierr = PetscNew(struct _StashMPIIJ, &stash); CHKERRQ(ierr);
  stash->rmap = 0;
  ierr = PetscLayoutReference(rmap, &(stash->rmap)); CHKERRQ(ierr);
  ierr = StashSeqIJCreate_Private(&(stash->astash)); CHKERRQ(ierr);
  ierr = StashSeqIJCreate_Private(&(stash->bstash)); CHKERRQ(ierr);
  stash->assembled = PETSC_TRUE;
  *_stash = stash;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "StashMPIIJDestroy_Private"
PetscErrorCode StashMPIIJDestroy_Private(StashMPIIJ *_stash) {
  PetscErrorCode ierr;
  StashMPIIJ stash = *_stash;
  PetscFunctionBegin;
  ierr = PetscLayoutDestroy(&(stash->rmap));          CHKERRQ(ierr);
  ierr = StashSeqIJDestroy_Private(&(stash->astash)); CHKERRQ(ierr);
  ierr = StashSeqIJDestroy_Private(&(stash->bstash)); CHKERRQ(ierr);
  ierr = PetscFree(stash);                            CHKERRQ(ierr);
  *_stash = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "StashMPIIJClear_Private"
PetscErrorCode StashMPIIJClear_Private(StashMPIIJ stash) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = StashSeqIJClear_Private(stash->astash); CHKERRQ(ierr);
  ierr = StashSeqIJClear_Private(stash->bstash); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




#undef  __FUNCT__
#define __FUNCT__ "StashMPIIJSetPreallocation_Private"
PetscErrorCode StashMPIIJSetPreallocation_Private(StashMPIIJ stash, PetscInt asize, PetscInt bsize) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = StashSeqIJSetPreallocation_Private(stash->astash,asize); CHKERRQ(ierr);
  ierr = StashSeqIJSetPreallocation_Private(stash->bstash,bsize); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "StashMPIIJGetMultivalued_Private"
PetscErrorCode StashMPIIJGetMultivalued_Private(StashMPIIJ stash, PetscBool *_multivalued) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = StashSeqIJGetMultivalued_Private(stash->astash, _multivalued); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "StashMPIIJSetMultivalued_Private"
PetscErrorCode StashMPIIJSetMultivalued_Private(StashMPIIJ stash, PetscBool multivalued) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = StashSeqIJSetMultivalued_Private(stash->astash, multivalued); CHKERRQ(ierr);
  ierr = StashSeqIJSetMultivalued_Private(stash->bstash, multivalued); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "StashMPIIJExtend_Private"
PetscErrorCode StashMPIIJExtend_Private(StashMPIIJ stash, PetscInt len, const PetscInt *ixidx, const PetscInt *iyidx) {
  PetscErrorCode ierr;
  PetscInt i;
  PetscFunctionBegin;
  for(i = 0; i < len; ++i) {
    if(ixidx[i] >= stash->rmap->rstart && ixidx[i] < stash->rmap->rend) {
      ierr = StashSeqIJExtend_Private(stash->astash,1,ixidx+i,iyidx+i); CHKERRQ(ierr);
    }
    else if(ixidx[i] && ixidx[i] < stash->rmap->N) {
      ierr = StashSeqIJExtend_Private(stash->bstash,1,ixidx+i,iyidx+i); CHKERRQ(ierr);      
    }
    else SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "I index %D at position %D is out of range [0,%D)", ixidx[i],i,stash->rmap->N);
  }
  stash->assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "StashMPIIJGetIndices_Private"
PetscErrorCode StashMPIIJGetIndices_Private(StashMPIIJ stash, PetscInt *_alen, PetscInt **_aixidx, PetscInt **_aiyidx, PetscInt *_blen, PetscInt **_bixidx, PetscInt **_biyidx) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(!stash->assembled) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Indices requested from an unassembled stash");
  ierr = StashSeqIJGetIndices_Private(stash->astash, _alen,_aixidx, _aiyidx); CHKERRQ(ierr);
  ierr = StashSeqIJGetIndices_Private(stash->bstash, _blen,_bixidx, _biyidx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "StashMPIIJGetIndicesMerged_Private"
PetscErrorCode StashMPIIJGetIndicesMerged_Private(StashMPIIJ stash, PetscInt *_len, PetscInt **_ixidx, PetscInt **_iyidx) {
  PetscErrorCode ierr;
  PetscInt len, alen, *aixidx = PETSC_NULL, *aiyidx = PETSC_NULL, blen, *bixidx = PETSC_NULL, *biyidx = PETSC_NULL;
  PetscFunctionBegin;
  if(!stash->assembled) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Indices requested from an unassembled stash");
  if(!_len && !_ixidx && !_iyidx) PetscFunctionReturn(0);

  ierr = StashMPIIJGetIndices_Private(stash, &alen, PETSC_NULL, PETSC_NULL, &blen, PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);
  len = alen + blen;
  if(_len) *_len = len;

  if((!_ixidx && !_iyidx) || (!alen && !blen)) PetscFunctionReturn(0);
  
  if(!_ixidx || !_iyidx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Output arrays must be null or non-null together");

  if(!alen) {
    /* Nothing to merge from the left, so get all of the indices from the right. */
    ierr = StashMPIIJGetIndices_Private(stash,PETSC_NULL,PETSC_NULL,PETSC_NULL,_len,_ixidx,_iyidx); CHKERRQ(ierr);
  }
  else if(!blen) {
    /* Nothing to merge from the right, so get all of the indices from the left. */
    ierr = StashMPIIJGetIndices_Private(stash,_len,_ixidx,_iyidx,PETSC_NULL,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  }
  else {
    /* Retrieve the indices into temporary arrays to hold the indices prior to merging. */
    ierr = StashMPIIJGetIndices_Private(stash,&alen,&aixidx,&aiyidx,&blen,&bixidx,&biyidx);  CHKERRQ(ierr);
    /* Merge. */
    ierr = PetscMergeIntArrayPair(alen,aixidx,aiyidx,blen,bixidx,biyidx,_len,_ixidx,_iyidx); CHKERRQ(ierr);
    /* Clean up. */
    ierr = PetscFree(aixidx); CHKERRQ(ierr);
    ierr = PetscFree(aiyidx); CHKERRQ(ierr);
    ierr = PetscFree(bixidx); CHKERRQ(ierr);
    ierr = PetscFree(biyidx); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "StashMPIIJAssemble_Private"
PetscErrorCode StashMPIIJAssemble_Private(StashMPIIJ stash) {
  PetscErrorCode ierr;
  MPI_Comm comm = stash->rmap->comm;
  PetscMPIInt size, rank, tag_ij, nsends, nrecvs, *plengths, *sstarts = PETSC_NULL, 
    *rnodes, *rlengths, *rstarts = PETSC_NULL, rlengthtotal;
  MPI_Request *recv_reqs_ij, *send_reqs;
  MPI_Status  recv_status, *send_statuses;
  PetscInt len, *ixidx = PETSC_NULL, *iyidx = PETSC_NULL;
  PetscInt *owner = PETSC_NULL, p, **rindices = PETSC_NULL, *sindices= PETSC_NULL;
  PetscInt low, high, idx, lastidx, count, i, j;
  PetscFunctionBegin;

  if(stash->assembled) PetscFunctionReturn(0);

  /* Comm parameters */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = StashSeqIJGetIndices_Private(stash->bstash, &len, &ixidx, &iyidx); CHKERRQ(ierr);

  /* Each processor ships off its ixidx[j] and iyidx[j] */
  /*  first count number of contributors to each processor */
  ierr  = PetscMalloc2(size,PetscMPIInt,&plengths,len,PetscInt,&owner);CHKERRQ(ierr);
  ierr  = PetscMemzero(plengths,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  lastidx = -1;
  count   = 0;
  p       = 0;
  low = 0; high = size-1;
  for(i = 0; i < len; ++i) {
    idx = ixidx[i];
    if(idx < 0 || idx >= stash->rmap->range[size]) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %D out of range [0,%D)", idx, stash->rmap->range[size]);
    if(i) {
      if(idx > lastidx) {
        /* lower bound is still valid, but the upper bound might not be.*/
        /*
         range is ordered, hence, is a subsequence of the integers.
         Thus, the distance between idx and lastidx in the range is no greater
         than the distance between them within the integers: idx - lastidx.
         Therefore, high raised by idx-lastidx is a valid upper bound on idx.
         */
        high = PetscMin(size, high+(idx-lastidx));
        /* p is the largest index in range whose value does not
           exceed last; since idx > lastidx, idx is located above p
           within range.
         */
        low = p; 
      }
      if(idx < lastidx) {
        /* upper bound is still valid, but the lower bound might not be.*/
        /*
         range is ordered, hence, is a subsequence of the integers.
         Thus, the distance between idx and lastidx in range is no greater
         than the distance between them within the integers: lastidx - idx.
         Therefore, low lowered by idx-lastidx is a valid upper bound on idx.
         */
        low = PetscMax(0,low+idx-lastidx);
        /* p is the largest range index whose value does not exceed lastidx;
         since idx < lastidx, idx is located no higher than p within range */
        high = p;
      }
    }/* if(i) */
    lastidx = idx;
    while((high) - (low) > 1) {
      p = (high+low)/2;
      if(i < stash->rmap->range[p]) {
        high = p;
      }
      else {
        low = p;
      }
    }
    plengths[p]++; 
    owner[i] = p; 
  }/* for(i=0; i < len; ++i) */

  nsends = 0;  for (p=0; p<size; ++p) { nsends += (plengths[p] > 0);} 
    
  /* inform other processors of number of messages and max length*/
  ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,plengths,&nrecvs);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nsends,nrecvs,plengths,&rnodes,&rlengths);CHKERRQ(ierr);
  /* Sort on the the receive nodes, so we can store the received ix indices in the order they were globally specified. */
  ierr = PetscSortMPIIntWithArray(nrecvs,rnodes,rlengths); CHKERRQ(ierr);
  /* sending/receiving pairs (ixidx[i],iyidx[i]) */
  for (i=0; i<nrecvs; ++i) rlengths[i] *=2;

  ierr = PetscCommGetNewTag(stash->rmap->comm, &tag_ij);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tag_ij,nrecvs,rnodes,rlengths,&rindices,&recv_reqs_ij);CHKERRQ(ierr);
  ierr = PetscFree(rnodes);CHKERRQ(ierr);

  for (i=0; i<nrecvs; ++i) rlengths[i] /=2;

  /* prepare send buffers and offsets.
      sindices is the index send buffer; 
      sstarts[p] gives the starting offset for values going to the pth processor, if any;
      because PAIRS of indices are sent from the same buffer, the index offset is 2*sstarts[p].
  */
  ierr     = PetscMalloc((size+1)*sizeof(PetscMPIInt),&sstarts);  CHKERRQ(ierr);
  ierr     = PetscMalloc(2*len*sizeof(PetscInt),&sindices);       CHKERRQ(ierr);

  /* Compute buffer offsets for the segments of data going to different processors,
     and zero out plengths: they will be used below as running counts when packing data
     into send buffers; as a result of that, plengths are recomputed by the end of the loop.
   */
  sstarts[0] = 0;
  for (p=0; p<size; ++p) { 
    sstarts[p+1] = sstarts[p] + plengths[p];
    plengths[p] = 0;
  }

  /* Now pack the indices into the appropriate buffer segments. */
  count = 0;
  for(i = 0; i < len; ++i) {
    p = owner[count];
    /* All ixidx indices first, then all iyidx: a remnant of the code that handled both 1- and 2-index cases.*/
    sindices[2*sstarts[p]+plengths[p]]                           = ixidx[i];
    sindices[2*sstarts[p]+(sstarts[p+1]-sstarts[p])+plengths[p]] = iyidx[i];
    ++plengths[p];
    ++count;
  }

  /* Allocate a send requests for the indices */
  ierr     = PetscMalloc(nsends*sizeof(MPI_Request),&send_reqs);  CHKERRQ(ierr);
  /* Post sends */
  for (p=0,count=0; p<size; ++p) {
    if (plengths[p]) {
      ierr = MPI_Isend(sindices+2*sstarts[p],2*plengths[p],MPIU_INT,p,tag_ij,comm,send_reqs+count++);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(plengths,owner);CHKERRQ(ierr);
  ierr = PetscFree(sstarts);CHKERRQ(ierr);

  /* Prepare to receive indices and values. */
  /* Compute the offsets of the individual received segments in the unified index/value arrays. */
  ierr = PetscMalloc(sizeof(PetscMPIInt)*(nrecvs+1), &rstarts); CHKERRQ(ierr);
  rstarts[0] = 0;
  for(j = 0; j < nrecvs; ++j) rstarts[j+1] = rstarts[j] + rlengths[j];


  /*  Wait on index receives and insert them the received indices into the local stash, as necessary. */
  count = nrecvs; 
  rlengthtotal = 0;
  while (count) {
    PetscMPIInt n,k;
    ierr = MPI_Waitany(nrecvs,recv_reqs_ij,&k,&recv_status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    rlengthtotal += n/2;
    count--;
    ierr = StashSeqIJExtend_Private(stash->astash,rlengths[k],rindices[k],rindices[k]+rlengths[k]); CHKERRQ(ierr);    
  }
  if (rstarts[nrecvs] != rlengthtotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Total message lengths %D not as expected %D",rlengthtotal,rstarts[nrecvs]);
  ierr = PetscFree(ixidx); CHKERRQ(ierr);
  ierr = PetscFree(iyidx); CHKERRQ(ierr);
  ierr = StashSeqIJClear_Private(stash->bstash);                          CHKERRQ(ierr);

  ierr = PetscFree(rlengths);    CHKERRQ(ierr);
  ierr = PetscFree(rstarts);     CHKERRQ(ierr);
  ierr = PetscFree(rindices[0]); CHKERRQ(ierr);
  ierr = PetscFree(rindices);    CHKERRQ(ierr);
  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc(sizeof(MPI_Status)*nsends,&send_statuses);     CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_reqs,send_statuses);               CHKERRQ(ierr);
    ierr = PetscFree(send_statuses);                                  CHKERRQ(ierr);
  }
  ierr = PetscFree(sindices);                                         CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


