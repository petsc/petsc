#define PETSCDM_DLL

#include "private/saimpl.h"  /*I "petscsa.h"  I*/

typedef struct {
  /* The following is used for mapping. */
  SA edges;
  PetscInt *supp;    /* support        -- global input indices from ownership range with nonempty images */
  PetscInt m;        /* size of suppot -- number of input indices with nonempty images */
  PetscInt *image;   /* image          -- distinct global output indices */
  PetscInt n;        /* size of image -- number of distinct global output indices */
  PetscInt *ij;      /* concatenated local images of ALL local input elements (i.e., all indices from the local ownership range), sorted within each image */
  PetscInt *ijlen;   /* image segment boundaries for each local input index */
  PetscInt maxijlen; /* max image segment size */
  /* The following is used for binning. */
  PetscInt *offset, *count; 
} SAMapping_Graph;

/*
 Increment ii by the number of unique elements of segment a[0,i-1] of a SORTED array a.
 */
#define PetscIntArrayCountUnique(a, i, ii)  \
{                                           \
  if(i) {                                   \
    PetscInt k = 0;                         \
    ++(ii);                                 \
    while(++k < (i))                        \
      if ((a)[k] != (a)[k-1]) {             \
        ++ii;                               \
      }                                     \
  }                                         \
}

/*
 Copy unique elements of segment a[0,i-1] of a SORTED array a, to aa[ii0,ii1-1]:
 i is an input, and ii is an input (with value ii0) and output (ii1), counting 
 the number of unique elements copied.
 */
#define PetscIntArrayCopyUnique(a, i, aa, ii)\
{                               \
  if(i) {                       \
    PetscInt k = 0;             \
    (aa)[(ii)] = (a)[k];        \
    ++(ii);                     \
    while (++k < (i))           \
      if ((a)[k] != (a)[k-1]) { \
        (aa)[(ii)] = (a)[k];    \
        ++(ii);                 \
      }                         \
  }                             \
}

/*
 Copy unique elements of segment a[0,i-1] of a SORTED array a, to aa[ii0,ii1-1]:
 i is an input, and ii is an input (with value ii0) and output (ii1), counting 
 the number of unique elements copied.  For each copied a, copy the corresponding
 b to bb.
 */
#define PetscIntArrayCopyUniqueWithScalar(a, b, i, aa, bb, ii)     \
{                               \
  if(i) {                       \
    PetscInt k = 0;             \
    (aa)[(ii)] = (a)[k];        \
    ++(ii);                     \
    while (++k < (i))           \
      if ((a)[k] != (a)[k-1]) { \
        (aa)[(ii)] = (a)[k];    \
        (bb)[(ii)] = (b)[k];    \
        ++(ii);                 \
      }                         \
  }                             \
}

/*
 Locate index i in the table table of length tablen.  If i is found in table,
 ii is its index, between 0 and tablen; otherwise, ii == -1. 
 count,last,low,high are auxiliary variables that help speed up the search when
 it is carried out repeatedly (e.g., for all i in an array); these variables
 should be passed back to this macro for each search iteration and not altered
 between the macro invocations.
 */
#define SAMappingGraphLocalizeIndex(tablen, table, i,ii,count,last,low,high) \
{                                                                       \
  PetscBool _9_found = PETSC_FALSE;                                     \
  /* Convert to local by searching through mapg->supp. */               \
  if((count) > 0) {                                                     \
    /* last and ii have valid previous values, that can be used to take \
     advantage of the already known information about the table. */     \
    if((i) > (last)) {                                                  \
      /* lower bound is still valid, but the upper bound might not be.*/ \
      /*                                                                \
       table is ordered, hence, is a subsequence of the integers.       \
       Thus, the distance between ind and last in table is no greater   \
       than the distance between them within the integers: ind - last.  \
       Therefore, high raised by ind-last is a valid upper bound on ind. \
       */                                                               \
      (high) = PetscMin((mapg)->m, (high)+((i)-(last)));                \
      /* ii is the largest index in the table whose value does not      \
       exceed last; since i > last, i is located above ii within        \
       table */                                                         \
      (low) = (ii);                                                     \
    }                                                                   \
    if((i) < (last)) {                                                  \
      /* upper bound is still valid, but the lower bound might not be.*/ \
      /*                                                                \
       table is ordered, hence, is a subsequence of the integers.       \
       Thus, the distance between i and last in table is no greater     \
       than the distance between them within the integers: last - i.    \
       Therefore, low lowered by i-last is a valid upper bound on i.    \
       */                                                               \
      (low) = PetscMax(0,(low)+((i)-last));                             \
      /* ii is the largest index of the table entry not exceeding last; \
       since i < last, i is located no higher than ii within table */   \
      (high) = (ii);                                                    \
    }                                                                   \
  }/* if((count) > 0) */                                                \
  else {                                                                \
    (low) = 0;                                                          \
    (high) = (tablen);                                                  \
  }                                                                     \
  (last) = (i);                                                         \
  while((high) - (low) > 5) {                                           \
    (ii) = ((high)+(low))/2;                                            \
    if((i) < (table)[ii]) {                                             \
      (high) = (ii);                                                    \
    }                                                                   \
    else {                                                              \
      (low) = (ii);                                                     \
    }                                                                   \
  }                                                                     \
  (ii) = (low);                                                         \
  while((ii) < (high) && (table)[(ii)] <= (i)) {                        \
    if((i) == (table)[(ii)]) {                                          \
      _9_found = PETSC_TRUE;                                            \
      break;                                                            \
    }                                                                   \
    ++(ii);                                                             \
  }                                                                     \
  if(!_9_found) (ii) = -1;                                              \
}
  

/* 
 Locate all integers from array iarr of length len in table of length tablen.
 The indices of the located integers -- their locations in table -- are
 stored in iiarr of length len.  
 If drop == PETSC_TRUE:
  - if an integer is not found in table, it is omitted and upon completion 
    iilen has the number of located indices; iilen <= ilen in this case. 
 If drop == PETSC_FALE:
  - if an integer is not found in table, it is replaced by -1; iilen == ilen
    upon completion.
 */
#define SAMappingGraphLocalizeIndices(tablen,table,ilen,iarr,iilen,iiarr,drop) \
do {                                                                       \
  PetscInt _10_last,_10_low = 0,_10_high = (tablen), _10_k, _10_ind;    \
  (iilen) = 0;                                                          \
  for(_10_k = 0; _10_k < (ilen); ++_10_k) {                             \
    SAMappingGraphLocalizeIndex((tablen),(table),(iarr)[_10_k],_10_ind,_10_k,_10_last,_10_low,_10_high); \
    if(_10_ind != -1 && !(drop)) (iiarr)[(iilen)++]  = _10_ind;         \
  }                                                                     \
} while(0)

#define SAMappingGraphOrderPointers_Private(i,w,j,mask,index,off,ii,ww,jj) \
do{                                                                     \
  if((index) == SA_I) {                                            \
    (ii) = (i);                                                         \
    if(((mask) & SA_J)&&(j)) (jj) = (j)+(off);                     \
    else (jj) = PETSC_NULL;                                             \
  }                                                                     \
  else {                                                                \
    (ii) = (j);                                                         \
    if(((mask) & SA_I)&&(i)) (jj) = (i)+(off);                     \
    else (jj) = PETSC_NULL;                                             \
  }                                                                     \
  if(((mask) & SA_W) && (w)) (ww) = (w);                           \
  else (ww) = PETSC_NULL;                                               \
} while(0)



#undef __FUNCT__  
#define __FUNCT__ "SAMappingGraphMap_Private"
static PetscErrorCode SAMappingGraphMap_Private(SAMapping map, PetscInt insize, const PetscInt inidxi[], const PetscReal inval[], const PetscInt inidxj[], PetscInt *outsize, PetscInt outidxi[], PetscScalar outval[], PetscInt outidxj[], PetscInt outsizes[], PetscBool local, PetscBool drop) 
{
  SAMapping_Graph *mapg = (SAMapping_Graph*)map->data;
  PetscInt i,j,k,last,low,high,ind,count;
  PetscFunctionBegin;

  j = 0;
  count = 0;
  for(i = 0; i < insize; ++i) {
    if(!local) {
      /* Convert to local by searching through mapg->supp. */
      SAMappingGraphLocalizeIndex(mapg->m,mapg->supp,inidxi[i],ind,count,last,low,high);
    }/* if(!local) */
    else {
      ind = inidxi[i];
    }
    if((ind < 0 || ind > mapg->m)){
      if(!drop) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %D at %D not in the support", inidxi[i], count);
      if(outsizes) outsizes[count] = 0;
      continue;
    }
    if(outidxi || (inval && outval) || (inidxj && outidxj) ) {
      for(k = mapg->ijlen[ind]; k < mapg->ijlen[ind+1]; ++k) {
        if(outidxi)         outidxi[j] = mapg->image[mapg->ij[k]];
        if(inidxj&&outidxj) outidxj[j] = inidxj[i];
        if(inval&&outval)   outval[j]  = inval[i];
        ++j;
      }
    }
    else {
      j += mapg->ijlen[ind+1]-mapg->ijlen[ind];
    }
    if(outsizes) outsizes[count] = (mapg->ijlen[ind+1]-mapg->ijlen[ind]);
    ++count;
  }/* for(i = 0; i < len; ++i) */
  if(outsize) *outsize = j;
  PetscFunctionReturn(0);
}/* SAMappingGraphMap_Private() */


#undef __FUNCT__  
#define __FUNCT__ "SAMappingGraphMapSA_Private"
static PetscErrorCode SAMappingGraphMapSA_Private(SAMapping map, SA inarr, SAIndex index, SA outarr, PetscInt **_outsizes, PetscBool *_own_outsizes, PetscBool local, PetscBool drop) 
{
  PetscErrorCode ierr;
  PetscInt *inidxi = PETSC_NULL, *inidxj = PETSC_NULL, outsize, *outidxi = PETSC_NULL, *outidxj = PETSC_NULL, outlen, outsizesarr[128],*outsizes;
  PetscScalar *inval = PETSC_NULL, *outval = PETSC_NULL;
  SALink  inlink;
  SAHunk  inhunk, outhunk;
  PetscFunctionBegin;
  /**/
  if(index != SA_I && index != SA_J) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "SA components %D have no index %D", inarr->mask, index);
  /* Determine the size of the output. */
  inlink = inarr->first;
  outsize = 0;
  while(inlink) {
    inhunk = inlink->hunk;
    SAMappingGraphOrderPointers_Private(inhunk->i,inhunk->w,inhunk->j,inhunk->mask,index,0,inidxi,inval,inidxj);
    ierr = SAMappingGraphMap_Private(map,inhunk->length,inidxi,inval,inidxj,&outlen,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_FALSE,drop); CHKERRQ(ierr);
    outsize += outlen;
    inlink = inlink->next;
  }
  /* Get space for the output. */
  ierr = SAHunkCreate(outsize,outarr->mask, &outhunk); CHKERRQ(ierr);
  /* Get space for outsizes, if necessary. */
  if(_outsizes) {
    if(inarr->length <= 128) {
      outsizes = outsizesarr;
      *_own_outsizes = PETSC_FALSE;
    }
    else {
      ierr = PetscMalloc(sizeof(PetscInt)*inarr->length, &outsizes); CHKERRQ(ierr);
      *_own_outsizes = PETSC_TRUE;
    }
  }
  else {
    outsizes = PETSC_NULL;
  }
  /* Do the mapping. */
  outsize = 0;
  inlink = inarr->first;
  while(inlink) {
    inhunk = inlink->hunk;
    SAMappingGraphOrderPointers_Private(inhunk->i,inhunk->w,inhunk->j,inhunk->mask,index,0,inidxi,inval,inidxj);
    SAMappingGraphOrderPointers_Private(outhunk->i,outhunk->w,outhunk->j,outhunk->mask,index,outsize,outidxi,outval,outidxj);
    ierr = SAMappingGraphMap_Private(map,inhunk->length,inidxi,inval,inidxj,&outlen,outidxi,outval,outidxj,outsizes,local,drop); CHKERRQ(ierr);
    outsize += outlen;
    if(outsizes) outsizes += inhunk->length;
    inlink = inlink->next;
  }
  ierr = SAAddHunk(outarr, outhunk); CHKERRQ(ierr);
  if(_outsizes) *_outsizes = outsizes;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingMap_Graph"
static PetscErrorCode SAMappingMap_Graph(SAMapping map, SA inarr, SAIndex index, SA outarr) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  if(index != SA_I && index != SA_J) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among SA components %D", index, inarr->mask);
  ierr = SAMappingGraphMapSA_Private(map,inarr,index,outarr,PETSC_NULL,PETSC_NULL,PETSC_FALSE,PETSC_TRUE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingMapSplit_Graph"
static PetscErrorCode SAMappingMapSplit_Graph(SAMapping map, SA inarr, SAIndex index, SA *outarrs) 
{
  PetscErrorCode ierr;
  PetscInt *outsizes;
  PetscBool own_outsizes;
  SA outarr;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  if(index != SA_I && index != SA_J) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "SA components %D don't contain index %D", inarr->mask, index);
  ierr = SACreate(inarr->mask, &outarr);                                                                     CHKERRQ(ierr);
  ierr = SAMappingGraphMapSA_Private(map,inarr,index,outarr,&outsizes,&own_outsizes,PETSC_FALSE,PETSC_TRUE); CHKERRQ(ierr);
  ierr = SASplit(outarr, inarr->length, outsizes, outarr->mask, outarrs);                                    CHKERRQ(ierr);
  ierr = SADestroy(outarr);                                                                                  CHKERRQ(ierr);
  if(own_outsizes) {
    ierr = PetscFree(outsizes); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingMapLocal_Graph"
static PetscErrorCode SAMappingMapLocal_Graph(SAMapping map, SA inarr, SAIndex index, SA outarr) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  if(index != SA_I && index != SA_J) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among SA components %D", index, inarr->mask);
  ierr = SAMappingGraphMapSA_Private(map,inarr,index,outarr,PETSC_NULL,PETSC_NULL,PETSC_TRUE,PETSC_TRUE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SAMappingMapSplitLocal_Graph"
static PetscErrorCode SAMappingMapSplitLocal_Graph(SAMapping map, SA inarr, SAIndex index, SA *outarrs) 
{
  PetscErrorCode ierr;
  PetscInt *outsizes;
  PetscBool own_outsizes;
  SA outarr;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  if(index != SA_I && index != SA_J) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "SA components %D don't contain index %D", inarr->mask, index);
  ierr = SACreate(inarr->mask, &outarr);                                                                    CHKERRQ(ierr);
  ierr = SAMappingGraphMapSA_Private(map,inarr,index,outarr,&outsizes,&own_outsizes,PETSC_TRUE,PETSC_TRUE); CHKERRQ(ierr);
  ierr = SASplit(outarr, inarr->length, outsizes, outarr->mask, outarrs);                                   CHKERRQ(ierr);
  ierr = SADestroy(outarr);                                                                                 CHKERRQ(ierr);
  if(own_outsizes) {
    ierr = PetscFree(outsizes); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SAMappingGraphBinSA_Private"
static PetscErrorCode SAMappingGraphBinSA_Private(SAMapping map, SA inarr, PetscInt index, SA outarr, const PetscInt *_outsizes[], PetscBool local, PetscBool drop) 
{
  PetscErrorCode ierr;
  SAMapping_Graph *mapg = (SAMapping_Graph*)map->data;
  PetscInt      *binoff, *bincount, i,j,k,count,last,low,high,ind, *inidxi,*inidxj,*outidxi,*outidxj;
  PetscReal     *inval, *outval;
  SALink link;
  SAHunk inhunk, outhunk;

  PetscFunctionBegin;
  /* We'll need to count contributions to each "bin" and the offset of each bin in outidx, etc. */
  /* Allocate the bin offset array, if necessary. */
  if(!mapg->offset) {
    ierr = PetscMalloc((mapg->n+1)*sizeof(PetscInt), &(mapg->offset)); CHKERRQ(ierr);
  }
  binoff = mapg->offset;
  if(!mapg->count) {
    ierr = PetscMalloc(mapg->n*sizeof(PetscInt), &(mapg->count)); CHKERRQ(ierr);
  }
  bincount = mapg->count;
  /* Now compute bin offsets */
  for(j = 0; j < mapg->n; ++j) {
    binoff[j] = 0;
  }
  binoff[mapg->n] = 0;
  count = 0;
  link = inarr->first;
  while(link) {
    inhunk = link->hunk;
    SAMappingGraphOrderPointers_Private(inhunk->i,inhunk->w,inhunk->j,inhunk->mask,index,0,inidxi,inval,inidxj);
    for(i = 0; i < inhunk->length; ++i) {
      if(!local) {
        /* Convert to local by searching through mapg->supp. */
        SAMappingGraphLocalizeIndex(mapg->m,mapg->supp,inidxi[i],ind,count,last,low,high);
      }/* if(!local) */
      else {
        ind = inidxi[i];
      }
      if((ind < 0 || ind > mapg->m)){
        if(!drop) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Array index %D at %D not in the support",inidxi[i],count);
        else continue;
      }
      for(k = mapg->ijlen[ind]; k < mapg->ijlen[ind+1]; ++k) {
        ++(binoff[mapg->ij[k]+1]);
      }
    }/* for(i = 0; i < hunk->length; ++i) */
    for(j = 0; j < mapg->n; ++j) {
      binoff[j+1] += binoff[j];
    }
    link = link->next;
  }/* while(link) */
  /* Allocate space for output. */
  ierr = SAHunkCreate(binoff[mapg->n],inarr->mask,&outhunk); CHKERRQ(ierr);
  SAMappingGraphOrderPointers_Private(outhunk->i,outhunk->w,outhunk->j,outhunk->mask,index,0,outidxi,outval,outidxj);
  /* Now bin the input indices and values. */
  if(outidxi || (inval && outval) || (inidxj && outidxj) ) {
    for(j = 0; j < mapg->n; ++j) {
      bincount[j] = 0;
    }
    count = 0;
    link = inarr->first;
    while(link) {
      inhunk = link->hunk;
      SAMappingGraphOrderPointers_Private(inhunk->i,inhunk->w,inhunk->j,inhunk->mask,index,0,inidxi,inval,inidxj);
      for(i = 0; i < inhunk->length; ++i) {
        if(!local) {
          /* Convert to local by searching through mapg->supp. */
          SAMappingGraphLocalizeIndex(mapg->m,mapg->supp,inidxi[i],ind,count,last,low,high);
        }/* if(!local) */
        else {
          ind = inidxi[i];
        }
        if((ind < 0 || ind > mapg->m)){
          if(!drop) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Array index %D at %D not in the support",inidxi[i],count);
          else continue;
        }
        for(k = mapg->ijlen[ind]; k < mapg->ijlen[ind+1]; ++k) {
          j = mapg->ij[k];
          if(outidxi)            outidxi[binoff[j]+bincount[j]] = inidxi[i];
          if(outval && inval)    outval[binoff[j] +bincount[j]] = inval[i];
          if(outidxj && inidxj)  outidxj[binoff[j]+bincount[j]] = inidxj[i];
          ++bincount[j];
        }
        ++count;
      }/* for(i = 0; i < hunk->length; ++i) */
    }/* if(outidxi || (inval && outval) || (inidxj && outidxj)) */
    link = link->next;
  }/* while(link) */
  if(_outsizes) *_outsizes = bincount;
  PetscFunctionReturn(0);

}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingBin_Graph"
static PetscErrorCode SAMappingBin_Graph(SAMapping map, SA inarr, SAIndex index, SA outarr) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  if(index != SA_I && index != SA_J) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among SA components %D", index, inarr->mask);
  ierr = SAMappingGraphBinSA_Private(map, inarr, index, outarr, PETSC_NULL, PETSC_FALSE, PETSC_TRUE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingBinSplit_Graph"
static PetscErrorCode SAMappingBinSplit_Graph(SAMapping map, SA inarr, SAIndex index, SA *outarrs) 
{
  PetscErrorCode ierr;
  SAMapping_Graph *mapg = (SAMapping_Graph*)map->data;
  const PetscInt *outsizes;
  SA outarr;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  if(index != SA_I && index != SA_J) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among SA components %D", index, inarr->mask);
  ierr = SACreate(inarr->mask, &outarr);                                                             CHKERRQ(ierr);
  ierr = SAMappingGraphBinSA_Private(map, inarr, index, outarr, &outsizes, PETSC_FALSE, PETSC_TRUE); CHKERRQ(ierr);
  ierr = SASplit(outarr, mapg->n, outsizes, outarr->mask, outarrs);                                  CHKERRQ(ierr);
  ierr = SADestroy(outarr);                                                                          CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingBinLocal_Graph"
static PetscErrorCode SAMappingBinLocal_Graph(SAMapping map, SA inarr, SAIndex index, SA outarr) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  if(index != SA_I && index != SA_J) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among SA components %D", index, inarr->mask);
  ierr = SAMappingGraphBinSA_Private(map, inarr, index, outarr, PETSC_NULL, PETSC_TRUE, PETSC_TRUE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SAMappingBinSplitLocal_Graph"
static PetscErrorCode SAMappingBinSplitLocal_Graph(SAMapping map, SA inarr, SAIndex index, SA *outarrs) 
{
  PetscErrorCode ierr;
  SAMapping_Graph *mapg = (SAMapping_Graph*)map->data;
  const PetscInt *outsizes;
  SA outarr;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  if(index != SA_I && index != SA_J) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among SA components %D", index, inarr->mask);
  ierr = SACreate(inarr->mask, &outarr);                                                            CHKERRQ(ierr);
  ierr = SAMappingGraphBinSA_Private(map, inarr, index, outarr, &outsizes, PETSC_TRUE, PETSC_TRUE); CHKERRQ(ierr);
  ierr = SASplit(outarr, mapg->n, outsizes, outarr->mask, outarrs);                                 CHKERRQ(ierr);
  ierr = SADestroy(outarr);                                                                         CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Clears everything, but edges. */
#undef __FUNCT__
#define __FUNCT__ "SAMappingGraphClear_Private"
static PetscErrorCode SAMappingGraphClear_Private(SAMapping map)
{
  SAMapping_Graph   *mapg  = (SAMapping_Graph*)(map->data);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(mapg->supp) {
    ierr = PetscFree(mapg->supp);   CHKERRQ(ierr);
  }
  if(mapg->image) {
    ierr = PetscFree(mapg->image);  CHKERRQ(ierr);
  }
  if(mapg->ij) {
    ierr = PetscFree(mapg->ij);     CHKERRQ(ierr);
  }
  if(mapg->ijlen) {
    ierr = PetscFree(mapg->ijlen);  CHKERRQ(ierr);
  }
  if(mapg->offset) {
    ierr = PetscFree(mapg->offset); CHKERRQ(ierr);
  }
  if(mapg->count) {
    ierr = PetscFree(mapg->count);  CHKERRQ(ierr);
  }
  mapg->m = mapg->n = mapg->maxijlen = 0;
  PetscFunctionReturn(0);
}


/*
 Sort ix indices, if necessary.
 If ix duplicates exist, arrange iy indices in segments corresponding 
 to the images of the same input element. Remove iy duplicates from each 
 image segment and record the number of images in ijlen.  Convert the iy
 indices to a local numbering with the corresponding global indices stored
 in globals.
*/
#undef __FUNCT__  
#define __FUNCT__ "SAMappingGraphAssembleEdgesLocal_Private"
static PetscErrorCode SAMappingGraphAssembleEdgesLocal_Private(SAMapping map, SA edges)
{
  PetscErrorCode ierr;
  SAMapping_Graph *mapg = (SAMapping_Graph *) map->data;
  PetscInt *ixidx, *iyidx, ind,start, end, i, j, totalnij, totalnij2,maxnij, nij, m,n,*ij, *ijlen, *supp, *image, len;
  PetscBool xincreasing;
  SA medges;
  SAHunk hunk;
 
  PetscFunctionBegin;
  /* Warning: in this subroutine we manipulate SA internals directly. */
  /* Clear the old data. */
  ierr = SAMappingGraphClear_Private(map); CHKERRQ(ierr);
  len = edges->length;
  if(!len) {
    PetscFunctionReturn(0);
  }
  /* Merge the edges, if necessary. */
  if(edges->first != edges->last) {
    ierr = SAMerge(edges, &medges); CHKERRQ(ierr);
  }
  else {
    medges = edges;
  }
  if(medges->first->hunk->refcnt > 1) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpectedly high reference count on the edge SA hunk: %D", medges->first->hunk->refcnt);
  }
  hunk = medges->first->hunk;
  /* We manipulate the hunk data directly. */
  ixidx = hunk->i;
  iyidx = hunk->j;


  /* Determine whether ixidx is strictly increasing.  */
  ind = ixidx[0];
  xincreasing = PETSC_TRUE;
  for(i = 1; i < len; ++i) {
    if(ixidx[i] <= ind) {
      xincreasing = PETSC_FALSE;
      break;
    }
  }
  if(!xincreasing) {
    /* ixidx is not strictly increasing, no rearrangement of ixidx is also necessary. */
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
      PetscIntArrayCountUnique(iyidx+start,end-start,nij);
      totalnij += nij;
      maxnij = PetscMax(maxnij, nij);
    }
    start = end;
  }
  /* 
   Now we know the size of the support -- m, and the total size of concatenated image segments -- totalnij. 
   Allocate an array for recording the support indices -- supp.
   Allocate an array for recording the images of each support index -- ij.
   Allocate an array for counting the number of images for each support index -- ijlen.
   */
  ierr = PetscMalloc(sizeof(PetscInt)*(m+1), &ijlen);  CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*(totalnij),&ij); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*m, &supp); CHKERRQ(ierr);

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
    PetscIntArrayCopyUnique(iyidx+start,end-start,ij+j,nij);
    supp[i] = ixidx[start]; 
    j += nij;
    ++i;
    ijlen[i] = j;
    start = end;
  }
  /* (A) Construct image -- the set of unique iyidx indices. */
  /* (B) Endow the image with a local numbering.  */
  /* (C) Then convert ij to this local numbering. */

  /* (A) */
  ierr = PetscSortInt(len,iyidx); CHKERRQ(ierr);
  n = 0;
  PetscIntArrayCountUnique(iyidx,len,n);
  ierr = PetscMalloc(sizeof(PetscInt)*n, &image);   CHKERRQ(ierr);
  n = 0;
  PetscIntArrayCopyUnique(iyidx,len,image,n);
  /* (B) */
  ierr = PetscSortInt(n, image); CHKERRQ(ierr);
  /* (C) */
  /* 
   This is done by going through ij and for each k in it doing a binary search in image. 
   The result of the search is ind(k) -- the position of k in image. 
   Then ind(k) replace k in ij.
   */
  totalnij2 = 0;
  SAMappingGraphLocalizeIndices(n,image,totalnij,ij,totalnij2,ij,PETSC_TRUE); CHKERRQ(ierr);
  if(totalnij!=totalnij2) {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of image indices befow %D and after %D localization don't match", totalnij,totalnij2);
  }
  mapg->supp     = supp;
  mapg->m        = m;
  mapg->image    = image;
  mapg->n        = n;
  mapg->ij       = ij;
  mapg->ijlen    = ijlen;
  mapg->maxijlen = maxnij;

  if(edges != medges) {
    ierr = SADestroy(medges); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* SAMappingGraphAssembleEdgesLocal_Private() */


#undef  __FUNCT__
#define __FUNCT__ "SAMappingGraphAddEdgesSA"
PetscErrorCode SAMappingGraphAddEdgesSA(SAMapping map, SA edges) 
{
  SAMapping_Graph *mapg = (SAMapping_Graph*)(map->data);
  PetscErrorCode ierr;
  SALink link;
  SAHunk hunk;
  PetscFunctionBegin;
  PetscFunctionBegin;
  
  SAMappingCheckType(map, SA_MAPPING_GRAPH,1);
  PetscValidPointer(edges,2);
  link = edges->first;
  while(link) {
    hunk = link->hunk;
    ierr = PetscCheckIntArrayRange(hunk->length, hunk->i, 0, map->xlayout->N, PETSC_TRUE, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscCheckIntArrayRange(hunk->length, hunk->j, 0, map->ylayout->N, PETSC_TRUE, PETSC_NULL); CHKERRQ(ierr);
    link = link->next;
  ierr = SAAddArray(mapg->edges, edges); CHKERRQ(ierr);
  map->assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMappingGraphAddEdgesIS"
PetscErrorCode SAMappingGraphAddEdgesIS(SAMapping map, IS inix, IS iniy) 
{
  PetscErrorCode ierr;
  SAMapping_Graph   *mapg = (SAMapping_Graph *)(map->data);
  IS ix, iy;
  const PetscInt *ixidx, *iyidx;
  PetscInt len;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH,1);
  /*
      if input or output vertices are not defined, assume they are the total domain or range.
  */
  if(!inix) {
    ierr = ISCreateStride(((PetscObject)map)->comm,map->xlayout->n,map->xlayout->rstart,1,&(ix));CHKERRQ(ierr);
  }
  else ix = inix;
  if(!iy) {
    ierr = ISCreateStride(((PetscObject)map)->comm,map->ylayout->n,map->ylayout->rstart,1,&(iy));CHKERRQ(ierr);
  }
  else iy = iniy;
#if defined(PETSC_USE_DEBUG)
  /* Consistency checks. */
  /* Make sure the IS sizes are compatible */
  {
    PetscInt nix,niy;
    ierr = ISGetLocalSize(ix,&nix);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&niy);CHKERRQ(ierr);
    if (nix != niy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local IS sizes don't match");
    ierr = PetscCheckISRange(ix, 0, map->xlayout->N, PETSC_TRUE, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscCheckISRange(iy, 0, map->ylayout->N, PETSC_TRUE, PETSC_NULL); CHKERRQ(ierr);
  }
#endif
  ierr = ISGetLocalSize(ix,&len);  CHKERRQ(ierr);
  ierr = ISGetIndices(ix, &ixidx); CHKERRQ(ierr);
  ierr = ISGetIndices(iy, &iyidx); CHKERRQ(ierr);
  ierr = SAAddData(mapg->edges, len, ixidx, PETSC_NULL, iyidx); CHKERRQ(ierr);
  ierr = ISRestoreIndices(ix,&ixidx); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iy,&iyidx); CHKERRQ(ierr);
  if(!inix) {
    ierr = ISDestroy(ix); CHKERRQ(ierr);
  }
  if(!iniy) {
    ierr = ISDestroy(iy); CHKERRQ(ierr);
  }
  map->assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMappingGraphAddEdges"
PetscErrorCode SAMappingGraphAddEdges(SAMapping map, PetscInt len, const PetscInt *ii, const PetscInt *jj) {
  SAMapping_Graph *mapg = (SAMapping_Graph*)(map->data);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscFunctionBegin;
  
  SAMappingCheckType(map, SA_MAPPING_GRAPH,1);
  PetscValidPointer(ii,3);
  PetscValidPointer(jj,4);
  ierr = PetscCheckIntArrayRange(len, ii, 0, map->xlayout->N, PETSC_TRUE, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscCheckIntArrayRange(len, jj, 0, map->ylayout->N, PETSC_TRUE, PETSC_NULL); CHKERRQ(ierr);
  ierr = SAAddData(mapg->edges, len, ii, PETSC_NULL, jj); CHKERRQ(ierr);
  map->assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}/* SAMappingGraphAddEdges() */


#undef  __FUNCT__
#define __FUNCT__ "SAMappingGraphGetEdgeCount_Private"
static PetscErrorCode SAMappingGraphGetEdgeCount_Private(SAMapping map, PetscInt *_len)
{
  PetscErrorCode ierr;
  SAMapping_Graph   *mapg = (SAMapping_Graph *)(map->data);
  PetscInt len;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  if(!map->assembled) {
    ierr = SAGetLength(mapg->edges, &len);             CHKERRQ(ierr);
  }
  else len = mapg->n;
  *_len = len;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SAMappingGraphGetEdges_Private"
static PetscErrorCode SAMappingGraphGetEdges_Private(SAMapping map, PetscInt *ix, PetscInt *iy) 
{
  PetscErrorCode ierr;
  SAMapping_Graph   *mapg = (SAMapping_Graph *)(map->data);
  PetscInt len,i,k;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  if(!map->assembled) {
    ierr = SAGetLength(mapg->edges, &len);             CHKERRQ(ierr);
  }
  else len = mapg->n;
  if(!map->assembled) {
    ierr = SAGetData(mapg->edges, ix, PETSC_NULL, iy); CHKERRQ(ierr);
  }
  else {
    for(i = 0; i < mapg->m; ++i) {
      for(k = mapg->ijlen[i]; k < mapg->ijlen[i+1]; ++k) {
        if(ix) ix[k] = mapg->supp[i];
        if(iy) iy[k] = mapg->image[mapg->ij[k]];
      }
    }
  }
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SAMappingGraphGetEdges"
PetscErrorCode SAMappingGraphGetEdges(SAMapping map, PetscInt *_len, PetscInt **_ix, PetscInt **_iy) 
{
  PetscErrorCode ierr;
  PetscInt len, *ix = PETSC_NULL, *iy = PETSC_NULL;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  ierr = SAMappingGraphGetEdgeCount_Private(map, &len); CHKERRQ(ierr);
  if(!len) {
    if(_len) *_len = len;
    if(_ix)  *_ix  = PETSC_NULL;
    if(_iy)  *_iy  = PETSC_NULL;
    PetscFunctionReturn(0);
  }
  if(_len) *_len = len;
  if(!_ix && !_iy) PetscFunctionReturn(0);
  if(_ix) {
    ierr = PetscMalloc(sizeof(PetscInt)*len, _ix); CHKERRQ(ierr);
    ix = *_ix;
  }
  if(_iy) {
    ierr = PetscMalloc(sizeof(PetscInt)*len, _iy); CHKERRQ(ierr);
    iy = *_iy;
  }
  ierr = SAMappingGraphGetEdges_Private(map, ix,iy); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SAMappingGraphGetEdgesIS"
PetscErrorCode SAMappingGraphGetEdgesIS(SAMapping map, IS *_ix, IS *_iy) 
{
  PetscErrorCode ierr;
  PetscInt len, *ixidx, *iyidx, **_ixidx = PETSC_NULL, **_iyidx = PETSC_NULL;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  if(_ix) _ixidx = &ixidx;
  if(_iy) _iyidx = &iyidx;
  ierr = SAMappingGraphGetEdges(map, &len, _ixidx,_iyidx); CHKERRQ(ierr);
  if(_ix) {
    ierr = ISCreateGeneral(map->xlayout->comm, len, ixidx, PETSC_OWN_POINTER, _ix); CHKERRQ(ierr);
  }
  if(_iy) {
    ierr = ISCreateGeneral(map->ylayout->comm, len, iyidx, PETSC_OWN_POINTER, _iy); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMappingGraphGetEdgesSA"
PetscErrorCode SAMappingGraphGetEdgesSA(SAMapping map, SA *_edges) 
{
  PetscErrorCode ierr;
  SAMapping_Graph   *mapg = (SAMapping_Graph *)(map->data);
  SA     edges;
  SAHunk hunk;
  PetscFunctionBegin;
  SAMappingCheckType(map, SA_MAPPING_GRAPH, 1);
  if(!_edges) PetscFunctionReturn(0);
  ierr = SACreate(SA_I|SA_J, &edges); CHKERRQ(ierr);
  if(!map->assembled) {
    ierr = SAAddArray(edges, mapg->edges);      CHKERRQ(ierr);
  }
  else {
    ierr = SAGetHunk(edges, mapg->n, &hunk);                 CHKERRQ(ierr);
    ierr = SAMappingGraphGetEdges_Private(map, hunk->i, hunk->j); CHKERRQ(ierr);
  }
  *_edges = edges;
  PetscFunctionReturn(0);
}




#undef __FUNCT__
#define __FUNCT__ "SAMappingAssemblyBegin_Graph"
static PetscErrorCode SAMappingAssemblyBegin_Graph(SAMapping map)
{
  SAMapping_Graph   *mapg  = (SAMapping_Graph*)(map->data);
  PetscErrorCode ierr;
  PetscMPIInt    xsize;

  PetscFunctionBegin;
  ierr = SAMappingGraphClear_Private(map); CHKERRQ(ierr);

  ierr = MPI_Comm_size(map->xlayout->comm, &xsize); CHKERRQ(ierr);
  if(xsize > 1) {
    SA aedges;
    /* Assembled edges across the xlayout comm. */
    ierr = SACreate(mapg->edges->mask, &aedges);                             CHKERRQ(ierr);
    ierr = SAAssemble(mapg->edges, SA_I, map->xlayout, aedges);         CHKERRQ(ierr);
    /* Assemble edges locally. */
    ierr = SAMappingGraphAssembleEdgesLocal_Private(map,aedges);                  CHKERRQ(ierr);
    ierr = SADestroy(aedges);                                                CHKERRQ(ierr);
  }
  else {
    /* Assemble edges locally. */
    ierr = SAMappingGraphAssembleEdgesLocal_Private(map, mapg->edges);            CHKERRQ(ierr);
  }
  ierr = SAClear(mapg->edges); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* SAMappingAssemblyBegin_Graph() */


#undef __FUNCT__
#define __FUNCT__ "SAMappingAssemblyEnd_Graph"
static PetscErrorCode SAMappingAssemblyEnd_Graph(SAMapping map)
{
  PetscFunctionBegin;
  /* Currently a noop */
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SAMappingGetSupport_Graph"
PetscErrorCode SAMappingGetSupport_Graph(SAMapping map, PetscInt *_len, PetscInt **_supp) 
{
  PetscErrorCode ierr;
  SAMapping_Graph *mapg = (SAMapping_Graph *)(map->data);
  PetscFunctionBegin;
  if(_len) *_len = mapg->m;
  if(_supp) {
    ierr = PetscMalloc(sizeof(PetscInt)*mapg->m, _supp);              CHKERRQ(ierr);
    ierr = PetscMemcpy(*_supp, mapg->supp, sizeof(PetscInt)*mapg->m); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMappingGetSupportIS_Graph"
PetscErrorCode SAMappingGetSupportIS_Graph(SAMapping map, IS *supp) 
{
  PetscErrorCode ierr;
  SAMapping_Graph         *mapg = (SAMapping_Graph *)(map->data);
  PetscFunctionBegin;
  ierr = ISCreateGeneral(map->xlayout->comm, mapg->m, mapg->supp, PETSC_COPY_VALUES, supp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMappingGetSupportSA_Graph"
PetscErrorCode SAMappingGetSupportSA_Graph(SAMapping map, SA *_supp) 
{
  PetscErrorCode ierr;
  SAMapping_Graph         *mapg = (SAMapping_Graph *)(map->data);
  SA supp;
  PetscFunctionBegin;
  ierr = SACreate(SA_I, &supp);                                   CHKERRQ(ierr);
  ierr = SAAddData(supp, mapg->m, mapg->supp, PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);
  *_supp = supp;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SAMappingGetImage_Graph"
PetscErrorCode SAMappingGetImage_Graph(SAMapping map, PetscInt *_len, PetscInt **_image) 
{
  PetscErrorCode ierr;
  SAMapping_Graph *mapg = (SAMapping_Graph *)(map->data);
  PetscFunctionBegin;
  if(_len) *_len = mapg->n;
  if(_image) {
    ierr = PetscMalloc(sizeof(PetscInt)*mapg->n, _image);               CHKERRQ(ierr);
    ierr = PetscMemcpy(*_image, mapg->image, sizeof(PetscInt)*mapg->n); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMappingGetImageIS_Graph"
PetscErrorCode SAMappingGetImageIS_Graph(SAMapping map, IS *image) 
{
  PetscErrorCode ierr;
  SAMapping_Graph         *mapg = (SAMapping_Graph *)(map->data);
  PetscFunctionBegin;
  ierr = ISCreateGeneral(map->ylayout->comm, mapg->n, mapg->image, PETSC_COPY_VALUES, image); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMappingGetImageSA_Graph"
PetscErrorCode SAMappingGetImageSA_Graph(SAMapping map, SA *_image) 
{
  PetscErrorCode ierr;
  SAMapping_Graph         *mapg = (SAMapping_Graph *)(map->data);
  SA image;
  PetscFunctionBegin;
  ierr = SACreate(SA_I, &image);                                    CHKERRQ(ierr);
  ierr = SAAddData(image, mapg->m, mapg->image, PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);
  *_image = image;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SAMappingGetMaxImageSize_Graph"
PetscErrorCode SAMappingGetMaxImageSize_Graph(SAMapping map, PetscInt *maxsize)
{
  SAMapping_Graph *mapg = (SAMapping_Graph *)(map->data);
  PetscFunctionBegin;
  SAMappingCheckType(map,SA_MAPPING_GRAPH,1);
  *maxsize = mapg->maxijlen;
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "SAMappingGetOperator_Graph"
PetscErrorCode SAMappingGetOperator_Graph(SAMapping map, Mat *mat)
{
  PetscErrorCode ierr;
  PetscMPIInt xsize, ysize;
  Vec x, y;
  IS ix,iy;
  VecScatter scatter;
  PetscFunctionBegin;
  SAMappingCheckType(map,SA_MAPPING_GRAPH,1);
 
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
  ierr = SAMappingGraphGetEdgesIS(map, &ix, &iy);                    CHKERRQ(ierr);
  ierr = VecScatterCreate(x,ix, y,iy, &scatter);                   CHKERRQ(ierr);
  ierr = MatCreateScatter(((PetscObject)mat)->comm, scatter, mat); CHKERRQ(ierr);
  ierr = ISDestroy(&ix); CHKERRQ(ierr);
  ierr = ISDestroy(&iy); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* SAMappingGetOperator_Graph() */


#undef  __FUNCT__
#define __FUNCT__ "SAMappingView_Graph"
static PetscErrorCode SAMappingView_Graph(SAMapping map, PetscViewer v) 
{
  PetscFunctionBegin;
  SAMappingCheckType(map,SA_MAPPING_GRAPH,1);
  /* FIX: actually implement this */
  PetscFunctionReturn(0);
}/* SAMappingView_Graph() */



#undef  __FUNCT__
#define __FUNCT__ "SAMappingInvert_Graph"
static PetscErrorCode SAMappingInvert_Graph(SAMapping map, SAMapping *_imap) 
{
  SAMapping imap;
  PetscErrorCode ierr;
  IS ix, iy;
  PetscFunctionBegin;
  ierr = SAMappingCreate(((PetscObject)map)->comm, &imap);                                             CHKERRQ(ierr);
  ierr = SAMappingSetSizes(imap, map->xlayout->n, map->ylayout->n, map->xlayout->N, map->ylayout->N);  CHKERRQ(ierr);
  ierr = SAMappingGraphGetEdgesIS(map, &ix, &iy);                                                      CHKERRQ(ierr);
  ierr = SAMappingGraphAddEdgesIS(imap, iy,ix);                                                        CHKERRQ(ierr);
  ierr = ISDestroy(&ix);                      CHKERRQ(ierr);
  ierr = ISDestroy(&iy);                      CHKERRQ(ierr);
  ierr = SAMappingAssemblyBegin(imap);                                                                 CHKERRQ(ierr);
  ierr = SAMappingAssemblyEnd(imap);                                                                   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* SAMappingInvert_Graph() */

#undef  __FUNCT__
#define __FUNCT__ "SAMappingPushforward_Graph_Graph"
PetscErrorCode SAMappingPushforward_Graph_Graph(SAMapping map1, SAMapping map2, SAMapping *_map3) 
{
  PetscErrorCode ierr;
  SAMapping map3;
  PetscInt nsupp1, nsupp2, nsupp3, *supp1, *supp2, *supp3, imgsize1, *imgsizes1, imgsize2, *imgsizes2, *image1, *image2, *ixidx, *iyidx;
  PetscInt count, i1,i2,i1low,i1high,i2low,i2high,k;
  PetscFunctionBegin;
  SAMappingCheckType(map1,SA_MAPPING_GRAPH,1);
  SAMappingCheckType(map2,SA_MAPPING_GRAPH,2);
  PetscCheckSameComm(map1,1,map2,2);
  PetscValidPointer(_map3,3);
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
  ierr = SAMappingGetSupport(map1,  &nsupp1, &supp1);  CHKERRQ(ierr);
  ierr = SAMappingGetSupport(map2,  &nsupp2, &supp2);  CHKERRQ(ierr);
  /* Avoid computing the intersection, which may be unscalable in storage. */
  /* 
   Count the number of images of the intersection of supports under the "upward" (1) and "rightward" (2) maps. 
   It is done this way: supp1 is mapped by map2 obtaining offsets2, and supp2 is mapped by map1 obtaining offsets1.
   */
  ierr = PetscMalloc2(nsupp1, PetscInt, &imgsizes2, nsupp2, PetscInt, &imgsizes1);                                        CHKERRQ(ierr);
  ierr = SAMappingGraphMap_Private(map1,nsupp2,supp2,PETSC_NULL,PETSC_NULL,&imgsize1,PETSC_NULL,PETSC_NULL,PETSC_NULL,imgsizes1,PETSC_FALSE,PETSC_FALSE); CHKERRQ(ierr);
  ierr = SAMappingGraphMap_Private(map2,nsupp1,supp1,PETSC_NULL,PETSC_NULL,&imgsize2,PETSC_NULL,PETSC_NULL,PETSC_NULL,imgsizes2,PETSC_FALSE,PETSC_FALSE); CHKERRQ(ierr);
  /* Count the number of supp1 indices with nonzero images in map2 -- that's the size of the intersection. */
  nsupp3 = 0;
  for(k = 0; k < nsupp1; ++k) nsupp3 += (imgsizes1[k]>0);
#if defined(PETSC_USE_DEBUG)
  /* Now count the number of supp2 indices with nonzero images in map1: should be the same. */
  {
    PetscInt nsupp3_2 = 0;
    for(k = 0; k < nsupp2; ++k) nsupp3_2 += (imgsizes2[k]>0);
    if(nsupp3 != nsupp3_2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Intersections supports different in map1: %D and map2: %D", nsupp3, nsupp3_2);
  }
#endif
  /* Allocate indices for the intersection. */
  ierr = PetscMalloc(sizeof(PetscInt)*nsupp3, &supp3); CHKERRQ(ierr);
  nsupp3 = 0;
  for(k = 0; k < nsupp2; ++k) {
    if(imgsizes1[k]) {
      supp3[nsupp3] = supp2[k];
      ++nsupp3;
    }
  }
  ierr = PetscFree(supp1);                      CHKERRQ(ierr);
  ierr = PetscFree(supp2);                      CHKERRQ(ierr);
       
  /* 
   Now allocate the image storage and map the supp3 to obtain the "up" (1) and "right" (2) images.
   Recall that imgsizes1 are allocated for supp2, and imgsizes2 for supp1.
   */
  ierr = PetscMalloc2(imgsize1,PetscInt,&image1,imgsize2,PetscInt,&image2); CHKERRQ(ierr);
  ierr = SAMappingGraphMap_Private(map1,nsupp3,supp3,PETSC_NULL,PETSC_NULL,&imgsize1,image1,PETSC_NULL,PETSC_NULL,imgsizes1,PETSC_FALSE,PETSC_FALSE);  CHKERRQ(ierr);
  ierr = SAMappingGraphMap_Private(map2,nsupp3,supp3,PETSC_NULL,PETSC_NULL,&imgsize2,image2,PETSC_NULL,PETSC_NULL,imgsizes2,PETSC_FALSE,PETSC_FALSE);  CHKERRQ(ierr);
  ierr = PetscFree(supp3);  CHKERRQ(ierr);

  /* Count the total number of arrows to add to the pushed forward SAMapping. */
  count = 0;
  for(k = 0; k < nsupp3; ++k) {
    count += (imgsizes1[k])*(imgsizes2[k]);
  }
  /* Allocate storage for the composed indices. */
  ierr = PetscMalloc2(count, PetscInt, &ixidx, count, PetscInt, &iyidx); CHKERRQ(ierr);
  count= 0;
  i1low = 0;
  i2low = 0;
  for(k = 0; k < nsupp3; ++k) {
    i1high = i1low + imgsizes1[k];
    i2high = i2low + imgsizes2[k];
    for(i1 = i1low; i1 < i1high; ++i1) {
      for(i2 = i2low; i1 < i2high; ++i2) {
        ixidx[count] = image1[i1];
        iyidx[count] = image2[i2];
        ++count;
      }
    }
    i1low = i1high;
    i2low = i2high;
  }
  ierr = PetscFree2(image1,image2);       CHKERRQ(ierr);
  ierr = PetscFree2(imgsizes1,imgsizes2); CHKERRQ(ierr);
  /* Now construct the new SAMapping. */
  ierr = SAMappingCreate(((PetscObject)map1)->comm, &map3);                                               CHKERRQ(ierr);
  ierr = SAMappingSetType(map3,SA_MAPPING_GRAPH);                                                         CHKERRQ(ierr);
  ierr = SAMappingSetSizes(map3, map1->ylayout->n, map2->ylayout->n, map1->ylayout->N, map2->ylayout->N); CHKERRQ(ierr);
  ierr = SAMappingGraphAddEdges(map3,count,ixidx,iyidx);                                                  CHKERRQ(ierr);
  ierr = SAMappingAssemblyBegin(map3); CHKERRQ(ierr);
  ierr = SAMappingAssemblyEnd(map3);   CHKERRQ(ierr);
  ierr = PetscFree2(ixidx,iyidx);      CHKERRQ(ierr);

  *_map3 = map3;
  PetscFunctionReturn(0);
}





#undef  __FUNCT__
#define __FUNCT__ "SAMappingPullback_Graph_Graph"
PetscErrorCode SAMappingPullback_Graph_Graph(SAMapping map1, SAMapping map2, SAMapping *_map3) 
{
  PetscErrorCode ierr;
  SAMapping imap1,map3;

  PetscFunctionBegin;
  SAMappingCheckType(map1,SA_MAPPING_GRAPH,1);
  SAMappingCheckType(map2,SA_MAPPING_GRAPH,3);
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
  ierr = SAMappingInvert_Graph(map1, &imap1);               CHKERRQ(ierr);
  ierr = SAMappingPushforward_Graph_Graph(imap1, map2, &map3); CHKERRQ(ierr);
  ierr = SAMappingDestroy(imap1);                        CHKERRQ(ierr);
  *_map3 = map3;
  PetscFunctionReturn(0);
}



#undef  __FUNCT__
#define __FUNCT__ "SAMappingDestroy_Graph"
PetscErrorCode SAMappingDestroy_Graph(SAMapping map) {
  PetscErrorCode ierr;
  SAMapping_Graph          *mapg = (SAMapping_Graph *)(map->data);
  
  PetscFunctionBegin;
  ierr = SAMappingGraphClear_Private(map); CHKERRQ(ierr);
  ierr = SADestroy(mapg->edges);      CHKERRQ(ierr);
  ierr = PetscFree(mapg);                  CHKERRQ(ierr);
  map->data = PETSC_NULL;
  
  map->setup = PETSC_FALSE;
  map->assembled = PETSC_FALSE;

  ierr = PetscObjectChangeTypeName((PetscObject)map,0); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)map,"SAMappingPullback_graph_graph_C", "",PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)map,"SAMappingPushforward_graph_graph_C", "",PETSC_NULL); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* SAMappingDestroy_Graph() */

EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "SAMappingCreate_Graph"
PetscErrorCode SAMappingCreate_Graph(SAMapping map) {
  PetscErrorCode ierr;
  SAMapping_Graph  *mapg;
  PetscFunctionBegin;
  ierr = PetscNewLog(map, SAMapping_Graph, &mapg); CHKERRQ(ierr);
  ierr = SACreate(SA_I|SA_J, &(mapg->edges)); CHKERRQ(ierr);
  map->data = (void*)mapg;

  map->ops->view                 = SAMappingView_Graph;
  map->ops->setup                = SAMappingSetUp_SAMapping;
  map->ops->assemblybegin        = SAMappingAssemblyBegin_Graph;
  map->ops->assemblyend          = SAMappingAssemblyEnd_Graph;
  map->ops->getsupport           = SAMappingGetSupport_Graph;
  map->ops->getsupportis         = SAMappingGetSupportIS_Graph;
  map->ops->getsupportsa         = SAMappingGetSupportSA_Graph;
  map->ops->getimage             = SAMappingGetImage_Graph;
  map->ops->getimageis           = SAMappingGetImageIS_Graph;
  map->ops->getimagesa           = SAMappingGetImageSA_Graph;
  map->ops->getmaximagesize      = SAMappingGetMaxImageSize_Graph;
  map->ops->maplocal             = SAMappingMapLocal_Graph;
  map->ops->map                  = SAMappingMap_Graph;
  map->ops->binlocal             = SAMappingBinLocal_Graph;
  map->ops->bin                  = SAMappingBin_Graph;
  map->ops->mapsplitlocal        = SAMappingMapSplitLocal_Graph;
  map->ops->mapsplit             = SAMappingMapSplit_Graph;
  map->ops->binsplitlocal        = SAMappingBinSplitLocal_Graph;
  map->ops->binsplit             = SAMappingBinSplit_Graph;
  map->ops->invert               = SAMappingInvert_Graph;
  map->ops->getoperator          = SAMappingGetOperator_Graph;

  map->setup     = PETSC_FALSE;
  map->assembled = PETSC_FALSE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)map, 
                                           "SAMappingPullback_graph_grah_C", "SAMappingPullback_ismappinggraph_ismappinggraph", 
                                           SAMappingPullback_Graph_Graph); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)map, 
                                           "SAMappingPushforward_graph_graph_C", "SAMappingPushforward_ismappinggraph_ismappinggraph", 
                                           SAMappingPushforward_Graph_Graph); CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)map, SA_MAPPING_GRAPH); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* SAMappingCreate_Graph() */
EXTERN_C_END
