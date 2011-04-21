#define PETSCDM_DLL

#include "private/ismapimpl.h"  /*I "petscdm.h"  I*/

typedef struct {
  /* The following is used for mapping. */
  ISArray edges;
  PetscInt *supp;    /* support        -- global input indices from ownership range with nonempty images */
  PetscInt m;        /* size of suppot -- number of input indices with nonempty images */
  PetscInt *image;   /* image          -- distinct global output indices */
  PetscInt n;        /* size of image -- number of distinct global output indices */
  PetscInt *ij;      /* concatenated local images of ALL local input elements (i.e., all indices from the local ownership range), sorted within each image */
  PetscInt *ijlen;   /* image segment boundaries for each local input index */
  PetscInt maxijlen; /* max image segment size */
  /* The following is used for binning. */
  PetscInt *offset, *count; 
} ISMapping_Graph;

#define ISMappingGraphLocalize(tablen, table, i,ii,found,count,last,low,high) \
        /* Convert to local by searching through mapg->supp. */               \
        (found) = PETSC_FALSE;                                                \
        if((count) > 0) {                                                     \
          /* last and ii have valid previous values, that can be used to take \
             advantage of the already known information about the table. */   \
          if((i) > (last)) {                                                  \
            /* lower bound is still valid, but the upper bound might not be.*/\
            /*                                                                \
             table is ordered, hence, is a subsequence of the integers.       \
             Thus, the distance between ind and last in table is no greater   \
             than the distance between them within the integers: ind - last.  \
             Therefore, high raised by ind-last is a valid upper bound on ind.\
             */                                                               \
             (high) = PetscMin((mapg)->m, (high)+((i)-(last)));               \
            /* ii is the largest index in the table whose value does not      \
               exceed last; since i > last, i is located above ii within      \
               table */                                                       \
            (low) = (ii);                                                     \
          }                                                                   \
          if((i) < (last)) {                                                  \
            /* upper bound is still valid, but the lower bound might not be.*/\
            /*                                                                \
             table is ordered, hence, is a subsequence of the integers.       \
             Thus, the distance between i and last in table is no greater     \
             than the distance between them within the integers: last - i.    \
             Therefore, low lowered by i-last is a valid upper bound on i.    \
             */                                                               \
            (low) = PetscMax(0,(low)+((i)-last));                             \
            /* ii is the largest index of the table entry not exceeding last; \
              since i < last, i is located no higher than ii within table */  \
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
            (found) = PETSC_TRUE;                                             \
            break;                                                            \
          }                                                                   \
          ++(ii);                                                             \
        }

#define ISMappingGraphGetHunkPointers(hunk,mask,index,ii,ww,jj) \
  if((index) == ISARRAY_I) { \
    (ii) = (hunk)->i;       \
    if((mask) & ISARRAY_J){ \
      (jj) = (hunk)->j;     \
    }                       \
  }                         \
  else {                    \
    (ii) = (hunk)->j;       \
    if((mask) & ISARRAY_I) {\
      (jj) = (hunk)->i;     \
    }                       \
  }                         \
  if((mask) & ISARRAY_W) {  \
    (ww) = (hunk)->w;       \
  }


#undef __FUNCT__  
#define __FUNCT__ "ISMappingGraphMap_Private"
static PetscErrorCode ISMappingGraphMap_Private(ISMapping map, ISArray inarr, PetscInt index, PetscInt *outsize, PetscInt outidxi[], PetscScalar outval[], PetscInt outidxj[], PetscInt offsets[], PetscBool local, PetscBool drop) 
{
  ISMapping_Graph *mapg = (ISMapping_Graph*)map->data;
  const PetscInt *inidxi, *inidxj = PETSC_NULL;
  const PetscScalar *inval; 
  PetscInt     i,j,k,count;
  PetscBool    found;
  PetscInt     last,low,high,ind;
  ISArrayHunk  hunk;
  PetscFunctionBegin;

  if(offsets) offsets[0] = 0;
  j = 0;
  hunk = inarr->first;
  count = 0;
  while(hunk) {
    ISMappingGraphGetHunkPointers(hunk,inarr->mask,index,inidxi,inval,inidxj);
    for(i = 0; i < hunk->length; ++i) {
      if(!local) {
        /* Convert to local by searching through mapg->supp. */
        ISMappingGraphLocalize(mapg->m,mapg->supp,inidxi[i],ind,found,count,last,low,high);
        if(!found) ind = -1;
      }/* if(!local) */
      else {
        ind = inidxi[i];
      }
      if((ind < 0 || ind > mapg->m)){
        if(!drop) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %D at %D not in the support", inidxi[i], count);
        if(offsets) offsets[count+1] = offsets[count];
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
      if(offsets) offsets[count+1] = offsets[count] + (mapg->ijlen[ind+1]-mapg->ijlen[ind]);
      ++count;
    }/* for(i = 0; i < hunk->length; ++i) */
    hunk = hunk->next;
  }/* while(hunk) */
  if(outsize) *outsize = j;
  PetscFunctionReturn(0);
}/* ISMappingGraphMap_Private() */



#undef __FUNCT__  
#define __FUNCT__ "ISMappingMap_Graph"
static PetscErrorCode ISMappingMap_Graph(ISMapping map, ISArray inarr, ISArrayIndex index, ISArray outarr) 
{
  PetscErrorCode ierr;
  PetscInt outsize, *outidxi = PETSC_NULL, *outidxj = PETSC_NULL;
  PetscScalar *outval = PETSC_NULL;
  ISArrayHunk  outhunk;
  PetscFunctionBegin;
  ISMappingCheckType(map, IS_MAPPING_GRAPH, 1);
  if(index != ISARRAY_I && index != ISARRAY_J) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among ISArray components %D", index, inarr->mask);
  /* Determine the size of the output. */
  ierr = ISMappingGraphMap_Private(map,inarr,index,&outsize,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_FALSE,PETSC_FALSE); CHKERRQ(ierr);
  /* Get space for the output. */
  ierr = ISArrayClear(outarr);                    CHKERRQ(ierr);
  ierr = ISArrayGetHunk(outarr,outsize,&outhunk); CHKERRQ(ierr);
  ISMappingGraphGetHunkPointers(outhunk,outarr->mask,index,outidxi,outval,outidxj);
  ierr = ISMappingGraphMap_Private(map,inarr,index,PETSC_NULL,outidxi,outval,outidxj,PETSC_NULL,PETSC_FALSE,PETSC_FALSE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingMap_Graph() */


#undef __FUNCT__  
#define __FUNCT__ "ISMappingMapLocal_Graph"
static PetscErrorCode ISMappingMapLocal_Graph(ISMapping map, ISArray inarr, ISArrayIndex index, ISArray outarr) 
{
  PetscErrorCode ierr;
  PetscInt outsize, *outidxi = PETSC_NULL, *outidxj = PETSC_NULL;
  PetscScalar *outval = PETSC_NULL;
  ISArrayHunk  outhunk;
  PetscFunctionBegin;
  ISMappingCheckType(map, IS_MAPPING_GRAPH, 1);
  if(index != ISARRAY_I && index != ISARRAY_J) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among ISArray components %D", index, inarr->mask);
  /* Determine the size of the output. */
  ierr = ISMappingGraphMap_Private(map,inarr,index,&outsize,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_TRUE,PETSC_FALSE); CHKERRQ(ierr);
  /* Get space for the output. */
  ierr = ISArrayClear(outarr);                    CHKERRQ(ierr);
  ierr = ISArrayGetHunk(outarr,outsize,&outhunk); CHKERRQ(ierr);
  ISMappingGraphGetHunkPointers(outhunk,outarr->mask,index,outidxi,outval,outidxj);
  ierr = ISMappingGraphMap_Private(map,inarr,index,PETSC_NULL,outidxi,outval,outidxj,PETSC_NULL,PETSC_TRUE,PETSC_FALSE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingMapLocal_Graph() */


#undef __FUNCT__  
#define __FUNCT__ "ISMappingGraphBin_Private"
static PetscErrorCode ISMappingGraphBin_Private(ISMapping map, ISArray inarr, PetscInt index, PetscInt *outsize, PetscInt outidxi[], PetscScalar outval[], PetscInt outidxj[], const PetscInt *offsets[], PetscBool local, PetscBool drop) 
{
  PetscErrorCode ierr;
  ISMapping_Graph *mapg = (ISMapping_Graph*)map->data;
  PetscInt      *binoff, *bincount;
  const PetscInt *inidxi, *inidxj = PETSC_NULL;
  const PetscScalar *inval;
  PetscInt     i,j,k,count;
  PetscInt     last,low,high,ind;
  PetscBool    found;
  ISArrayHunk   hunk;
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
  for(j = 0; j < mapg->n; ++j) {
    binoff[j] = 0;
    bincount[j] = 0;
  }
  binoff[mapg->n] = 0;
  /* Now compute bin offsets */
  count = 0;
  hunk = inarr->first;
  while(hunk) {
    ISMappingGraphGetHunkPointers(hunk,inarr->mask,index,inidxi,inval,inidxj);
    for(i = 0; i < hunk->length; ++i) {
      if(!local) {
        /* Convert to local by searching through mapg->supp. */
        ISMappingGraphLocalize(mapg->m,mapg->supp,inidxi[i],ind,found,count,last,low,high);
        if(!found) ind = -1;
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
  }/* while(hunk) */
  /* Now bin the input indices and values. */
  if(outidxi || (inval && outval) || (inidxj && outidxj) ) {
    for(j = 0; j < mapg->n; ++j) {
      bincount[j] = 0;
    }
    count = 0;
    hunk = inarr->first;
    while(hunk) {
      ISMappingGraphGetHunkPointers(hunk,inarr->mask,index,inidxi,inval,inidxj);
      for(i = 0; i < hunk->length; ++i) {
        if(!local) {
          /* Convert to local by searching through mapg->supp. */
          ISMappingGraphLocalize(mapg->m,mapg->supp,inidxi[i],ind,found,count,last,low,high);
          if(!found) ind = -1;
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
      }/* for(i = 0; i < hunk->length; ++i) */
    }/* if(outidxi || (inval && outval) || (inidxj && outidxj)) */
  }/* while(hunk) */
  if(outsize) *outsize = binoff[mapg->n];
  if(offsets) *offsets = binoff;
  PetscFunctionReturn(0);
}/* ISMappingGraphBin_Private() */

#undef __FUNCT__  
#define __FUNCT__ "ISMappingBin_Graph"
static PetscErrorCode ISMappingBin_Graph(ISMapping map, ISArray inarr, ISArrayIndex index, ISArray outarr) 
{
  PetscErrorCode ierr;
  PetscInt outsize, *outidxi = PETSC_NULL, *outidxj = PETSC_NULL;
  PetscScalar *outval = PETSC_NULL;
  ISArrayHunk  outhunk;
  PetscFunctionBegin;
  ISMappingCheckType(map, IS_MAPPING_GRAPH, 1);
  if(index != ISARRAY_I && index != ISARRAY_J) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among ISArray components %D", index, inarr->mask);
  /* Determine the size of the output: it's the same as the size of the output for mapping inarr, which is faster than ISMappingGraphBin_Private. */
  ierr = ISMappingGraphMap_Private(map,inarr,index,&outsize,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_FALSE,PETSC_FALSE); CHKERRQ(ierr);
  /* Get space for the output. */
  ierr = ISArrayClear(outarr);                    CHKERRQ(ierr);
  ierr = ISArrayGetHunk(outarr,outsize,&outhunk); CHKERRQ(ierr);
  ISMappingGraphGetHunkPointers(outhunk,outarr->mask,index,outidxi,outval,outidxj);
  ierr = ISMappingGraphBin_Private(map,inarr,index,PETSC_NULL,outidxi,outval,outidxj,PETSC_NULL,PETSC_FALSE,PETSC_FALSE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingBin_Graph() */

#undef __FUNCT__  
#define __FUNCT__ "ISMappingBinLocal_Graph"
static PetscErrorCode ISMappingBinLocal_Graph(ISMapping map, ISArray inarr, ISArrayIndex index, ISArray outarr) 
{
  PetscErrorCode ierr;
  PetscInt outsize, *outidxi = PETSC_NULL, *outidxj = PETSC_NULL;
  PetscScalar *outval = PETSC_NULL;
  ISArrayHunk  outhunk;
  PetscFunctionBegin;
  ISMappingCheckType(map, IS_MAPPING_GRAPH, 1);
  if(index != ISARRAY_I && index != ISARRAY_J) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among ISArray components %D", index, inarr->mask);
  /* Determine the size of the output: it's the same as the size of the output for mapping inarr, which is faster than ISMappingGraphBin_Private. */
  ierr = ISMappingGraphMap_Private(map,inarr,index,&outsize,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_TRUE,PETSC_FALSE); CHKERRQ(ierr);
  /* Get space for the output. */
  ierr = ISArrayClear(outarr);                    CHKERRQ(ierr);
  ierr = ISArrayGetHunk(outarr,outsize,&outhunk); CHKERRQ(ierr);
  ISMappingGraphGetHunkPointers(outhunk,outarr->mask,index,outidxi,outval,outidxj);
  ierr = ISMappingGraphBin_Private(map,inarr,index,PETSC_NULL,outidxi,outval,outidxj,PETSC_NULL,PETSC_TRUE,PETSC_FALSE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingBinLocal_Graph() */


#undef __FUNCT__  
#define __FUNCT__ "ISMappingMapSplit_Graph"
static PetscErrorCode ISMappingMapSplit_Graph(ISMapping map, ISArray inarr, ISArrayIndex index, ISArray *outarrs) 
{
  PetscErrorCode ierr;
  PetscInt outsize, *outidxi = PETSC_NULL, *outidxj = PETSC_NULL;
  PetscScalar *outval = PETSC_NULL;
  ISArrayHunk  outhunk, subhunk;
  PetscInt offsets[1024], *off, i;
  PetscFunctionBegin;
  ISMappingCheckType(map, IS_MAPPING_GRAPH, 1);
  if(index != ISARRAY_I && index != ISARRAY_J) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among ISArray components %D", index, inarr->mask);
  /* Determine the size of the output. */
  ierr = ISMappingGraphMap_Private(map,inarr,index,&outsize,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_FALSE,PETSC_FALSE); CHKERRQ(ierr);
  /* Create a new hunk for the output: the hunk is then split into subhunks that go into individual subarrays. */
  ierr = ISArrayHunkCreate(outsize,inarr->mask,&outhunk); CHKERRQ(ierr);
  /* Allocate space for offsets, if necessary. */
  if(inarr->length < 1024) {
    off = offsets;
  }
  else {
    ierr = PetscMalloc(sizeof(PetscInt)*(inarr->length+1), &off); CHKERRQ(ierr);
  }
  ISMappingGraphGetHunkPointers(outhunk,inarr->mask,index,outidxi,outval,outidxj);
  ierr = ISMappingGraphMap_Private(map,inarr,index,PETSC_NULL,outidxi,outval,outidxj,off,PETSC_FALSE,PETSC_FALSE); CHKERRQ(ierr);
  /* Break output up into subarray.s */
  for(i = 0; i < inarr->length; ++i) {
    ierr = ISArrayClear(outarrs[i]);                                   CHKERRQ(ierr);
    ierr = ISArrayHunkGetSubHunk(outhunk,off[i+1]-off[i],inarr->mask,&subhunk); CHKERRQ(ierr);
    ierr = ISArrayAddHunk(outarrs[i],subhunk);                                   CHKERRQ(ierr);
  }
  if(off != offsets) {
    ierr = PetscFree(off); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* ISMappingMapSplit_Graph() */


#undef __FUNCT__  
#define __FUNCT__ "ISMappingMapSplitLocal_Graph"
static PetscErrorCode ISMappingMapSplitLocal_Graph(ISMapping map, ISArray inarr, ISArrayIndex index, ISArray *outarrs) 
{
  PetscErrorCode ierr;
  PetscInt outsize, *outidxi = PETSC_NULL, *outidxj = PETSC_NULL;
  PetscScalar *outval = PETSC_NULL;
  ISArrayHunk  outhunk, subhunk;
  PetscInt offsets[1024], *off, i;
  PetscFunctionBegin;
  ISMappingCheckType(map, IS_MAPPING_GRAPH, 1);
  if(index != ISARRAY_I && index != ISARRAY_J) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among ISArray components %D", index, inarr->mask);
  /* Determine the size of the output. */
  ierr = ISMappingGraphMap_Private(map,inarr,index,&outsize,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_TRUE,PETSC_FALSE); CHKERRQ(ierr);
  /* Create a new hunk for the output: the hunk is then split into subhunks that go into individual subarrays. */
  ierr = ISArrayHunkCreate(outsize,inarr->mask,&outhunk); CHKERRQ(ierr);
  /* Allocate space for offsets, if necessary. */
  if(inarr->length < 1024) {
    off = offsets;
  }
  else {
    ierr = PetscMalloc(sizeof(PetscInt)*(inarr->length+1), &off); CHKERRQ(ierr);
  }
  ISMappingGraphGetHunkPointers(outhunk,inarr->mask,index,outidxi,outval,outidxj);
  ierr = ISMappingGraphMap_Private(map,inarr,index,PETSC_NULL,outidxi,outval,outidxj,off,PETSC_TRUE,PETSC_FALSE); CHKERRQ(ierr);
  /* Break output up into subarray.s */
  for(i = 0; i < inarr->length; ++i) {
    ierr = ISArrayClear(outarrs[i]);                                            CHKERRQ(ierr);
    ierr = ISArrayHunkGetSubHunk(outhunk,off[i+1]-off[i],inarr->mask,&subhunk); CHKERRQ(ierr);
    ierr = ISArrayAddHunk(outarrs[i],subhunk);                                   CHKERRQ(ierr);
  }
  if(off != offsets) {
    ierr = PetscFree(off); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* ISMappingMapSplitLocal_Graph() */


#undef __FUNCT__  
#define __FUNCT__ "ISMappingBinSplit_Graph"
static PetscErrorCode ISMappingBinSplit_Graph(ISMapping map, ISArray inarr, ISArrayIndex index, ISArray *outarrs) 
{
  PetscErrorCode ierr;
  PetscInt outsize, *outidxi = PETSC_NULL, *outidxj = PETSC_NULL;
  PetscScalar *outval = PETSC_NULL;
  ISArrayHunk  outhunk, subhunk;
  const PetscInt *off;
  PetscInt i;
  PetscFunctionBegin;
  ISMappingCheckType(map, IS_MAPPING_GRAPH, 1);
  if(index != ISARRAY_I && index != ISARRAY_J) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among ISArray components %D", index, inarr->mask);
  /* Determine the size of the output: it's the same as the size of the output for mapping inarr, which is faster than ISMappingGraphBin_Private. */
  ierr = ISMappingGraphMap_Private(map,inarr,index,&outsize,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_FALSE,PETSC_FALSE); CHKERRQ(ierr);
  /* Create a new hunk for the output: the hunk is then split into subhunks that go into individual subarrays. */
  ierr = ISArrayHunkCreate(outsize,inarr->mask,&outhunk); CHKERRQ(ierr);
  ISMappingGraphGetHunkPointers(outhunk,inarr->mask,index,outidxi,outval,outidxj);
  ierr = ISMappingGraphBin_Private(map,inarr,index,PETSC_NULL,outidxi,outval,outidxj,&off,PETSC_FALSE,PETSC_FALSE); CHKERRQ(ierr);
  /* Break output up into subarray.s */
  for(i = 0; i < inarr->length; ++i) {
    ierr = ISArrayClear(outarrs[i]);                                            CHKERRQ(ierr);
    ierr = ISArrayHunkGetSubHunk(outhunk,off[i+1]-off[i],inarr->mask,&subhunk); CHKERRQ(ierr);
    ierr = ISArrayAddHunk(outarrs[i],subhunk);                                  CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* ISMappingBinSplit_Graph() */


#undef __FUNCT__  
#define __FUNCT__ "ISMappingBinSplitLocal_Graph"
static PetscErrorCode ISMappingBinSplitLocal_Graph(ISMapping map, ISArray inarr, ISArrayIndex index, ISArray *outarrs) 
{
  PetscErrorCode ierr;
  PetscInt outsize, *outidxi = PETSC_NULL, *outidxj = PETSC_NULL, i;
  PetscScalar *outval = PETSC_NULL;
  ISArrayHunk  outhunk, subhunk;
  const PetscInt *off;
  PetscFunctionBegin;
  ISMappingCheckType(map, IS_MAPPING_GRAPH, 1);
  if(index != ISARRAY_I && index != ISARRAY_J) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid index %D", index);
  if(!(inarr->mask & index)) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index %D not present among ISArray components %D", index, inarr->mask);
  /* Determine the size of the output: it's the same as the size of the output for mapping inarr, which is faster than ISMappingGraphBin_Private. */
  ierr = ISMappingGraphMap_Private(map,inarr,index,&outsize,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_TRUE,PETSC_FALSE); CHKERRQ(ierr);
  /* Create a new hunk for the output: the hunk is then split into subhunks that go into individual subarrays. */
  ierr = ISArrayHunkCreate(outsize,inarr->mask,&outhunk); CHKERRQ(ierr);
  ISMappingGraphGetHunkPointers(outhunk,inarr->mask,index,outidxi,outval,outidxj);
  ierr = ISMappingGraphBin_Private(map,inarr,index,PETSC_NULL,outidxi,outval,outidxj,&off,PETSC_TRUE,PETSC_FALSE); CHKERRQ(ierr);
  /* Break output up into subarray.s */
  for(i = 0; i < inarr->length; ++i) {
    ierr = ISArrayClear(outarrs[i]);                                            CHKERRQ(ierr);
    ierr = ISArrayHunkGetSubHunk(outhunk,off[i+1]-off[i],inarr->mask,&subhunk); CHKERRQ(ierr);
    ierr = ISArrayAddHunk(outarrs[i],subhunk);                                  CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* ISMappingBinSplitLocal_Graph() */



/*
 Sort ix indices, if necessary.
 If ix duplicates exist, arrange iy indices in segments corresponding 
 to the images of the same input element. Remove iy duplicates from each 
 image segment and record the number of images in ijlen.  Convert the iy
 indices to a local numbering with the corresponding global indices stored
 in globals.
*/
#undef __FUNCT__  
#define __FUNCT__ "ISMappingGraphAssembleLocal_Private"
static PetscErrorCode ISMappingGraphAssembleLocal_Private(ISMapping map, PetscInt len, const PetscInt ixidx_const[], const PetscInt iyidx_const[], ISMapping_Graph *mapg)
{
  PetscErrorCode ierr;
  PetscInt *ixidx, *iyidx;
  PetscInt ind,start, end, i, j, totalnij,maxnij, nij, m,n;
  PetscBool xincreasing;
  PetscInt *ij, *ijlen, *supp, *image;
  PetscFunctionBegin;

  /* Assume ixidx_const and iyidx_const have the same size. */
  if(!len) {
    mapg->m        = 0;
    mapg->supp     = PETSC_NULL;
    mapg->n        = 0;
    mapg->image    = PETSC_NULL;
    mapg->ij       = PETSC_NULL;
    mapg->ijlen    = PETSC_NULL;
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
  ierr = ISMappingGraph_LocateIndices(n,image,totalnij,ij,&totalnij,ij,PETSC_TRUE); CHKERRQ(ierr);
  mapg->supp     = supp;
  mapg->m        = m;
  mapg->image    = image;
  mapg->n        = n;
  mapg->ij       = ij;
  mapg->ijlen    = ijlen;
  mapg->maxijlen = maxnij;
  PetscFunctionReturn(0);
}/* ISMappingGraph_AssembleLocal() */


#undef  __FUNCT__
#define __FUNCT__ "ISMappingGraphAddEdgeArray"
PetscErrorCode ISMappingGraphAddEdgeArray(ISMapping map, ISArray edges) {
  ISMapping_Graph *mapg = (ISMapping_Graph*)(map->data);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscFunctionBegin;
  
  ISMappingCheckType(map, IS_MAPPING_GRAPH,1);
  if(mapg->ix) {
    ierr = ISDestroy(mapg->ix); CHKERRQ(ierr);
  }
  mapg->ix = ix; 
  if(ix) {ierr = PetscObjectReference((PetscObject)ix); CHKERRQ(ierr);}
  if(mapg->iy) {
    ierr = ISDestroy(mapg->iy); CHKERRQ(ierr);
  }
  mapg->iy = iy; 
  if(iy){ierr = PetscObjectReference((PetscObject)iy); CHKERRQ(ierr);}
  map->assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}/* ISMappingAddEdgeArray() */

#undef  __FUNCT__
#define __FUNCT__ "ISMappingGraphGetEdgeArray"
PetscErrorCode ISMappingGraphGetEdgeArray(ISMapping map, ISArray *_edges) {
  PetscErrorCode ierr;
  ISMapping_Graph   *mapg = (ISMapping_Graph *)(map->data);
  PetscInt len;
  PetscInt i,j;
  ISArray edges;
  ISArrayHunk hunk;
  PetscFunctionBegin;
  ISMappingCheckType(map, IS_MAPPING_GRAPH, 1);
  PetscValidPointer(_edges,2);
  if(mapg->edges) {
    ierr = ISArrayDuplicate(mapg->edges, _edges); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ISMappingCheckAssembled(map, PETSC_TRUE,1);
  len = mapg->ijlen[mapg->m];
  ierr = ISArrayCreate(ISARRAY_I|ISARRAY_J, &edges);         CHKERRQ(ierr);
  ierr = ISArrayHunkCreate(len, ISARRAY_I|ISARRAY_J, &hunk); CHKERRQ(ierr);
  for(i = 0; i < mapg->m; ++i) {
    for(k = mapg->ijlen[i]; k < mapg->ijlen[i+1]; ++k) {
      hunk->i[k] = mapg->supp[i];
      hunk->j[k] = mapg->image[mapg->ij[k]]; /* Could PetscMemcpy map->image to hunk->j instead, but keep this for clarity. */
    }
  }
  ierr = ISArrayAddHunk(arr,hunk); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingGraphGetEdgeArray() */

#undef __FUNCT__
#define __FUNCT__ "ISMappingAssemblyBegin_Graph"
static PetscErrorCode ISMappingAssemblyBegin_Graph(ISMapping map)
{
  ISMapping_Graph   *mapg  = (ISMapping_Graph*)(map->data);
  PetscErrorCode ierr;
  PetscMPIInt    xsize;

  PetscFunctionBegin;
  /*
      if input or output vertices are not defined, assume they are the total domain or range.
  */
  if(!mapg->ix) {
    ierr = ISCreateStride(((PetscObject)map)->comm,map->xlayout->n,map->xlayout->rstart,1,&(mapg->ix));CHKERRQ(ierr);
  }
  if(!mapg->iy) {
    ierr = ISCreateStride(((PetscObject)map)->comm,map->ylayout->n,map->ylayout->rstart,1,&(mapg->iy));CHKERRQ(ierr);
  }
#if defined(PETSC_USE_DEBUG)
  /* Consistency checks. */
  /* Make sure the IS sizes are compatible */
  {
    PetscInt nix, niy;
    ierr = ISGetLocalSize(mapg->ix,&nix);CHKERRQ(ierr);
    ierr = ISGetLocalSize(mapg->iy,&niy);CHKERRQ(ierr);
    if (nix != niy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local IS sizes don't match");
    ierr = ISMappingGraph_CheckISRange(mapg->ix, 0, map->xlayout->N, PETSC_TRUE, PETSC_NULL); CHKERRQ(ierr);
    ierr = ISMappingGraph_CheckISRange(mapg->iy, 0, map->ylayout->N, PETSC_TRUE, PETSC_NULL); CHKERRQ(ierr);
  }
#endif

  if(mapg->supp) {
    ierr = PetscFree(mapg->supp); CHKERRQ(ierr);
  }
  if(mapg->image) {
    ierr = PetscFree(mapg->image); CHKERRQ(ierr);
  }
  if(mapg->ij) {
    ierr = PetscFree(mapg->ij); CHKERRQ(ierr);
  }

  ierr = MPI_Comm_size(map->xlayout->comm, &xsize); CHKERRQ(ierr);
  if(xsize > 1) {
    PetscInt len, alen;
    const PetscInt *ixidx, *iyidx;
    PetscInt *aixidx, *aiyidx;
    ierr = ISGetLocalSize(mapg->ix, &len); CHKERRQ(ierr);
    ierr = ISGetIndices(mapg->ix, &ixidx); CHKERRQ(ierr);
    ierr = ISGetIndices(mapg->iy, &iyidx); CHKERRQ(ierr);
    /* Assemble edges in parallel. */
    ierr = ISMappingGraph_AssembleMPI(map, len, ixidx, iyidx, &alen, &aixidx, &aiyidx); CHKERRQ(ierr);
    ierr = ISRestoreIndices(mapg->ix, &ixidx); CHKERRQ(ierr);
    ierr = ISRestoreIndices(mapg->iy, &iyidx); CHKERRQ(ierr);
    /* Assemble edges locally. */
    ierr = ISMappingGraph_AssembleLocal(map, alen, aixidx, aiyidx, mapg); CHKERRQ(ierr);
  }
  else {
    PetscInt len;
    const PetscInt *ixidx, *iyidx;
    ierr = ISGetLocalSize(mapg->ix, &len); CHKERRQ(ierr);
    ierr = ISGetIndices(mapg->ix, &ixidx); CHKERRQ(ierr);
    ierr = ISGetIndices(mapg->iy, &iyidx); CHKERRQ(ierr);
     /* Assemble edges locally. */
    ierr = ISMappingGraph_AssembleLocal(map, len, ixidx, iyidx, mapg); CHKERRQ(ierr);

    ierr = ISRestoreIndices(mapg->ix, &ixidx); CHKERRQ(ierr);
    ierr = ISRestoreIndices(mapg->iy, &iyidx); CHKERRQ(ierr);
  }
  ierr = ISDestroy(mapg->ix); CHKERRQ(ierr);
  mapg->ix = PETSC_NULL;
  ierr = ISDestroy(mapg->iy); CHKERRQ(ierr);
  mapg->iy = PETSC_NULL;
  PetscFunctionReturn(0);
}/* ISMappingAssemblyBegin_Graph() */


#undef __FUNCT__
#define __FUNCT__ "ISMappingAssemblyEnd_Graph"
static PetscErrorCode ISMappingAssemblyEnd_Graph(ISMapping map)
{
  PetscFunctionBegin;
  /* Currently a noop */
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISMappingGetSupportIS_Graph"
PetscErrorCode ISMappingGetSupportIS_Graph(ISMapping map, IS *supp) {
  PetscErrorCode ierr;
  ISMapping_Graph         *mapg = (ISMapping_Graph *)(map->data);
  PetscFunctionBegin;
  ierr = ISCreateGeneral(map->xlayout->comm, mapg->m, mapg->supp, PETSC_COPY_VALUES, supp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingGetSupportIS_Graph() */


#undef  __FUNCT__
#define __FUNCT__ "ISMappingGetImageIS_Graph"
PetscErrorCode ISMappingGetImageIS_Graph(ISMapping map, IS *image) {
  PetscErrorCode ierr;
  ISMapping_Graph         *mapg = (ISMapping_Graph *)(map->data);
  PetscFunctionBegin;
  ierr = ISCreateGeneral(map->ylayout->comm, mapg->n, mapg->image, PETSC_COPY_VALUES, image); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingGetImageIS_Graph() */


#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetMaxImageSizeLocal_Graph"
PetscErrorCode ISMappingGetMaxImageSizeLocal_Graph(ISMapping map, PetscInt *maxsize)
{
  ISMapping_Graph *mapg = (ISMapping_Graph *)(map->data);
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_GRAPH,1);
  *maxsize = mapg->maxijlen;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetImageSizeLocal_Graph"
PetscErrorCode ISMappingGetImageSizeLocal_Graph(ISMapping map, PetscInt *size)
{
  ISMapping_Graph *mapg = (ISMapping_Graph *)(map->data);
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_GRAPH,1);
  *size = mapg->n;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetSupportSizeLocal_Graph"
PetscErrorCode ISMappingGetSupportSizeLocal_Graph(ISMapping map, PetscInt *size)
{
  ISMapping_Graph *mapg = (ISMapping_Graph *)(map->data);
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_GRAPH,1);
  *size = mapg->m;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetOperator_Graph"
PetscErrorCode ISMappingGetOperator_Graph(ISMapping map, Mat *mat)
{
  PetscErrorCode ierr;
  PetscMPIInt xsize, ysize;
  Vec x, y;
  IS ix,iy;
  VecScatter scatter;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_GRAPH,1);
 
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
  ierr = ISMappingGraphGetEdges(map, &ix, &iy);                   CHKERRQ(ierr);
  ierr = VecScatterCreate(x,ix, y,iy, &scatter);               CHKERRQ(ierr);
  ierr = MatCreateScatter(((PetscObject)mat)->comm, scatter, mat); CHKERRQ(ierr);
  ierr = ISDestroy(ix); CHKERRQ(ierr);
  ierr = ISDestroy(iy); CHKERRQ(ierr);
  ierr = VecDestroy(x); CHKERRQ(ierr);
  ierr = VecDestroy(y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingGetOperator_Graph() */


#undef  __FUNCT__
#define __FUNCT__ "ISMappingView_Graph"
static PetscErrorCode ISMappingView_Graph(ISMapping map, PetscViewer v) 
{
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_GRAPH,1);
  /* FIX: actually implement this */
  PetscFunctionReturn(0);
}/* ISMappingView_Graph() */



#undef  __FUNCT__
#define __FUNCT__ "ISMappingInvert_Graph"
static PetscErrorCode ISMappingInvert_Graph(ISMapping map, ISMapping *imap) 
{
  PetscErrorCode ierr;
  IS ix, iy;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_GRAPH,1);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ierr = ISMappingCreate(((PetscObject)map)->comm, imap); CHKERRQ(ierr);
  ierr = ISMappingSetSizes(*imap, map->xlayout->n, map->ylayout->n, map->xlayout->N, map->ylayout->N); CHKERRQ(ierr);
  ierr = ISMappingGraphGetEdges(map, &ix,&iy);  CHKERRQ(ierr);
  ierr = ISMappingGraphSetEdges(*imap,iy, ix);  CHKERRQ(ierr);
  ierr = ISDestroy(ix);                      CHKERRQ(ierr);
  ierr = ISDestroy(iy);                      CHKERRQ(ierr);
  ierr = ISMappingAssemblyBegin(*imap);      CHKERRQ(ierr);
  ierr = ISMappingAssemblyEnd(*imap);        CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingInvert_Graph() */


#undef  __FUNCT__
#define __FUNCT__ "ISMappingPushforward_Graph_Graph"
PetscErrorCode ISMappingPushforward_Graph_Graph(ISMapping map1, ISMapping map2, ISMapping *_map3) 
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
  ISMappingCheckType(map1,IS_MAPPING_GRAPH,1);
  ISMappingCheckType(map2,IS_MAPPING_GRAPH,3);
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
  ierr = ISMappingSetType(map3,IS_MAPPING_GRAPH);                                                            CHKERRQ(ierr);
  ierr = ISMappingSetSizes(map3, map1->ylayout->n, map2->ylayout->n, map1->ylayout->N, map2->ylayout->N); CHKERRQ(ierr);
  ierr = ISMappingGraphSetEdges(map3,ix,iy); CHKERRQ(ierr);
  ierr = ISMappingAssemblyBegin(map3);    CHKERRQ(ierr);
  ierr = ISMappingAssemblyEnd(map3);      CHKERRQ(ierr);
  ierr = ISDestroy(ix);                   CHKERRQ(ierr);
  ierr = ISDestroy(iy);                   CHKERRQ(ierr);

  *_map3 = map3;
  PetscFunctionReturn(0);
}/* ISMappingPushforward_Graph_Graph() */


#undef  __FUNCT__
#define __FUNCT__ "ISMappingPullback_Graph_Graph"
PetscErrorCode ISMappingPullback_Graph_Graph(ISMapping map1, ISMapping map2, ISMapping *_map3) 
{
  PetscErrorCode ierr;
  ISMapping imap1,map3;
  PetscFunctionBegin;
  ISMappingCheckType(map1,IS_MAPPING_GRAPH,1);
  ISMappingCheckType(map2,IS_MAPPING_GRAPH,3);
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
  ierr = ISMappingInvert_Graph(map1, &imap1);               CHKERRQ(ierr);
  ierr = ISMappingPushforward_Graph_Graph(imap1, map2, &map3); CHKERRQ(ierr);
  ierr = ISMappingDestroy(imap1);                        CHKERRQ(ierr);
  *_map3 = map3;
  PetscFunctionReturn(0);
}



#undef  __FUNCT__
#define __FUNCT__ "ISMappingDestroy_Graph"
PetscErrorCode ISMappingDestroy_Graph(ISMapping map) {
  PetscErrorCode ierr;
  ISMapping_Graph          *mapg = (ISMapping_Graph *)(map->data);
  
  PetscFunctionBegin;
  if(mapg) {
    if(mapg->ijlen) {
      ierr = PetscFree(mapg->ijlen); CHKERRQ(ierr);
    }
    if(mapg->ij) {
      ierr = PetscFree(mapg->ij); CHKERRQ(ierr);
    }    
    if(mapg->image) {
      ierr = PetscFree(mapg->image); CHKERRQ(ierr);
    }
    if(mapg->supp) {
      ierr = PetscFree(mapg->supp); CHKERRQ(ierr);
    }
    if(mapg->ix) {
      ierr = ISDestroy(mapg->ix);   CHKERRQ(ierr);
    }
    if(mapg->iy) {
      ierr = ISDestroy(mapg->iy);   CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(mapg); CHKERRQ(ierr);
  map->data = PETSC_NULL;
  
  map->setup = PETSC_FALSE;
  map->assembled = PETSC_FALSE;

  ierr = PetscObjectChangeTypeName((PetscObject)map,0); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)map,"ISMappingPullback_graph_graph_C", "",PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)map,"ISMappingPushforward_graph_graph_C", "",PETSC_NULL); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingDestroy_Graph() */

EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "ISMappingCreate_Graph"
PetscErrorCode ISMappingCreate_Graph(ISMapping map) {
  PetscErrorCode ierr;
  ISMapping_Graph  *mapg;
  PetscFunctionBegin;
  ierr = PetscNewLog(map, ISMapping_Graph, &mapg); CHKERRQ(ierr);
  map->data = (void*)mapg;

  map->ops->view                 = ISMappingView_Graph;
  map->ops->setup                = ISMappingSetUp_ISMapping;
  map->ops->assemblybegin        = ISMappingAssemblyBegin_Graph;
  map->ops->assemblyend          = ISMappingAssemblyEnd_Graph;
  map->ops->getsupportis         = ISMappingGetSupportIS_Graph;
  map->ops->getsupportsizelocal  = ISMappingGetSupportSizeLocal_Graph;
  map->ops->getimageis           = ISMappingGetSupportIS_Graph;
  map->ops->getimagesizelocal    = ISMappingGetImageSizeLocal_Graph;
  map->ops->getmaximagesizelocal = ISMappingGetMaxImageSizeLocal_Graph;
  map->ops->maplocal             = ISMappingMapLocal_Graph;
  map->ops->map                  = ISMappingMap_Graph;
  map->ops->binlocal             = ISMappingBinLocal_Graph;
  map->ops->bin                  = ISMappingBin_Graph;
  map->ops->mapsplitlocal        = ISMappingMapSplitLocal_Graph;
  map->ops->mapsplit             = ISMappingMapSplit_Graph;
  map->ops->binsplitlocal        = ISMappingBinSplitLocal_Graph;
  map->ops->binsplit             = ISMappingBinSplit_Graph;
  map->ops->invert               = ISMappingInvert_Graph;
  map->ops->getoperator          = ISMappingGetOperator_Graph;

  map->setup     = PETSC_FALSE;
  map->assembled = PETSC_FALSE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)map, 
                                           "ISMappingPullback_graph_grah_C", "ISMappingPullback_ismappinggraph_ismappinggraph", 
                                           ISMappingPullback_Graph_Graph); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)map, 
                                           "ISMappingPushforward_graph_graph_C", "ISMappingPushforward_ismappinggraph_ismappinggraph", 
                                           ISMappingPushforward_Graph_Graph); CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)map, IS_MAPPING_GRAPH); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingCreate_Graph() */
EXTERN_C_END
