#define PETSCVEC_DLL

#include "petscis.h"    /*I "petscis.h"  I*/

const char *ISColoringTypes[] = {"global","ghosted","ISColoringType","IS_COLORING_",0};

#undef __FUNCT__  
#define __FUNCT__ "ISColoringDestroy"
/*@
   ISColoringDestroy - Destroys a coloring context.

   Collective on ISColoring

   Input Parameter:
.  iscoloring - the coloring context

   Level: advanced

.seealso: ISColoringView(), MatGetColoring()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISColoringDestroy(ISColoring iscoloring)
{
  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(iscoloring,1);
  if (--iscoloring->refct > 0) PetscFunctionReturn(0);

  if (iscoloring->is) {
    for (i=0; i<iscoloring->n; i++) {
      ierr = ISDestroy(iscoloring->is[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iscoloring->is);CHKERRQ(ierr);
  }
  ierr = PetscFree(iscoloring->colors);CHKERRQ(ierr);
  PetscCommDestroy(&iscoloring->comm);
  ierr = PetscFree(iscoloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISColoringView"
/*@C
   ISColoringView - Views a coloring context.

   Collective on ISColoring

   Input Parameters:
+  iscoloring - the coloring context
-  viewer - the viewer

   Level: advanced

.seealso: ISColoringDestroy(), ISColoringGetIS(), MatGetColoring()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISColoringView(ISColoring iscoloring,PetscViewer viewer)
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscTruth     iascii;
  IS             *is;

  PetscFunctionBegin;
  PetscValidPointer(iscoloring,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(iscoloring->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    MPI_Comm    comm;
    PetscMPIInt rank;
    ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of colors %d\n",rank,iscoloring->n);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for ISColoring",((PetscObject)viewer)->type_name);
  }

  ierr = ISColoringGetIS(iscoloring,PETSC_IGNORE,&is);CHKERRQ(ierr);
  for (i=0; i<iscoloring->n; i++) {
    ierr = ISView(iscoloring->is[i],viewer);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISColoringGetIS"
/*@C
   ISColoringGetIS - Extracts index sets from the coloring context

   Collective on ISColoring 

   Input Parameter:
.  iscoloring - the coloring context

   Output Parameters:
+  nn - number of index sets in the coloring context
-  is - array of index sets

   Level: advanced

.seealso: ISColoringRestoreIS(), ISColoringView()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISColoringGetIS(ISColoring iscoloring,PetscInt *nn,IS *isis[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(iscoloring,1);

  if (nn)  *nn  = iscoloring->n;
  if (isis) {
    if (!iscoloring->is) {
      PetscInt        *mcolors,**ii,nc = iscoloring->n,i,base, n = iscoloring->N;
      ISColoringValue *colors = iscoloring->colors;
      IS              *is;

#if defined(PETSC_USE_DEBUG)
      for (i=0; i<n; i++) {
        if (((PetscInt)colors[i]) >= nc) {
          SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Coloring is our of range index %d value %d number colors %d",(int)i,(int)colors[i],(int)nc);
        }
      }
#endif
                      
      /* generate the lists of nodes for each color */
      ierr = PetscMalloc(nc*sizeof(PetscInt),&mcolors);CHKERRQ(ierr);
      ierr = PetscMemzero(mcolors,nc*sizeof(PetscInt));CHKERRQ(ierr);
      for (i=0; i<n; i++) {
	mcolors[colors[i]]++;
      }

      ierr = PetscMalloc(nc*sizeof(PetscInt*),&ii);CHKERRQ(ierr);
      ierr = PetscMalloc(n*sizeof(PetscInt),&ii[0]);CHKERRQ(ierr);
      for (i=1; i<nc; i++) {
	ii[i] = ii[i-1] + mcolors[i-1];
      }
      ierr = PetscMemzero(mcolors,nc*sizeof(PetscInt));CHKERRQ(ierr);

      if (iscoloring->ctype == IS_COLORING_GLOBAL){
        ierr = MPI_Scan(&iscoloring->N,&base,1,MPIU_INT,MPI_SUM,iscoloring->comm);CHKERRQ(ierr);
        base -= iscoloring->N;
        for (i=0; i<n; i++) {
          ii[colors[i]][mcolors[colors[i]]++] = i + base; /* global idx */
        }
      } else if (iscoloring->ctype == IS_COLORING_GHOSTED){
        for (i=0; i<n; i++) {
          ii[colors[i]][mcolors[colors[i]]++] = i;   /* local idx */
        }
      } else {
        SETERRQ(PETSC_ERR_SUP,"Not provided for this ISColoringType type");
      }
     
      ierr = PetscMalloc(nc*sizeof(IS),&is);CHKERRQ(ierr);
      for (i=0; i<nc; i++) {
	ierr = ISCreateGeneral(iscoloring->comm,mcolors[i],ii[i],is+i);CHKERRQ(ierr);
      }

      iscoloring->is   = is;
      ierr = PetscFree(ii[0]);CHKERRQ(ierr);
      ierr = PetscFree(ii);CHKERRQ(ierr);
      ierr = PetscFree(mcolors);CHKERRQ(ierr);
    }
    *isis = iscoloring->is;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISColoringRestoreIS"
/*@C
   ISColoringRestoreIS - Restores the index sets extracted from the coloring context

   Collective on ISColoring 

   Input Parameter:
+  iscoloring - the coloring context
-  is - array of index sets

   Level: advanced

.seealso: ISColoringGetIS(), ISColoringView()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISColoringRestoreIS(ISColoring iscoloring,IS *is[])
{
  PetscFunctionBegin;
  PetscValidPointer(iscoloring,1);
  
  /* currently nothing is done here */

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "ISColoringCreate"
/*@C
    ISColoringCreate - Generates an ISColoring context from lists (provided 
    by each processor) of colors for each node.

    Collective on MPI_Comm

    Input Parameters:
+   comm - communicator for the processors creating the coloring
.   ncolors - max color value
.   n - number of nodes on this processor
-   colors - array containing the colors for this processor, color
             numbers begin at 0. In C/C++ this array must have been obtained with PetscMalloc()
             and should NOT be freed (The ISColoringDestroy() will free it).

    Output Parameter:
.   iscoloring - the resulting coloring data structure

    Options Database Key:
.   -is_coloring_view - Activates ISColoringView()

   Level: advanced
   
    Notes: By default sets coloring type to  IS_COLORING_GLOBAL

.seealso: MatColoringCreate(), ISColoringView(), ISColoringDestroy(), ISColoringSetType()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISColoringCreate(MPI_Comm comm,PetscInt ncolors,PetscInt n,const ISColoringValue colors[],ISColoring *iscoloring)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag;
  PetscInt       base,top,i;
  PetscInt       nc,ncwork;
  PetscTruth     flg = PETSC_FALSE;
  MPI_Status     status;

  PetscFunctionBegin;
  if (ncolors != PETSC_DECIDE && ncolors > IS_COLORING_MAX) {
    if (ncolors > 65535) {
      SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Max color value exeeds 65535 limit. This number is unrealistic. Perhaps a bug in code?\nCurrent max: %d user rewuested: %d",IS_COLORING_MAX,ncolors);
    } else {
      SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Max color value exeeds limit. Perhaps reconfigure PETSc with --with-is-color-value-type=short?\n Current max: %d user rewuested: %d",IS_COLORING_MAX,ncolors);
    }
  }
  ierr = PetscNew(struct _n_ISColoring,iscoloring);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(comm,&(*iscoloring)->comm,&tag);CHKERRQ(ierr);
  comm = (*iscoloring)->comm;

  /* compute the number of the first node on my processor */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* should use MPI_Scan() */
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    base = 0;
    top  = n;
  } else {
    ierr = MPI_Recv(&base,1,MPIU_INT,rank-1,tag,comm,&status);CHKERRQ(ierr);
    top = base+n;
  }
  if (rank < size-1) {
    ierr = MPI_Send(&top,1,MPIU_INT,rank+1,tag,comm);CHKERRQ(ierr);
  }

  /* compute the total number of colors */
  ncwork = 0;
  for (i=0; i<n; i++) {
    if (ncwork < colors[i]) ncwork = colors[i];
  }
  ncwork++;
  ierr = MPI_Allreduce(&ncwork,&nc,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
  if (nc > ncolors) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Number of colors passed in %D is less then the actual number of colors in array %D",ncolors,nc);
  (*iscoloring)->n      = nc;
  (*iscoloring)->is     = 0;
  (*iscoloring)->colors = (ISColoringValue *)colors;
  (*iscoloring)->N      = n;
  (*iscoloring)->refct  = 1;
  (*iscoloring)->ctype  = IS_COLORING_GLOBAL;

  ierr = PetscOptionsGetTruth(PETSC_NULL,"-is_coloring_view",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIGetStdout((*iscoloring)->comm,&viewer);CHKERRQ(ierr);
    ierr = ISColoringView(*iscoloring,viewer);CHKERRQ(ierr);
  }
  ierr = PetscInfo1(0,"Number of colors %d\n",nc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISPartitioningToNumbering"
/*@
    ISPartitioningToNumbering - Takes an ISPartitioning and on each processor
    generates an IS that contains a new global node number for each index based
    on the partitioing.

    Collective on IS

    Input Parameters
.   partitioning - a partitioning as generated by MatPartitioningApply()

    Output Parameter:
.   is - on each processor the index set that defines the global numbers 
         (in the new numbering) for all the nodes currently (before the partitioning) 
         on that processor

   Level: advanced

.seealso: MatPartitioningCreate(), AOCreateBasic(), ISPartitioningCount()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISPartitioningToNumbering(IS part,IS *is)
{
  MPI_Comm       comm;
  PetscInt       i,np,npt,n,*starts = PETSC_NULL,*sums = PETSC_NULL,*lsizes = PETSC_NULL,*newi = PETSC_NULL;
  const PetscInt *indices = PETSC_NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)part,&comm);CHKERRQ(ierr);

  /* count the number of partitions, i.e., virtual processors */
  ierr = ISGetLocalSize(part,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(part,&indices);CHKERRQ(ierr);
  np = 0;
  for (i=0; i<n; i++) {
    np = PetscMax(np,indices[i]);
  }
  ierr = MPI_Allreduce(&np,&npt,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
  np = npt+1; /* so that it looks like a MPI_Comm_size output */

  /*
        lsizes - number of elements of each partition on this particular processor
        sums - total number of "previous" nodes for any particular partition
        starts - global number of first element in each partition on this processor
  */
  ierr   = PetscMalloc3(np,PetscInt,&lsizes,np,PetscInt,&starts,np,PetscInt,&sums);CHKERRQ(ierr);
  ierr   = PetscMemzero(lsizes,np*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    lsizes[indices[i]]++;
  }  
  ierr = MPI_Allreduce(lsizes,sums,np,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  ierr = MPI_Scan(lsizes,starts,np,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  for (i=0; i<np; i++) {
    starts[i] -= lsizes[i];
  }
  for (i=1; i<np; i++) {
    sums[i]    += sums[i-1];
    starts[i]  += sums[i-1];
  }

  /* 
      For each local index give it the new global number
  */
  ierr = PetscMalloc(n*sizeof(PetscInt),&newi);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    newi[i] = starts[indices[i]]++;
  }
  ierr = PetscFree3(lsizes,starts,sums);CHKERRQ(ierr);

  ierr = ISRestoreIndices(part,&indices);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n,newi,is);CHKERRQ(ierr);
  ierr = PetscFree(newi);CHKERRQ(ierr);
  ierr = ISSetPermutation(*is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISPartitioningCount"
/*@
    ISPartitioningCount - Takes a ISPartitioning and determines the number of 
    resulting elements on each (partition) process

    Collective on IS

    Input Parameters:
+   partitioning - a partitioning as generated by MatPartitioningApply()
-   len - length of the array count, this is the total number of partitions

    Output Parameter:
.   count - array of length size, to contain the number of elements assigned
        to each partition, where size is the number of partitions generated
         (see notes below).

   Level: advanced

    Notes:
        By default the number of partitions generated (and thus the length
        of count) is the size of the communicator associated with IS,
        but it can be set by MatPartitioningSetNParts. The resulting array
        of lengths can for instance serve as input of PCBJacobiSetTotalBlocks.


.seealso: MatPartitioningCreate(), AOCreateBasic(), ISPartitioningToNumbering(),
        MatPartitioningSetNParts()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISPartitioningCount(IS part,PetscInt len,PetscInt count[])
{
  MPI_Comm       comm;
  PetscInt       i,n,*lsizes;
  const PetscInt *indices;
  PetscErrorCode ierr;
  PetscMPIInt    npp;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)part,&comm);CHKERRQ(ierr);
  if (len == PETSC_DEFAULT) {
    PetscMPIInt size;
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    len  = (PetscInt) size;
  }

  /* count the number of partitions */
  ierr = ISGetLocalSize(part,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(part,&indices);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  { 
    PetscInt np = 0,npt;
    for (i=0; i<n; i++) {
      np = PetscMax(np,indices[i]);
    }    
    ierr = MPI_Allreduce(&np,&npt,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
    np = npt+1; /* so that it looks like a MPI_Comm_size output */
    if (np > len) SETERRQ2(PETSC_ERR_ARG_SIZ,"Length of count array %D is less than number of partitions %D",len,np);
  }
#endif

  /*
        lsizes - number of elements of each partition on this particular processor
        sums - total number of "previous" nodes for any particular partition
        starts - global number of first element in each partition on this processor
  */
  ierr = PetscMalloc(len*sizeof(PetscInt),&lsizes);CHKERRQ(ierr);
  ierr = PetscMemzero(lsizes,len*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    lsizes[indices[i]]++;
  }  
  ierr = ISRestoreIndices(part,&indices);CHKERRQ(ierr);
  npp  = PetscMPIIntCast(len);
  ierr = MPI_Allreduce(lsizes,count,npp,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  ierr = PetscFree(lsizes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISAllGather"
/*@
    ISAllGather - Given an index set (IS) on each processor, generates a large 
    index set (same on each processor) by concatenating together each
    processors index set.

    Collective on IS

    Input Parameter:
.   is - the distributed index set

    Output Parameter:
.   isout - the concatenated index set (same on all processors)

    Notes: 
    ISAllGather() is clearly not scalable for large index sets.

    The IS created on each processor must be created with a common
    communicator (e.g., PETSC_COMM_WORLD). If the index sets were created 
    with PETSC_COMM_SELF, this routine will not work as expected, since 
    each process will generate its own new IS that consists only of
    itself.

    The communicator for this new IS is PETSC_COMM_SELF

    Level: intermediate

    Concepts: gather^index sets
    Concepts: index sets^gathering to all processors
    Concepts: IS^gathering to all processors

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlock(), ISAllGatherIndices()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISAllGather(IS is,IS *isout)
{
  PetscErrorCode ierr;
  PetscInt       *indices,n,i,N,step,first;
  const PetscInt *lindices;
  MPI_Comm       comm;
  PetscMPIInt    size,*sizes = PETSC_NULL,*offsets = PETSC_NULL,nn;
  PetscTruth     stride;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(isout,2);

  ierr = PetscObjectGetComm((PetscObject)is,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISStride(is,&stride);CHKERRQ(ierr);
  if (size == 1 && stride) { /* should handle parallel ISStride also */
    ierr = ISStrideGetInfo(is,&first,&step);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,n,first,step,isout);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc2(size,PetscMPIInt,&sizes,size,PetscMPIInt,&offsets);CHKERRQ(ierr);
  
    nn   = PetscMPIIntCast(n);
    ierr = MPI_Allgather(&nn,1,MPI_INT,sizes,1,MPI_INT,comm);CHKERRQ(ierr);
    offsets[0] = 0;
    for (i=1;i<size; i++) offsets[i] = offsets[i-1] + sizes[i-1];
    N = offsets[size-1] + sizes[size-1];
    
    ierr = PetscMalloc(N*sizeof(PetscInt),&indices);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&lindices);CHKERRQ(ierr);
    ierr = MPI_Allgatherv((void*)lindices,nn,MPIU_INT,indices,sizes,offsets,MPIU_INT,comm);CHKERRQ(ierr); 
    ierr = ISRestoreIndices(is,&lindices);CHKERRQ(ierr);
    ierr = PetscFree2(sizes,offsets);CHKERRQ(ierr);

    ierr = ISCreateGeneral(PETSC_COMM_SELF,N,indices,isout);CHKERRQ(ierr);
    ierr = PetscFree(indices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISAllGatherIndices"
/*@C
    ISAllGatherIndices - Given a a set of integers on each processor, generates a large 
    set (same on each processor) by concatenating together each processors integers

    Collective on MPI_Comm

    Input Parameter:
+   comm - communicator to share the indices
.   n - local size of set
-   lindices - local indices

    Output Parameter:
+   outN - total number of indices
-   outindices - all of the integers

    Notes: 
    ISAllGatherIndices() is clearly not scalable for large index sets.


    Level: intermediate

    Concepts: gather^index sets
    Concepts: index sets^gathering to all processors
    Concepts: IS^gathering to all processors

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlock(), ISAllGather()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISAllGatherIndices(MPI_Comm comm,PetscInt n,const PetscInt lindices[],PetscInt *outN,PetscInt *outindices[])
{
  PetscErrorCode ierr;
  PetscInt       *indices,i,N;
  PetscMPIInt    size,*sizes = PETSC_NULL,*offsets = PETSC_NULL,nn;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscMalloc2(size,PetscMPIInt,&sizes,size,PetscMPIInt,&offsets);CHKERRQ(ierr);
  
  nn   = PetscMPIIntCast(n);
  ierr = MPI_Allgather(&nn,1,MPI_INT,sizes,1,MPI_INT,comm);CHKERRQ(ierr);
  offsets[0] = 0;
  for (i=1;i<size; i++) offsets[i] = offsets[i-1] + sizes[i-1];
  N    = offsets[size-1] + sizes[size-1];

  ierr = PetscMalloc(N*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  ierr = MPI_Allgatherv((void*)lindices,nn,MPIU_INT,indices,sizes,offsets,MPIU_INT,comm);CHKERRQ(ierr); 
  ierr = PetscFree2(sizes,offsets);CHKERRQ(ierr);

  *outindices = indices;
  if (outN) *outN = N;
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "ISAllGatherColors"
/*@C
    ISAllGatherColors - Given a a set of colors on each processor, generates a large 
    set (same on each processor) by concatenating together each processors colors

    Collective on MPI_Comm

    Input Parameter:
+   comm - communicator to share the indices
.   n - local size of set
-   lindices - local colors

    Output Parameter:
+   outN - total number of indices
-   outindices - all of the colors

    Notes: 
    ISAllGatherColors() is clearly not scalable for large index sets.


    Level: intermediate

    Concepts: gather^index sets
    Concepts: index sets^gathering to all processors
    Concepts: IS^gathering to all processors

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlock(), ISAllGather(), ISAllGatherIndices()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISAllGatherColors(MPI_Comm comm,PetscInt n,ISColoringValue *lindices,PetscInt *outN,ISColoringValue *outindices[])
{
  ISColoringValue *indices;
  PetscErrorCode  ierr;
  PetscInt        i,N;
  PetscMPIInt     size,*offsets = PETSC_NULL,*sizes = PETSC_NULL, nn = n;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscMalloc2(size,PetscMPIInt,&sizes,size,PetscMPIInt,&offsets);CHKERRQ(ierr);
  
  ierr = MPI_Allgather(&nn,1,MPI_INT,sizes,1,MPI_INT,comm);CHKERRQ(ierr);
  offsets[0] = 0;
  for (i=1;i<size; i++) offsets[i] = offsets[i-1] + sizes[i-1];
  N    = offsets[size-1] + sizes[size-1];
  ierr = PetscFree2(sizes,offsets);CHKERRQ(ierr);

  ierr = PetscMalloc((N+1)*sizeof(ISColoringValue),&indices);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(lindices,(PetscMPIInt)n,MPIU_COLORING_VALUE,indices,sizes,offsets,MPIU_COLORING_VALUE,comm);CHKERRQ(ierr); 

  *outindices = indices;
  if (outN) *outN = N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISComplement"
/*@
    ISComplement - Given an index set (IS) generates the complement index set. That is all
       all indices that are NOT in the given set.

    Collective on IS

    Input Parameter:
+   is - the index set
.   nmin - the first index desired in the local part of the complement
-   nmax - the largest index desired in the local part of the complement (note that all indices in is must be greater or equal to nmin and less than nmax)

    Output Parameter:
.   isout - the complement

    Notes:  The communicator for this new IS is the same as for the input IS

      For a parallel IS, this will generate the local part of the complement on each process

      To generate the entire complement (on each process) of a parallel IS, first call ISAllGather() and then
    call this routine.

    Level: intermediate

    Concepts: gather^index sets
    Concepts: index sets^gathering to all processors
    Concepts: IS^gathering to all processors

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlock(), ISAllGatherIndices(), ISAllGather()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISComplement(IS is,PetscInt nmin,PetscInt nmax,IS *isout)
{
  PetscErrorCode ierr;
  const PetscInt *indices;
  PetscInt       n,i,j,unique,cnt,*nindices;
  PetscTruth     sorted;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(isout,3);
  if (nmin < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"nmin %D cannot be negative",nmin);
  if (nmin > nmax) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"nmin %D cannot be greater than nmax %D",nmin,nmax);
  ierr = ISSorted(is,&sorted);CHKERRQ(ierr);
  if (!sorted) SETERRQ(PETSC_ERR_ARG_WRONG,"Index set must be sorted");

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&indices);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  for (i=0; i<n; i++) {
    if (indices[i] <  nmin) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Index %D's value %D is smaller than minimum given %D",i,indices[i],nmin);
    if (indices[i] >= nmax) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Index %D's value %D is larger than maximum given %D",i,indices[i],nmax);
  }
#endif
  /* Count number of unique entries */
  unique = (n>0);
  for (i=0; i<n-1; i++) {
    if (indices[i+1] != indices[i]) unique++;
  }
  ierr = PetscMalloc((nmax-nmin-unique)*sizeof(PetscInt),&nindices);CHKERRQ(ierr);
  cnt = 0;
  for (i=nmin,j=0; i<nmax; i++) {
    if (j<n && i==indices[j]) do { j++; } while (j<n && i==indices[j]);
    else nindices[cnt++] = i;
  }
  if (cnt != nmax-nmin-unique) SETERRQ2(PETSC_ERR_PLIB,"Number of entries found in complement %D does not match expected %D",cnt,nmax-nmin-unique);
  ierr = ISCreateGeneralNC(((PetscObject)is)->comm,cnt,nindices,isout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
