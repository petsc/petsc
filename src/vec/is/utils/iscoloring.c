/*$Id: iscoloring.c,v 1.64 2000/10/24 20:24:56 bsmith Exp bsmith $*/

#include "petscsys.h"   /*I "petscsys.h" I*/
#include "petscis.h"    /*I "petscis.h"  I*/

#undef __FUNC__  
#define __FUNC__ "ISColoringDestroy"
/*@C
   ISColoringDestroy - Destroys a coloring context.

   Collective on ISColoring

   Input Parameter:
.  iscoloring - the coloring context

   Level: advanced

.seealso: ISColoringView(), MatGetColoring()
@*/
int ISColoringDestroy(ISColoring iscoloring)
{
  int i,ierr;

  PetscFunctionBegin;
  PetscValidPointer(iscoloring);

  for (i=0; i<iscoloring->n; i++) {
    ierr = ISDestroy(iscoloring->is[i]);CHKERRQ(ierr);
  }
  PetscCommDestroy_Private(&iscoloring->comm);
  ierr = PetscFree(iscoloring->is);CHKERRQ(ierr);
  ierr = PetscFree(iscoloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISColoringView"
/*@C
   ISColoringView - Views a coloring context.

   Collective on ISColoring

   Input Parameters:
+  iscoloring - the coloring context
-  viewer - the viewer

   Level: advanced

.seealso: ISColoringDestroy(), ISColoringGetIS(), MatGetColoring()
@*/
int ISColoringView(ISColoring iscoloring,PetscViewer viewer)
{
  int        i,ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  PetscValidPointer(iscoloring);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(iscoloring->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    MPI_Comm comm;
    int      rank;
    ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of colors %d\n",rank,iscoloring->n);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported for ISColoring",((PetscObject)viewer)->type_name);
  }

  for (i=0; i<iscoloring->n; i++) {
    ierr = ISView(iscoloring->is[i],viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISColoringGetIS"
/*@C
   ISColoringGetIS - Extracts index sets from the coloring context

   Collective on ISColoring 

   Input Parameter:
.  iscoloring - the coloring context

   Output Parameters:
+  n - number of index sets in the coloring context
-  is - array of index sets

   Level: advanced

.seealso: ISColoringRestoreIS(), ISColoringView()
@*/
int ISColoringGetIS(ISColoring iscoloring,int *n,IS *is[])
{
  PetscFunctionBegin;
  PetscValidPointer(iscoloring);

  if (n)  *n  = iscoloring->n;
  if (is) *is = iscoloring->is;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISColoringRestoreIS"
/*@C
   ISColoringGetIS - Restores the index sets extracted from the coloring context

   Collective on ISColoring 

   Input Parameter:
+  iscoloring - the coloring context
-  is - array of index sets

   Level: advanced

.seealso: ISColoringGetIS(), ISColoringView()
@*/
int ISColoringRestoreIS(ISColoring iscoloring,IS *is[])
{
  PetscFunctionBegin;
  PetscValidPointer(iscoloring);
  
  /* currently nothing is done here */

  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "ISColoringCreate"
/*@C
    ISColoringCreate - Generates an ISColoring context from lists (provided 
    by each processor) of colors for each node.

    Collective on MPI_Comm

    Input Parameters:
+   comm - communicator for the processors creating the coloring
.   n - number of nodes on this processor
-   colors - array containing the colors for this processor, color
             numbers begin at 0.

    Output Parameter:
.   iscoloring - the resulting coloring data structure

    Options Database Key:
.   -is_coloring_view - Activates ISColoringView()

   Level: advanced

.seealso: MatColoringCreate(), ISColoringView(), ISColoringDestroy()
@*/
int ISColoringCreate(MPI_Comm comm,int n,const int colors[],ISColoring *iscoloring)
{
  int        ierr,size,rank,base,top,tag,nc,ncwork,*mcolors,**ii,i;
  PetscTruth flg;
  MPI_Status status;
  IS         *is;

  PetscFunctionBegin;
  ierr = PetscNew(struct _p_ISColoring,iscoloring);CHKERRQ(ierr);
  ierr = PetscCommDuplicate_Private(comm,&(*iscoloring)->comm,&tag);CHKERRQ(ierr);
  comm = (*iscoloring)->comm;

  /* compute the number of the first node on my processor */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* should use MPI_Scan() */
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    base = 0;
    top  = n;
  } else {
    ierr = MPI_Recv(&base,1,MPI_INT,rank-1,tag,comm,&status);CHKERRQ(ierr);
    top = base+n;
  }
  if (rank < size-1) {
    ierr = MPI_Send(&top,1,MPI_INT,rank+1,tag,comm);CHKERRQ(ierr);
  }

  /* compute the total number of colors */
  ncwork = 0;
  for (i=0; i<n; i++) {
    if (ncwork < colors[i]) ncwork = colors[i];
  }
  ncwork++;
  ierr = MPI_Allreduce(&ncwork,&nc,1,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);

  /* generate the lists of nodes for each color */
  ierr = PetscMalloc((nc+1)*sizeof(int),&mcolors);CHKERRQ(ierr);
  ierr = PetscMemzero(mcolors,nc*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    mcolors[colors[i]]++;
  }

  ierr = PetscMalloc((nc+1)*sizeof(int*),&ii);CHKERRQ(ierr);
  ierr = PetscMalloc((n+1)*sizeof(int),&ii[0]);CHKERRQ(ierr);
  for (i=1; i<nc; i++) {
    ii[i] = ii[i-1] + mcolors[i-1];
  }
  ierr = PetscMemzero(mcolors,nc*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ii[colors[i]][mcolors[colors[i]]++] = i + base;
  }
  ierr = PetscMalloc((nc+1)*sizeof(IS),&is);CHKERRQ(ierr);
  for (i=0; i<nc; i++) {
    ierr = ISCreateGeneral(comm,mcolors[i],ii[i],is+i);CHKERRQ(ierr);
  }

  (*iscoloring)->n    = nc;
  (*iscoloring)->is   = is;

  ierr = PetscFree(ii[0]);CHKERRQ(ierr);
  ierr = PetscFree(ii);CHKERRQ(ierr);
  ierr = PetscFree(mcolors);CHKERRQ(ierr);


  ierr = PetscOptionsHasName(PETSC_NULL,"-is_coloring_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = ISColoringView(*iscoloring,PETSC_VIEWER_STDOUT_((*iscoloring)->comm));CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISPartitioningToNumbering"
/*@C
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

.seealso: MatPartitioningCreate(), AOCreateBasic(), ISPartioningCount()

@*/
int ISPartitioningToNumbering(IS part,IS *is)
{
  MPI_Comm comm;
  int      i,ierr,rank,size,*indices,np,n,*starts,*sums,*lsizes,*newi;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)part,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* count the number of partitions, make sure <= size */
  ierr = ISGetLocalSize(part,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(part,&indices);CHKERRQ(ierr);
  np = 0;
  for (i=0; i<n; i++) {
    np = PetscMax(np,indices[i]);
  }  
  if (np >= size) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Number of partitions %d larger than number of processors %d",np,size);
  }

  /*
        lsizes - number of elements of each partition on this particular processor
        sums - total number of "previous" nodes for any particular partition
        starts - global number of first element in each partition on this processor
  */
  ierr   = PetscMalloc(3*size*sizeof(int),&lsizes);CHKERRQ(ierr);
  starts = lsizes + size;
  sums   = starts + size;
  ierr   = PetscMemzero(lsizes,size*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    lsizes[indices[i]]++;
  }  
  ierr = MPI_Allreduce(lsizes,sums,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  ierr = MPI_Scan(lsizes,starts,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  for (i=0; i<size; i++) {
    starts[i] -= lsizes[i];
  }
  for (i=1; i<size; i++) {
    sums[i]    += sums[i-1];
    starts[i]  += sums[i-1];
  }

  /* 
      For each local index give it the new global number
  */
  ierr = PetscMalloc((n+1)*sizeof(int),&newi);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    newi[i] = starts[indices[i]]++;
  }
  ierr = PetscFree(lsizes);CHKERRQ(ierr);

  ierr = ISRestoreIndices(part,&indices);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n,newi,is);CHKERRQ(ierr);
  ierr = PetscFree(newi);CHKERRQ(ierr);
  ierr = ISSetPermutation(*is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISPartitioningCount"
/*@C
    ISPartitioningCount - Takes a ISPartitioning and determines the number of 
    resulting elements on each processor

    Collective on IS

    Input Parameters:
.   partitioning - a partitioning as generated by MatPartitioningApply()

    Output Parameter:
.   count - array of length size of communicator associated with IS, contains 
           the number of elements assigned to each processor

   Level: advanced

.seealso: MatPartitioningCreate(), AOCreateBasic(), ISPartitioningToNumbering()

@*/
int ISPartitioningCount(IS part,int count[])
{
  MPI_Comm comm;
  int      i,ierr,size,*indices,np,n,*lsizes;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)part,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* count the number of partitions,make sure <= size */
  ierr = ISGetLocalSize(part,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(part,&indices);CHKERRQ(ierr);
  np = 0;
  for (i=0; i<n; i++) {
    np = PetscMax(np,indices[i]);
  }  
  if (np >= size) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Number of partitions %d larger than number of processors %d",np,size);
  }

  /*
        lsizes - number of elements of each partition on this particular processor
        sums - total number of "previous" nodes for any particular partition
        starts - global number of first element in each partition on this processor
  */
  ierr = PetscMalloc(size*sizeof(int),&lsizes);CHKERRQ(ierr);
  ierr   = PetscMemzero(lsizes,size*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    lsizes[indices[i]]++;
  }  
  ierr = ISRestoreIndices(part,&indices);CHKERRQ(ierr);
  ierr = MPI_Allreduce(lsizes,count,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  ierr = PetscFree(lsizes);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISAllGather"
/*@C
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

    Level: intermediate

    Concepts: gather^index sets
    Concepts: index sets^gathering to all processors
    Concepts: IS^gathering to all processors

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlock()
@*/
int ISAllGather(IS is,IS *isout)
{
  int      *indices,*sizes,size,*offsets,n,*lindices,i,N,ierr;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = PetscObjectGetComm((PetscObject)is,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscMalloc(2*size*sizeof(int),&sizes);CHKERRQ(ierr);
  offsets = sizes + size;
  
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = MPI_Allgather(&n,1,MPI_INT,sizes,1,MPI_INT,comm);CHKERRQ(ierr);
  offsets[0] = 0;
  for (i=1;i<size; i++) offsets[i] = offsets[i-1] + sizes[i-1];
  N = offsets[size-1] + sizes[size-1];

  ierr = PetscMalloc((N+1)*sizeof(int),&indices);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&lindices);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(lindices,n,MPI_INT,indices,sizes,offsets,MPI_INT,comm);CHKERRQ(ierr); 
  ierr = ISRestoreIndices(is,&lindices);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF,N,indices,isout);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);

  ierr = PetscFree(sizes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




