
#include <petsc/private/isimpl.h>    /*I "petscis.h"  I*/
#include <petscviewer.h>
#include <petscsf.h>

const char *const ISColoringTypes[] = {"global","ghosted","ISColoringType","IS_COLORING_",NULL};

PetscErrorCode ISColoringReference(ISColoring coloring)
{
  PetscFunctionBegin;
  coloring->refct++;
  PetscFunctionReturn(0);
}

/*@C

    ISColoringSetType - indicates if the coloring is for the local representation (including ghost points) or the global representation

   Collective on coloring

   Input Parameters:
+    coloring - the coloring object
-    type - either IS_COLORING_LOCAL or IS_COLORING_GLOBAL

   Notes:
     With IS_COLORING_LOCAL the coloring is in the numbering of the local vector, for IS_COLORING_GLOBAL it is in the number of the global vector

   Level: intermediate

.seealso: MatFDColoringCreate(), ISColoring, ISColoringCreate(), IS_COLORING_LOCAL, IS_COLORING_GLOBAL, ISColoringGetType()

@*/
PetscErrorCode ISColoringSetType(ISColoring coloring,ISColoringType type)
{
  PetscFunctionBegin;
  coloring->ctype = type;
  PetscFunctionReturn(0);
}

/*@C

    ISColoringGetType - gets if the coloring is for the local representation (including ghost points) or the global representation

   Collective on coloring

   Input Parameter:
.   coloring - the coloring object

   Output Parameter:
.    type - either IS_COLORING_LOCAL or IS_COLORING_GLOBAL

   Level: intermediate

.seealso: MatFDColoringCreate(), ISColoring, ISColoringCreate(), IS_COLORING_LOCAL, IS_COLORING_GLOBAL, ISColoringSetType()

@*/
PetscErrorCode ISColoringGetType(ISColoring coloring,ISColoringType *type)
{
  PetscFunctionBegin;
  *type = coloring->ctype;
  PetscFunctionReturn(0);
}

/*@
   ISColoringDestroy - Destroys a coloring context.

   Collective on ISColoring

   Input Parameter:
.  iscoloring - the coloring context

   Level: advanced

.seealso: ISColoringView(), MatColoring
@*/
PetscErrorCode  ISColoringDestroy(ISColoring *iscoloring)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*iscoloring) PetscFunctionReturn(0);
  PetscValidPointer((*iscoloring),1);
  if (--(*iscoloring)->refct > 0) {*iscoloring = NULL; PetscFunctionReturn(0);}

  if ((*iscoloring)->is) {
    for (i=0; i<(*iscoloring)->n; i++) {
      ierr = ISDestroy(&(*iscoloring)->is[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree((*iscoloring)->is);CHKERRQ(ierr);
  }
  if ((*iscoloring)->allocated) {ierr = PetscFree((*iscoloring)->colors);CHKERRQ(ierr);}
  ierr = PetscCommDestroy(&(*iscoloring)->comm);CHKERRQ(ierr);
  ierr = PetscFree((*iscoloring));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  ISColoringViewFromOptions - Processes command line options to determine if/how an ISColoring object is to be viewed.

  Collective on ISColoring

  Input Parameters:
+ obj   - the ISColoring object
. prefix - prefix to use for viewing, or NULL to use prefix of 'mat'
- optionname - option to activate viewing

  Level: intermediate

  Developer Note: This cannot use PetscObjectViewFromOptions() because ISColoring is not a PetscObject

*/
PetscErrorCode ISColoringViewFromOptions(ISColoring obj,PetscObject bobj,const char optionname[])
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg;
  PetscViewerFormat format;
  char              *prefix;

  PetscFunctionBegin;
  prefix = bobj ? bobj->prefix : NULL;
  ierr   = PetscOptionsGetViewer(obj->comm,NULL,prefix,optionname,&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = ISColoringView(obj,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   ISColoringView - Views a coloring context.

   Collective on ISColoring

   Input Parameters:
+  iscoloring - the coloring context
-  viewer - the viewer

   Level: advanced

.seealso: ISColoringDestroy(), ISColoringGetIS(), MatColoring
@*/
PetscErrorCode  ISColoringView(ISColoring iscoloring,PetscViewer viewer)
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscBool      iascii;
  IS             *is;

  PetscFunctionBegin;
  PetscValidPointer(iscoloring,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(iscoloring->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    MPI_Comm    comm;
    PetscMPIInt size,rank;

    ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"ISColoring Object: %d MPI processes\n",size);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"ISColoringType: %s\n",ISColoringTypes[iscoloring->ctype]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of colors %" PetscInt_FMT "\n",rank,iscoloring->n);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  }

  ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,PETSC_IGNORE,&is);CHKERRQ(ierr);
  for (i=0; i<iscoloring->n; i++) {
    ierr = ISView(iscoloring->is[i],viewer);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   ISColoringGetColors - Returns an array with the color for each node

   Not Collective

   Input Parameter:
.  iscoloring - the coloring context

   Output Parameters:
+  n - number of nodes
.  nc - number of colors
-  colors - color for each node

   Level: advanced

.seealso: ISColoringRestoreIS(), ISColoringView(), ISColoringGetIS()
@*/
PetscErrorCode  ISColoringGetColors(ISColoring iscoloring,PetscInt *n,PetscInt *nc,const ISColoringValue **colors)
{
  PetscFunctionBegin;
  PetscValidPointer(iscoloring,1);

  if (n) *n = iscoloring->N;
  if (nc) *nc = iscoloring->n;
  if (colors) *colors = iscoloring->colors;
  PetscFunctionReturn(0);
}

/*@C
   ISColoringGetIS - Extracts index sets from the coloring context. Each is contains the nodes of one color

   Collective on ISColoring

   Input Parameters:
+  iscoloring - the coloring context
-  mode - if this value is PETSC_OWN_POINTER then the caller owns the pointer and must free the array of IS and each IS in the array

   Output Parameters:
+  nn - number of index sets in the coloring context
-  is - array of index sets

   Level: advanced

.seealso: ISColoringRestoreIS(), ISColoringView(), ISColoringGetColoring()
@*/
PetscErrorCode  ISColoringGetIS(ISColoring iscoloring,PetscCopyMode mode, PetscInt *nn,IS *isis[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(iscoloring,1);

  if (nn) *nn = iscoloring->n;
  if (isis) {
    if (!iscoloring->is) {
      PetscInt        *mcolors,**ii,nc = iscoloring->n,i,base, n = iscoloring->N;
      ISColoringValue *colors = iscoloring->colors;
      IS              *is;

      if (PetscDefined(USE_DEBUG)) {
        for (i=0; i<n; i++) {
          PetscAssertFalse(((PetscInt)colors[i]) >= nc,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coloring is our of range index %d value %d number colors %d",(int)i,(int)colors[i],(int)nc);
        }
      }

      /* generate the lists of nodes for each color */
      ierr = PetscCalloc1(nc,&mcolors);CHKERRQ(ierr);
      for (i=0; i<n; i++) mcolors[colors[i]]++;

      ierr = PetscMalloc1(nc,&ii);CHKERRQ(ierr);
      ierr = PetscMalloc1(n,&ii[0]);CHKERRQ(ierr);
      for (i=1; i<nc; i++) ii[i] = ii[i-1] + mcolors[i-1];
      ierr = PetscArrayzero(mcolors,nc);CHKERRQ(ierr);

      if (iscoloring->ctype == IS_COLORING_GLOBAL) {
        ierr = MPI_Scan(&iscoloring->N,&base,1,MPIU_INT,MPI_SUM,iscoloring->comm);CHKERRMPI(ierr);
        base -= iscoloring->N;
        for (i=0; i<n; i++) ii[colors[i]][mcolors[colors[i]]++] = i + base; /* global idx */
      } else if (iscoloring->ctype == IS_COLORING_LOCAL) {
        for (i=0; i<n; i++) ii[colors[i]][mcolors[colors[i]]++] = i;   /* local idx */
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not provided for this ISColoringType type");

      ierr = PetscMalloc1(nc,&is);CHKERRQ(ierr);
      for (i=0; i<nc; i++) {
        ierr = ISCreateGeneral(iscoloring->comm,mcolors[i],ii[i],PETSC_COPY_VALUES,is+i);CHKERRQ(ierr);
      }

      if (mode != PETSC_OWN_POINTER) iscoloring->is = is;
      *isis = is;
      ierr = PetscFree(ii[0]);CHKERRQ(ierr);
      ierr = PetscFree(ii);CHKERRQ(ierr);
      ierr = PetscFree(mcolors);CHKERRQ(ierr);
    } else {
      *isis = iscoloring->is;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   ISColoringRestoreIS - Restores the index sets extracted from the coloring context

   Collective on ISColoring

   Input Parameters:
+  iscoloring - the coloring context
.  mode - who retains ownership of the is
-  is - array of index sets

   Level: advanced

.seealso: ISColoringGetIS(), ISColoringView()
@*/
PetscErrorCode  ISColoringRestoreIS(ISColoring iscoloring,PetscCopyMode mode,IS *is[])
{
  PetscFunctionBegin;
  PetscValidPointer(iscoloring,1);

  /* currently nothing is done here */
  PetscFunctionReturn(0);
}

/*@
    ISColoringCreate - Generates an ISColoring context from lists (provided
    by each processor) of colors for each node.

    Collective

    Input Parameters:
+   comm - communicator for the processors creating the coloring
.   ncolors - max color value
.   n - number of nodes on this processor
.   colors - array containing the colors for this processor, color numbers begin at 0.
-   mode - see PetscCopyMode for meaning of this flag.

    Output Parameter:
.   iscoloring - the resulting coloring data structure

    Options Database Key:
.   -is_coloring_view - Activates ISColoringView()

   Level: advanced

    Notes:
    By default sets coloring type to  IS_COLORING_GLOBAL

.seealso: MatColoringCreate(), ISColoringView(), ISColoringDestroy(), ISColoringSetType()

@*/
PetscErrorCode  ISColoringCreate(MPI_Comm comm,PetscInt ncolors,PetscInt n,const ISColoringValue colors[],PetscCopyMode mode,ISColoring *iscoloring)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag;
  PetscInt       base,top,i;
  PetscInt       nc,ncwork;
  MPI_Status     status;

  PetscFunctionBegin;
  if (ncolors != PETSC_DECIDE && ncolors > IS_COLORING_MAX) {
    PetscAssertFalse(ncolors > PETSC_MAX_UINT16,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Max color value exceeds %d limit. This number is unrealistic. Perhaps a bug in code?\nCurrent max: %d user requested: %" PetscInt_FMT,PETSC_MAX_UINT16,PETSC_IS_COLORING_MAX,ncolors);
    else                 SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Max color value exceeds limit. Perhaps reconfigure PETSc with --with-is-color-value-type=short?\n Current max: %d user requested: %" PetscInt_FMT,PETSC_IS_COLORING_MAX,ncolors);
  }
  ierr = PetscNew(iscoloring);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(comm,&(*iscoloring)->comm,&tag);CHKERRQ(ierr);
  comm = (*iscoloring)->comm;

  /* compute the number of the first node on my processor */
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  /* should use MPI_Scan() */
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (rank == 0) {
    base = 0;
    top  = n;
  } else {
    ierr = MPI_Recv(&base,1,MPIU_INT,rank-1,tag,comm,&status);CHKERRMPI(ierr);
    top  = base+n;
  }
  if (rank < size-1) {
    ierr = MPI_Send(&top,1,MPIU_INT,rank+1,tag,comm);CHKERRMPI(ierr);
  }

  /* compute the total number of colors */
  ncwork = 0;
  for (i=0; i<n; i++) {
    if (ncwork < colors[i]) ncwork = colors[i];
  }
  ncwork++;
  ierr = MPIU_Allreduce(&ncwork,&nc,1,MPIU_INT,MPI_MAX,comm);CHKERRMPI(ierr);
  PetscAssertFalse(nc > ncolors,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of colors passed in %" PetscInt_FMT " is less then the actual number of colors in array %" PetscInt_FMT,ncolors,nc);
  (*iscoloring)->n      = nc;
  (*iscoloring)->is     = NULL;
  (*iscoloring)->N      = n;
  (*iscoloring)->refct  = 1;
  (*iscoloring)->ctype  = IS_COLORING_GLOBAL;
  if (mode == PETSC_COPY_VALUES) {
    ierr = PetscMalloc1(n,&(*iscoloring)->colors);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)(*iscoloring),n*sizeof(ISColoringValue));CHKERRQ(ierr);
    ierr = PetscArraycpy((*iscoloring)->colors,colors,n);CHKERRQ(ierr);
    (*iscoloring)->allocated = PETSC_TRUE;
  } else if (mode == PETSC_OWN_POINTER) {
    (*iscoloring)->colors    = (ISColoringValue*)colors;
    (*iscoloring)->allocated = PETSC_TRUE;
  } else {
    (*iscoloring)->colors    = (ISColoringValue*)colors;
    (*iscoloring)->allocated = PETSC_FALSE;
  }
  ierr = ISColoringViewFromOptions(*iscoloring,NULL,"-is_coloring_view");CHKERRQ(ierr);
  ierr = PetscInfo(0,"Number of colors %" PetscInt_FMT "\n",nc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    ISBuildTwoSided - Takes an IS that describes where we will go. Generates an IS that contains new numbers from remote or local
    on the IS.

    Collective on IS

    Input Parameters:
+   ito - an IS describes where we will go. Negative target rank will be ignored
-   toindx - an IS describes what indices should send. NULL means sending natural numbering

    Output Parameter:
.   rows - contains new numbers from remote or local

   Level: advanced

.seealso: MatPartitioningCreate(), ISPartitioningToNumbering(), ISPartitioningCount()

@*/
PetscErrorCode  ISBuildTwoSided(IS ito,IS toindx, IS *rows)
{
   const PetscInt *ito_indices,*toindx_indices;
   PetscInt       *send_indices,rstart,*recv_indices,nrecvs,nsends;
   PetscInt       *tosizes,*fromsizes,i,j,*tosizes_tmp,*tooffsets_tmp,ito_ln;
   PetscMPIInt    *toranks,*fromranks,size,target_rank,*fromperm_newtoold,nto,nfrom;
   PetscLayout     isrmap;
   MPI_Comm        comm;
   PetscSF         sf;
   PetscSFNode    *iremote;
   PetscErrorCode  ierr;

   PetscFunctionBegin;
   ierr = PetscObjectGetComm((PetscObject)ito,&comm);CHKERRQ(ierr);
   ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
   ierr = ISGetLocalSize(ito,&ito_ln);CHKERRQ(ierr);
   ierr = ISGetLayout(ito,&isrmap);CHKERRQ(ierr);
   ierr = PetscLayoutGetRange(isrmap,&rstart,NULL);CHKERRQ(ierr);
   ierr = ISGetIndices(ito,&ito_indices);CHKERRQ(ierr);
   ierr = PetscCalloc2(size,&tosizes_tmp,size+1,&tooffsets_tmp);CHKERRQ(ierr);
   for (i=0; i<ito_ln; i++) {
     if (ito_indices[i]<0) continue;
     else PetscAssertFalse(ito_indices[i]>=size,comm,PETSC_ERR_ARG_OUTOFRANGE,"target rank %" PetscInt_FMT " is larger than communicator size %d ",ito_indices[i],size);
     tosizes_tmp[ito_indices[i]]++;
   }
   nto = 0;
   for (i=0; i<size; i++) {
     tooffsets_tmp[i+1] = tooffsets_tmp[i]+tosizes_tmp[i];
     if (tosizes_tmp[i]>0) nto++;
   }
   ierr = PetscCalloc2(nto,&toranks,2*nto,&tosizes);CHKERRQ(ierr);
   nto  = 0;
   for (i=0; i<size; i++) {
     if (tosizes_tmp[i]>0) {
       toranks[nto]     = i;
       tosizes[2*nto]   = tosizes_tmp[i];/* size */
       tosizes[2*nto+1] = tooffsets_tmp[i];/* offset */
       nto++;
     }
   }
   nsends = tooffsets_tmp[size];
   ierr   = PetscCalloc1(nsends,&send_indices);CHKERRQ(ierr);
   if (toindx) {
     ierr = ISGetIndices(toindx,&toindx_indices);CHKERRQ(ierr);
   }
   for (i=0; i<ito_ln; i++) {
     if (ito_indices[i]<0) continue;
     target_rank = ito_indices[i];
     send_indices[tooffsets_tmp[target_rank]] = toindx? toindx_indices[i]:(i+rstart);
     tooffsets_tmp[target_rank]++;
   }
   if (toindx) {
     ierr = ISRestoreIndices(toindx,&toindx_indices);CHKERRQ(ierr);
   }
   ierr = ISRestoreIndices(ito,&ito_indices);CHKERRQ(ierr);
   ierr = PetscFree2(tosizes_tmp,tooffsets_tmp);CHKERRQ(ierr);
   ierr = PetscCommBuildTwoSided(comm,2,MPIU_INT,nto,toranks,tosizes,&nfrom,&fromranks,&fromsizes);CHKERRQ(ierr);
   ierr = PetscFree2(toranks,tosizes);CHKERRQ(ierr);
   ierr = PetscMalloc1(nfrom,&fromperm_newtoold);CHKERRQ(ierr);
   for (i=0; i<nfrom; i++) fromperm_newtoold[i] = i;
   ierr   = PetscSortMPIIntWithArray(nfrom,fromranks,fromperm_newtoold);CHKERRQ(ierr);
   nrecvs = 0;
   for (i=0; i<nfrom; i++) nrecvs += fromsizes[i*2];
   ierr   = PetscCalloc1(nrecvs,&recv_indices);CHKERRQ(ierr);
   ierr   = PetscMalloc1(nrecvs,&iremote);CHKERRQ(ierr);
   nrecvs = 0;
   for (i=0; i<nfrom; i++) {
     for (j=0; j<fromsizes[2*fromperm_newtoold[i]]; j++) {
       iremote[nrecvs].rank    = fromranks[i];
       iremote[nrecvs++].index = fromsizes[2*fromperm_newtoold[i]+1]+j;
     }
   }
   ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
   ierr = PetscSFSetGraph(sf,nsends,nrecvs,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
   ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
   /* how to put a prefix ? */
   ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
   ierr = PetscSFBcastBegin(sf,MPIU_INT,send_indices,recv_indices,MPI_REPLACE);CHKERRQ(ierr);
   ierr = PetscSFBcastEnd(sf,MPIU_INT,send_indices,recv_indices,MPI_REPLACE);CHKERRQ(ierr);
   ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
   ierr = PetscFree(fromranks);CHKERRQ(ierr);
   ierr = PetscFree(fromsizes);CHKERRQ(ierr);
   ierr = PetscFree(fromperm_newtoold);CHKERRQ(ierr);
   ierr = PetscFree(send_indices);CHKERRQ(ierr);
   if (rows) {
     ierr = PetscSortInt(nrecvs,recv_indices);CHKERRQ(ierr);
     ierr = ISCreateGeneral(comm,nrecvs,recv_indices,PETSC_OWN_POINTER,rows);CHKERRQ(ierr);
   } else {
     ierr = PetscFree(recv_indices);CHKERRQ(ierr);
   }
   PetscFunctionReturn(0);
}

/*@
    ISPartitioningToNumbering - Takes an ISPartitioning and on each processor
    generates an IS that contains a new global node number for each index based
    on the partitioing.

    Collective on IS

    Input Parameters:
.   partitioning - a partitioning as generated by MatPartitioningApply()
                   or MatPartitioningApplyND()

    Output Parameter:
.   is - on each processor the index set that defines the global numbers
         (in the new numbering) for all the nodes currently (before the partitioning)
         on that processor

   Level: advanced

.seealso: MatPartitioningCreate(), AOCreateBasic(), ISPartitioningCount()

@*/
PetscErrorCode  ISPartitioningToNumbering(IS part,IS *is)
{
  MPI_Comm       comm;
  IS             ndorder;
  PetscInt       i,np,npt,n,*starts = NULL,*sums = NULL,*lsizes = NULL,*newi = NULL;
  const PetscInt *indices = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,IS_CLASSID,1);
  PetscValidPointer(is,2);
  /* see if the partitioning comes from nested dissection */
  ierr = PetscObjectQuery((PetscObject)part,"_petsc_matpartitioning_ndorder",(PetscObject*)&ndorder);CHKERRQ(ierr);
  if (ndorder) {
    ierr = PetscObjectReference((PetscObject)ndorder);CHKERRQ(ierr);
    *is  = ndorder;
    PetscFunctionReturn(0);
  }

  ierr = PetscObjectGetComm((PetscObject)part,&comm);CHKERRQ(ierr);
  /* count the number of partitions, i.e., virtual processors */
  ierr = ISGetLocalSize(part,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(part,&indices);CHKERRQ(ierr);
  np   = 0;
  for (i=0; i<n; i++) np = PetscMax(np,indices[i]);
  ierr = MPIU_Allreduce(&np,&npt,1,MPIU_INT,MPI_MAX,comm);CHKERRMPI(ierr);
  np   = npt+1; /* so that it looks like a MPI_Comm_size output */

  /*
        lsizes - number of elements of each partition on this particular processor
        sums - total number of "previous" nodes for any particular partition
        starts - global number of first element in each partition on this processor
  */
  ierr = PetscMalloc3(np,&lsizes,np,&starts,np,&sums);CHKERRQ(ierr);
  ierr = PetscArrayzero(lsizes,np);CHKERRQ(ierr);
  for (i=0; i<n; i++) lsizes[indices[i]]++;
  ierr = MPIU_Allreduce(lsizes,sums,np,MPIU_INT,MPI_SUM,comm);CHKERRMPI(ierr);
  ierr = MPI_Scan(lsizes,starts,np,MPIU_INT,MPI_SUM,comm);CHKERRMPI(ierr);
  for (i=0; i<np; i++) starts[i] -= lsizes[i];
  for (i=1; i<np; i++) {
    sums[i]   += sums[i-1];
    starts[i] += sums[i-1];
  }

  /*
      For each local index give it the new global number
  */
  ierr = PetscMalloc1(n,&newi);CHKERRQ(ierr);
  for (i=0; i<n; i++) newi[i] = starts[indices[i]]++;
  ierr = PetscFree3(lsizes,starts,sums);CHKERRQ(ierr);

  ierr = ISRestoreIndices(part,&indices);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n,newi,PETSC_OWN_POINTER,is);CHKERRQ(ierr);
  ierr = ISSetPermutation(*is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    ISPartitioningCount - Takes a ISPartitioning and determines the number of
    resulting elements on each (partition) process

    Collective on IS

    Input Parameters:
+   partitioning - a partitioning as generated by MatPartitioningApply() or
                   MatPartitioningApplyND()
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
        If the partitioning has been obtained by MatPartitioningApplyND(),
        the returned count does not include the separators.

.seealso: MatPartitioningCreate(), AOCreateBasic(), ISPartitioningToNumbering(),
        MatPartitioningSetNParts(), MatPartitioningApply(), MatPartitioningApplyND()

@*/
PetscErrorCode  ISPartitioningCount(IS part,PetscInt len,PetscInt count[])
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
    ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
    len  = (PetscInt) size;
  }

  /* count the number of partitions */
  ierr = ISGetLocalSize(part,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(part,&indices);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    PetscInt np = 0,npt;
    for (i=0; i<n; i++) np = PetscMax(np,indices[i]);
    ierr = MPIU_Allreduce(&np,&npt,1,MPIU_INT,MPI_MAX,comm);CHKERRMPI(ierr);
    np   = npt+1; /* so that it looks like a MPI_Comm_size output */
    PetscAssertFalse(np > len,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Length of count array %" PetscInt_FMT " is less than number of partitions %" PetscInt_FMT,len,np);
  }

  /*
        lsizes - number of elements of each partition on this particular processor
        sums - total number of "previous" nodes for any particular partition
        starts - global number of first element in each partition on this processor
  */
  ierr = PetscCalloc1(len,&lsizes);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (indices[i] > -1) lsizes[indices[i]]++;
  }
  ierr = ISRestoreIndices(part,&indices);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(len,&npp);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(lsizes,count,npp,MPIU_INT,MPI_SUM,comm);CHKERRMPI(ierr);
  ierr = PetscFree(lsizes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlock()
@*/
PetscErrorCode  ISAllGather(IS is,IS *isout)
{
  PetscErrorCode ierr;
  PetscInt       *indices,n,i,N,step,first;
  const PetscInt *lindices;
  MPI_Comm       comm;
  PetscMPIInt    size,*sizes = NULL,*offsets = NULL,nn;
  PetscBool      stride;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(isout,2);

  ierr = PetscObjectGetComm((PetscObject)is,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)is,ISSTRIDE,&stride);CHKERRQ(ierr);
  if (size == 1 && stride) { /* should handle parallel ISStride also */
    ierr = ISStrideGetInfo(is,&first,&step);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,n,first,step,isout);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc2(size,&sizes,size,&offsets);CHKERRQ(ierr);

    ierr       = PetscMPIIntCast(n,&nn);CHKERRQ(ierr);
    ierr       = MPI_Allgather(&nn,1,MPI_INT,sizes,1,MPI_INT,comm);CHKERRMPI(ierr);
    offsets[0] = 0;
    for (i=1; i<size; i++) {
      PetscInt s = offsets[i-1] + sizes[i-1];
      ierr = PetscMPIIntCast(s,&offsets[i]);CHKERRQ(ierr);
    }
    N = offsets[size-1] + sizes[size-1];

    ierr = PetscMalloc1(N,&indices);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&lindices);CHKERRQ(ierr);
    ierr = MPI_Allgatherv((void*)lindices,nn,MPIU_INT,indices,sizes,offsets,MPIU_INT,comm);CHKERRMPI(ierr);
    ierr = ISRestoreIndices(is,&lindices);CHKERRQ(ierr);
    ierr = PetscFree2(sizes,offsets);CHKERRQ(ierr);

    ierr = ISCreateGeneral(PETSC_COMM_SELF,N,indices,PETSC_OWN_POINTER,isout);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
    ISAllGatherColors - Given a a set of colors on each processor, generates a large
    set (same on each processor) by concatenating together each processors colors

    Collective

    Input Parameters:
+   comm - communicator to share the indices
.   n - local size of set
-   lindices - local colors

    Output Parameters:
+   outN - total number of indices
-   outindices - all of the colors

    Notes:
    ISAllGatherColors() is clearly not scalable for large index sets.

    Level: intermediate

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlock(), ISAllGather()
@*/
PetscErrorCode  ISAllGatherColors(MPI_Comm comm,PetscInt n,ISColoringValue *lindices,PetscInt *outN,ISColoringValue *outindices[])
{
  ISColoringValue *indices;
  PetscErrorCode  ierr;
  PetscInt        i,N;
  PetscMPIInt     size,*offsets = NULL,*sizes = NULL, nn = n;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = PetscMalloc2(size,&sizes,size,&offsets);CHKERRQ(ierr);

  ierr       = MPI_Allgather(&nn,1,MPI_INT,sizes,1,MPI_INT,comm);CHKERRMPI(ierr);
  offsets[0] = 0;
  for (i=1; i<size; i++) offsets[i] = offsets[i-1] + sizes[i-1];
  N    = offsets[size-1] + sizes[size-1];
  ierr = PetscFree2(sizes,offsets);CHKERRQ(ierr);

  ierr = PetscMalloc1(N+1,&indices);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(lindices,(PetscMPIInt)n,MPIU_COLORING_VALUE,indices,sizes,offsets,MPIU_COLORING_VALUE,comm);CHKERRMPI(ierr);

  *outindices = indices;
  if (outN) *outN = N;
  PetscFunctionReturn(0);
}

/*@
    ISComplement - Given an index set (IS) generates the complement index set. That is all
       all indices that are NOT in the given set.

    Collective on IS

    Input Parameters:
+   is - the index set
.   nmin - the first index desired in the local part of the complement
-   nmax - the largest index desired in the local part of the complement (note that all indices in is must be greater or equal to nmin and less than nmax)

    Output Parameter:
.   isout - the complement

    Notes:
    The communicator for this new IS is the same as for the input IS

      For a parallel IS, this will generate the local part of the complement on each process

      To generate the entire complement (on each process) of a parallel IS, first call ISAllGather() and then
    call this routine.

    Level: intermediate

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlock(), ISAllGather()
@*/
PetscErrorCode  ISComplement(IS is,PetscInt nmin,PetscInt nmax,IS *isout)
{
  PetscErrorCode ierr;
  const PetscInt *indices;
  PetscInt       n,i,j,unique,cnt,*nindices;
  PetscBool      sorted;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(isout,4);
  PetscAssertFalse(nmin < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nmin %" PetscInt_FMT " cannot be negative",nmin);
  PetscAssertFalse(nmin > nmax,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nmin %" PetscInt_FMT " cannot be greater than nmax %" PetscInt_FMT,nmin,nmax);
  ierr = ISSorted(is,&sorted);CHKERRQ(ierr);
  PetscAssertFalse(!sorted,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Index set must be sorted");

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&indices);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    for (i=0; i<n; i++) {
      PetscAssertFalse(indices[i] <  nmin,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %" PetscInt_FMT "'s value %" PetscInt_FMT " is smaller than minimum given %" PetscInt_FMT,i,indices[i],nmin);
      PetscAssertFalse(indices[i] >= nmax,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %" PetscInt_FMT "'s value %" PetscInt_FMT " is larger than maximum given %" PetscInt_FMT,i,indices[i],nmax);
    }
  }
  /* Count number of unique entries */
  unique = (n>0);
  for (i=0; i<n-1; i++) {
    if (indices[i+1] != indices[i]) unique++;
  }
  ierr = PetscMalloc1(nmax-nmin-unique,&nindices);CHKERRQ(ierr);
  cnt  = 0;
  for (i=nmin,j=0; i<nmax; i++) {
    if (j<n && i==indices[j]) do { j++; } while (j<n && i==indices[j]);
    else nindices[cnt++] = i;
  }
  PetscAssertFalse(cnt != nmax-nmin-unique,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Number of entries found in complement %" PetscInt_FMT " does not match expected %" PetscInt_FMT,cnt,nmax-nmin-unique);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is),cnt,nindices,PETSC_OWN_POINTER,isout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
