/*$Id: isltog.c,v 1.65 2001/05/21 14:16:29 bsmith Exp $*/

#include "petscconfig.h"
#include "petscsys.h"   /*I "petscsys.h" I*/
#include "src/vec/is/isimpl.h"    /*I "petscis.h"  I*/

EXTERN int VecInitializePackage(char *);

#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingGetSize"
/*@C
    ISLocalToGlobalMappingGetSize - Gets the local size of a local to global mapping.

    Not Collective

    Input Parameter:
.   ltog - local to global mapping

    Output Parameter:
.   n - the number of entries in the local mapping

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
int ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping mapping,int *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE);
  *n = mapping->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingView"
/*@C
    ISLocalToGlobalMappingView - View a local to global mapping

    Not Collective

    Input Parameters:
+   ltog - local to global mapping
-   viewer - viewer

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
int ISLocalToGlobalMappingView(ISLocalToGlobalMapping mapping,PetscViewer viewer)
{
  int        i,ierr,rank;
  PetscTruth isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(mapping->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);

  ierr = MPI_Comm_rank(mapping->comm,&rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    for (i=0; i<mapping->n; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %d %d\n",rank,i,mapping->indices[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported for ISLocalToGlobalMapping",((PetscObject)viewer)->type_name);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingCreateIS"
/*@C
    ISLocalToGlobalMappingCreateIS - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not collective

    Input Parameter:
.   is - index set containing the global numbers for each local

    Output Parameter:
.   mapping - new mapping data structure

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
int ISLocalToGlobalMappingCreateIS(IS is,ISLocalToGlobalMapping *mapping)
{
  int      n,*indices,ierr;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = PetscObjectGetComm((PetscObject)is,&comm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&indices);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm,n,indices,mapping);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&indices);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingCreate"
/*@C
    ISLocalToGlobalMappingCreate - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not Collective, but communicator may have more than one process

    Input Parameters:
+   comm - MPI communicator
.   n - the number of local elements
-   indices - the global index for each local element

    Output Parameter:
.   mapping - new mapping data structure

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS()
@*/
int ISLocalToGlobalMappingCreate(MPI_Comm cm,int n,const int indices[],ISLocalToGlobalMapping *mapping)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidIntPointer(indices);
  PetscValidPointer(mapping);
  *mapping = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = VecInitializePackage(PETSC_NULL);                                                                CHKERRQ(ierr);
#endif

  PetscHeaderCreate(*mapping,_p_ISLocalToGlobalMapping,int,IS_LTOGM_COOKIE,0,"ISLocalToGlobalMapping",
                    cm,ISLocalToGlobalMappingDestroy,ISLocalToGlobalMappingView);
  PetscLogObjectCreate(*mapping);
  PetscLogObjectMemory(*mapping,sizeof(struct _p_ISLocalToGlobalMapping)+n*sizeof(int));

  (*mapping)->n       = n;
  ierr = PetscMalloc((n+1)*sizeof(int),&(*mapping)->indices);CHKERRQ(ierr);
  ierr = PetscMemcpy((*mapping)->indices,indices,n*sizeof(int));CHKERRQ(ierr);

  /*
      Do not create the global to local mapping. This is only created if 
     ISGlobalToLocalMapping() is called 
  */
  (*mapping)->globals = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingBlock"
/*@C
    ISLocalToGlobalMappingBlock - Creates a blocked index version of an 
       ISLocalToGlobalMapping that is appropriate for MatSetLocalToGlobalMappingBlock()
       and VecSetLocalToGlobalMappingBlock().

    Not Collective, but communicator may have more than one process

    Input Parameters:
+    inmap - original point-wise mapping
-    bs - block size

    Output Parameter:
.   outmap - block based mapping

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingCreateIS()
@*/
int ISLocalToGlobalMappingBlock(ISLocalToGlobalMapping inmap,int bs,ISLocalToGlobalMapping *outmap)
{
  int ierr,*ii,i,n;

  PetscFunctionBegin;

  if (bs > 1) {
    n    = inmap->n/bs;
    ierr = PetscMalloc(n*sizeof(int),&ii);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ii[i] = inmap->indices[bs*i]/bs;
    }
    ierr = ISLocalToGlobalMappingCreate(inmap->comm,n,ii,outmap);CHKERRQ(ierr);
    ierr = PetscFree(ii);CHKERRQ(ierr);
  } else {
    *outmap = inmap;
    ierr    = PetscObjectReference((PetscObject)inmap);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingDestroy"
/*@
   ISLocalToGlobalMappingDestroy - Destroys a mapping between a local (0 to n)
   ordering and a global parallel ordering.

   Note Collective

   Input Parameters:
.  mapping - mapping data structure

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate()
@*/
int ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping mapping)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidPointer(mapping);
  if (--mapping->refct > 0) PetscFunctionReturn(0);
  if (mapping->refct < 0) {
    SETERRQ(1,"Mapping already destroyed");
  }

  ierr = PetscFree(mapping->indices);CHKERRQ(ierr);
  if (mapping->globals) {ierr = PetscFree(mapping->globals);CHKERRQ(ierr);}
  PetscLogObjectDestroy(mapping);
  PetscHeaderDestroy(mapping);
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingApplyIS"
/*@
    ISLocalToGlobalMappingApplyIS - Creates from an IS in the local numbering
    a new index set using the global numbering defined in an ISLocalToGlobalMapping
    context.

    Not collective

    Input Parameters:
+   mapping - mapping between local and global numbering
-   is - index set in local numbering

    Output Parameters:
.   newis - index set in global numbering

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy(), ISGlobalToLocalMappingApply()
@*/
int ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping mapping,IS is,IS *newis)
{
  int ierr,n,i,*idxin,*idxmap,*idxout,Nmax = mapping->n;

  PetscFunctionBegin;
  PetscValidPointer(mapping);
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidPointer(newis);

  ierr   = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr   = ISGetIndices(is,&idxin);CHKERRQ(ierr);
  idxmap = mapping->indices;
  
  ierr = PetscMalloc((n+1)*sizeof(int),&idxout);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (idxin[i] >= Nmax) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Local index %d too large %d (max) at %d",idxin[i],Nmax,i);
    idxout[i] = idxmap[idxin[i]];
  }
  ierr = ISRestoreIndices(is,&idxin);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,idxout,newis);CHKERRQ(ierr);
  ierr = PetscFree(idxout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   ISLocalToGlobalMappingApply - Takes a list of integers in a local numbering
   and converts them to the global numbering.

   Not collective

   Input Parameters:
+  mapping - the local to global mapping context
.  N - number of integers
-  in - input indices in local numbering

   Output Parameter:
.  out - indices in global numbering

   Synopsis:
   int ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,int N,int in[],int out[])

   Notes: 
   The in and out array parameters may be identical.

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate(),ISLocalToGlobalMappingDestroy(), 
          ISLocalToGlobalMappingApplyIS(),AOCreateBasic(),AOApplicationToPetsc(),
          AOPetscToApplication(), ISGlobalToLocalMappingApply()

    Concepts: mapping^local to global

M*/

/* -----------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "ISGlobalToLocalMappingSetUp_Private"
/*
    Creates the global fields in the ISLocalToGlobalMapping structure
*/
static int ISGlobalToLocalMappingSetUp_Private(ISLocalToGlobalMapping mapping)
{
  int ierr,i,*idx = mapping->indices,n = mapping->n,end,start,*globals;

  PetscFunctionBegin;
  end   = 0;
  start = 100000000;

  for (i=0; i<n; i++) {
    if (idx[i] < 0) continue;
    if (idx[i] < start) start = idx[i];
    if (idx[i] > end)   end   = idx[i];
  }
  if (start > end) {start = 0; end = -1;}
  mapping->globalstart = start;
  mapping->globalend   = end;

  ierr             = PetscMalloc((end-start+2)*sizeof(int),&globals);CHKERRQ(ierr);
  mapping->globals = globals;
  for (i=0; i<end-start+1; i++) {
    globals[i] = -1;
  }
  for (i=0; i<n; i++) {
    if (idx[i] < 0) continue;
    globals[idx[i] - start] = i;
  }

  PetscLogObjectMemory(mapping,(end-start+1)*sizeof(int));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGlobalToLocalMappingApply"
/*@
    ISGlobalToLocalMappingApply - Provides the local numbering for a list of integers
    specified with a global numbering.

    Not collective

    Input Parameters:
+   mapping - mapping between local and global numbering
.   type - IS_GTOLM_MASK - replaces global indices with no local value with -1
           IS_GTOLM_DROP - drops the indices with no local value from the output list
.   n - number of global indices to map
-   idx - global indices to map

    Output Parameters:
+   nout - number of indices in output array (if type == IS_GTOLM_MASK then nout = n)
-   idxout - local index of each global index, one must pass in an array long enough 
             to hold all the indices. You can call ISGlobalToLocalMappingApply() with 
             idxout == PETSC_NULL to determine the required length (returned in nout)
             and then allocate the required space and call ISGlobalToLocalMappingApply()
             a second time to set the values.

    Notes:
    Either nout or idxout may be PETSC_NULL. idx and idxout may be identical.

    This is not scalable in memory usage. Each processor requires O(Nglobal) size 
    array to compute these.

    Level: advanced

    Concepts: mapping^global to local

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()
@*/
int ISGlobalToLocalMappingApply(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingType type,
                                  int n,const int idx[],int *nout,int idxout[])
{
  int i,ierr,*globals,nf = 0,tmp,start,end;

  PetscFunctionBegin;
  if (!mapping->globals) {
    ierr = ISGlobalToLocalMappingSetUp_Private(mapping);CHKERRQ(ierr);
  }
  globals = mapping->globals;
  start   = mapping->globalstart;
  end     = mapping->globalend;

  if (type == IS_GTOLM_MASK) {
    if (idxout) {
      for (i=0; i<n; i++) {
        if (idx[i] < 0) idxout[i] = idx[i]; 
        else if (idx[i] < start) idxout[i] = -1;
        else if (idx[i] > end)   idxout[i] = -1;
        else                     idxout[i] = globals[idx[i] - start];
      }
    }
    if (nout) *nout = n;
  } else {
    if (idxout) {
      for (i=0; i<n; i++) {
        if (idx[i] < 0) continue;
        if (idx[i] < start) continue;
        if (idx[i] > end) continue;
        tmp = globals[idx[i] - start];
        if (tmp < 0) continue;
        idxout[nf++] = tmp;
      }
    } else {
      for (i=0; i<n; i++) {
        if (idx[i] < 0) continue;
        if (idx[i] < start) continue;
        if (idx[i] > end) continue;
        tmp = globals[idx[i] - start];
        if (tmp < 0) continue;
        nf++;
      }
    }
    if (nout) *nout = nf;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingGetInfo"
/*@C
    ISLocalToGlobalMappingGetInfo - Gets the neighbor information for each processor and 
     each index shared by more than one processor 

    Collective on ISLocalToGlobalMapping

    Input Parameters:
.   mapping - the mapping from local to global indexing

    Output Parameter:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each subdomain (processor)
-   indices - indices of local nodes shared with neighbor (sorted by global numbering)

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingRestoreInfo()
@*/
int ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping mapping,int *nproc,int **procs,int **numprocs,int ***indices)
{
  int         i,n = mapping->n,ierr,Ng,ng,max = 0,*lindices = mapping->indices;
  int         size,rank,*nprocs,*owner,nsends,*sends,j,*starts,*work,nmax,nrecvs,*recvs,proc;
  int         tag1,tag2,tag3,cnt,*len,*source,imdex,scale,*ownedsenders,*nownedsenders,rstart,nowned;
  int         node,nownedm,nt,*sends2,nsends2,*starts2,*lens2,*dest,nrecvs2,*starts3,*recvs2,k,*bprocs,*tmp;
  int         first_procs,first_numprocs,*first_indices;
  MPI_Request *recv_waits,*send_waits;
  MPI_Status  recv_status,*send_status,*recv_statuses;
  MPI_Comm    comm = mapping->comm;
  PetscTruth  debug = PETSC_FALSE;

  PetscFunctionBegin;
  ierr   = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr   = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (size == 1) {
    *nproc         = 0;
    *procs         = PETSC_NULL;
    ierr           = PetscMalloc(sizeof(int),numprocs);CHKERRQ(ierr);
    (*numprocs)[0] = 0;
    ierr           = PetscMalloc(sizeof(int*),indices);CHKERRQ(ierr); 
    (*indices)[0]  = PETSC_NULL;
    PetscFunctionReturn(0);
  }

  ierr = PetscOptionsHasName(PETSC_NULL,"-islocaltoglobalmappinggetinfo_debug",&debug);CHKERRQ(ierr);

  /*
    Notes on ISLocalToGlobalMappingGetInfo

    globally owned node - the nodes that have been assigned to this processor in global
           numbering, just for this routine.

    nontrivial globally owned node - node assigned to this processor that is on a subdomain
           boundary (i.e. is has more than one local owner)

    locally owned node - node that exists on this processors subdomain

    nontrivial locally owned node - node that is not in the interior (i.e. has more than one
           local subdomain
  */
  ierr = PetscObjectGetNewTag((PetscObject)mapping,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)mapping,&tag2);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)mapping,&tag3);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    if (lindices[i] > max) max = lindices[i];
  }
  ierr   = MPI_Allreduce(&max,&Ng,1,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
  Ng++;
  ierr   = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr   = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  scale  = Ng/size + 1;
  ng     = scale; if (rank == size-1) ng = Ng - scale*(size-1); ng = PetscMax(1,ng);
  rstart = scale*rank;

  /* determine ownership ranges of global indices */
  ierr = PetscMalloc((2*size+1)*sizeof(int),&nprocs);CHKERRQ(ierr);
  ierr = PetscMemzero(nprocs,2*size*sizeof(int));CHKERRQ(ierr);

  /* determine owners of each local node  */
  ierr = PetscMalloc((n+1)*sizeof(int),&owner);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    proc              = lindices[i]/scale; /* processor that globally owns this index */
    nprocs[size+proc] = 1;                 /* processor globally owns at least one of ours */
    owner[i]          = proc;              
    nprocs[proc]++;                        /* count of how many that processor globally owns of ours */
  }
  nsends = 0; for (i=0; i<size; i++) nsends += nprocs[size + i];
  PetscLogInfo(0,"ISLocalToGlobalMappingGetInfo: Number of global owners for my local data %d\n",nsends);

  /* inform other processors of number of messages and max length*/
  ierr = PetscMalloc(2*size*sizeof(int),&work);CHKERRQ(ierr);
  ierr   = MPI_Allreduce(nprocs,work,2*size,MPI_INT,PetscMaxSum_Op,comm);CHKERRQ(ierr);
  nmax   = work[rank];
  nrecvs = work[size+rank]; 
  ierr   = PetscFree(work);CHKERRQ(ierr);
  PetscLogInfo(0,"ISLocalToGlobalMappingGetInfo: Number of local owners for my global data %d\n",nrecvs);

  /* post receives for owned rows */
  ierr = PetscMalloc((2*nrecvs+1)*(nmax+1)*sizeof(int),&recvs);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(recvs+2*nmax*i,2*nmax,MPI_INT,MPI_ANY_SOURCE,tag1,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* pack messages containing lists of local nodes to owners */
  ierr       = PetscMalloc((2*n+1)*sizeof(int),&sends);CHKERRQ(ierr);
  ierr       = PetscMalloc((size+1)*sizeof(int),&starts);CHKERRQ(ierr);
  starts[0]  = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + 2*nprocs[i-1];} 
  for (i=0; i<n; i++) {
    sends[starts[owner[i]]++] = lindices[i];
    sends[starts[owner[i]]++] = i;
  }
  ierr = PetscFree(owner);CHKERRQ(ierr);
  starts[0]  = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + 2*nprocs[i-1];} 

  /* send the messages */
  ierr = PetscMalloc((nsends+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc((nsends+1)*sizeof(int),&dest);CHKERRQ(ierr);
  cnt = 0;
  for (i=0; i<size; i++) {
    if (nprocs[i]) {
      ierr      = MPI_Isend(sends+starts[i],2*nprocs[i],MPI_INT,i,tag1,comm,send_waits+cnt);CHKERRQ(ierr);
      dest[cnt] = i;
      cnt++;
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  /* wait on receives */
  ierr = PetscMalloc((2*nrecvs+1)*sizeof(int),&source);CHKERRQ(ierr);
  len  = source + nrecvs;
  cnt  = nrecvs; 
  ierr = PetscMalloc((ng+1)*sizeof(int),&nownedsenders);CHKERRQ(ierr);
  ierr = PetscMemzero(nownedsenders,ng*sizeof(int));CHKERRQ(ierr);
  while (cnt) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr           = MPI_Get_count(&recv_status,MPI_INT,&len[imdex]);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    len[imdex]     = len[imdex]/2;
    /* count how many local owners for each of my global owned indices */
    for (i=0; i<len[imdex]; i++) nownedsenders[recvs[2*imdex*nmax+2*i]-rstart]++;
    cnt--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);

  /* count how many globally owned indices are on an edge multiplied by how many processors own them. */
  nowned  = 0;
  nownedm = 0;
  for (i=0; i<ng; i++) {
    if (nownedsenders[i] > 1) {nownedm += nownedsenders[i]; nowned++;}
  }

  /* create single array to contain rank of all local owners of each globally owned index */
  ierr      = PetscMalloc((nownedm+1)*sizeof(int),&ownedsenders);CHKERRQ(ierr);
  ierr      = PetscMalloc((ng+1)*sizeof(int),&starts);CHKERRQ(ierr);
  starts[0] = 0;
  for (i=1; i<ng; i++) {
    if (nownedsenders[i-1] > 1) starts[i] = starts[i-1] + nownedsenders[i-1];
    else starts[i] = starts[i-1];
  }

  /* for each nontrival globally owned node list all arriving processors */
  for (i=0; i<nrecvs; i++) {
    for (j=0; j<len[i]; j++) {
      node = recvs[2*i*nmax+2*j]-rstart;
      if (nownedsenders[node] > 1) {
        ownedsenders[starts[node]++] = source[i];
      }
    }
  }

  if (debug) { /* -----------------------------------  */
    starts[0]    = 0;
    for (i=1; i<ng; i++) {
      if (nownedsenders[i-1] > 1) starts[i] = starts[i-1] + nownedsenders[i-1];
      else starts[i] = starts[i-1];
    }
    for (i=0; i<ng; i++) {
      if (nownedsenders[i] > 1) {
        ierr = PetscSynchronizedPrintf(comm,"[%d] global node %d local owner processors: ",rank,i+rstart);CHKERRQ(ierr);
        for (j=0; j<nownedsenders[i]; j++) {
          ierr = PetscSynchronizedPrintf(comm,"%d ",ownedsenders[starts[i]+j]);CHKERRQ(ierr);
        }
        ierr = PetscSynchronizedPrintf(comm,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);
  }/* -----------------------------------  */

  /* wait on original sends */
  if (nsends) {
    ierr = PetscMalloc(nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);CHKERRQ(ierr);
  ierr = PetscFree(sends);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);

  /* pack messages to send back to local owners */
  starts[0]    = 0;
  for (i=1; i<ng; i++) {
    if (nownedsenders[i-1] > 1) starts[i] = starts[i-1] + nownedsenders[i-1];
    else starts[i] = starts[i-1];
  }
  nsends2 = nrecvs;
  ierr    = PetscMalloc((nsends2+1)*sizeof(int),&nprocs);CHKERRQ(ierr); /* length of each message */
  for (i=0; i<nrecvs; i++) {
    nprocs[i] = 1;
    for (j=0; j<len[i]; j++) {
      node = recvs[2*i*nmax+2*j]-rstart;
      if (nownedsenders[node] > 1) {
        nprocs[i] += 2 + nownedsenders[node];
      }
    }
  }
  nt = 0; for (i=0; i<nsends2; i++) nt += nprocs[i];
  ierr = PetscMalloc((nt+1)*sizeof(int),&sends2);CHKERRQ(ierr); 
  ierr = PetscMalloc((nsends2+1)*sizeof(int),&starts2);CHKERRQ(ierr);
  starts2[0] = 0; for (i=1; i<nsends2; i++) starts2[i] = starts2[i-1] + nprocs[i-1];
  /*
     Each message is 1 + nprocs[i] long, and consists of 
       (0) the number of nodes being sent back 
       (1) the local node number,
       (2) the number of processors sharing it,
       (3) the processors sharing it
  */
  for (i=0; i<nsends2; i++) {
    cnt = 1;
    sends2[starts2[i]] = 0;
    for (j=0; j<len[i]; j++) {
      node = recvs[2*i*nmax+2*j]-rstart;
      if (nownedsenders[node] > 1) {
        sends2[starts2[i]]++;
        sends2[starts2[i]+cnt++] = recvs[2*i*nmax+2*j+1];
        sends2[starts2[i]+cnt++] = nownedsenders[node];
        ierr = PetscMemcpy(&sends2[starts2[i]+cnt],&ownedsenders[starts[node]],nownedsenders[node]*sizeof(int));CHKERRQ(ierr);
        cnt += nownedsenders[node];
      }
    }
  }

  /* send the message lengths */
  for (i=0; i<nsends2; i++) {
    ierr = MPI_Send(&nprocs[i],1,MPI_INT,source[i],tag2,comm);CHKERRQ(ierr);
  }

  /* receive the message lengths */
  nrecvs2 = nsends;
  ierr = PetscMalloc((nrecvs2+1)*sizeof(int),&lens2);CHKERRQ(ierr);  
  ierr = PetscMalloc((nrecvs2+1)*sizeof(int),&starts3);CHKERRQ(ierr);  
  nt      = 0;
  for (i=0; i<nrecvs2; i++) {
    ierr =  MPI_Recv(&lens2[i],1,MPI_INT,dest[i],tag2,comm,&recv_status);CHKERRQ(ierr);
    nt   += lens2[i];
  }
  starts3[0] = 0;
  for (i=0; i<nrecvs2-1; i++) {
    starts3[i+1] = starts3[i] + lens2[i];
  }
  ierr = PetscMalloc((nt+1)*sizeof(int),&recvs2);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs2+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs2; i++) {
    ierr = MPI_Irecv(recvs2+starts3[i],lens2[i],MPI_INT,dest[i],tag3,comm,recv_waits+i);CHKERRQ(ierr);
  }
  
  /* send the messages */
  ierr = PetscMalloc((nsends2+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  for (i=0; i<nsends2; i++) {
    ierr = MPI_Isend(sends2+starts2[i],nprocs[i],MPI_INT,source[i],tag3,comm,send_waits+i);CHKERRQ(ierr);
  }

  /* wait on receives */
  ierr = PetscMalloc((nrecvs2+1)*sizeof(MPI_Status),&recv_statuses);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrecvs2,recv_waits,recv_statuses);CHKERRQ(ierr);
  ierr = PetscFree(recv_statuses);CHKERRQ(ierr);
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);

  if (debug) { /* -----------------------------------  */
    cnt = 0;
    for (i=0; i<nrecvs2; i++) {
      nt = recvs2[cnt++];
      for (j=0; j<nt; j++) {
        ierr = PetscSynchronizedPrintf(comm,"[%d] local node %d number of subdomains %d: ",rank,recvs2[cnt],recvs2[cnt+1]);CHKERRQ(ierr);
        for (k=0; k<recvs2[cnt+1]; k++) {
          ierr = PetscSynchronizedPrintf(comm,"%d ",recvs2[cnt+2+k]);CHKERRQ(ierr);
        }
        cnt += 2 + recvs2[cnt+1];
        ierr = PetscSynchronizedPrintf(comm,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);
  } /* -----------------------------------  */

  /* count number subdomains for each local node */
  ierr = PetscMalloc(size*sizeof(int),&nprocs);CHKERRQ(ierr);
  ierr = PetscMemzero(nprocs,size*sizeof(int));CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<nrecvs2; i++) {
    nt = recvs2[cnt++];
    for (j=0; j<nt; j++) {
      for (k=0; k<recvs2[cnt+1]; k++) {
        nprocs[recvs2[cnt+2+k]]++;
      }
      cnt += 2 + recvs2[cnt+1];
    }
  }
  nt = 0; for (i=0; i<size; i++) nt += (nprocs[i] > 0);
  *nproc    = nt;
  ierr = PetscMalloc((nt+1)*sizeof(int),procs);CHKERRQ(ierr);
  ierr = PetscMalloc((nt+1)*sizeof(int),numprocs);CHKERRQ(ierr);
  ierr = PetscMalloc((nt+1)*sizeof(int*),indices);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(int),&bprocs);CHKERRQ(ierr);
  cnt       = 0;
  for (i=0; i<size; i++) {
    if (nprocs[i] > 0) {
      bprocs[i]        = cnt;
      (*procs)[cnt]    = i;
      (*numprocs)[cnt] = nprocs[i];
      ierr             = PetscMalloc(nprocs[i]*sizeof(int),&(*indices)[cnt]);CHKERRQ(ierr);
      cnt++;
    }
  }

  /* make the list of subdomains for each nontrivial local node */
  ierr = PetscMemzero(*numprocs,nt*sizeof(int));CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<nrecvs2; i++) {
    nt = recvs2[cnt++];
    for (j=0; j<nt; j++) {
      for (k=0; k<recvs2[cnt+1]; k++) {
        (*indices)[bprocs[recvs2[cnt+2+k]]][(*numprocs)[bprocs[recvs2[cnt+2+k]]]++] = recvs2[cnt];
      }
      cnt += 2 + recvs2[cnt+1];
    }
  }
  ierr = PetscFree(bprocs);CHKERRQ(ierr);
  ierr = PetscFree(recvs2);CHKERRQ(ierr);

  /* sort the node indexing by their global numbers */
  nt = *nproc;
  for (i=0; i<nt; i++) {
    ierr = PetscMalloc(((*numprocs)[i])*sizeof(int),&tmp);CHKERRQ(ierr);
    for (j=0; j<(*numprocs)[i]; j++) {
      tmp[j] = lindices[(*indices)[i][j]];
    }
    ierr = PetscSortIntWithArray((*numprocs)[i],tmp,(*indices)[i]);CHKERRQ(ierr); 
    ierr = PetscFree(tmp);CHKERRQ(ierr);
  }

  if (debug) { /* -----------------------------------  */
    nt = *nproc;
    for (i=0; i<nt; i++) {
      ierr = PetscSynchronizedPrintf(comm,"[%d] subdomain %d number of indices %d: ",rank,(*procs)[i],(*numprocs)[i]);CHKERRQ(ierr);
      for (j=0; j<(*numprocs)[i]; j++) {
        ierr = PetscSynchronizedPrintf(comm,"%d ",(*indices)[i][j]);CHKERRQ(ierr);
      }
      ierr = PetscSynchronizedPrintf(comm,"\n");CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);
  } /* -----------------------------------  */

  /* wait on sends */
  if (nsends2) {
    ierr = PetscMalloc(nsends2*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends2,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }

  ierr = PetscFree(starts3);CHKERRQ(ierr);
  ierr = PetscFree(dest);CHKERRQ(ierr);
  ierr = PetscFree(send_waits);CHKERRQ(ierr);

  ierr = PetscFree(nownedsenders);CHKERRQ(ierr);
  ierr = PetscFree(ownedsenders);CHKERRQ(ierr);
  ierr = PetscFree(starts);CHKERRQ(ierr);
  ierr = PetscFree(starts2);CHKERRQ(ierr);
  ierr = PetscFree(lens2);CHKERRQ(ierr);

  ierr = PetscFree(source);CHKERRQ(ierr);
  ierr = PetscFree(recvs);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
  ierr = PetscFree(sends2);CHKERRQ(ierr);

  /* put the information about myself as the first entry in the list */
  first_procs    = (*procs)[0];
  first_numprocs = (*numprocs)[0];
  first_indices  = (*indices)[0];
  for (i=0; i<*nproc; i++) {
    if ((*procs)[i] == rank) {
      (*procs)[0]    = (*procs)[i];
      (*numprocs)[0] = (*numprocs)[i];
      (*indices)[0]  = (*indices)[i];
      (*procs)[i]    = first_procs; 
      (*numprocs)[i] = first_numprocs;
      (*indices)[i]  = first_indices;
      break;
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingRestoreInfo"
/*@C
    ISLocalToGlobalMappingRestoreInfo - Frees the memory allocated by ISLocalToGlobalMappingGetInfo()

    Collective on ISLocalToGlobalMapping

    Input Parameters:
.   mapping - the mapping from local to global indexing

    Output Parameter:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each processor
-   indices - indices of local nodes shared with neighbor (sorted by global numbering)

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetInfo()
@*/
int ISLocalToGlobalMappingRestoreInfo(ISLocalToGlobalMapping mapping,int *nproc,int **procs,int **numprocs,int ***indices)
{
  int ierr,i;

  PetscFunctionBegin;
  if (*procs) {ierr = PetscFree(*procs);CHKERRQ(ierr);}
  if (*numprocs) {ierr = PetscFree(*numprocs);CHKERRQ(ierr);}
  if (*indices) {
    if ((*indices)[0]) {ierr = PetscFree((*indices)[0]);CHKERRQ(ierr);}
    for (i=1; i<*nproc; i++) {
      if ((*indices)[i]) {ierr = PetscFree((*indices)[i]);CHKERRQ(ierr);}
    }
    ierr = PetscFree(*indices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


















