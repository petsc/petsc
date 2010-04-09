#define PETSCVEC_DLL

#include "petscvec.h"   /*I "petscvec.h" I*/
#include "private/isimpl.h"    /*I "petscis.h"  I*/

PetscCookie PETSCVEC_DLLEXPORT IS_LTOGM_COOKIE;

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
PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping mapping,PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,1);
  PetscValidIntPointer(n,2);
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
PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingView(ISLocalToGlobalMapping mapping,PetscViewer viewer)
{
  PetscInt        i;
  PetscMPIInt     rank;
  PetscTruth      iascii;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)mapping)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);

  ierr = MPI_Comm_rank(((PetscObject)mapping)->comm,&rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    for (i=0; i<mapping->n; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %d %d\n",rank,i,mapping->indices[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for ISLocalToGlobalMapping",((PetscObject)viewer)->type_name);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingCreateIS"
/*@
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
PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingCreateIS(IS is,ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;
  PetscInt       n;
  const PetscInt *indices;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(mapping,2);

  ierr = PetscObjectGetComm((PetscObject)is,&comm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&indices);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm,n,indices,mapping);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&indices);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingCreate"
/*@
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

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreateNC()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingCreate(MPI_Comm cm,PetscInt n,const PetscInt indices[],ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;
  PetscInt       *in;

  PetscFunctionBegin;
  if (n) PetscValidIntPointer(indices,3);
  PetscValidPointer(mapping,4);
  ierr = PetscMalloc(n*sizeof(PetscInt),&in);CHKERRQ(ierr);
  ierr = PetscMemcpy(in,indices,n*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateNC(cm,n,in,mapping);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingCreateNC"
/*@C
    ISLocalToGlobalMappingCreateNC - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not Collective, but communicator may have more than one process

    Input Parameters:
+   comm - MPI communicator
.   n - the number of local elements
-   indices - the global index for each local element

    Output Parameter:
.   mapping - new mapping data structure

    Level: developer

    Notes: Does not copy the indices, just keeps the pointer to the indices. The ISLocalToGlobalMappingDestroy()
    will free the space so it must be obtained with PetscMalloc() and it must not be freed elsewhere.

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingCreateNC(MPI_Comm cm,PetscInt n,const PetscInt indices[],ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n) PetscValidIntPointer(indices,3);
  PetscValidPointer(mapping,4);
  *mapping = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = ISInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(*mapping,_p_ISLocalToGlobalMapping,int,IS_LTOGM_COOKIE,0,"ISLocalToGlobalMapping",
			   cm,ISLocalToGlobalMappingDestroy,ISLocalToGlobalMappingView);CHKERRQ(ierr);

  (*mapping)->n       = n;
  (*mapping)->indices = (PetscInt*)indices;
  ierr = PetscLogObjectMemory(*mapping,n*sizeof(PetscInt));CHKERRQ(ierr);

  /*
    Do not create the global to local mapping. This is only created if 
    ISGlobalToLocalMapping() is called 
  */
  (*mapping)->globals = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingBlock"
/*@
    ISLocalToGlobalMappingBlock - Creates a blocked index version of an 
       ISLocalToGlobalMapping that is appropriate for MatSetLocalToGlobalMappingBlock()
       and VecSetLocalToGlobalMappingBlock().

    Not Collective, but communicator may have more than one process

    Input Parameters:
+    inmap - original point-wise mapping
-    bs - block size

    Output Parameter:
.   outmap - block based mapping; the indices are relative to BLOCKS, not individual vector or matrix entries.

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingCreateIS()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingBlock(ISLocalToGlobalMapping inmap,PetscInt bs,ISLocalToGlobalMapping *outmap)
{
  PetscErrorCode ierr;
  PetscInt       *ii,i,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(inmap,IS_LTOGM_COOKIE,1);
  PetscValidPointer(outmap,1);
  if (bs > 1) {
    n    = inmap->n/bs;
    if (n*bs != inmap->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Pointwise mapping length is not divisible by block size");
    ierr = PetscMalloc(n*sizeof(PetscInt),&ii);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ii[i] = inmap->indices[bs*i]/bs;
    }
    ierr = ISLocalToGlobalMappingCreate(((PetscObject)inmap)->comm,n,ii,outmap);CHKERRQ(ierr);
    ierr = PetscFree(ii);CHKERRQ(ierr);
  } else {
    ierr    = PetscObjectReference((PetscObject)inmap);CHKERRQ(ierr);
    *outmap = inmap;
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
PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,1);
  if (--((PetscObject)mapping)->refct > 0) PetscFunctionReturn(0);
  ierr = PetscFree(mapping->indices);CHKERRQ(ierr);
  ierr = PetscFree(mapping->globals);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(mapping);CHKERRQ(ierr);
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
PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping mapping,IS is,IS *newis)
{
  PetscErrorCode ierr;
  PetscInt       n,i,*idxmap,*idxout,Nmax = mapping->n;
  const PetscInt *idxin;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,2);
  PetscValidPointer(newis,3);

  ierr   = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr   = ISGetIndices(is,&idxin);CHKERRQ(ierr);
  idxmap = mapping->indices;
  
  ierr = PetscMalloc(n*sizeof(PetscInt),&idxout);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (idxin[i] >= Nmax) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Local index %d too large %d (max) at %d",idxin[i],Nmax-1,i);
    idxout[i] = idxmap[idxin[i]];
  }
  ierr = ISRestoreIndices(is,&idxin);CHKERRQ(ierr);
  ierr = ISCreateGeneralNC(PETSC_COMM_SELF,n,idxout,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   ISLocalToGlobalMappingApply - Takes a list of integers in a local numbering
   and converts them to the global numbering.

   Synopsis:
   PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,int N,int in[],int out[])

   Not collective

   Input Parameters:
+  mapping - the local to global mapping context
.  N - number of integers
-  in - input indices in local numbering

   Output Parameter:
.  out - indices in global numbering

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
static PetscErrorCode ISGlobalToLocalMappingSetUp_Private(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode ierr;
  PetscInt       i,*idx = mapping->indices,n = mapping->n,end,start,*globals;

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

  ierr             = PetscMalloc((end-start+2)*sizeof(PetscInt),&globals);CHKERRQ(ierr);
  mapping->globals = globals;
  for (i=0; i<end-start+1; i++) {
    globals[i] = -1;
  }
  for (i=0; i<n; i++) {
    if (idx[i] < 0) continue;
    globals[idx[i] - start] = i;
  }

  ierr = PetscLogObjectMemory(mapping,(end-start+1)*sizeof(PetscInt));CHKERRQ(ierr);
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
PetscErrorCode PETSCVEC_DLLEXPORT ISGlobalToLocalMappingApply(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingType type,
                                  PetscInt n,const PetscInt idx[],PetscInt *nout,PetscInt idxout[])
{
  PetscInt       i,*globals,nf = 0,tmp,start,end;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,1);
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
-   indices - indices of nodes (in local numbering) shared with neighbors (sorted by global numbering)

    Level: advanced

    Concepts: mapping^local to global

    Fortran Usage: 
$        ISLocalToGlobalMpngGetInfoSize(ISLocalToGlobalMapping,PetscInt nproc,PetscInt numprocmax,ierr) followed by 
$        ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,PetscInt nproc, PetscInt procs[nproc],PetscInt numprocs[nproc],
          PetscInt indices[nproc][numprocmax],ierr)
        There is no ISLocalToGlobalMappingRestoreInfo() in Fortran. You must make sure that procs[], numprocs[] and 
        indices[][] are large enough arrays, either by allocating them dynamically or defining static ones large enough.


.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingRestoreInfo()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag1,tag2,tag3,*len,*source,imdex;
  PetscInt       i,n = mapping->n,Ng,ng,max = 0,*lindices = mapping->indices;
  PetscInt       *nprocs,*owner,nsends,*sends,j,*starts,nmax,nrecvs,*recvs,proc;
  PetscInt       cnt,scale,*ownedsenders,*nownedsenders,rstart,nowned;
  PetscInt       node,nownedm,nt,*sends2,nsends2,*starts2,*lens2,*dest,nrecvs2,*starts3,*recvs2,k,*bprocs,*tmp;
  PetscInt       first_procs,first_numprocs,*first_indices;
  MPI_Request    *recv_waits,*send_waits;
  MPI_Status     recv_status,*send_status,*recv_statuses;
  MPI_Comm       comm = ((PetscObject)mapping)->comm;
  PetscTruth     debug = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,1);
  ierr   = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr   = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (size == 1) {
    *nproc         = 0;
    *procs         = PETSC_NULL;
    ierr           = PetscMalloc(sizeof(PetscInt),numprocs);CHKERRQ(ierr);
    (*numprocs)[0] = 0;
    ierr           = PetscMalloc(sizeof(PetscInt*),indices);CHKERRQ(ierr); 
    (*indices)[0]  = PETSC_NULL;
    PetscFunctionReturn(0);
  }

  ierr = PetscOptionsGetTruth(PETSC_NULL,"-islocaltoglobalmappinggetinfo_debug",&debug,PETSC_NULL);CHKERRQ(ierr);

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
  ierr   = MPI_Allreduce(&max,&Ng,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
  Ng++;
  ierr   = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr   = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  scale  = Ng/size + 1;
  ng     = scale; if (rank == size-1) ng = Ng - scale*(size-1); ng = PetscMax(1,ng);
  rstart = scale*rank;

  /* determine ownership ranges of global indices */
  ierr = PetscMalloc(2*size*sizeof(PetscInt),&nprocs);CHKERRQ(ierr);
  ierr = PetscMemzero(nprocs,2*size*sizeof(PetscInt));CHKERRQ(ierr);

  /* determine owners of each local node  */
  ierr = PetscMalloc(n*sizeof(PetscInt),&owner);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    proc             = lindices[i]/scale; /* processor that globally owns this index */
    nprocs[2*proc+1] = 1;                 /* processor globally owns at least one of ours */
    owner[i]         = proc;              
    nprocs[2*proc]++;                     /* count of how many that processor globally owns of ours */
  }
  nsends = 0; for (i=0; i<size; i++) nsends += nprocs[2*i+1];
  ierr = PetscInfo1(mapping,"Number of global owners for my local data %d\n",nsends);CHKERRQ(ierr);

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nrecvs);CHKERRQ(ierr);
  ierr = PetscInfo1(mapping,"Number of local owners for my global data %d\n",nrecvs);CHKERRQ(ierr);

  /* post receives for owned rows */
  ierr = PetscMalloc((2*nrecvs+1)*(nmax+1)*sizeof(PetscInt),&recvs);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(recvs+2*nmax*i,2*nmax,MPIU_INT,MPI_ANY_SOURCE,tag1,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* pack messages containing lists of local nodes to owners */
  ierr       = PetscMalloc((2*n+1)*sizeof(PetscInt),&sends);CHKERRQ(ierr);
  ierr       = PetscMalloc((size+1)*sizeof(PetscInt),&starts);CHKERRQ(ierr);
  starts[0]  = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + 2*nprocs[2*i-2];} 
  for (i=0; i<n; i++) {
    sends[starts[owner[i]]++] = lindices[i];
    sends[starts[owner[i]]++] = i;
  }
  ierr = PetscFree(owner);CHKERRQ(ierr);
  starts[0]  = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + 2*nprocs[2*i-2];} 

  /* send the messages */
  ierr = PetscMalloc((nsends+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc((nsends+1)*sizeof(PetscInt),&dest);CHKERRQ(ierr);
  cnt = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i]) {
      ierr      = MPI_Isend(sends+starts[i],2*nprocs[2*i],MPIU_INT,i,tag1,comm,send_waits+cnt);CHKERRQ(ierr);
      dest[cnt] = i;
      cnt++;
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  /* wait on receives */
  ierr = PetscMalloc((nrecvs+1)*sizeof(PetscMPIInt),&source);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs+1)*sizeof(PetscMPIInt),&len);CHKERRQ(ierr);
  cnt  = nrecvs; 
  ierr = PetscMalloc((ng+1)*sizeof(PetscInt),&nownedsenders);CHKERRQ(ierr);
  ierr = PetscMemzero(nownedsenders,ng*sizeof(PetscInt));CHKERRQ(ierr);
  while (cnt) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr           = MPI_Get_count(&recv_status,MPIU_INT,&len[imdex]);CHKERRQ(ierr);
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
  ierr      = PetscMalloc((nownedm+1)*sizeof(PetscInt),&ownedsenders);CHKERRQ(ierr);
  ierr      = PetscMalloc((ng+1)*sizeof(PetscInt),&starts);CHKERRQ(ierr);
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
  ierr    = PetscMalloc((nsends2+1)*sizeof(PetscInt),&nprocs);CHKERRQ(ierr); /* length of each message */
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
  ierr = PetscMalloc((nt+1)*sizeof(PetscInt),&sends2);CHKERRQ(ierr); 
  ierr = PetscMalloc((nsends2+1)*sizeof(PetscInt),&starts2);CHKERRQ(ierr);
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
        ierr = PetscMemcpy(&sends2[starts2[i]+cnt],&ownedsenders[starts[node]],nownedsenders[node]*sizeof(PetscInt));CHKERRQ(ierr);
        cnt += nownedsenders[node];
      }
    }
  }

  /* receive the message lengths */
  nrecvs2 = nsends;
  ierr = PetscMalloc((nrecvs2+1)*sizeof(PetscInt),&lens2);CHKERRQ(ierr);  
  ierr = PetscMalloc((nrecvs2+1)*sizeof(PetscInt),&starts3);CHKERRQ(ierr);  
  ierr = PetscMalloc((nrecvs2+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs2; i++) {
    ierr = MPI_Irecv(&lens2[i],1,MPIU_INT,dest[i],tag2,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* send the message lengths */
  for (i=0; i<nsends2; i++) {
    ierr = MPI_Send(&nprocs[i],1,MPIU_INT,source[i],tag2,comm);CHKERRQ(ierr);
  }

  /* wait on receives of lens */
  if (nrecvs2) {
    ierr = PetscMalloc(nrecvs2*sizeof(MPI_Status),&recv_statuses);CHKERRQ(ierr);
    ierr = MPI_Waitall(nrecvs2,recv_waits,recv_statuses);CHKERRQ(ierr);
    ierr = PetscFree(recv_statuses);CHKERRQ(ierr);
  }
  ierr = PetscFree(recv_waits);

  starts3[0] = 0;
  nt         = 0;
  for (i=0; i<nrecvs2-1; i++) {
    starts3[i+1] = starts3[i] + lens2[i];
    nt          += lens2[i];
  }
  nt += lens2[nrecvs2-1];

  ierr = PetscMalloc((nt+1)*sizeof(PetscInt),&recvs2);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs2+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs2; i++) {
    ierr = MPI_Irecv(recvs2+starts3[i],lens2[i],MPIU_INT,dest[i],tag3,comm,recv_waits+i);CHKERRQ(ierr);
  }
  
  /* send the messages */
  ierr = PetscMalloc((nsends2+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  for (i=0; i<nsends2; i++) {
    ierr = MPI_Isend(sends2+starts2[i],nprocs[i],MPIU_INT,source[i],tag3,comm,send_waits+i);CHKERRQ(ierr);
  }

  /* wait on receives */
  if (nrecvs2) {
    ierr = PetscMalloc(nrecvs2*sizeof(MPI_Status),&recv_statuses);CHKERRQ(ierr);
    ierr = MPI_Waitall(nrecvs2,recv_waits,recv_statuses);CHKERRQ(ierr);
    ierr = PetscFree(recv_statuses);CHKERRQ(ierr);
  }
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
  ierr = PetscMalloc(size*sizeof(PetscInt),&nprocs);CHKERRQ(ierr);
  ierr = PetscMemzero(nprocs,size*sizeof(PetscInt));CHKERRQ(ierr);
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
  ierr = PetscMalloc((nt+1)*sizeof(PetscInt),procs);CHKERRQ(ierr);
  ierr = PetscMalloc((nt+1)*sizeof(PetscInt),numprocs);CHKERRQ(ierr);
  ierr = PetscMalloc((nt+1)*sizeof(PetscInt*),indices);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscInt),&bprocs);CHKERRQ(ierr);
  cnt       = 0;
  for (i=0; i<size; i++) {
    if (nprocs[i] > 0) {
      bprocs[i]        = cnt;
      (*procs)[cnt]    = i;
      (*numprocs)[cnt] = nprocs[i];
      ierr             = PetscMalloc(nprocs[i]*sizeof(PetscInt),&(*indices)[cnt]);CHKERRQ(ierr);
      cnt++;
    }
  }

  /* make the list of subdomains for each nontrivial local node */
  ierr = PetscMemzero(*numprocs,nt*sizeof(PetscInt));CHKERRQ(ierr);
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
    ierr = PetscMalloc(((*numprocs)[i])*sizeof(PetscInt),&tmp);CHKERRQ(ierr);
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
  ierr = PetscFree(len);CHKERRQ(ierr);
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
PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingRestoreInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscFree(*procs);CHKERRQ(ierr);
  ierr = PetscFree(*numprocs);CHKERRQ(ierr);
  if (*indices) {
    ierr = PetscFree((*indices)[0]);CHKERRQ(ierr);
    for (i=1; i<*nproc; i++) {
      ierr = PetscFree((*indices)[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(*indices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
