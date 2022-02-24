/* Mainly for MPI_Isend in SFBASIC. Once SFNEIGHBOR, SFALLGHATERV etc have a persistent version,
   we can also do abstractions like Prepare/StartCommunication.
*/

#include <../src/vec/is/sf/impls/basic/sfpack.h>

/* Start MPI requests. If use non-GPU aware MPI, we might need to copy data from device buf to host buf */
static PetscErrorCode PetscSFLinkStartRequests_MPI(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscMPIInt       nreqs;
  MPI_Request       *reqs = NULL;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscInt          buflen;

  PetscFunctionBegin;
  buflen = (direction == PETSCSF_ROOT2LEAF) ? sf->leafbuflen[PETSCSF_REMOTE] : bas->rootbuflen[PETSCSF_REMOTE];
  if (buflen) {
    if (direction == PETSCSF_ROOT2LEAF) {
      nreqs = sf->nleafreqs;
      CHKERRQ(PetscSFLinkGetMPIBuffersAndRequests(sf,link,direction,NULL,NULL,NULL,&reqs));
    } else { /* leaf to root */
      nreqs = bas->nrootreqs;
      CHKERRQ(PetscSFLinkGetMPIBuffersAndRequests(sf,link,direction,NULL,NULL,&reqs,NULL));
    }
    CHKERRMPI(MPI_Startall_irecv(buflen,link->unit,nreqs,reqs));
  }

  buflen = (direction == PETSCSF_ROOT2LEAF) ? bas->rootbuflen[PETSCSF_REMOTE] : sf->leafbuflen[PETSCSF_REMOTE];
  if (buflen) {
    if (direction == PETSCSF_ROOT2LEAF) {
      nreqs  = bas->nrootreqs;
      CHKERRQ(PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/*device2host before sending */));
      CHKERRQ(PetscSFLinkGetMPIBuffersAndRequests(sf,link,direction,NULL,NULL,&reqs,NULL));
    } else { /* leaf to root */
      nreqs  = sf->nleafreqs;
      CHKERRQ(PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE));
      CHKERRQ(PetscSFLinkGetMPIBuffersAndRequests(sf,link,direction,NULL,NULL,NULL,&reqs));
    }
    CHKERRQ(PetscSFLinkSyncStreamBeforeCallMPI(sf,link,direction));
    CHKERRMPI(MPI_Startall_isend(buflen,link->unit,nreqs,reqs));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFLinkWaitRequests_MPI(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic        *bas = (PetscSF_Basic*)sf->data;
  const PetscMemType   rootmtype_mpi = link->rootmtype_mpi,leafmtype_mpi = link->leafmtype_mpi;
  const PetscInt       rootdirect_mpi = link->rootdirect_mpi,leafdirect_mpi = link->leafdirect_mpi;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Waitall(bas->nrootreqs,link->rootreqs[direction][rootmtype_mpi][rootdirect_mpi],MPI_STATUSES_IGNORE));
  CHKERRMPI(MPI_Waitall(sf->nleafreqs, link->leafreqs[direction][leafmtype_mpi][leafdirect_mpi],MPI_STATUSES_IGNORE));
  if (direction == PETSCSF_ROOT2LEAF) {
    CHKERRQ(PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_FALSE/* host2device after recving */));
  } else {
    CHKERRQ(PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_FALSE));
  }
  PetscFunctionReturn(0);
}

/*
   The routine Creates a communication link for the given operation. It first looks up its link cache. If
   there is a free & suitable one, it uses it. Otherwise it creates a new one.

   A link contains buffers and MPI requests for send/recv. It also contains pack/unpack routines to pack/unpack
   root/leafdata to/from these buffers. Buffers are allocated at our discretion. When we find root/leafata
   can be directly passed to MPI, we won't allocate them. Even we allocate buffers, we only allocate
   those that are needed by the given `sfop` and `op`, in other words, we do lazy memory-allocation.

   The routine also allocates buffers on CPU when one does not use gpu-aware MPI but data is on GPU.

   In SFBasic, MPI requests are persistent. They are init'ed until we try to get requests from a link.

   The routine is shared by SFBasic and SFNeighbor based on the fact they all deal with sparse graphs and
   need pack/unpack data.
*/
PetscErrorCode PetscSFLinkCreate_MPI(PetscSF sf,MPI_Datatype unit,PetscMemType xrootmtype,const void *rootdata,PetscMemType xleafmtype,const void *leafdata,MPI_Op op,PetscSFOperation sfop,PetscSFLink *mylink)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscInt          i,j,k,nrootreqs,nleafreqs,nreqs;
  PetscSFLink       *p,link;
  PetscSFDirection  direction;
  MPI_Request       *reqs = NULL;
  PetscBool         match,rootdirect[2],leafdirect[2];
  PetscMemType      rootmtype = PetscMemTypeHost(xrootmtype) ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE; /* Convert to 0/1 as we will use it in subscript */
  PetscMemType      leafmtype = PetscMemTypeHost(xleafmtype) ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE;
  PetscMemType      rootmtype_mpi,leafmtype_mpi;   /* mtypes seen by MPI */
  PetscInt          rootdirect_mpi,leafdirect_mpi; /* root/leafdirect seen by MPI*/

  PetscFunctionBegin;

  /* Can we directly use root/leafdirect with the given sf, sfop and op? */
  for (i=PETSCSF_LOCAL; i<=PETSCSF_REMOTE; i++) {
    if (sfop == PETSCSF_BCAST) {
      rootdirect[i] = bas->rootcontig[i]; /* Pack roots */
      leafdirect[i] = (sf->leafcontig[i] && op == MPI_REPLACE) ? PETSC_TRUE : PETSC_FALSE;  /* Unpack leaves */
    } else if (sfop == PETSCSF_REDUCE) {
      leafdirect[i] = sf->leafcontig[i];  /* Pack leaves */
      rootdirect[i] = (bas->rootcontig[i] && op == MPI_REPLACE) ? PETSC_TRUE : PETSC_FALSE; /* Unpack roots */
    } else { /* PETSCSF_FETCH */
      rootdirect[i] = PETSC_FALSE; /* FETCH always need a separate rootbuf */
      leafdirect[i] = PETSC_FALSE; /* We also force allocating a separate leafbuf so that leafdata and leafupdate can share mpi requests */
    }
  }

  if (sf->use_gpu_aware_mpi) {
    rootmtype_mpi = rootmtype;
    leafmtype_mpi = leafmtype;
  } else {
    rootmtype_mpi = leafmtype_mpi = PETSC_MEMTYPE_HOST;
  }
  /* Will root/leafdata be directly accessed by MPI?  Without use_gpu_aware_mpi, device data is bufferred on host and then passed to MPI */
  rootdirect_mpi = rootdirect[PETSCSF_REMOTE] && (rootmtype_mpi == rootmtype)? 1 : 0;
  leafdirect_mpi = leafdirect[PETSCSF_REMOTE] && (leafmtype_mpi == leafmtype)? 1 : 0;

  direction = (sfop == PETSCSF_BCAST)? PETSCSF_ROOT2LEAF : PETSCSF_LEAF2ROOT;
  nrootreqs = bas->nrootreqs;
  nleafreqs = sf->nleafreqs;

  /* Look for free links in cache */
  for (p=&bas->avail; (link=*p); p=&link->next) {
    if (!link->use_nvshmem) { /* Only check with MPI links */
      CHKERRQ(MPIPetsc_Type_compare(unit,link->unit,&match));
      if (match) {
        /* If root/leafdata will be directly passed to MPI, test if the data used to initialized the MPI requests matches with the current.
           If not, free old requests. New requests will be lazily init'ed until one calls PetscSFLinkGetMPIBuffersAndRequests().
        */
        if (rootdirect_mpi && sf->persistent && link->rootreqsinited[direction][rootmtype][1] && link->rootdatadirect[direction][rootmtype] != rootdata) {
          reqs = link->rootreqs[direction][rootmtype][1]; /* Here, rootmtype = rootmtype_mpi */
          for (i=0; i<nrootreqs; i++) {if (reqs[i] != MPI_REQUEST_NULL) CHKERRMPI(MPI_Request_free(&reqs[i]));}
          link->rootreqsinited[direction][rootmtype][1] = PETSC_FALSE;
        }
        if (leafdirect_mpi && sf->persistent && link->leafreqsinited[direction][leafmtype][1] && link->leafdatadirect[direction][leafmtype] != leafdata) {
          reqs = link->leafreqs[direction][leafmtype][1];
          for (i=0; i<nleafreqs; i++) {if (reqs[i] != MPI_REQUEST_NULL) CHKERRMPI(MPI_Request_free(&reqs[i]));}
          link->leafreqsinited[direction][leafmtype][1] = PETSC_FALSE;
        }
        *p = link->next; /* Remove from available list */
        goto found;
      }
    }
  }

  CHKERRQ(PetscNew(&link));
  CHKERRQ(PetscSFLinkSetUp_Host(sf,link,unit));
  CHKERRQ(PetscCommGetNewTag(PetscObjectComm((PetscObject)sf),&link->tag)); /* One tag per link */

  nreqs = (nrootreqs+nleafreqs)*8;
  CHKERRQ(PetscMalloc1(nreqs,&link->reqs));
  for (i=0; i<nreqs; i++) link->reqs[i] = MPI_REQUEST_NULL; /* Initialized to NULL so that we know which need to be freed in Destroy */

  for (i=0; i<2; i++) { /* Two communication directions */
    for (j=0; j<2; j++) { /* Two memory types */
      for (k=0; k<2; k++) { /* root/leafdirect 0 or 1 */
        link->rootreqs[i][j][k] = link->reqs + nrootreqs*(4*i+2*j+k);
        link->leafreqs[i][j][k] = link->reqs + nrootreqs*8 + nleafreqs*(4*i+2*j+k);
      }
    }
  }
  link->StartCommunication    = PetscSFLinkStartRequests_MPI;
  link->FinishCommunication   = PetscSFLinkWaitRequests_MPI;

found:

#if defined(PETSC_HAVE_DEVICE)
  if ((PetscMemTypeDevice(xrootmtype) || PetscMemTypeDevice(xleafmtype)) && !link->deviceinited) {
    #if defined(PETSC_HAVE_CUDA)
      if (sf->backend == PETSCSF_BACKEND_CUDA)   CHKERRQ(PetscSFLinkSetUp_CUDA(sf,link,unit)); /* Setup streams etc */
    #endif
    #if defined(PETSC_HAVE_HIP)
      if (sf->backend == PETSCSF_BACKEND_HIP)    CHKERRQ(PetscSFLinkSetUp_HIP(sf,link,unit)); /* Setup streams etc */
    #endif
    #if defined(PETSC_HAVE_KOKKOS)
      if (sf->backend == PETSCSF_BACKEND_KOKKOS) CHKERRQ(PetscSFLinkSetUp_Kokkos(sf,link,unit));
    #endif
  }
#endif

  /* Allocate buffers along root/leafdata */
  for (i=PETSCSF_LOCAL; i<=PETSCSF_REMOTE; i++) {
    /* For local communication, buffers are only needed when roots and leaves have different mtypes */
    if (i == PETSCSF_LOCAL && rootmtype == leafmtype) continue;
    if (bas->rootbuflen[i]) {
      if (rootdirect[i]) { /* Aha, we disguise rootdata as rootbuf */
        link->rootbuf[i][rootmtype] = (char*)rootdata + bas->rootstart[i]*link->unitbytes;
      } else { /* Have to have a separate rootbuf */
        if (!link->rootbuf_alloc[i][rootmtype]) {
          CHKERRQ(PetscSFMalloc(sf,rootmtype,bas->rootbuflen[i]*link->unitbytes,(void**)&link->rootbuf_alloc[i][rootmtype]));
        }
        link->rootbuf[i][rootmtype] = link->rootbuf_alloc[i][rootmtype];
      }
    }

    if (sf->leafbuflen[i]) {
      if (leafdirect[i]) {
        link->leafbuf[i][leafmtype] = (char*)leafdata + sf->leafstart[i]*link->unitbytes;
      } else {
        if (!link->leafbuf_alloc[i][leafmtype]) {
          CHKERRQ(PetscSFMalloc(sf,leafmtype,sf->leafbuflen[i]*link->unitbytes,(void**)&link->leafbuf_alloc[i][leafmtype]));
        }
        link->leafbuf[i][leafmtype] = link->leafbuf_alloc[i][leafmtype];
      }
    }
  }

#if defined(PETSC_HAVE_DEVICE)
  /* Allocate buffers on host for buffering data on device in cast not use_gpu_aware_mpi */
  if (PetscMemTypeDevice(rootmtype) && PetscMemTypeHost(rootmtype_mpi)) {
    if (!link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST]) {
      CHKERRQ(PetscMalloc(bas->rootbuflen[PETSCSF_REMOTE]*link->unitbytes,&link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST]));
    }
    link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST] = link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST];
  }
  if (PetscMemTypeDevice(leafmtype) && PetscMemTypeHost(leafmtype_mpi)) {
    if (!link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST]) {
      CHKERRQ(PetscMalloc(sf->leafbuflen[PETSCSF_REMOTE]*link->unitbytes,&link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST]));
    }
    link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST] = link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST];
  }
#endif

  /* Set `current` state of the link. They may change between different SF invocations with the same link */
  if (sf->persistent) { /* If data is directly passed to MPI and inits MPI requests, record the data for comparison on future invocations */
    if (rootdirect_mpi) link->rootdatadirect[direction][rootmtype] = rootdata;
    if (leafdirect_mpi) link->leafdatadirect[direction][leafmtype] = leafdata;
  }

  link->rootdata = rootdata; /* root/leafdata are keys to look up links in PetscSFXxxEnd */
  link->leafdata = leafdata;
  for (i=PETSCSF_LOCAL; i<=PETSCSF_REMOTE; i++) {
    link->rootdirect[i] = rootdirect[i];
    link->leafdirect[i] = leafdirect[i];
  }
  link->rootdirect_mpi  = rootdirect_mpi;
  link->leafdirect_mpi  = leafdirect_mpi;
  link->rootmtype       = rootmtype;
  link->leafmtype       = leafmtype;
  link->rootmtype_mpi   = rootmtype_mpi;
  link->leafmtype_mpi   = leafmtype_mpi;

  link->next            = bas->inuse;
  bas->inuse            = link;
  *mylink               = link;
  PetscFunctionReturn(0);
}
