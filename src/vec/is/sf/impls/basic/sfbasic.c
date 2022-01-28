#include <../src/vec/is/sf/impls/basic/sfbasic.h>
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <petsc/private/viewerimpl.h>

/*===================================================================================*/
/*              SF public interface implementations                                  */
/*===================================================================================*/
PETSC_INTERN PetscErrorCode PetscSFSetUp_Basic(PetscSF sf)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscInt       *rlengths,*ilengths,i,nRemoteRootRanks,nRemoteLeafRanks;
  PetscMPIInt    rank,niranks,*iranks,tag;
  MPI_Comm       comm;
  MPI_Group      group;
  MPI_Request    *rootreqs,*leafreqs;

  PetscFunctionBegin;
  ierr = MPI_Comm_group(PETSC_COMM_SELF,&group);CHKERRMPI(ierr);
  ierr = PetscSFSetUpRanks(sf,group);CHKERRQ(ierr);
  ierr = MPI_Group_free(&group);CHKERRMPI(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)sf,&tag);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  /*
   * Inform roots about how many leaves and from which ranks
   */
  ierr = PetscMalloc1(sf->nranks,&rlengths);CHKERRQ(ierr);
  /* Determine number, sending ranks and length of incoming */
  for (i=0; i<sf->nranks; i++) {
    rlengths[i] = sf->roffset[i+1] - sf->roffset[i]; /* Number of roots referenced by my leaves; for rank sf->ranks[i] */
  }
  nRemoteRootRanks = sf->nranks-sf->ndranks;
  ierr = PetscCommBuildTwoSided(comm,1,MPIU_INT,nRemoteRootRanks,sf->ranks+sf->ndranks,rlengths+sf->ndranks,&niranks,&iranks,(void**)&ilengths);CHKERRQ(ierr);

  /* Sort iranks. See use of VecScatterGetRemoteOrdered_Private() in MatGetBrowsOfAoCols_MPIAIJ() on why.
     We could sort ranks there at the price of allocating extra working arrays. Presumably, niranks is
     small and the sorting is cheap.
   */
  ierr = PetscSortMPIIntWithIntArray(niranks,iranks,ilengths);CHKERRQ(ierr);

  /* Partition into distinguished and non-distinguished incoming ranks */
  bas->ndiranks = sf->ndranks;
  bas->niranks = bas->ndiranks + niranks;
  ierr = PetscMalloc2(bas->niranks,&bas->iranks,bas->niranks+1,&bas->ioffset);CHKERRQ(ierr);
  bas->ioffset[0] = 0;
  for (i=0; i<bas->ndiranks; i++) {
    bas->iranks[i] = sf->ranks[i];
    bas->ioffset[i+1] = bas->ioffset[i] + rlengths[i];
  }
  PetscAssertFalse(bas->ndiranks > 1 || (bas->ndiranks == 1 && bas->iranks[0] != rank),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Broken setup for shared ranks");
  for (; i<bas->niranks; i++) {
    bas->iranks[i] = iranks[i-bas->ndiranks];
    bas->ioffset[i+1] = bas->ioffset[i] + ilengths[i-bas->ndiranks];
  }
  bas->itotal = bas->ioffset[i];
  ierr = PetscFree(rlengths);CHKERRQ(ierr);
  ierr = PetscFree(iranks);CHKERRQ(ierr);
  ierr = PetscFree(ilengths);CHKERRQ(ierr);

  /* Send leaf identities to roots */
  nRemoteLeafRanks = bas->niranks-bas->ndiranks;
  ierr = PetscMalloc1(bas->itotal,&bas->irootloc);CHKERRQ(ierr);
  ierr = PetscMalloc2(nRemoteLeafRanks,&rootreqs,nRemoteRootRanks,&leafreqs);CHKERRQ(ierr);
  for (i=bas->ndiranks; i<bas->niranks; i++) {
    ierr = MPIU_Irecv(bas->irootloc+bas->ioffset[i],bas->ioffset[i+1]-bas->ioffset[i],MPIU_INT,bas->iranks[i],tag,comm,&rootreqs[i-bas->ndiranks]);CHKERRMPI(ierr);
  }
  for (i=0; i<sf->nranks; i++) {
    PetscInt npoints = sf->roffset[i+1] - sf->roffset[i];
    if (i < sf->ndranks) {
      PetscAssertFalse(sf->ranks[i] != rank,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot interpret distinguished leaf rank");
      PetscAssertFalse(bas->iranks[0] != rank,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot interpret distinguished root rank");
      PetscAssertFalse(npoints != bas->ioffset[1]-bas->ioffset[0],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Distinguished rank exchange has mismatched lengths");
      ierr = PetscArraycpy(bas->irootloc+bas->ioffset[0],sf->rremote+sf->roffset[i],npoints);CHKERRQ(ierr);
      continue;
    }
    ierr = MPIU_Isend(sf->rremote+sf->roffset[i],npoints,MPIU_INT,sf->ranks[i],tag,comm,&leafreqs[i-sf->ndranks]);CHKERRMPI(ierr);
  }
  ierr = MPI_Waitall(nRemoteLeafRanks,rootreqs,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);
  ierr = MPI_Waitall(nRemoteRootRanks,leafreqs,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);

  sf->nleafreqs  = nRemoteRootRanks;
  bas->nrootreqs = nRemoteLeafRanks;
  sf->persistent = PETSC_TRUE;

  /* Setup fields related to packing, such as rootbuflen[] */
  ierr = PetscSFSetUpPackFields(sf);CHKERRQ(ierr);
  ierr = PetscFree2(rootreqs,leafreqs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFReset_Basic(PetscSF sf)
{
  PetscErrorCode    ierr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscSFLink       link = bas->avail,next;

  PetscFunctionBegin;
  PetscAssertFalse(bas->inuse,PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"Outstanding operation has not been completed");
  ierr = PetscFree2(bas->iranks,bas->ioffset);CHKERRQ(ierr);
  ierr = PetscFree(bas->irootloc);CHKERRQ(ierr);

 #if defined(PETSC_HAVE_DEVICE)
  for (PetscInt i=0; i<2; i++) {ierr = PetscSFFree(sf,PETSC_MEMTYPE_DEVICE,bas->irootloc_d[i]);CHKERRQ(ierr);}
 #endif

 #if defined(PETSC_HAVE_NVSHMEM)
  ierr = PetscSFReset_Basic_NVSHMEM(sf);CHKERRQ(ierr);
 #endif

  for (; link; link=next) {next = link->next; ierr = PetscSFLinkDestroy(sf,link);CHKERRQ(ierr);}
  bas->avail = NULL;
  ierr = PetscSFResetPackFields(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFDestroy_Basic(PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFReset_Basic(sf);CHKERRQ(ierr);
  ierr = PetscFree(sf->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_SINGLE_LIBRARY)
#include <petscmat.h>

PETSC_INTERN PetscErrorCode PetscSFView_Basic_PatternAndSizes(PetscSF sf,PetscViewer viewer)
{
  PetscErrorCode       ierr;
  PetscSF_Basic        *bas = (PetscSF_Basic*)sf->data;
  PetscInt             i,nrootranks,ndrootranks;
  const PetscInt       *rootoffset;
  PetscMPIInt          rank,size;
  const PetscMPIInt    *rootranks;
  MPI_Comm             comm = PetscObjectComm((PetscObject)sf);
  PetscScalar          unitbytes;
  Mat                  A;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  /* PetscSFView is most useful for the SF used in VecScatterBegin/End in MatMult etc, where we do
    PetscSFBcast, i.e., roots send data to leaves.  We dump the communication pattern into a matrix
    in senders' view point: how many bytes I will send to my neighbors.

    Looking at a column of the matrix, one can also know how many bytes the rank will receive from others.

    If PetscSFLink bas->inuse is available, we can use that to get tree vertex size. But that would give
    different interpretations for the same SF for different data types. Since we most care about VecScatter,
    we uniformly treat each vertex as a PetscScalar.
  */
  unitbytes = (PetscScalar)sizeof(PetscScalar);

  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,&rootranks,&rootoffset,NULL);CHKERRQ(ierr);
  ierr = MatCreateAIJ(comm,1,1,size,size,1,NULL,nrootranks-ndrootranks,NULL,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"__petsc_internal__");CHKERRQ(ierr); /* To prevent the internal A from taking any command line options */
  for (i=0; i<nrootranks; i++) {
    ierr = MatSetValue(A,(PetscInt)rank,bas->iranks[i],(rootoffset[i+1]-rootoffset[i])*unitbytes,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PETSC_INTERN PetscErrorCode PetscSFView_Basic(PetscSF sf,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii && viewer->format != PETSC_VIEWER_ASCII_MATLAB) {ierr = PetscViewerASCIIPrintf(viewer,"  MultiSF sort=%s\n",sf->rankorder ? "rank-order" : "unordered");CHKERRQ(ierr);}
#if defined(PETSC_USE_SINGLE_LIBRARY)
  else {
    PetscBool  isdraw,isbinary;
    ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
    if ((isascii && viewer->format == PETSC_VIEWER_ASCII_MATLAB) || isdraw || isbinary) {
      ierr = PetscSFView_Basic_PatternAndSizes(sf,viewer);CHKERRQ(ierr);
    }
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastBegin_Basic(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFLink       link = NULL;

  PetscFunctionBegin;
  /* Create a communication link, which provides buffers, MPI requests etc (if MPI is used) */
  ierr = PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_BCAST,&link);CHKERRQ(ierr);
  /* Pack rootdata to rootbuf for remote communication */
  ierr = PetscSFLinkPackRootData(sf,link,PETSCSF_REMOTE,rootdata);CHKERRQ(ierr);
  /* Start communcation, e.g., post MPI_Isend */
  ierr = PetscSFLinkStartCommunication(sf,link,PETSCSF_ROOT2LEAF);CHKERRQ(ierr);
  /* Do local scatter (i.e., self to self communication), which overlaps with the remote communication above */
  ierr = PetscSFLinkScatterLocal(sf,link,PETSCSF_ROOT2LEAF,(void*)rootdata,leafdata,op);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFBcastEnd_Basic(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFLink       link = NULL;

  PetscFunctionBegin;
  /* Retrieve the link used in XxxBegin() with root/leafdata as key */
  ierr = PetscSFLinkGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  /* Finish remote communication, e.g., post MPI_Waitall */
  ierr = PetscSFLinkFinishCommunication(sf,link,PETSCSF_ROOT2LEAF);CHKERRQ(ierr);
  /* Unpack data in leafbuf to leafdata for remote communication */
  ierr = PetscSFLinkUnpackLeafData(sf,link,PETSCSF_REMOTE,leafdata,op);CHKERRQ(ierr);
  /* Recycle the link */
  ierr = PetscSFLinkReclaim(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Shared by ReduceBegin and FetchAndOpBegin */
PETSC_STATIC_INLINE PetscErrorCode PetscSFLeafToRootBegin_Basic(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op,PetscSFOperation sfop,PetscSFLink *out)
{
  PetscErrorCode    ierr;
  PetscSFLink       link = NULL;

  PetscFunctionBegin;
  ierr = PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,sfop,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkPackLeafData(sf,link,PETSCSF_REMOTE,leafdata);CHKERRQ(ierr);
  ierr = PetscSFLinkStartCommunication(sf,link,PETSCSF_LEAF2ROOT);CHKERRQ(ierr);
  *out = link;
  PetscFunctionReturn(0);
}

/* leaf -> root with reduction */
static PetscErrorCode PetscSFReduceBegin_Basic(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFLink       link = NULL;

  PetscFunctionBegin;
  ierr = PetscSFLeafToRootBegin_Basic(sf,unit,leafmtype,leafdata,rootmtype,rootdata,op,PETSCSF_REDUCE,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkScatterLocal(sf,link,PETSCSF_LEAF2ROOT,rootdata,(void*)leafdata,op);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFReduceEnd_Basic(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFLink       link = NULL;

  PetscFunctionBegin;
  ierr = PetscSFLinkGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkFinishCommunication(sf,link,PETSCSF_LEAF2ROOT);CHKERRQ(ierr);
  ierr = PetscSFLinkUnpackRootData(sf,link,PETSCSF_REMOTE,rootdata,op);CHKERRQ(ierr);
  ierr = PetscSFLinkReclaim(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Basic(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,void *rootdata,PetscMemType leafmtype,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFLink       link = NULL;

  PetscFunctionBegin;
  ierr = PetscSFLeafToRootBegin_Basic(sf,unit,leafmtype,leafdata,rootmtype,rootdata,op,PETSCSF_FETCH,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkFetchAndOpLocal(sf,link,rootdata,leafdata,leafupdate,op);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpEnd_Basic(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFLink       link = NULL;

  PetscFunctionBegin;
  ierr = PetscSFLinkGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  /* This implementation could be changed to unpack as receives arrive, at the cost of non-determinism */
  ierr = PetscSFLinkFinishCommunication(sf,link,PETSCSF_LEAF2ROOT);CHKERRQ(ierr);
  /* Do fetch-and-op, the (remote) update results are in rootbuf */
  ierr = PetscSFLinkFetchAndOpRemote(sf,link,rootdata,op);CHKERRQ(ierr);
  /* Bcast rootbuf to leafupdate */
  ierr = PetscSFLinkStartCommunication(sf,link,PETSCSF_ROOT2LEAF);CHKERRQ(ierr);
  ierr = PetscSFLinkFinishCommunication(sf,link,PETSCSF_ROOT2LEAF);CHKERRQ(ierr);
  /* Unpack and insert fetched data into leaves */
  ierr = PetscSFLinkUnpackLeafData(sf,link,PETSCSF_REMOTE,leafupdate,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFLinkReclaim(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFGetLeafRanks_Basic(PetscSF sf,PetscInt *niranks,const PetscMPIInt **iranks,const PetscInt **ioffset,const PetscInt **irootloc)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  if (niranks)  *niranks  = bas->niranks;
  if (iranks)   *iranks   = bas->iranks;
  if (ioffset)  *ioffset  = bas->ioffset;
  if (irootloc) *irootloc = bas->irootloc;
  PetscFunctionReturn(0);
}

/* An optimized PetscSFCreateEmbeddedRootSF. We aggresively make use of the established communication on sf.
   We need one bcast on sf, and no communication anymore to build the embedded sf. Note that selected[]
   was sorted before calling the routine.
 */
PETSC_INTERN PetscErrorCode PetscSFCreateEmbeddedRootSF_Basic(PetscSF sf,PetscInt nselected,const PetscInt *selected,PetscSF *newsf)
{
  PetscSF           esf;
  PetscInt          esf_nranks,esf_ndranks,*esf_roffset,*esf_rmine,*esf_rremote;
  PetscInt          i,j,p,q,nroots,esf_nleaves,*new_ilocal,nranks,ndranks,niranks,ndiranks,minleaf,maxleaf,maxlocal;
  char              *rootdata,*leafdata,*leafmem; /* Only stores 0 or 1, so we can save memory with char */
  PetscMPIInt       *esf_ranks;
  const PetscMPIInt *ranks,*iranks;
  const PetscInt    *roffset,*rmine,*rremote,*ioffset,*irootloc;
  PetscBool         connected;
  PetscSFNode       *new_iremote;
  PetscSF_Basic     *bas;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)sf),&esf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(esf);CHKERRQ(ierr);
  ierr = PetscSFSetType(esf,PETSCSFBASIC);CHKERRQ(ierr); /* This optimized routine can only create a basic sf */

  /* Find out which leaves are still connected to roots in the embedded sf by doing a Bcast */
  ierr = PetscSFGetGraph(sf,&nroots,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafRange(sf,&minleaf,&maxleaf);CHKERRQ(ierr);
  maxlocal = maxleaf - minleaf + 1;
  ierr = PetscCalloc2(nroots,&rootdata,maxlocal,&leafmem);CHKERRQ(ierr);
  leafdata = leafmem - minleaf;
  /* Tag selected roots */
  for (i=0; i<nselected; ++i) rootdata[selected[i]] = 1;

  ierr = PetscSFBcastBegin(sf,MPI_CHAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPI_CHAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nranks,&ndranks,&ranks,&roffset,&rmine,&rremote);CHKERRQ(ierr); /* Get send info */
  esf_nranks = esf_ndranks = esf_nleaves = 0;
  for (i=0; i<nranks; i++) {
    connected = PETSC_FALSE; /* Is this process still connected to this remote root rank? */
    for (j=roffset[i]; j<roffset[i+1]; j++) {if (leafdata[rmine[j]]) {esf_nleaves++; connected = PETSC_TRUE;}}
    if (connected) {esf_nranks++; if (i < ndranks) esf_ndranks++;}
  }

  /* Set graph of esf and also set up its outgoing communication (i.e., send info), which is usually done by PetscSFSetUpRanks */
  ierr = PetscMalloc1(esf_nleaves,&new_ilocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(esf_nleaves,&new_iremote);CHKERRQ(ierr);
  ierr = PetscMalloc4(esf_nranks,&esf_ranks,esf_nranks+1,&esf_roffset,esf_nleaves,&esf_rmine,esf_nleaves,&esf_rremote);CHKERRQ(ierr);
  p    = 0; /* Counter for connected root ranks */
  q    = 0; /* Counter for connected leaves */
  esf_roffset[0] = 0;
  for (i=0; i<nranks; i++) { /* Scan leaf data again to fill esf arrays */
    connected = PETSC_FALSE;
    for (j=roffset[i]; j<roffset[i+1]; j++) {
      if (leafdata[rmine[j]]) {
        esf_rmine[q]         = new_ilocal[q] = rmine[j];
        esf_rremote[q]       = rremote[j];
        new_iremote[q].index = rremote[j];
        new_iremote[q].rank  = ranks[i];
        connected            = PETSC_TRUE;
        q++;
      }
    }
    if (connected) {
      esf_ranks[p]     = ranks[i];
      esf_roffset[p+1] = q;
      p++;
    }
  }

  /* SetGraph internally resets the SF, so we only set its fields after the call */
  ierr           = PetscSFSetGraph(esf,nroots,esf_nleaves,new_ilocal,PETSC_OWN_POINTER,new_iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  esf->nranks    = esf_nranks;
  esf->ndranks   = esf_ndranks;
  esf->ranks     = esf_ranks;
  esf->roffset   = esf_roffset;
  esf->rmine     = esf_rmine;
  esf->rremote   = esf_rremote;
  esf->nleafreqs = esf_nranks - esf_ndranks;

  /* Set up the incoming communication (i.e., recv info) stored in esf->data, which is usually done by PetscSFSetUp_Basic */
  bas  = (PetscSF_Basic*)esf->data;
  ierr = PetscSFGetRootInfo_Basic(sf,&niranks,&ndiranks,&iranks,&ioffset,&irootloc);CHKERRQ(ierr); /* Get recv info */
  /* Embedded sf always has simpler communication than the original one. We might allocate longer arrays than needed here. But we
     we do not care since these arrays are usually short. The benefit is we can fill these arrays by just parsing irootloc once.
   */
  ierr = PetscMalloc2(niranks,&bas->iranks,niranks+1,&bas->ioffset);CHKERRQ(ierr);
  ierr = PetscMalloc1(ioffset[niranks],&bas->irootloc);CHKERRQ(ierr);
  bas->niranks = bas->ndiranks = bas->ioffset[0] = 0;
  p = 0; /* Counter for connected leaf ranks */
  q = 0; /* Counter for connected roots */
  for (i=0; i<niranks; i++) {
    connected = PETSC_FALSE; /* Is the current process still connected to this remote leaf rank? */
    for (j=ioffset[i]; j<ioffset[i+1]; j++) {
      if (rootdata[irootloc[j]]) {
        bas->irootloc[q++] = irootloc[j];
        connected = PETSC_TRUE;
      }
    }
    if (connected) {
      bas->niranks++;
      if (i<ndiranks) bas->ndiranks++; /* Note that order of ranks (including distinguished ranks) is kept */
      bas->iranks[p]    = iranks[i];
      bas->ioffset[p+1] = q;
      p++;
    }
  }
  bas->itotal     = q;
  bas->nrootreqs  = bas->niranks - bas->ndiranks;
  esf->persistent = PETSC_TRUE;
  /* Setup packing related fields */
  ierr = PetscSFSetUpPackFields(esf);CHKERRQ(ierr);

  /* Copy from PetscSFSetUp(), since this method wants to skip PetscSFSetUp(). */
#if defined(PETSC_HAVE_CUDA)
  if (esf->backend == PETSCSF_BACKEND_CUDA) {
    esf->ops->Malloc = PetscSFMalloc_CUDA;
    esf->ops->Free   = PetscSFFree_CUDA;
  }
#endif

#if defined(PETSC_HAVE_HIP)
  /* TODO: Needs debugging */
  if (esf->backend == PETSCSF_BACKEND_HIP) {
    esf->ops->Malloc = PetscSFMalloc_HIP;
    esf->ops->Free   = PetscSFFree_HIP;
  }
#endif

#if defined(PETSC_HAVE_KOKKOS)
  if (esf->backend == PETSCSF_BACKEND_KOKKOS) {
    esf->ops->Malloc = PetscSFMalloc_Kokkos;
    esf->ops->Free   = PetscSFFree_Kokkos;
  }
#endif
  esf->setupcalled = PETSC_TRUE; /* We have done setup ourselves! */
  ierr = PetscFree2(rootdata,leafmem);CHKERRQ(ierr);
  *newsf = esf;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscSFCreate_Basic(PetscSF sf)
{
  PetscSF_Basic  *dat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sf->ops->SetUp                = PetscSFSetUp_Basic;
  sf->ops->Reset                = PetscSFReset_Basic;
  sf->ops->Destroy              = PetscSFDestroy_Basic;
  sf->ops->View                 = PetscSFView_Basic;
  sf->ops->BcastBegin           = PetscSFBcastBegin_Basic;
  sf->ops->BcastEnd             = PetscSFBcastEnd_Basic;
  sf->ops->ReduceBegin          = PetscSFReduceBegin_Basic;
  sf->ops->ReduceEnd            = PetscSFReduceEnd_Basic;
  sf->ops->FetchAndOpBegin      = PetscSFFetchAndOpBegin_Basic;
  sf->ops->FetchAndOpEnd        = PetscSFFetchAndOpEnd_Basic;
  sf->ops->GetLeafRanks         = PetscSFGetLeafRanks_Basic;
  sf->ops->CreateEmbeddedRootSF = PetscSFCreateEmbeddedRootSF_Basic;

  ierr = PetscNewLog(sf,&dat);CHKERRQ(ierr);
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}
