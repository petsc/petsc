
#include <petsc/private/vecimpl.h>

#define DEFAULT_STASH_SIZE   100

/*
  VecStashCreate_Private - Creates a stash,currently used for all the parallel
  matrix implementations. The stash is where elements of a matrix destined
  to be stored on other processors are kept until matrix assembly is done.

  This is a simple minded stash. Simply adds entries to end of stash.

  Input Parameters:
  comm - communicator, required for scatters.
  bs   - stash block size. used when stashing blocks of values

  Output Parameters:
  stash    - the newly created stash
*/
PetscErrorCode VecStashCreate_Private(MPI_Comm comm,PetscInt bs,VecStash *stash)
{
  PetscErrorCode ierr;
  PetscInt       max,*opt,nopt;
  PetscBool      flg;

  PetscFunctionBegin;
  /* Require 2 tags, get the second using PetscCommGetNewTag() */
  stash->comm = comm;
  ierr = PetscCommGetNewTag(stash->comm,&stash->tag1);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(stash->comm,&stash->tag2);CHKERRQ(ierr);
  ierr = MPI_Comm_size(stash->comm,&stash->size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(stash->comm,&stash->rank);CHKERRMPI(ierr);

  nopt = stash->size;
  ierr = PetscMalloc1(nopt,&opt);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(NULL,NULL,"-vecstash_initial_size",opt,&nopt,&flg);CHKERRQ(ierr);
  if (flg) {
    if (nopt == 1)                max = opt[0];
    else if (nopt == stash->size) max = opt[stash->rank];
    else if (stash->rank < nopt)  max = opt[stash->rank];
    else                          max = 0; /* use default */
    stash->umax = max;
  } else {
    stash->umax = 0;
  }
  ierr = PetscFree(opt);CHKERRQ(ierr);

  if (bs <= 0) bs = 1;

  stash->bs       = bs;
  stash->nmax     = 0;
  stash->oldnmax  = 0;
  stash->n        = 0;
  stash->reallocs = -1;
  stash->idx      = NULL;
  stash->array    = NULL;

  stash->send_waits   = NULL;
  stash->recv_waits   = NULL;
  stash->send_status  = NULL;
  stash->nsends       = 0;
  stash->nrecvs       = 0;
  stash->svalues      = NULL;
  stash->rvalues      = NULL;
  stash->rmax         = 0;
  stash->nprocs       = NULL;
  stash->nprocessed   = 0;
  stash->donotstash   = PETSC_FALSE;
  stash->ignorenegidx = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*
   VecStashDestroy_Private - Destroy the stash
*/
PetscErrorCode VecStashDestroy_Private(VecStash *stash)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(stash->array,stash->idx);CHKERRQ(ierr);
  ierr = PetscFree(stash->bowners);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   VecStashScatterEnd_Private - This is called as the fial stage of
   scatter. The final stages of message passing is done here, and
   all the memory used for message passing is cleanedu up. This
   routine also resets the stash, and deallocates the memory used
   for the stash. It also keeps track of the current memory usage
   so that the same value can be used the next time through.
*/
PetscErrorCode VecStashScatterEnd_Private(VecStash *stash)
{
  PetscErrorCode ierr;
  PetscInt       nsends=stash->nsends,oldnmax;
  MPI_Status     *send_status;

  PetscFunctionBegin;
  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc1(2*nsends,&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(2*nsends,stash->send_waits,send_status);CHKERRMPI(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }

  /* Now update nmaxold to be app 10% more than max n, this way the
     wastage of space is reduced the next time this stash is used.
     Also update the oldmax, only if it increases */
  if (stash->n) {
    oldnmax = ((PetscInt)(stash->n * 1.1) + 5)*stash->bs;
    if (oldnmax > stash->oldnmax) stash->oldnmax = oldnmax;
  }

  stash->nmax       = 0;
  stash->n          = 0;
  stash->reallocs   = -1;
  stash->rmax       = 0;
  stash->nprocessed = 0;

  ierr = PetscFree2(stash->array,stash->idx);CHKERRQ(ierr);
  stash->array = NULL;
  stash->idx   = NULL;
  ierr = PetscFree(stash->send_waits);CHKERRQ(ierr);
  ierr = PetscFree(stash->recv_waits);CHKERRQ(ierr);
  ierr = PetscFree2(stash->svalues,stash->sindices);CHKERRQ(ierr);
  ierr = PetscFree2(stash->rvalues,stash->rindices);CHKERRQ(ierr);
  ierr = PetscFree(stash->nprocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   VecStashGetInfo_Private - Gets the relavant statistics of the stash

   Input Parameters:
   stash    - the stash
   nstash   - the size of the stash
   reallocs - the number of additional mallocs incurred.

*/
PetscErrorCode VecStashGetInfo_Private(VecStash *stash,PetscInt *nstash,PetscInt *reallocs)
{
  PetscFunctionBegin;
  if (nstash) *nstash = stash->n*stash->bs;
  if (reallocs) {
    if (stash->reallocs < 0) *reallocs = 0;
    else                     *reallocs = stash->reallocs;
  }
  PetscFunctionReturn(0);
}

/*
   VecStashSetInitialSize_Private - Sets the initial size of the stash

   Input Parameters:
   stash  - the stash
   max    - the value that is used as the max size of the stash.
            this value is used while allocating memory. It specifies
            the number of vals stored, even with the block-stash
*/
PetscErrorCode VecStashSetInitialSize_Private(VecStash *stash,PetscInt max)
{
  PetscFunctionBegin;
  stash->umax = max;
  PetscFunctionReturn(0);
}

/* VecStashExpand_Private - Expand the stash. This function is called
   when the space in the stash is not sufficient to add the new values
   being inserted into the stash.

   Input Parameters:
   stash - the stash
   incr  - the minimum increase requested

   Notes:
   This routine doubles the currently used memory.
*/
PetscErrorCode VecStashExpand_Private(VecStash *stash,PetscInt incr)
{
  PetscErrorCode ierr;
  PetscInt       *n_idx,newnmax,bs=stash->bs;
  PetscScalar    *n_array;

  PetscFunctionBegin;
  /* allocate a larger stash. */
  if (!stash->oldnmax && !stash->nmax) { /* new stash */
    if (stash->umax)                  newnmax = stash->umax/bs;
    else                              newnmax = DEFAULT_STASH_SIZE/bs;
  } else if (!stash->nmax) { /* resuing stash */
    if (stash->umax > stash->oldnmax) newnmax = stash->umax/bs;
    else                              newnmax = stash->oldnmax/bs;
  } else                              newnmax = stash->nmax*2;

  if (newnmax  < (stash->nmax + incr)) newnmax += 2*incr;

  ierr = PetscMalloc2(bs*newnmax,&n_array,newnmax,&n_idx);CHKERRQ(ierr);
  ierr = PetscMemcpy(n_array,stash->array,bs*stash->nmax*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemcpy(n_idx,stash->idx,stash->nmax*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscFree2(stash->array,stash->idx);CHKERRQ(ierr);

  stash->array = n_array;
  stash->idx   = n_idx;
  stash->nmax  = newnmax;
  stash->reallocs++;
  PetscFunctionReturn(0);
}
/*
  VecStashScatterBegin_Private - Initiates the transfer of values to the
  correct owners. This function goes through the stash, and check the
  owners of each stashed value, and sends the values off to the owner
  processors.

  Input Parameters:
  stash  - the stash
  owners - an array of size 'no-of-procs' which gives the ownership range
           for each node.

  Notes:
    The 'owners' array in the cased of the blocked-stash has the
  ranges specified blocked global indices, and for the regular stash in
  the proper global indices.
*/
PetscErrorCode VecStashScatterBegin_Private(VecStash *stash,PetscInt *owners)
{
  PetscErrorCode ierr;
  PetscMPIInt    size = stash->size,tag1=stash->tag1,tag2=stash->tag2;
  PetscInt       *owner,*start,*nprocs,nsends,nreceives;
  PetscInt       nmax,count,*sindices,*rindices,i,j,idx,bs=stash->bs,lastidx;
  PetscScalar    *rvalues,*svalues;
  MPI_Comm       comm = stash->comm;
  MPI_Request    *send_waits,*recv_waits;

  PetscFunctionBegin;
  /*  first count number of contributors to each processor */
  ierr = PetscCalloc1(2*size,&nprocs);CHKERRQ(ierr);
  ierr = PetscMalloc1(stash->n,&owner);CHKERRQ(ierr);

  j       = 0;
  lastidx = -1;
  for (i=0; i<stash->n; i++) {
    /* if indices are NOT locally sorted, need to start search at the beginning */
    if (lastidx > (idx = stash->idx[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[2*j]++; nprocs[2*j+1] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;
  for (i=0; i<size; i++) nsends += nprocs[2*i+1];

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nreceives);CHKERRQ(ierr);

  /* post receives:
     since we don't know how long each individual message is we
     allocate the largest needed buffer for each receive. Potentially
     this is a lot of wasted space.
  */
  ierr = PetscMalloc2(nreceives*nmax*bs,&rvalues,nreceives*nmax,&rindices);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*nreceives,&recv_waits);CHKERRQ(ierr);
  for (i=0,count=0; i<nreceives; i++) {
    ierr = MPI_Irecv(rvalues+bs*nmax*i,bs*nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag1,comm,recv_waits+count++);CHKERRMPI(ierr);
    ierr = MPI_Irecv(rindices+nmax*i,nmax,MPIU_INT,MPI_ANY_SOURCE,tag2,comm,recv_waits+count++);CHKERRMPI(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to
         the ith processor
  */
  ierr = PetscMalloc2(stash->n*bs,&svalues,stash->n,&sindices);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*nsends,&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&start);CHKERRQ(ierr);
  /* use 2 sends the first with all_v, the next with all_i */
  start[0] = 0;
  for (i=1; i<size; i++) start[i] = start[i-1] + nprocs[2*i-2];

  for (i=0; i<stash->n; i++) {
    j = owner[i];
    if (bs == 1) svalues[start[j]] = stash->array[i];
    else {
      ierr = PetscMemcpy(svalues+bs*start[j],stash->array+bs*i,bs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    sindices[start[j]] = stash->idx[i];
    start[j]++;
  }
  start[0] = 0;
  for (i=1; i<size; i++) start[i] = start[i-1] + nprocs[2*i-2];

  for (i=0,count=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      ierr = MPI_Isend(svalues+bs*start[i],bs*nprocs[2*i],MPIU_SCALAR,i,tag1,comm,send_waits+count++);CHKERRMPI(ierr);
      ierr = MPI_Isend(sindices+start[i],nprocs[2*i],MPIU_INT,i,tag2,comm,send_waits+count++);CHKERRMPI(ierr);
    }
  }
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(start);CHKERRQ(ierr);
  /* This memory is reused in scatter end  for a different purpose*/
  for (i=0; i<2*size; i++) nprocs[i] = -1;

  stash->nprocs     = nprocs;
  stash->svalues    = svalues;
  stash->sindices   = sindices;
  stash->rvalues    = rvalues;
  stash->rindices   = rindices;
  stash->nsends     = nsends;
  stash->nrecvs     = nreceives;
  stash->send_waits = send_waits;
  stash->recv_waits = recv_waits;
  stash->rmax       = nmax;
  PetscFunctionReturn(0);
}

/*
   VecStashScatterGetMesg_Private - This function waits on the receives posted
   in the function VecStashScatterBegin_Private() and returns one message at
   a time to the calling function. If no messages are left, it indicates this
   by setting flg = 0, else it sets flg = 1.

   Input Parameters:
   stash - the stash

   Output Parameters:
   nvals - the number of entries in the current message.
   rows  - an array of row indices (or blocked indices) corresponding to the values
   cols  - an array of columnindices (or blocked indices) corresponding to the values
   vals  - the values
   flg   - 0 indicates no more message left, and the current call has no values associated.
           1 indicates that the current call successfully received a message, and the
             other output parameters nvals,rows,cols,vals are set appropriately.
*/
PetscErrorCode VecStashScatterGetMesg_Private(VecStash *stash,PetscMPIInt *nvals,PetscInt **rows,PetscScalar **vals,PetscInt *flg)
{
  PetscErrorCode ierr;
  PetscMPIInt    i = 0; /* dummy value so MPI-Uni doesn't think it is not set */
  PetscInt       *flg_v;
  PetscInt       i1,i2,bs=stash->bs;
  MPI_Status     recv_status;
  PetscBool      match_found = PETSC_FALSE;

  PetscFunctionBegin;
  *flg = 0; /* When a message is discovered this is reset to 1 */
  /* Return if no more messages to process */
  if (stash->nprocessed == stash->nrecvs) PetscFunctionReturn(0);

  flg_v = stash->nprocs;
  /* If a matching pair of receives are found, process them, and return the data to
     the calling function. Until then keep receiving messages */
  while (!match_found) {
    ierr = MPI_Waitany(2*stash->nrecvs,stash->recv_waits,&i,&recv_status);CHKERRMPI(ierr);
    /* Now pack the received message into a structure which is useable by others */
    if (i % 2) {
      ierr = MPI_Get_count(&recv_status,MPIU_INT,nvals);CHKERRMPI(ierr);
      flg_v[2*recv_status.MPI_SOURCE+1] = i/2;
    } else {
      ierr = MPI_Get_count(&recv_status,MPIU_SCALAR,nvals);CHKERRMPI(ierr);
      flg_v[2*recv_status.MPI_SOURCE] = i/2;
      *nvals = *nvals/bs;
    }

    /* Check if we have both the messages from this proc */
    i1 = flg_v[2*recv_status.MPI_SOURCE];
    i2 = flg_v[2*recv_status.MPI_SOURCE+1];
    if (i1 != -1 && i2 != -1) {
      *rows = stash->rindices + i2*stash->rmax;
      *vals = stash->rvalues + i1*bs*stash->rmax;
      *flg  = 1;
      stash->nprocessed++;
      match_found = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}

/*
 * Sort the stash, removing duplicates (combining as appropriate).
 */
PetscErrorCode VecStashSortCompress_Private(VecStash *stash)
{
  PetscErrorCode ierr;
  PetscInt i,j,bs = stash->bs;

  PetscFunctionBegin;
  if (!stash->n) PetscFunctionReturn(0);
  if (bs == 1) {
    ierr = PetscSortIntWithScalarArray(stash->n,stash->idx,stash->array);CHKERRQ(ierr);
    for (i=1,j=0; i<stash->n; i++) {
      if (stash->idx[i] == stash->idx[j]) {
        switch (stash->insertmode) {
        case ADD_VALUES:
          stash->array[j] += stash->array[i];
          break;
        case INSERT_VALUES:
          stash->array[j] = stash->array[i];
          break;
        default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Insert mode not supported 0x%x",stash->insertmode);
        }
      } else {
        j++;
        stash->idx[j]   = stash->idx[i];
        stash->array[j] = stash->array[i];
      }
    }
    stash->n = j + 1;
  } else {                      /* block stash */
    PetscInt *perm = NULL;
    PetscScalar *arr;
    ierr = PetscMalloc2(stash->n,&perm,stash->n*bs,&arr);CHKERRQ(ierr);
    for (i=0; i<stash->n; i++) perm[i] = i;
    ierr = PetscSortIntWithArray(stash->n,stash->idx,perm);CHKERRQ(ierr);

    /* Out-of-place copy of arr */
    ierr = PetscMemcpy(arr,stash->array+perm[0]*bs,bs*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=1,j=0; i<stash->n; i++) {
      PetscInt k;
      if (stash->idx[i] == stash->idx[j]) {
        switch (stash->insertmode) {
        case ADD_VALUES:
          for (k=0; k<bs; k++) arr[j*bs+k] += stash->array[perm[i]*bs+k];
          break;
        case INSERT_VALUES:
          for (k=0; k<bs; k++) arr[j*bs+k] = stash->array[perm[i]*bs+k];
          break;
        default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Insert mode not supported 0x%x",stash->insertmode);
        }
      } else {
        j++;
        stash->idx[j] = stash->idx[i];
        for (k=0; k<bs; k++) arr[j*bs+k] = stash->array[perm[i]*bs+k];
      }
    }
    stash->n = j + 1;
    ierr = PetscMemcpy(stash->array,arr,stash->n*bs*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscFree2(perm,arr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecStashGetOwnerList_Private(VecStash *stash,PetscLayout map,PetscMPIInt *nowners,PetscMPIInt **owners)
{
  PetscInt       i,bs = stash->bs;
  PetscMPIInt    r;
  PetscSegBuffer seg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (bs != 1 && bs != map->bs) SETERRQ2(map->comm,PETSC_ERR_PLIB,"Stash block size %" PetscInt_FMT " does not match layout block size %" PetscInt_FMT,bs,map->bs);
  ierr = PetscSegBufferCreate(sizeof(PetscMPIInt),50,&seg);CHKERRQ(ierr);
  *nowners = 0;
  for (i=0,r=-1; i<stash->n; i++) {
    if (stash->idx[i]*bs >= map->range[r+1]) {
      PetscMPIInt *rank;
      ierr = PetscSegBufferGet(seg,1,&rank);CHKERRQ(ierr);
      ierr = PetscLayoutFindOwner(map,stash->idx[i]*bs,&r);CHKERRQ(ierr);
      *rank = r;
      (*nowners)++;
    }
  }
  ierr = PetscSegBufferExtractAlloc(seg,owners);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&seg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
