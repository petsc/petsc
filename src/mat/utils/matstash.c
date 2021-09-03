
#include <petsc/private/matimpl.h>

#define DEFAULT_STASH_SIZE   10000

static PetscErrorCode MatStashScatterBegin_Ref(Mat,MatStash*,PetscInt*);
PETSC_INTERN PetscErrorCode MatStashScatterGetMesg_Ref(MatStash*,PetscMPIInt*,PetscInt**,PetscInt**,PetscScalar**,PetscInt*);
PETSC_INTERN PetscErrorCode MatStashScatterEnd_Ref(MatStash*);
#if !defined(PETSC_HAVE_MPIUNI)
static PetscErrorCode MatStashScatterBegin_BTS(Mat,MatStash*,PetscInt*);
static PetscErrorCode MatStashScatterGetMesg_BTS(MatStash*,PetscMPIInt*,PetscInt**,PetscInt**,PetscScalar**,PetscInt*);
static PetscErrorCode MatStashScatterEnd_BTS(MatStash*);
#endif

/*
  MatStashCreate_Private - Creates a stash,currently used for all the parallel
  matrix implementations. The stash is where elements of a matrix destined
  to be stored on other processors are kept until matrix assembly is done.

  This is a simple minded stash. Simply adds entries to end of stash.

  Input Parameters:
  comm - communicator, required for scatters.
  bs   - stash block size. used when stashing blocks of values

  Output Parameters:
  stash    - the newly created stash
*/
PetscErrorCode MatStashCreate_Private(MPI_Comm comm,PetscInt bs,MatStash *stash)
{
  PetscErrorCode ierr;
  PetscInt       max,*opt,nopt,i;
  PetscBool      flg;

  PetscFunctionBegin;
  /* Require 2 tags,get the second using PetscCommGetNewTag() */
  stash->comm = comm;

  ierr = PetscCommGetNewTag(stash->comm,&stash->tag1);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(stash->comm,&stash->tag2);CHKERRQ(ierr);
  ierr = MPI_Comm_size(stash->comm,&stash->size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(stash->comm,&stash->rank);CHKERRMPI(ierr);
  ierr = PetscMalloc1(2*stash->size,&stash->flg_v);CHKERRQ(ierr);
  for (i=0; i<2*stash->size; i++) stash->flg_v[i] = -1;

  nopt = stash->size;
  ierr = PetscMalloc1(nopt,&opt);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(NULL,NULL,"-matstash_initial_size",opt,&nopt,&flg);CHKERRQ(ierr);
  if (flg) {
    if (nopt == 1)                max = opt[0];
    else if (nopt == stash->size) max = opt[stash->rank];
    else if (stash->rank < nopt)  max = opt[stash->rank];
    else                          max = 0; /* Use default */
    stash->umax = max;
  } else {
    stash->umax = 0;
  }
  ierr = PetscFree(opt);CHKERRQ(ierr);
  if (bs <= 0) bs = 1;

  stash->bs         = bs;
  stash->nmax       = 0;
  stash->oldnmax    = 0;
  stash->n          = 0;
  stash->reallocs   = -1;
  stash->space_head = NULL;
  stash->space      = NULL;

  stash->send_waits  = NULL;
  stash->recv_waits  = NULL;
  stash->send_status = NULL;
  stash->nsends      = 0;
  stash->nrecvs      = 0;
  stash->svalues     = NULL;
  stash->rvalues     = NULL;
  stash->rindices    = NULL;
  stash->nprocessed  = 0;
  stash->reproduce   = PETSC_FALSE;
  stash->blocktype   = MPI_DATATYPE_NULL;

  ierr = PetscOptionsGetBool(NULL,NULL,"-matstash_reproduce",&stash->reproduce,NULL);CHKERRQ(ierr);
#if !defined(PETSC_HAVE_MPIUNI)
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-matstash_legacy",&flg,NULL);CHKERRQ(ierr);
  if (!flg) {
    stash->ScatterBegin   = MatStashScatterBegin_BTS;
    stash->ScatterGetMesg = MatStashScatterGetMesg_BTS;
    stash->ScatterEnd     = MatStashScatterEnd_BTS;
    stash->ScatterDestroy = MatStashScatterDestroy_BTS;
  } else {
#endif
    stash->ScatterBegin   = MatStashScatterBegin_Ref;
    stash->ScatterGetMesg = MatStashScatterGetMesg_Ref;
    stash->ScatterEnd     = MatStashScatterEnd_Ref;
    stash->ScatterDestroy = NULL;
#if !defined(PETSC_HAVE_MPIUNI)
  }
#endif
  PetscFunctionReturn(0);
}

/*
   MatStashDestroy_Private - Destroy the stash
*/
PetscErrorCode MatStashDestroy_Private(MatStash *stash)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMatStashSpaceDestroy(&stash->space_head);CHKERRQ(ierr);
  if (stash->ScatterDestroy) {ierr = (*stash->ScatterDestroy)(stash);CHKERRQ(ierr);}

  stash->space = NULL;

  ierr = PetscFree(stash->flg_v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   MatStashScatterEnd_Private - This is called as the final stage of
   scatter. The final stages of message passing is done here, and
   all the memory used for message passing is cleaned up. This
   routine also resets the stash, and deallocates the memory used
   for the stash. It also keeps track of the current memory usage
   so that the same value can be used the next time through.
*/
PetscErrorCode MatStashScatterEnd_Private(MatStash *stash)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*stash->ScatterEnd)(stash);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatStashScatterEnd_Ref(MatStash *stash)
{
  PetscErrorCode ierr;
  PetscInt       nsends=stash->nsends,bs2,oldnmax,i;
  MPI_Status     *send_status;

  PetscFunctionBegin;
  for (i=0; i<2*stash->size; i++) stash->flg_v[i] = -1;
  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc1(2*nsends,&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(2*nsends,stash->send_waits,send_status);CHKERRMPI(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }

  /* Now update nmaxold to be app 10% more than max n used, this way the
     wastage of space is reduced the next time this stash is used.
     Also update the oldmax, only if it increases */
  if (stash->n) {
    bs2     = stash->bs*stash->bs;
    oldnmax = ((int)(stash->n * 1.1) + 5)*bs2;
    if (oldnmax > stash->oldnmax) stash->oldnmax = oldnmax;
  }

  stash->nmax       = 0;
  stash->n          = 0;
  stash->reallocs   = -1;
  stash->nprocessed = 0;

  ierr = PetscMatStashSpaceDestroy(&stash->space_head);CHKERRQ(ierr);

  stash->space = NULL;

  ierr = PetscFree(stash->send_waits);CHKERRQ(ierr);
  ierr = PetscFree(stash->recv_waits);CHKERRQ(ierr);
  ierr = PetscFree2(stash->svalues,stash->sindices);CHKERRQ(ierr);
  ierr = PetscFree(stash->rvalues[0]);CHKERRQ(ierr);
  ierr = PetscFree(stash->rvalues);CHKERRQ(ierr);
  ierr = PetscFree(stash->rindices[0]);CHKERRQ(ierr);
  ierr = PetscFree(stash->rindices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   MatStashGetInfo_Private - Gets the relavant statistics of the stash

   Input Parameters:
   stash    - the stash
   nstash   - the size of the stash. Indicates the number of values stored.
   reallocs - the number of additional mallocs incurred.

*/
PetscErrorCode MatStashGetInfo_Private(MatStash *stash,PetscInt *nstash,PetscInt *reallocs)
{
  PetscInt bs2 = stash->bs*stash->bs;

  PetscFunctionBegin;
  if (nstash) *nstash = stash->n*bs2;
  if (reallocs) {
    if (stash->reallocs < 0) *reallocs = 0;
    else                     *reallocs = stash->reallocs;
  }
  PetscFunctionReturn(0);
}

/*
   MatStashSetInitialSize_Private - Sets the initial size of the stash

   Input Parameters:
   stash  - the stash
   max    - the value that is used as the max size of the stash.
            this value is used while allocating memory.
*/
PetscErrorCode MatStashSetInitialSize_Private(MatStash *stash,PetscInt max)
{
  PetscFunctionBegin;
  stash->umax = max;
  PetscFunctionReturn(0);
}

/* MatStashExpand_Private - Expand the stash. This function is called
   when the space in the stash is not sufficient to add the new values
   being inserted into the stash.

   Input Parameters:
   stash - the stash
   incr  - the minimum increase requested

   Notes:
   This routine doubles the currently used memory.
 */
static PetscErrorCode MatStashExpand_Private(MatStash *stash,PetscInt incr)
{
  PetscErrorCode ierr;
  PetscInt       newnmax,bs2= stash->bs*stash->bs;

  PetscFunctionBegin;
  /* allocate a larger stash */
  if (!stash->oldnmax && !stash->nmax) { /* new stash */
    if (stash->umax)                  newnmax = stash->umax/bs2;
    else                              newnmax = DEFAULT_STASH_SIZE/bs2;
  } else if (!stash->nmax) { /* resuing stash */
    if (stash->umax > stash->oldnmax) newnmax = stash->umax/bs2;
    else                              newnmax = stash->oldnmax/bs2;
  } else                              newnmax = stash->nmax*2;
  if (newnmax  < (stash->nmax + incr)) newnmax += 2*incr;

  /* Get a MatStashSpace and attach it to stash */
  ierr = PetscMatStashSpaceGet(bs2,newnmax,&stash->space);CHKERRQ(ierr);
  if (!stash->space_head) { /* new stash or resuing stash->oldnmax */
    stash->space_head = stash->space;
  }

  stash->reallocs++;
  stash->nmax = newnmax;
  PetscFunctionReturn(0);
}
/*
  MatStashValuesRow_Private - inserts values into the stash. This function
  expects the values to be roworiented. Multiple columns belong to the same row
  can be inserted with a single call to this function.

  Input Parameters:
  stash  - the stash
  row    - the global row correspoiding to the values
  n      - the number of elements inserted. All elements belong to the above row.
  idxn   - the global column indices corresponding to each of the values.
  values - the values inserted
*/
PetscErrorCode MatStashValuesRow_Private(MatStash *stash,PetscInt row,PetscInt n,const PetscInt idxn[],const PetscScalar values[],PetscBool ignorezeroentries)
{
  PetscErrorCode     ierr;
  PetscInt           i,k,cnt = 0;
  PetscMatStashSpace space=stash->space;

  PetscFunctionBegin;
  /* Check and see if we have sufficient memory */
  if (!space || space->local_remaining < n) {
    ierr = MatStashExpand_Private(stash,n);CHKERRQ(ierr);
  }
  space = stash->space;
  k     = space->local_used;
  for (i=0; i<n; i++) {
    if (ignorezeroentries && values && values[i] == 0.0) continue;
    space->idx[k] = row;
    space->idy[k] = idxn[i];
    space->val[k] = values ? values[i] : 0.0;
    k++;
    cnt++;
  }
  stash->n               += cnt;
  space->local_used      += cnt;
  space->local_remaining -= cnt;
  PetscFunctionReturn(0);
}

/*
  MatStashValuesCol_Private - inserts values into the stash. This function
  expects the values to be columnoriented. Multiple columns belong to the same row
  can be inserted with a single call to this function.

  Input Parameters:
  stash   - the stash
  row     - the global row correspoiding to the values
  n       - the number of elements inserted. All elements belong to the above row.
  idxn    - the global column indices corresponding to each of the values.
  values  - the values inserted
  stepval - the consecutive values are sepated by a distance of stepval.
            this happens because the input is columnoriented.
*/
PetscErrorCode MatStashValuesCol_Private(MatStash *stash,PetscInt row,PetscInt n,const PetscInt idxn[],const PetscScalar values[],PetscInt stepval,PetscBool ignorezeroentries)
{
  PetscErrorCode     ierr;
  PetscInt           i,k,cnt = 0;
  PetscMatStashSpace space=stash->space;

  PetscFunctionBegin;
  /* Check and see if we have sufficient memory */
  if (!space || space->local_remaining < n) {
    ierr = MatStashExpand_Private(stash,n);CHKERRQ(ierr);
  }
  space = stash->space;
  k     = space->local_used;
  for (i=0; i<n; i++) {
    if (ignorezeroentries && values && values[i*stepval] == 0.0) continue;
    space->idx[k] = row;
    space->idy[k] = idxn[i];
    space->val[k] = values ? values[i*stepval] : 0.0;
    k++;
    cnt++;
  }
  stash->n               += cnt;
  space->local_used      += cnt;
  space->local_remaining -= cnt;
  PetscFunctionReturn(0);
}

/*
  MatStashValuesRowBlocked_Private - inserts blocks of values into the stash.
  This function expects the values to be roworiented. Multiple columns belong
  to the same block-row can be inserted with a single call to this function.
  This function extracts the sub-block of values based on the dimensions of
  the original input block, and the row,col values corresponding to the blocks.

  Input Parameters:
  stash  - the stash
  row    - the global block-row correspoiding to the values
  n      - the number of elements inserted. All elements belong to the above row.
  idxn   - the global block-column indices corresponding to each of the blocks of
           values. Each block is of size bs*bs.
  values - the values inserted
  rmax   - the number of block-rows in the original block.
  cmax   - the number of block-columns on the original block.
  idx    - the index of the current block-row in the original block.
*/
PetscErrorCode MatStashValuesRowBlocked_Private(MatStash *stash,PetscInt row,PetscInt n,const PetscInt idxn[],const PetscScalar values[],PetscInt rmax,PetscInt cmax,PetscInt idx)
{
  PetscErrorCode     ierr;
  PetscInt           i,j,k,bs2,bs=stash->bs,l;
  const PetscScalar  *vals;
  PetscScalar        *array;
  PetscMatStashSpace space=stash->space;

  PetscFunctionBegin;
  if (!space || space->local_remaining < n) {
    ierr = MatStashExpand_Private(stash,n);CHKERRQ(ierr);
  }
  space = stash->space;
  l     = space->local_used;
  bs2   = bs*bs;
  for (i=0; i<n; i++) {
    space->idx[l] = row;
    space->idy[l] = idxn[i];
    /* Now copy over the block of values. Store the values column oriented.
       This enables inserting multiple blocks belonging to a row with a single
       funtion call */
    array = space->val + bs2*l;
    vals  = values + idx*bs2*n + bs*i;
    for (j=0; j<bs; j++) {
      for (k=0; k<bs; k++) array[k*bs] = values ? vals[k] : 0.0;
      array++;
      vals += cmax*bs;
    }
    l++;
  }
  stash->n               += n;
  space->local_used      += n;
  space->local_remaining -= n;
  PetscFunctionReturn(0);
}

/*
  MatStashValuesColBlocked_Private - inserts blocks of values into the stash.
  This function expects the values to be roworiented. Multiple columns belong
  to the same block-row can be inserted with a single call to this function.
  This function extracts the sub-block of values based on the dimensions of
  the original input block, and the row,col values corresponding to the blocks.

  Input Parameters:
  stash  - the stash
  row    - the global block-row correspoiding to the values
  n      - the number of elements inserted. All elements belong to the above row.
  idxn   - the global block-column indices corresponding to each of the blocks of
           values. Each block is of size bs*bs.
  values - the values inserted
  rmax   - the number of block-rows in the original block.
  cmax   - the number of block-columns on the original block.
  idx    - the index of the current block-row in the original block.
*/
PetscErrorCode MatStashValuesColBlocked_Private(MatStash *stash,PetscInt row,PetscInt n,const PetscInt idxn[],const PetscScalar values[],PetscInt rmax,PetscInt cmax,PetscInt idx)
{
  PetscErrorCode     ierr;
  PetscInt           i,j,k,bs2,bs=stash->bs,l;
  const PetscScalar  *vals;
  PetscScalar        *array;
  PetscMatStashSpace space=stash->space;

  PetscFunctionBegin;
  if (!space || space->local_remaining < n) {
    ierr = MatStashExpand_Private(stash,n);CHKERRQ(ierr);
  }
  space = stash->space;
  l     = space->local_used;
  bs2   = bs*bs;
  for (i=0; i<n; i++) {
    space->idx[l] = row;
    space->idy[l] = idxn[i];
    /* Now copy over the block of values. Store the values column oriented.
     This enables inserting multiple blocks belonging to a row with a single
     funtion call */
    array = space->val + bs2*l;
    vals  = values + idx*bs2*n + bs*i;
    for (j=0; j<bs; j++) {
      for (k=0; k<bs; k++) array[k] = values ? vals[k] : 0.0;
      array += bs;
      vals  += rmax*bs;
    }
    l++;
  }
  stash->n               += n;
  space->local_used      += n;
  space->local_remaining -= n;
  PetscFunctionReturn(0);
}
/*
  MatStashScatterBegin_Private - Initiates the transfer of values to the
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
PetscErrorCode MatStashScatterBegin_Private(Mat mat,MatStash *stash,PetscInt *owners)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*stash->ScatterBegin)(mat,stash,owners);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatStashScatterBegin_Ref(Mat mat,MatStash *stash,PetscInt *owners)
{
  PetscInt           *owner,*startv,*starti,tag1=stash->tag1,tag2=stash->tag2,bs2;
  PetscInt           size=stash->size,nsends;
  PetscErrorCode     ierr;
  PetscInt           count,*sindices,**rindices,i,j,idx,lastidx,l;
  PetscScalar        **rvalues,*svalues;
  MPI_Comm           comm = stash->comm;
  MPI_Request        *send_waits,*recv_waits,*recv_waits1,*recv_waits2;
  PetscMPIInt        *sizes,*nlengths,nreceives;
  PetscInt           *sp_idx,*sp_idy;
  PetscScalar        *sp_val;
  PetscMatStashSpace space,space_next;

  PetscFunctionBegin;
  {                             /* make sure all processors are either in INSERTMODE or ADDMODE */
    InsertMode addv;
    ierr = MPIU_Allreduce((PetscEnum*)&mat->insertmode,(PetscEnum*)&addv,1,MPIU_ENUM,MPI_BOR,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
    if (addv == (ADD_VALUES|INSERT_VALUES)) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Some processors inserted others added");
    mat->insertmode = addv; /* in case this processor had no cache */
  }

  bs2 = stash->bs*stash->bs;

  /*  first count number of contributors to each processor */
  ierr = PetscCalloc1(size,&nlengths);CHKERRQ(ierr);
  ierr = PetscMalloc1(stash->n+1,&owner);CHKERRQ(ierr);

  i       = j    = 0;
  lastidx = -1;
  space   = stash->space_head;
  while (space) {
    space_next = space->next;
    sp_idx     = space->idx;
    for (l=0; l<space->local_used; l++) {
      /* if indices are NOT locally sorted, need to start search at the beginning */
      if (lastidx > (idx = sp_idx[l])) j = 0;
      lastidx = idx;
      for (; j<size; j++) {
        if (idx >= owners[j] && idx < owners[j+1]) {
          nlengths[j]++; owner[i] = j; break;
        }
      }
      i++;
    }
    space = space_next;
  }

  /* Now check what procs get messages - and compute nsends. */
  ierr = PetscCalloc1(size,&sizes);CHKERRQ(ierr);
  for (i=0, nsends=0; i<size; i++) {
    if (nlengths[i]) {
      sizes[i] = 1; nsends++;
    }
  }

  {PetscMPIInt *onodes,*olengths;
   /* Determine the number of messages to expect, their lengths, from from-ids */
   ierr = PetscGatherNumberOfMessages(comm,sizes,nlengths,&nreceives);CHKERRQ(ierr);
   ierr = PetscGatherMessageLengths(comm,nsends,nreceives,nlengths,&onodes,&olengths);CHKERRQ(ierr);
   /* since clubbing row,col - lengths are multiplied by 2 */
   for (i=0; i<nreceives; i++) olengths[i] *=2;
   ierr = PetscPostIrecvInt(comm,tag1,nreceives,onodes,olengths,&rindices,&recv_waits1);CHKERRQ(ierr);
   /* values are size 'bs2' lengths (and remove earlier factor 2 */
   for (i=0; i<nreceives; i++) olengths[i] = olengths[i]*bs2/2;
   ierr = PetscPostIrecvScalar(comm,tag2,nreceives,onodes,olengths,&rvalues,&recv_waits2);CHKERRQ(ierr);
   ierr = PetscFree(onodes);CHKERRQ(ierr);
   ierr = PetscFree(olengths);CHKERRQ(ierr);}

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to
         the ith processor
  */
  ierr = PetscMalloc2(bs2*stash->n,&svalues,2*(stash->n+1),&sindices);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*nsends,&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc2(size,&startv,size,&starti);CHKERRQ(ierr);
  /* use 2 sends the first with all_a, the next with all_i and all_j */
  startv[0] = 0; starti[0] = 0;
  for (i=1; i<size; i++) {
    startv[i] = startv[i-1] + nlengths[i-1];
    starti[i] = starti[i-1] + 2*nlengths[i-1];
  }

  i     = 0;
  space = stash->space_head;
  while (space) {
    space_next = space->next;
    sp_idx     = space->idx;
    sp_idy     = space->idy;
    sp_val     = space->val;
    for (l=0; l<space->local_used; l++) {
      j = owner[i];
      if (bs2 == 1) {
        svalues[startv[j]] = sp_val[l];
      } else {
        PetscInt    k;
        PetscScalar *buf1,*buf2;
        buf1 = svalues+bs2*startv[j];
        buf2 = space->val + bs2*l;
        for (k=0; k<bs2; k++) buf1[k] = buf2[k];
      }
      sindices[starti[j]]             = sp_idx[l];
      sindices[starti[j]+nlengths[j]] = sp_idy[l];
      startv[j]++;
      starti[j]++;
      i++;
    }
    space = space_next;
  }
  startv[0] = 0;
  for (i=1; i<size; i++) startv[i] = startv[i-1] + nlengths[i-1];

  for (i=0,count=0; i<size; i++) {
    if (sizes[i]) {
      ierr = MPI_Isend(sindices+2*startv[i],2*nlengths[i],MPIU_INT,i,tag1,comm,send_waits+count++);CHKERRMPI(ierr);
      ierr = MPI_Isend(svalues+bs2*startv[i],bs2*nlengths[i],MPIU_SCALAR,i,tag2,comm,send_waits+count++);CHKERRMPI(ierr);
    }
  }
#if defined(PETSC_USE_INFO)
  ierr = PetscInfo1(NULL,"No of messages: %d \n",nsends);CHKERRQ(ierr);
  for (i=0; i<size; i++) {
    if (sizes[i]) {
      ierr = PetscInfo2(NULL,"Mesg_to: %d: size: %d bytes\n",i,nlengths[i]*(bs2*sizeof(PetscScalar)+2*sizeof(PetscInt)));CHKERRQ(ierr);
    }
  }
#endif
  ierr = PetscFree(nlengths);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree2(startv,starti);CHKERRQ(ierr);
  ierr = PetscFree(sizes);CHKERRQ(ierr);

  /* recv_waits need to be contiguous for MatStashScatterGetMesg_Private() */
  ierr = PetscMalloc1(2*nreceives,&recv_waits);CHKERRQ(ierr);

  for (i=0; i<nreceives; i++) {
    recv_waits[2*i]   = recv_waits1[i];
    recv_waits[2*i+1] = recv_waits2[i];
  }
  stash->recv_waits = recv_waits;

  ierr = PetscFree(recv_waits1);CHKERRQ(ierr);
  ierr = PetscFree(recv_waits2);CHKERRQ(ierr);

  stash->svalues         = svalues;
  stash->sindices        = sindices;
  stash->rvalues         = rvalues;
  stash->rindices        = rindices;
  stash->send_waits      = send_waits;
  stash->nsends          = nsends;
  stash->nrecvs          = nreceives;
  stash->reproduce_count = 0;
  PetscFunctionReturn(0);
}

/*
   MatStashScatterGetMesg_Private - This function waits on the receives posted
   in the function MatStashScatterBegin_Private() and returns one message at
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
PetscErrorCode MatStashScatterGetMesg_Private(MatStash *stash,PetscMPIInt *nvals,PetscInt **rows,PetscInt **cols,PetscScalar **vals,PetscInt *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*stash->ScatterGetMesg)(stash,nvals,rows,cols,vals,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatStashScatterGetMesg_Ref(MatStash *stash,PetscMPIInt *nvals,PetscInt **rows,PetscInt **cols,PetscScalar **vals,PetscInt *flg)
{
  PetscErrorCode ierr;
  PetscMPIInt    i,*flg_v = stash->flg_v,i1,i2;
  PetscInt       bs2;
  MPI_Status     recv_status;
  PetscBool      match_found = PETSC_FALSE;

  PetscFunctionBegin;
  *flg = 0; /* When a message is discovered this is reset to 1 */
  /* Return if no more messages to process */
  if (stash->nprocessed == stash->nrecvs) PetscFunctionReturn(0);

  bs2 = stash->bs*stash->bs;
  /* If a matching pair of receives are found, process them, and return the data to
     the calling function. Until then keep receiving messages */
  while (!match_found) {
    if (stash->reproduce) {
      i    = stash->reproduce_count++;
      ierr = MPI_Wait(stash->recv_waits+i,&recv_status);CHKERRMPI(ierr);
    } else {
      ierr = MPI_Waitany(2*stash->nrecvs,stash->recv_waits,&i,&recv_status);CHKERRMPI(ierr);
    }
    if (recv_status.MPI_SOURCE < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Negative MPI source!");

    /* Now pack the received message into a structure which is usable by others */
    if (i % 2) {
      ierr = MPI_Get_count(&recv_status,MPIU_SCALAR,nvals);CHKERRMPI(ierr);

      flg_v[2*recv_status.MPI_SOURCE] = i/2;

      *nvals = *nvals/bs2;
    } else {
      ierr = MPI_Get_count(&recv_status,MPIU_INT,nvals);CHKERRMPI(ierr);

      flg_v[2*recv_status.MPI_SOURCE+1] = i/2;

      *nvals = *nvals/2; /* This message has both row indices and col indices */
    }

    /* Check if we have both messages from this proc */
    i1 = flg_v[2*recv_status.MPI_SOURCE];
    i2 = flg_v[2*recv_status.MPI_SOURCE+1];
    if (i1 != -1 && i2 != -1) {
      *rows = stash->rindices[i2];
      *cols = *rows + *nvals;
      *vals = stash->rvalues[i1];
      *flg  = 1;
      stash->nprocessed++;
      match_found = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}

#if !defined(PETSC_HAVE_MPIUNI)
typedef struct {
  PetscInt row;
  PetscInt col;
  PetscScalar vals[1];          /* Actually an array of length bs2 */
} MatStashBlock;

static PetscErrorCode MatStashSortCompress_Private(MatStash *stash,InsertMode insertmode)
{
  PetscErrorCode ierr;
  PetscMatStashSpace space;
  PetscInt n = stash->n,bs = stash->bs,bs2 = bs*bs,cnt,*row,*col,*perm,rowstart,i;
  PetscScalar **valptr;

  PetscFunctionBegin;
  ierr = PetscMalloc4(n,&row,n,&col,n,&valptr,n,&perm);CHKERRQ(ierr);
  for (space=stash->space_head,cnt=0; space; space=space->next) {
    for (i=0; i<space->local_used; i++) {
      row[cnt] = space->idx[i];
      col[cnt] = space->idy[i];
      valptr[cnt] = &space->val[i*bs2];
      perm[cnt] = cnt;          /* Will tell us where to find valptr after sorting row[] and col[] */
      cnt++;
    }
  }
  if (cnt != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatStash n %D, but counted %D entries",n,cnt);
  ierr = PetscSortIntWithArrayPair(n,row,col,perm);CHKERRQ(ierr);
  /* Scan through the rows, sorting each one, combining duplicates, and packing send buffers */
  for (rowstart=0,cnt=0,i=1; i<=n; i++) {
    if (i == n || row[i] != row[rowstart]) {         /* Sort the last row. */
      PetscInt colstart;
      ierr = PetscSortIntWithArray(i-rowstart,&col[rowstart],&perm[rowstart]);CHKERRQ(ierr);
      for (colstart=rowstart; colstart<i;) { /* Compress multiple insertions to the same location */
        PetscInt j,l;
        MatStashBlock *block;
        ierr = PetscSegBufferGet(stash->segsendblocks,1,&block);CHKERRQ(ierr);
        block->row = row[rowstart];
        block->col = col[colstart];
        ierr = PetscArraycpy(block->vals,valptr[perm[colstart]],bs2);CHKERRQ(ierr);
        for (j=colstart+1; j<i && col[j] == col[colstart]; j++) { /* Add any extra stashed blocks at the same (row,col) */
          if (insertmode == ADD_VALUES) {
            for (l=0; l<bs2; l++) block->vals[l] += valptr[perm[j]][l];
          } else {
            ierr = PetscArraycpy(block->vals,valptr[perm[j]],bs2);CHKERRQ(ierr);
          }
        }
        colstart = j;
      }
      rowstart = i;
    }
  }
  ierr = PetscFree4(row,col,valptr,perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatStashBlockTypeSetUp(MatStash *stash)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stash->blocktype == MPI_DATATYPE_NULL) {
    PetscInt     bs2 = PetscSqr(stash->bs);
    PetscMPIInt  blocklens[2];
    MPI_Aint     displs[2];
    MPI_Datatype types[2],stype;
    /* Note that DummyBlock is a type having standard layout, even when PetscScalar is C++ std::complex.
       std::complex itself has standard layout, so does DummyBlock, recursively.
       To be compatible with C++ std::complex, complex implementations on GPUs must also have standard layout,
       though they can have different alignment, e.g, 16 bytes for double complex, instead of 8 bytes as in GCC stdlibc++.
       offsetof(type, member) only requires type to have standard layout. Ref. https://en.cppreference.com/w/cpp/types/offsetof.

       We can test if std::complex has standard layout with the following code:
       #include <iostream>
       #include <type_traits>
       #include <complex>
       int main() {
         std::cout << std::boolalpha;
         std::cout << std::is_standard_layout<std::complex<double> >::value << '\n';
       }
       Output: true
     */
    struct DummyBlock {PetscInt row,col; PetscScalar vals;};

    stash->blocktype_size = offsetof(struct DummyBlock,vals) + bs2*sizeof(PetscScalar);
    if (stash->blocktype_size % sizeof(PetscInt)) { /* Implies that PetscInt is larger and does not satisfy alignment without padding */
      stash->blocktype_size += sizeof(PetscInt) - stash->blocktype_size % sizeof(PetscInt);
    }
    ierr = PetscSegBufferCreate(stash->blocktype_size,1,&stash->segsendblocks);CHKERRQ(ierr);
    ierr = PetscSegBufferCreate(stash->blocktype_size,1,&stash->segrecvblocks);CHKERRQ(ierr);
    ierr = PetscSegBufferCreate(sizeof(MatStashFrame),1,&stash->segrecvframe);CHKERRQ(ierr);
    blocklens[0] = 2;
    blocklens[1] = bs2;
    displs[0] = offsetof(struct DummyBlock,row);
    displs[1] = offsetof(struct DummyBlock,vals);
    types[0] = MPIU_INT;
    types[1] = MPIU_SCALAR;
    ierr = MPI_Type_create_struct(2,blocklens,displs,types,&stype);CHKERRMPI(ierr);
    ierr = MPI_Type_commit(&stype);CHKERRMPI(ierr);
    ierr = MPI_Type_create_resized(stype,0,stash->blocktype_size,&stash->blocktype);CHKERRMPI(ierr);
    ierr = MPI_Type_commit(&stash->blocktype);CHKERRMPI(ierr);
    ierr = MPI_Type_free(&stype);CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

/* Callback invoked after target rank has initiatied receive of rendezvous message.
 * Here we post the main sends.
 */
static PetscErrorCode MatStashBTSSend_Private(MPI_Comm comm,const PetscMPIInt tag[],PetscMPIInt rankid,PetscMPIInt rank,void *sdata,MPI_Request req[],void *ctx)
{
  MatStash *stash = (MatStash*)ctx;
  MatStashHeader *hdr = (MatStashHeader*)sdata;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (rank != stash->sendranks[rankid]) SETERRQ3(comm,PETSC_ERR_PLIB,"BTS Send rank %d does not match sendranks[%d] %d",rank,rankid,stash->sendranks[rankid]);
  ierr = MPI_Isend(stash->sendframes[rankid].buffer,hdr->count,stash->blocktype,rank,tag[0],comm,&req[0]);CHKERRMPI(ierr);
  stash->sendframes[rankid].count = hdr->count;
  stash->sendframes[rankid].pending = 1;
  PetscFunctionReturn(0);
}

/* Callback invoked by target after receiving rendezvous message.
 * Here we post the main recvs.
 */
static PetscErrorCode MatStashBTSRecv_Private(MPI_Comm comm,const PetscMPIInt tag[],PetscMPIInt rank,void *rdata,MPI_Request req[],void *ctx)
{
  MatStash *stash = (MatStash*)ctx;
  MatStashHeader *hdr = (MatStashHeader*)rdata;
  MatStashFrame *frame;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSegBufferGet(stash->segrecvframe,1,&frame);CHKERRQ(ierr);
  ierr = PetscSegBufferGet(stash->segrecvblocks,hdr->count,&frame->buffer);CHKERRQ(ierr);
  ierr = MPI_Irecv(frame->buffer,hdr->count,stash->blocktype,rank,tag[0],comm,&req[0]);CHKERRMPI(ierr);
  frame->count = hdr->count;
  frame->pending = 1;
  PetscFunctionReturn(0);
}

/*
 * owners[] contains the ownership ranges; may be indexed by either blocks or scalars
 */
static PetscErrorCode MatStashScatterBegin_BTS(Mat mat,MatStash *stash,PetscInt owners[])
{
  PetscErrorCode ierr;
  size_t nblocks;
  char *sendblocks;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) { /* make sure all processors are either in INSERTMODE or ADDMODE */
    InsertMode addv;
    ierr = MPIU_Allreduce((PetscEnum*)&mat->insertmode,(PetscEnum*)&addv,1,MPIU_ENUM,MPI_BOR,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
    if (addv == (ADD_VALUES|INSERT_VALUES)) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Some processors inserted others added");
  }

  ierr = MatStashBlockTypeSetUp(stash);CHKERRQ(ierr);
  ierr = MatStashSortCompress_Private(stash,mat->insertmode);CHKERRQ(ierr);
  ierr = PetscSegBufferGetSize(stash->segsendblocks,&nblocks);CHKERRQ(ierr);
  ierr = PetscSegBufferExtractInPlace(stash->segsendblocks,&sendblocks);CHKERRQ(ierr);
  if (stash->first_assembly_done) { /* Set up sendhdrs and sendframes for each rank that we sent before */
    PetscInt i;
    size_t b;
    for (i=0,b=0; i<stash->nsendranks; i++) {
      stash->sendframes[i].buffer = &sendblocks[b*stash->blocktype_size];
      /* sendhdr is never actually sent, but the count is used by MatStashBTSSend_Private */
      stash->sendhdr[i].count = 0; /* Might remain empty (in which case we send a zero-sized message) if no values are communicated to that process */
      for (; b<nblocks; b++) {
        MatStashBlock *sendblock_b = (MatStashBlock*)&sendblocks[b*stash->blocktype_size];
        if (PetscUnlikely(sendblock_b->row < owners[stash->sendranks[i]])) SETERRQ2(stash->comm,PETSC_ERR_ARG_WRONG,"MAT_SUBSET_OFF_PROC_ENTRIES set, but row %D owned by %d not communicated in initial assembly",sendblock_b->row,stash->sendranks[i]);
        if (sendblock_b->row >= owners[stash->sendranks[i]+1]) break;
        stash->sendhdr[i].count++;
      }
    }
  } else {                      /* Dynamically count and pack (first time) */
    PetscInt sendno;
    size_t i,rowstart;

    /* Count number of send ranks and allocate for sends */
    stash->nsendranks = 0;
    for (rowstart=0; rowstart<nblocks;) {
      PetscInt owner;
      MatStashBlock *sendblock_rowstart = (MatStashBlock*)&sendblocks[rowstart*stash->blocktype_size];
      ierr = PetscFindInt(sendblock_rowstart->row,stash->size+1,owners,&owner);CHKERRQ(ierr);
      if (owner < 0) owner = -(owner+2);
      for (i=rowstart+1; i<nblocks; i++) { /* Move forward through a run of blocks with the same owner */
        MatStashBlock *sendblock_i = (MatStashBlock*)&sendblocks[i*stash->blocktype_size];
        if (sendblock_i->row >= owners[owner+1]) break;
      }
      stash->nsendranks++;
      rowstart = i;
    }
    ierr = PetscMalloc3(stash->nsendranks,&stash->sendranks,stash->nsendranks,&stash->sendhdr,stash->nsendranks,&stash->sendframes);CHKERRQ(ierr);

    /* Set up sendhdrs and sendframes */
    sendno = 0;
    for (rowstart=0; rowstart<nblocks;) {
      PetscInt owner;
      MatStashBlock *sendblock_rowstart = (MatStashBlock*)&sendblocks[rowstart*stash->blocktype_size];
      ierr = PetscFindInt(sendblock_rowstart->row,stash->size+1,owners,&owner);CHKERRQ(ierr);
      if (owner < 0) owner = -(owner+2);
      stash->sendranks[sendno] = owner;
      for (i=rowstart+1; i<nblocks; i++) { /* Move forward through a run of blocks with the same owner */
        MatStashBlock *sendblock_i = (MatStashBlock*)&sendblocks[i*stash->blocktype_size];
        if (sendblock_i->row >= owners[owner+1]) break;
      }
      stash->sendframes[sendno].buffer = sendblock_rowstart;
      stash->sendframes[sendno].pending = 0;
      stash->sendhdr[sendno].count = i - rowstart;
      sendno++;
      rowstart = i;
    }
    if (sendno != stash->nsendranks) SETERRQ2(stash->comm,PETSC_ERR_PLIB,"BTS counted %D sendranks, but %D sends",stash->nsendranks,sendno);
  }

  /* Encode insertmode on the outgoing messages. If we want to support more than two options, we would need a new
   * message or a dummy entry of some sort. */
  if (mat->insertmode == INSERT_VALUES) {
    size_t i;
    for (i=0; i<nblocks; i++) {
      MatStashBlock *sendblock_i = (MatStashBlock*)&sendblocks[i*stash->blocktype_size];
      sendblock_i->row = -(sendblock_i->row+1);
    }
  }

  if (stash->first_assembly_done) {
    PetscMPIInt i,tag;
    ierr = PetscCommGetNewTag(stash->comm,&tag);CHKERRQ(ierr);
    for (i=0; i<stash->nrecvranks; i++) {
      ierr = MatStashBTSRecv_Private(stash->comm,&tag,stash->recvranks[i],&stash->recvhdr[i],&stash->recvreqs[i],stash);CHKERRQ(ierr);
    }
    for (i=0; i<stash->nsendranks; i++) {
      ierr = MatStashBTSSend_Private(stash->comm,&tag,i,stash->sendranks[i],&stash->sendhdr[i],&stash->sendreqs[i],stash);CHKERRQ(ierr);
    }
    stash->use_status = PETSC_TRUE; /* Use count from message status. */
  } else {
    ierr = PetscCommBuildTwoSidedFReq(stash->comm,1,MPIU_INT,stash->nsendranks,stash->sendranks,(PetscInt*)stash->sendhdr,
                                      &stash->nrecvranks,&stash->recvranks,(PetscInt*)&stash->recvhdr,1,&stash->sendreqs,&stash->recvreqs,
                                      MatStashBTSSend_Private,MatStashBTSRecv_Private,stash);CHKERRQ(ierr);
    ierr = PetscMalloc2(stash->nrecvranks,&stash->some_indices,stash->nrecvranks,&stash->some_statuses);CHKERRQ(ierr);
    stash->use_status = PETSC_FALSE; /* Use count from header instead of from message. */
  }

  ierr = PetscSegBufferExtractInPlace(stash->segrecvframe,&stash->recvframes);CHKERRQ(ierr);
  stash->recvframe_active     = NULL;
  stash->recvframe_i          = 0;
  stash->some_i               = 0;
  stash->some_count           = 0;
  stash->recvcount            = 0;
  stash->first_assembly_done  = mat->assembly_subset; /* See the same logic in VecAssemblyBegin_MPI_BTS */
  stash->insertmode           = &mat->insertmode;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatStashScatterGetMesg_BTS(MatStash *stash,PetscMPIInt *n,PetscInt **row,PetscInt **col,PetscScalar **val,PetscInt *flg)
{
  PetscErrorCode ierr;
  MatStashBlock *block;

  PetscFunctionBegin;
  *flg = 0;
  while (!stash->recvframe_active || stash->recvframe_i == stash->recvframe_count) {
    if (stash->some_i == stash->some_count) {
      if (stash->recvcount == stash->nrecvranks) PetscFunctionReturn(0); /* Done */
      ierr = MPI_Waitsome(stash->nrecvranks,stash->recvreqs,&stash->some_count,stash->some_indices,stash->use_status?stash->some_statuses:MPI_STATUSES_IGNORE);CHKERRMPI(ierr);
      stash->some_i = 0;
    }
    stash->recvframe_active = &stash->recvframes[stash->some_indices[stash->some_i]];
    stash->recvframe_count = stash->recvframe_active->count; /* From header; maximum count */
    if (stash->use_status) { /* Count what was actually sent */
      ierr = MPI_Get_count(&stash->some_statuses[stash->some_i],stash->blocktype,&stash->recvframe_count);CHKERRMPI(ierr);
    }
    if (stash->recvframe_count > 0) { /* Check for InsertMode consistency */
      block = (MatStashBlock*)&((char*)stash->recvframe_active->buffer)[0];
      if (PetscUnlikely(*stash->insertmode == NOT_SET_VALUES)) *stash->insertmode = block->row < 0 ? INSERT_VALUES : ADD_VALUES;
      if (PetscUnlikely(*stash->insertmode == INSERT_VALUES && block->row >= 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Assembling INSERT_VALUES, but rank %d requested ADD_VALUES",stash->recvranks[stash->some_indices[stash->some_i]]);
      if (PetscUnlikely(*stash->insertmode == ADD_VALUES && block->row < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Assembling ADD_VALUES, but rank %d requested INSERT_VALUES",stash->recvranks[stash->some_indices[stash->some_i]]);
    }
    stash->some_i++;
    stash->recvcount++;
    stash->recvframe_i = 0;
  }
  *n = 1;
  block = (MatStashBlock*)&((char*)stash->recvframe_active->buffer)[stash->recvframe_i*stash->blocktype_size];
  if (block->row < 0) block->row = -(block->row + 1);
  *row = &block->row;
  *col = &block->col;
  *val = block->vals;
  stash->recvframe_i++;
  *flg = 1;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatStashScatterEnd_BTS(MatStash *stash)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Waitall(stash->nsendranks,stash->sendreqs,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);
  if (stash->first_assembly_done) { /* Reuse the communication contexts, so consolidate and reset segrecvblocks  */
    void *dummy;
    ierr = PetscSegBufferExtractInPlace(stash->segrecvblocks,&dummy);CHKERRQ(ierr);
  } else {                      /* No reuse, so collect everything. */
    ierr = MatStashScatterDestroy_BTS(stash);CHKERRQ(ierr);
  }

  /* Now update nmaxold to be app 10% more than max n used, this way the
     wastage of space is reduced the next time this stash is used.
     Also update the oldmax, only if it increases */
  if (stash->n) {
    PetscInt bs2     = stash->bs*stash->bs;
    PetscInt oldnmax = ((int)(stash->n * 1.1) + 5)*bs2;
    if (oldnmax > stash->oldnmax) stash->oldnmax = oldnmax;
  }

  stash->nmax       = 0;
  stash->n          = 0;
  stash->reallocs   = -1;
  stash->nprocessed = 0;

  ierr = PetscMatStashSpaceDestroy(&stash->space_head);CHKERRQ(ierr);

  stash->space = NULL;

  PetscFunctionReturn(0);
}

PetscErrorCode MatStashScatterDestroy_BTS(MatStash *stash)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSegBufferDestroy(&stash->segsendblocks);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&stash->segrecvframe);CHKERRQ(ierr);
  stash->recvframes = NULL;
  ierr = PetscSegBufferDestroy(&stash->segrecvblocks);CHKERRQ(ierr);
  if (stash->blocktype != MPI_DATATYPE_NULL) {
    ierr = MPI_Type_free(&stash->blocktype);CHKERRMPI(ierr);
  }
  stash->nsendranks = 0;
  stash->nrecvranks = 0;
  ierr = PetscFree3(stash->sendranks,stash->sendhdr,stash->sendframes);CHKERRQ(ierr);
  ierr = PetscFree(stash->sendreqs);CHKERRQ(ierr);
  ierr = PetscFree(stash->recvreqs);CHKERRQ(ierr);
  ierr = PetscFree(stash->recvranks);CHKERRQ(ierr);
  ierr = PetscFree(stash->recvhdr);CHKERRQ(ierr);
  ierr = PetscFree2(stash->some_indices,stash->some_statuses);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
