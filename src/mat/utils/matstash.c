#define PETSCMAT_DLL

#include "src/mat/matimpl.h"

/*
       The input to the stash is ALWAYS in MatScalar precision, and the 
    internal storage and output is also in MatScalar.
*/
#define DEFAULT_STASH_SIZE   10000

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
#undef __FUNCT__  
#define __FUNCT__ "MatStashCreate_Private"
PetscErrorCode MatStashCreate_Private(MPI_Comm comm,PetscInt bs,MatStash *stash)
{
  PetscErrorCode ierr;
  PetscInt       max,*opt,nopt;
  PetscTruth     flg;

  PetscFunctionBegin;
  /* Require 2 tags,get the second using PetscCommGetNewTag() */
  stash->comm = comm;
  ierr = PetscCommGetNewTag(stash->comm,&stash->tag1);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(stash->comm,&stash->tag2);CHKERRQ(ierr);
  ierr = MPI_Comm_size(stash->comm,&stash->size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(stash->comm,&stash->rank);CHKERRQ(ierr);

  nopt = stash->size;
  ierr = PetscMalloc(nopt*sizeof(PetscInt),&opt);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(PETSC_NULL,"-matstash_initial_size",opt,&nopt,&flg);CHKERRQ(ierr);
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

  stash->bs       = bs;
  stash->nmax     = 0;
  stash->oldnmax  = 0;
  stash->n        = 0;
  stash->reallocs = -1;
  stash->idx      = 0;
  stash->idy      = 0;
  stash->array    = 0;

  stash->send_waits  = 0;
  stash->recv_waits  = 0;
  stash->send_status = 0;
  stash->nsends      = 0;
  stash->nrecvs      = 0;
  stash->svalues     = 0;
  stash->rvalues     = 0;
  stash->rindices    = 0;
  stash->nprocs      = 0;
  stash->nprocessed  = 0;
  PetscFunctionReturn(0);
}

/* 
   MatStashDestroy_Private - Destroy the stash
*/
#undef __FUNCT__  
#define __FUNCT__ "MatStashDestroy_Private"
PetscErrorCode MatStashDestroy_Private(MatStash *stash)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stash->array) {
    ierr = PetscFree(stash->array);CHKERRQ(ierr);
    stash->array = 0;
  }
  PetscFunctionReturn(0);
}

/* 
   MatStashScatterEnd_Private - This is called as the fial stage of
   scatter. The final stages of messagepassing is done here, and
   all the memory used for messagepassing is cleanedu up. This
   routine also resets the stash, and deallocates the memory used
   for the stash. It also keeps track of the current memory usage
   so that the same value can be used the next time through.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatStashScatterEnd_Private"
PetscErrorCode MatStashScatterEnd_Private(MatStash *stash)
{ 
  PetscErrorCode ierr;
  int         nsends=stash->nsends,bs2,oldnmax;
  MPI_Status  *send_status;

  PetscFunctionBegin;
  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc(2*nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(2*nsends,stash->send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }

  /* Now update nmaxold to be app 10% more than max n used, this way the
     wastage of space is reduced the next time this stash is used.
     Also update the oldmax, only if it increases */
  if (stash->n) {
    bs2      = stash->bs*stash->bs;
    oldnmax  = ((int)(stash->n * 1.1) + 5)*bs2;
    if (oldnmax > stash->oldnmax) stash->oldnmax = oldnmax;
  }

  stash->nmax       = 0;
  stash->n          = 0;
  stash->reallocs   = -1;
  stash->nprocessed = 0;

  if (stash->array) {
    ierr         = PetscFree(stash->array);CHKERRQ(ierr);
    stash->array = 0;
    stash->idx   = 0;
    stash->idy   = 0;
  }
  if (stash->send_waits) {
    ierr = PetscFree(stash->send_waits);CHKERRQ(ierr);
    stash->send_waits = 0;
  }
  if (stash->recv_waits) {
    ierr = PetscFree(stash->recv_waits);CHKERRQ(ierr);
    stash->recv_waits = 0;
  } 
  if (stash->svalues) {
    ierr = PetscFree(stash->svalues);CHKERRQ(ierr);
    stash->svalues = 0;
  }
  if (stash->rvalues) {
    ierr = PetscFree(stash->rvalues);CHKERRQ(ierr);
    stash->rvalues = 0;
  }
  if (stash->rindices) {
    ierr = PetscFree(stash->rindices);CHKERRQ(ierr);
    stash->rindices = 0;
  }
  if (stash->nprocs) {
    ierr = PetscFree(stash->nprocs);CHKERRQ(ierr);
    stash->nprocs = 0;
  }

  PetscFunctionReturn(0);
}

/* 
   MatStashGetInfo_Private - Gets the relavant statistics of the stash

   Input Parameters:
   stash    - the stash
   nstash   - the size of the stash. Indicates the number of values stored.
   reallocs - the number of additional mallocs incurred.
   
*/
#undef __FUNCT__  
#define __FUNCT__ "MatStashGetInfo_Private"
PetscErrorCode MatStashGetInfo_Private(MatStash *stash,PetscInt *nstash,PetscInt *reallocs)
{
  PetscInt bs2 = stash->bs*stash->bs;

  PetscFunctionBegin;
  if (nstash) *nstash   = stash->n*bs2;
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
#undef __FUNCT__  
#define __FUNCT__ "MatStashSetInitialSize_Private"
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
#undef __FUNCT__  
#define __FUNCT__ "MatStashExpand_Private"
static PetscErrorCode MatStashExpand_Private(MatStash *stash,PetscInt incr)
{ 
  PetscErrorCode ierr;
  PetscInt       *n_idx,*n_idy,newnmax,bs2;
  MatScalar *n_array;

  PetscFunctionBegin;
  /* allocate a larger stash */
  bs2     = stash->bs*stash->bs; 
  if (!stash->oldnmax && !stash->nmax) { /* new stash */
    if (stash->umax)                  newnmax = stash->umax/bs2;             
    else                              newnmax = DEFAULT_STASH_SIZE/bs2;
  } else if (!stash->nmax) { /* resuing stash */ 
    if (stash->umax > stash->oldnmax) newnmax = stash->umax/bs2;
    else                              newnmax = stash->oldnmax/bs2;
  } else                              newnmax = stash->nmax*2;
  if (newnmax  < (stash->nmax + incr)) newnmax += 2*incr;

  ierr  = PetscMalloc((newnmax)*(2*sizeof(PetscInt)+bs2*sizeof(MatScalar)),&n_array);CHKERRQ(ierr);
  n_idx = (PetscInt*)(n_array + bs2*newnmax);
  n_idy = (PetscInt*)(n_idx + newnmax);
  ierr  = PetscMemcpy(n_array,stash->array,bs2*stash->nmax*sizeof(MatScalar));CHKERRQ(ierr);
  ierr  = PetscMemcpy(n_idx,stash->idx,stash->nmax*sizeof(PetscInt));CHKERRQ(ierr);
  ierr  = PetscMemcpy(n_idy,stash->idy,stash->nmax*sizeof(PetscInt));CHKERRQ(ierr);
  if (stash->array) {ierr = PetscFree(stash->array);CHKERRQ(ierr);}
  stash->array   = n_array; 
  stash->idx     = n_idx; 
  stash->idy     = n_idy;
  stash->nmax    = newnmax;
  stash->reallocs++;
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
#undef __FUNCT__  
#define __FUNCT__ "MatStashValuesRow_Private"
PetscErrorCode MatStashValuesRow_Private(MatStash *stash,PetscInt row,PetscInt n,const PetscInt idxn[],const MatScalar values[])
{
  PetscErrorCode ierr;
  PetscInt i; 

  PetscFunctionBegin;
  /* Check and see if we have sufficient memory */
  if ((stash->n + n) > stash->nmax) {
    ierr = MatStashExpand_Private(stash,n);CHKERRQ(ierr);
  }
  for (i=0; i<n; i++) {
    stash->idx[stash->n]   = row;
    stash->idy[stash->n]   = idxn[i];
    stash->array[stash->n] = values[i];
    stash->n++;
  }
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
#undef __FUNCT__  
#define __FUNCT__ "MatStashValuesCol_Private"
PetscErrorCode MatStashValuesCol_Private(MatStash *stash,PetscInt row,PetscInt n,const PetscInt idxn[],const MatScalar values[],PetscInt stepval)
{
  PetscErrorCode ierr;
  PetscInt i; 

  PetscFunctionBegin;
  /* Check and see if we have sufficient memory */
  if ((stash->n + n) > stash->nmax) {
    ierr = MatStashExpand_Private(stash,n);CHKERRQ(ierr);
  }
  for (i=0; i<n; i++) {
    stash->idx[stash->n]   = row;
    stash->idy[stash->n]   = idxn[i];
    stash->array[stash->n] = values[i*stepval];
    stash->n++;
  }
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
  cmax   - the number of block-columsn on the original block.
  idx    - the index of the current block-row in the original block.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatStashValuesRowBlocked_Private"
PetscErrorCode MatStashValuesRowBlocked_Private(MatStash *stash,PetscInt row,PetscInt n,const PetscInt idxn[],const MatScalar values[],PetscInt rmax,PetscInt cmax,PetscInt idx)
{
  PetscErrorCode ierr;
  PetscInt i,j,k,bs2,bs=stash->bs; 
  const MatScalar *vals;
  MatScalar       *array;

  PetscFunctionBegin;
  bs2 = bs*bs;
  if ((stash->n+n) > stash->nmax) {
    ierr = MatStashExpand_Private(stash,n);CHKERRQ(ierr);
  }
  for (i=0; i<n; i++) {
    stash->idx[stash->n]   = row;
    stash->idy[stash->n] = idxn[i];
    /* Now copy over the block of values. Store the values column oriented.
       This enables inserting multiple blocks belonging to a row with a single
       funtion call */
    array = stash->array + bs2*stash->n;
    vals  = values + idx*bs2*n + bs*i;
    for (j=0; j<bs; j++) {
      for (k=0; k<bs; k++) {array[k*bs] = vals[k];}
      array += 1;
      vals  += cmax*bs;
    }
    stash->n++;
  }
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
  cmax   - the number of block-columsn on the original block.
  idx    - the index of the current block-row in the original block.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatStashValuesColBlocked_Private"
PetscErrorCode MatStashValuesColBlocked_Private(MatStash *stash,PetscInt row,PetscInt n,const PetscInt idxn[],const MatScalar values[],PetscInt rmax,PetscInt cmax,PetscInt idx)
{
  PetscErrorCode ierr;
  PetscInt i,j,k,bs2,bs=stash->bs; 
  const MatScalar *vals;
  MatScalar       *array;

  PetscFunctionBegin;
  bs2 = bs*bs;
  if ((stash->n+n) > stash->nmax) {
    ierr = MatStashExpand_Private(stash,n);CHKERRQ(ierr);
  }
  for (i=0; i<n; i++) {
    stash->idx[stash->n]   = row;
    stash->idy[stash->n] = idxn[i];
    /* Now copy over the block of values. Store the values column oriented.
     This enables inserting multiple blocks belonging to a row with a single
     funtion call */
    array = stash->array + bs2*stash->n;
    vals  = values + idx*bs + bs2*rmax*i;
    for (j=0; j<bs; j++) {
      for (k=0; k<bs; k++) {array[k] = vals[k];}
      array += bs;
      vals  += rmax*bs;
    }
    stash->n++;
  }
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

  Notes: The 'owners' array in the cased of the blocked-stash has the 
  ranges specified blocked global indices, and for the regular stash in
  the proper global indices.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatStashScatterBegin_Private"
PetscErrorCode MatStashScatterBegin_Private(MatStash *stash,PetscInt *owners)
{ 
  PetscInt       *owner,*startv,*starti,tag1=stash->tag1,tag2=stash->tag2,bs2;
  PetscInt       size=stash->size,nsends;
  PetscErrorCode ierr;
  PetscInt       count,*sindices,**rindices,i,j,idx,lastidx;
  MatScalar      **rvalues,*svalues;
  MPI_Comm       comm = stash->comm;
  MPI_Request    *send_waits,*recv_waits,*recv_waits1,*recv_waits2;
  PetscMPIInt    *nprocs,*nlengths,nreceives;

  PetscFunctionBegin;

  bs2   = stash->bs*stash->bs;
  /*  first count number of contributors to each processor */
  ierr  = PetscMalloc(2*size*sizeof(PetscMPIInt),&nprocs);CHKERRQ(ierr);
  ierr  = PetscMemzero(nprocs,2*size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr  = PetscMalloc((stash->n+1)*sizeof(PetscInt),&owner);CHKERRQ(ierr);

  nlengths = nprocs+size;
  j        = 0;
  lastidx  = -1;
  for (i=0; i<stash->n; i++) {
    /* if indices are NOT locally sorted, need to start search at the beginning */
    if (lastidx > (idx = stash->idx[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nlengths[j]++; owner[i] = j; break;
      }
    }
  }
  /* Now check what procs get messages - and compute nsends. */
  for (i=0, nsends=0 ; i<size; i++) { 
    if (nlengths[i]) { nprocs[i] = 1; nsends ++;}
  }

  { int  *onodes,*olengths;
  /* Determine the number of messages to expect, their lengths, from from-ids */
  ierr = PetscGatherNumberOfMessages(comm,nprocs,nlengths,&nreceives);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nsends,nreceives,nlengths,&onodes,&olengths);CHKERRQ(ierr);
  /* since clubbing row,col - lengths are multiplied by 2 */
  for (i=0; i<nreceives; i++) olengths[i] *=2;
  ierr = PetscPostIrecvInt(comm,tag1,nreceives,onodes,olengths,&rindices,&recv_waits1);CHKERRQ(ierr);
  /* values are size 'bs2' lengths (and remove earlier factor 2 */
  for (i=0; i<nreceives; i++) olengths[i] = olengths[i]*bs2/2;
  ierr = PetscPostIrecvScalar(comm,tag2,nreceives,onodes,olengths,&rvalues,&recv_waits2);CHKERRQ(ierr);
  ierr = PetscFree(onodes);CHKERRQ(ierr);
  ierr = PetscFree(olengths);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  ierr     = PetscMalloc((stash->n+1)*(bs2*sizeof(MatScalar)+2*sizeof(PetscInt)),&svalues);CHKERRQ(ierr);
  sindices = (PetscInt*)(svalues + bs2*stash->n);
  ierr     = PetscMalloc(2*(nsends+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  ierr     = PetscMalloc(2*size*sizeof(PetscInt),&startv);CHKERRQ(ierr);
  starti   = startv + size;
  /* use 2 sends the first with all_a, the next with all_i and all_j */
  startv[0]  = 0; starti[0] = 0;
  for (i=1; i<size; i++) { 
    startv[i] = startv[i-1] + nlengths[i-1];
    starti[i] = starti[i-1] + nlengths[i-1]*2;
  } 
  for (i=0; i<stash->n; i++) {
    j = owner[i];
    if (bs2 == 1) {
      svalues[startv[j]]              = stash->array[i];
    } else {
      PetscInt       k;
      MatScalar *buf1,*buf2;
      buf1 = svalues+bs2*startv[j];
      buf2 = stash->array+bs2*i;
      for (k=0; k<bs2; k++){ buf1[k] = buf2[k]; }
    }
    sindices[starti[j]]               = stash->idx[i];
    sindices[starti[j]+nlengths[j]]   = stash->idy[i];
    startv[j]++;
    starti[j]++;
  }
  startv[0] = 0;
  for (i=1; i<size; i++) { startv[i] = startv[i-1] + nlengths[i-1];} 

  for (i=0,count=0; i<size; i++) {
    if (nprocs[i]) {
      ierr = MPI_Isend(sindices+2*startv[i],2*nlengths[i],MPIU_INT,i,tag1,comm,send_waits+count++);CHKERRQ(ierr);
      ierr = MPI_Isend(svalues+bs2*startv[i],bs2*nlengths[i],MPIU_MATSCALAR,i,tag2,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
#if defined(PETSC_USE_VERBOSE)
  ierr = PetscLogInfo((0,"MatStashScatterBegin_Private: No of messages: %d \n",nsends));CHKERRQ(ierr);
  for (i=0; i<size; i++) {
    if (nprocs[i]) {
      ierr = PetscLogInfo((0,"MatStashScatterBegin_Private: Mesg_to: %d: size: %d \n",i,nlengths[i]*bs2*sizeof(MatScalar)+2*sizeof(PetscInt)));CHKERRQ(ierr);
    }
  }
#endif
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(startv);CHKERRQ(ierr);
  /* This memory is reused in scatter end  for a different purpose*/
  for (i=0; i<2*size; i++) nprocs[i] = -1;
  stash->nprocs      = nprocs;
  
  /* recv_waits need to be contiguous for MatStashScatterGetMesg_Private() */
  ierr  = PetscMalloc((nreceives+1)*2*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);

  for (i=0; i<nreceives; i++) { 
    recv_waits[2*i]   = recv_waits1[i];
    recv_waits[2*i+1] = recv_waits2[i];
  }
  stash->recv_waits = recv_waits;
  ierr = PetscFree(recv_waits1);CHKERRQ(ierr);
  ierr = PetscFree(recv_waits2);CHKERRQ(ierr);

  stash->svalues    = svalues;    stash->rvalues     = rvalues;
  stash->rindices   = rindices;   stash->send_waits  = send_waits;
  stash->nsends     = nsends;     stash->nrecvs      = nreceives;
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
#undef __FUNCT__  
#define __FUNCT__ "MatStashScatterGetMesg_Private"
PetscErrorCode MatStashScatterGetMesg_Private(MatStash *stash,PetscMPIInt *nvals,PetscInt **rows,PetscInt** cols,MatScalar **vals,PetscInt *flg)
{
  PetscErrorCode ierr;
  PetscMPIInt    i,*flg_v,i1,i2;
  PetscInt       bs2;
  MPI_Status     recv_status;
  PetscTruth     match_found = PETSC_FALSE;

  PetscFunctionBegin;

  *flg = 0; /* When a message is discovered this is reset to 1 */
  /* Return if no more messages to process */
  if (stash->nprocessed == stash->nrecvs) { PetscFunctionReturn(0); } 

  flg_v = stash->nprocs;
  bs2   = stash->bs*stash->bs;
  /* If a matching pair of receieves are found, process them, and return the data to
     the calling function. Until then keep receiving messages */
  while (!match_found) {
    ierr = MPI_Waitany(2*stash->nrecvs,stash->recv_waits,&i,&recv_status);CHKERRQ(ierr);
    /* Now pack the received message into a structure which is useable by others */
    if (i % 2) { 
      ierr = MPI_Get_count(&recv_status,MPIU_MATSCALAR,nvals);CHKERRQ(ierr);
      flg_v[2*recv_status.MPI_SOURCE] = i/2; 
      *nvals = *nvals/bs2; 
    } else { 
      ierr = MPI_Get_count(&recv_status,MPIU_INT,nvals);CHKERRQ(ierr);
      flg_v[2*recv_status.MPI_SOURCE+1] = i/2; 
      *nvals = *nvals/2; /* This message has both row indices and col indices */
    }
    
    /* Check if we have both the messages from this proc */
    i1 = flg_v[2*recv_status.MPI_SOURCE];
    i2 = flg_v[2*recv_status.MPI_SOURCE+1];
    if (i1 != -1 && i2 != -1) {
      *rows       = stash->rindices[i2];
      *cols       = *rows + *nvals;
      *vals       = stash->rvalues[i1];
      *flg        = 1;
      stash->nprocessed ++;
      match_found = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}
