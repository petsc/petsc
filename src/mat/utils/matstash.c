#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: stash.c,v 1.21 1999/03/09 21:33:15 balay Exp balay $";
#endif

#include "src/mat/matimpl.h"

#define DEFAULT_STASH_SIZE   10000
/*
   This stash is currently used for all the parallel matrix implementations.
   The stash is where elements of a matrix destined to be stored on other 
   processors are kept until matrix assembly is done.

   This is a simple minded stash. Simply add entry to end of stash.
*/

#undef __FUNC__  
#define __FUNC__ "StashCreate_Private"
int StashCreate_Private(MPI_Comm comm,int bs_stash, int bs_mat, Stash *stash)
{
  int ierr,flg,max=DEFAULT_STASH_SIZE;

  PetscFunctionBegin;
  /* Require 2 tags, get the second using PetscCommGetNewTag() */
  ierr = PetscCommDuplicate_Private(comm,&stash->comm,&stash->tag1);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(stash->comm,&stash->tag2); CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-stash_initial_size",&max,&flg);CHKERRQ(ierr);
  ierr = StashSetInitialSize_Private(stash,max); CHKERRQ(ierr);
  ierr = MPI_Comm_size(stash->comm,&stash->size); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(stash->comm,&stash->rank); CHKERRQ(ierr);

  if (bs_stash <= 0) bs_stash = 1;
  if (bs_mat   <= 0) bs_mat   = 1;

  stash->bs_stash = bs_stash;
  stash->bs_mat   = bs_mat;
  stash->nmax     = 0;
  stash->n        = 0;
  stash->reallocs = 0;
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
  stash->rmax        = 0;
  stash->nprocs      = 0;
  stash->nprocessed  = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "StashDestroy_Private"
int StashDestroy_Private(Stash *stash)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscCommDestroy_Private(&stash->comm); CHKERRQ(ierr);
  if (stash->array) {PetscFree(stash->array); stash->array = 0;}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "StashScatterEnd_Private"
int StashScatterEnd_Private(Stash *stash)
{ 
  int         nsends=stash->nsends,ierr;
  MPI_Status  *send_status;

  PetscFunctionBegin;
  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *)PetscMalloc(2*nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    ierr        = MPI_Waitall(2*nsends,stash->send_waits,send_status);CHKERRQ(ierr);
    PetscFree(send_status);
  }

  /* Now update nmaxold to be app 10% more than nmax, this way the
     wastage of space is reduced the next time this stash is used */
  stash->oldnmax    = (int)(stash->nmax * 1.1) + 5;
  stash->nmax       = 0;
  stash->n          = 0;
  stash->reallocs   = 0;
  stash->rmax       = 0;
  stash->nprocessed = 0;

  if (stash->array) {
    PetscFree(stash->array); 
    stash->array = 0;
    stash->idx   = 0;
    stash->idy   = 0;
  }
  if (stash->send_waits)  {PetscFree(stash->send_waits);stash->send_waits = 0;}
  if (stash->recv_waits)  {PetscFree(stash->recv_waits);stash->recv_waits = 0;} 
  if (stash->svalues)     {PetscFree(stash->svalues);stash->svalues = 0;}
  if (stash->rvalues)     {PetscFree(stash->rvalues); stash->rvalues = 0;}
  if (stash->nprocs)      {PetscFree(stash->nprocs); stash->nprocs = 0;}

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "StashInfo_Private"
int StashInfo_Private(Stash *stash)
{
  PetscFunctionBegin;
  PLogInfo(0,"StashInfo_Private:Stash size %d, mallocs incured %d\n",stash->n,stash->reallocs);
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ "StashSetInitialSize_Private"
int StashSetInitialSize_Private(Stash *stash,int max)
{
  PetscFunctionBegin;
  stash->oldnmax = max;
  stash->nmax    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "StashExpand_Private"
static int StashExpand_Private(Stash *stash)
{ 
  int    *n_idx,*n_idy,newnmax,bs2;
  Scalar *n_array;

  PetscFunctionBegin;
  /* allocate a larger stash */
  if (stash->nmax == 0) newnmax = stash->oldnmax;
  else                  newnmax = stash->nmax*2;
  
  bs2     = stash->bs_stash*stash->bs_stash; 
  n_array = (Scalar *)PetscMalloc((newnmax)*(2*sizeof(int)+bs2*sizeof(Scalar)));CHKPTRQ(n_array);
  n_idx   = (int *) (n_array + bs2*newnmax);
  n_idy   = (int *) (n_idx + newnmax);
  PetscMemcpy(n_array,stash->array,bs2*stash->nmax*sizeof(Scalar));
  PetscMemcpy(n_idx,stash->idx,stash->nmax*sizeof(int));
  PetscMemcpy(n_idy,stash->idy,stash->nmax*sizeof(int));
  if (stash->array) PetscFree(stash->array);
  stash->array   = n_array; 
  stash->idx     = n_idx; 
  stash->idy     = n_idy;
  stash->nmax    = newnmax;
  stash->oldnmax = newnmax;
  stash->reallocs++;
  PetscFunctionReturn(0);
}

/* 
    Should do this properly. With a sorted array.
*/
#undef __FUNC__  
#define __FUNC__ "StashValues_Private"
int StashValues_Private(Stash *stash,int row,int n, int *idxn,Scalar *values)
{
  int    ierr,i; 
  Scalar val;

  PetscFunctionBegin;
  for ( i=0; i<n; i++ ) {
    if ( stash->n == stash->nmax ) {
      ierr = StashExpand_Private(stash); CHKERRQ(ierr);
    }
    stash->idx[stash->n]   = row;
    stash->idy[stash->n]   = idxn[i];
    stash->array[stash->n] = values[i];
    stash->n++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "StashValuesBlocked_Private"
int StashValuesBlocked_Private(Stash *stash,int row,int n,int *idxn,Scalar *values,
                               int rmax,int cmax,int roworiented,int idx)
{
  int    ierr,i,j,k,found,bs2,bs=stash->bs_stash; 
  Scalar *vals,*array;

  /* stepval gives the offset that one should use to access the next line of 
     a given block of values */

  PetscFunctionBegin;
  bs2 = bs*bs;
  for ( i=0; i<n; i++ ) {
    if ( stash->n == stash->nmax ) {
      ierr = StashExpand_Private(stash); CHKERRQ(ierr);
    }
    stash->idx[stash->n]   = row;
    stash->idy[stash->n] = idxn[i];
    /* Now copy over the block of values. Store the values column oriented.
     This enables inserting multiple blocks belonging to a row with a single
     funtion call */
    if (roworiented) {
      array = stash->array + bs2*stash->n;
      vals  = values + idx*bs2*n + bs*i;
      for ( j=0; j<bs; j++ ) {
        for ( k=0; k<bs; k++ ) {array[k*bs] = vals[k];}
        array += 1;
        vals  += cmax*bs;
      }
    } else {
      array = stash->array + bs2*stash->n;
      vals  = values + idx*bs + bs2*rmax*i;
      for ( j=0; j<bs; j++ ) {
        for ( k=0; k<bs; k++ ) {array[k] = vals[k];}
        array += bs;
        vals  += rmax*bs;
      }
    }
    stash->n++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "StashScatterBegin_Private"
int StashScatterBegin_Private(Stash *stash,int *owners)
{ 
  int         *owner,*startv,*starti,tag1=stash->tag1,tag2=stash->tag2,bs2;
  int         rank=stash->rank,size=stash->size,*nprocs,*procs,nsends,nreceives;
  int         nmax,*work,count,ierr,*sindices,*rindices,i,j,idx,mult;
  Scalar      *rvalues,*svalues;
  MPI_Comm    comm = stash->comm;
  MPI_Request *send_waits,*recv_waits;

  PetscFunctionBegin;

  bs2 = stash->bs_stash*stash->bs_stash;
  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc( (stash->n+1)*sizeof(int) ); CHKPTRQ(owner);

  /* if blockstash, then the the owners and row,col indices 
   correspond to reduced indices */
  if (stash->bs_stash == 1) mult = stash->bs_mat;
  else                      mult = 1;

  for ( i=0; i<stash->n; i++ ) {
    idx = stash->idx[i];
    for ( j=0; j<size; j++ ) {
      if (idx >= mult*owners[j] && idx < mult*owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 
  
  /* inform other processors of number of messages and max length*/
  work = (int *)PetscMalloc(size*sizeof(int)); CHKPTRQ(work);
  ierr = MPI_Allreduce(procs,work,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  nreceives = work[rank]; 
  ierr = MPI_Allreduce(nprocs,work,size,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
  nmax = work[rank];
  PetscFree(work);
  /* post receives: 
     since we don't know how long each individual message is we 
     allocate the largest needed buffer for each receive. Potentially 
     this is a lot of wasted space.
  */
  rvalues    = (Scalar *)PetscMalloc((nreceives+1)*(nmax+1)*(bs2*sizeof(Scalar)+2*sizeof(int)));CHKPTRQ(rvalues);
  rindices   = (int *) (rvalues + bs2*nreceives*nmax);
  recv_waits = (MPI_Request *)PetscMalloc((nreceives+1)*2*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for ( i=0,count=0; i<nreceives; i++ ) {
    ierr = MPI_Irecv(rvalues+bs2*nmax*i,bs2*nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag1,comm,
                     recv_waits+count++); CHKERRQ(ierr);
    ierr = MPI_Irecv(rindices+2*nmax*i,2*nmax,MPI_INT,MPI_ANY_SOURCE,tag2,comm,
                     recv_waits+count++); CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues    = (Scalar *)PetscMalloc((stash->n+1)*(bs2*sizeof(Scalar)+2*sizeof(int)));CHKPTRQ(svalues);
  sindices   = (int *) (svalues + bs2*stash->n);
  send_waits = (MPI_Request *) PetscMalloc(2*(nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  startv     = (int *) PetscMalloc(2*size*sizeof(int) ); CHKPTRQ(startv);
  starti     = startv + size;
  /* use 2 sends the first with all_a, the next with all_i and all_j */
  startv[0]  = 0; starti[0] = 0;
  for ( i=1; i<size; i++ ) { 
    startv[i] = startv[i-1] + nprocs[i-1];
    starti[i] = starti[i-1] + nprocs[i-1]*2;
  } 
  for ( i=0; i<stash->n; i++ ) {
    j = owner[i];
    if (bs2 == 1) {
      svalues[startv[j]]              = stash->array[i];
    } else {
      PetscMemcpy(svalues+bs2*startv[j],stash->array+bs2*i,bs2*sizeof(Scalar));
    }
    sindices[starti[j]]             = stash->idx[i];
    sindices[starti[j]+nprocs[j]]   = stash->idy[i];
    startv[j]++;
    starti[j]++;
  }
  startv[0] = 0;
  for ( i=1; i<size; i++ ) { startv[i] = startv[i-1] + nprocs[i-1];} 
  for ( i=0,count=0; i<size; i++ ) {
    if (procs[i]) {
      ierr = MPI_Isend(svalues+bs2*startv[i],bs2*nprocs[i],MPIU_SCALAR,i,tag1,comm,
                       send_waits+count++);CHKERRQ(ierr);
      ierr = MPI_Isend(sindices+2*startv[i],2*nprocs[i],MPI_INT,i,tag2,comm,
                       send_waits+count++);CHKERRQ(ierr);
    }
  }
  PetscFree(owner);
  PetscFree(startv); 
  /* This memory is reused in scatter end  for a different purpose*/
  for (i=0; i<2*size; i++ ) nprocs[i] = -1;
  stash->nprocs      = nprocs;

  stash->svalues    = svalues;    stash->rvalues    = rvalues;
  stash->nsends     = nsends;     stash->nrecvs     = nreceives;
  stash->send_waits = send_waits; stash->recv_waits = recv_waits;
  stash->rmax       = nmax;
  PetscFunctionReturn(0);
}

/* 
   This function waits on the receives posted in the function
   StashScatterBegin_Private() and returns one message at a time to
   the calling function. If no messages are left, it indicates this by
   setting flg = 0, else it sets flg = 1.
*/
#undef __FUNC__  
#define __FUNC__ "StashScatterGetMesg_Private"
int StashScatterGetMesg_Private(Stash *stash,int *nvals,int **rows,int** cols,Scalar **vals,int *flg)
{
  int         i,ierr,size=stash->size,*flg_v,*flg_i;
  int         i1,i2,*rindices,match_found=0,bs2;
  MPI_Status  recv_status;

  PetscFunctionBegin;

  *flg = 0; /* When a message is discovered this is reset to 1 */
  /* Return if no more messages to process */
  if (stash->nprocessed == stash->nrecvs) { PetscFunctionReturn(0); } 

  flg_v = stash->nprocs;
  flg_i = flg_v + size;
  bs2   = stash->bs_stash*stash->bs_stash;
  /* If a matching pair of receieves are found, process them, and return the data to
     the calling function. Until then keep receiving messages */
  while (!match_found) {
    ierr = MPI_Waitany(2*stash->nrecvs,stash->recv_waits,&i,&recv_status);CHKERRQ(ierr);
    /* Now pack the received message into a structure which is useable by others */
    if (i % 2) { 
      ierr = MPI_Get_count(&recv_status,MPI_INT,nvals);CHKERRQ(ierr);
      flg_i[recv_status.MPI_SOURCE] = i/2; 
      *nvals = *nvals/2; /* This message has both row indices and col indices */
    } else { 
      ierr = MPI_Get_count(&recv_status,MPIU_SCALAR,nvals);CHKERRQ(ierr);
      flg_v[recv_status.MPI_SOURCE] = i/2; 
      *nvals = *nvals/bs2; 
    }
    
    /* Check if we have both the messages from this proc */
    i1 = flg_v[recv_status.MPI_SOURCE];
    i2 = flg_i[recv_status.MPI_SOURCE];
    if (i1 != -1 && i2 != -1) {
      rindices    = (int *) (stash->rvalues + bs2*stash->rmax*stash->nrecvs);
      *rows       = rindices + 2*i2*stash->rmax;
      *cols       = *rows + *nvals;
      *vals       = stash->rvalues + i1*bs2*stash->rmax;
      *flg        = 1;
      stash->nprocessed ++;
      match_found = 1;
    }
  }
  PetscFunctionReturn(0);
}
