#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: stash.c,v 1.20 1999/02/25 22:48:34 balay Exp balay $";
#endif

#include "src/vec/vecimpl.h"
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
int StashCreate_Private(MPI_Comm comm,int bs, Stash *stash)
{
  int ierr,flg,max=DEFAULT_STASH_SIZE;

  PetscFunctionBegin;
  /* Require 2 tags, get the second using PetscCommGetNewTag() */
  ierr = PetscCommDuplicate_Private(comm,&stash->comm,&stash->tag1);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(comm,&stash->tag2); CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-stash_initial_size",&max,&flg);CHKERRQ(ierr);
  ierr = StashSetInitialSize_Private(stash,max); CHKERRQ(ierr);

  if (bs <= 0) bs = 1;
  stash->bs       = bs;
  stash->nmax     = 0;
  stash->n        = 0;
  stash->reallocs = 0;
  stash->idx      = 0;
  stash->idy      = 0;
  stash->array    = 0;

  stash->send_waits = 0;
  stash->recv_waits = 0;
  stash->nsends     = 0;
  stash->nrecvs     = 0;
  stash->svalues    = 0;
  stash->rvalues    = 0;
  stash->rmax       = 0;
  stash->rdata      = 0;
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
#define __FUNC__ "StashReset_Private"
int StashReset_Private(Stash *stash)
{
  PetscFunctionBegin;
  /* Now update nmaxold to be app 10% more than nmax, this way the
     wastage of space is reduced the next time this stash is used */
  stash->oldnmax  = (int)(stash->nmax * 1.1) + 5;
  stash->nmax     = 0;
  stash->n        = 0;
  stash->reallocs = 0;
  stash->rmax     = 0;

  if (stash->array) {
    PetscFree(stash->array); 
    stash->array = 0;
    stash->idx   = 0;
    stash->idy   = 0;
  }
  if (stash->send_waits) {PetscFree(stash->send_waits);stash->send_waits = 0;}
  if (stash->recv_waits) {PetscFree(stash->recv_waits);stash->recv_waits =0;} 
  if (stash->svalues)    {PetscFree(stash->svalues);stash->svalues = 0;}
  if (stash->rvalues)    {PetscFree(stash->rvalues); stash->rvalues = 0;}
  if (stash->rdata)      {PetscFree(stash->rdata); stash->rdata = 0;}

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
  int    *n_idx,*n_idy,newnmax; 
  Scalar *n_array;

  PetscFunctionBegin;
  /* allocate a larger stash */
  if (stash->nmax == 0) newnmax = stash->oldnmax;
  else                  newnmax = stash->nmax *2;
  
  n_array = (Scalar *)PetscMalloc((newnmax)*(2*sizeof(int)+sizeof(Scalar)));CHKPTRQ(n_array);
  n_idx = (int *) (n_array + newnmax);
  n_idy = (int *) (n_idx + newnmax);
  PetscMemcpy(n_array,stash->array,stash->nmax*sizeof(Scalar));
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
int StashValues_Private(Stash *stash,int row,int n, int *idxn,Scalar *values,InsertMode addv)
{
  int    ierr,i,found; 
  Scalar val;

  PetscFunctionBegin;
  for ( i=0; i<n; i++ ) {
    found = 0;
    val = *values++;
    if (!found) { /* not found so add to end */
      if ( stash->n == stash->nmax ) {
        ierr = StashExpand_Private(stash); CHKERRQ(ierr);
      }
      stash->array[stash->n] = val;
      stash->idx[stash->n]   = row;
      stash->idy[stash->n++] = idxn[i];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "StashScatterBegin_Private"
int StashScatterBegin_Private(Stash *stash,int *owners)
{ 
  MPI_Comm    comm = stash->comm;
  MPI_Request *send_waits,*recv_waits;
  Scalar      *rvalues,*svalues;
  int         *owner,*startv,*starti,tag1=stash->tag1,tag2=stash->tag2;
  int         rank,size,*nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  int         count,ierr,*sindices,*rindices,bs=stash->bs;

  PetscFunctionBegin;

  ierr = MPI_Comm_size(comm,&size); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank); CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc( (stash->n+1)*sizeof(int) ); CHKPTRQ(owner);
  for ( i=0; i<stash->n; i++ ) {
    idx = stash->idx[i];
    for ( j=0; j<size; j++ ) {
      if (idx >= bs*owners[j] && idx < bs*owners[j+1]) {
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
  rvalues    = (Scalar *)PetscMalloc((nreceives+1)*(nmax+1)*(sizeof(Scalar)+2*sizeof(int)));
  CHKPTRQ(rvalues);
  rindices   = (int *) (rvalues + nreceives*nmax);
  recv_waits = (MPI_Request *) PetscMalloc((nreceives+1)*2*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0,count=0; i<nreceives; i++ ) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag1,comm,
                     recv_waits+count++); CHKERRQ(ierr);
    ierr = MPI_Irecv(rindices+2*nmax*i,2*nmax,MPI_INT,MPI_ANY_SOURCE,tag2,comm,
                     recv_waits+count++); CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues    = (Scalar *) PetscMalloc((stash->n+1)*(sizeof(Scalar)+2*sizeof(int)));
  CHKPTRQ(svalues);
  sindices   = (int *) (svalues + stash->n);
  send_waits = (MPI_Request *) PetscMalloc(2*(nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  startv     = (int *) PetscMalloc(2*size*sizeof(int) ); CHKPTRQ(startv);
  starti     = startv + size;
  /* use 2 sends the first with all_a, the next with all_i and then all_j */
  startv[0]  = 0; starti[0] = 0;
  for ( i=1; i<size; i++ ) { 
    startv[i] = startv[i-1] + nprocs[i-1];
    starti[i] = starti[i-1] + nprocs[i-1]*2;
  } 
  for ( i=0; i<stash->n; i++ ) {
    j = owner[i];
    svalues[startv[j]]              = stash->array[i];
    sindices[starti[j]]             = stash->idx[i];
    sindices[starti[j]+nprocs[j]]   = stash->idy[i];
    startv[j]++;
    starti[j]++;
  }
  startv[0] = 0;
  for ( i=1; i<size; i++ ) { startv[i] = startv[i-1] + nprocs[i-1];} 
  for ( i=0,count=0; i<size; i++ ) {
    if (procs[i]) {
      ierr = MPI_Isend(svalues+startv[i],nprocs[i],MPIU_SCALAR,i,tag1,comm,
                       send_waits+count++);CHKERRQ(ierr);
      ierr = MPI_Isend(sindices+2*startv[i],2*nprocs[i],MPI_INT,i,tag2,comm,
                       send_waits+count++);CHKERRQ(ierr);
    }
  }
  PetscFree(owner);
  PetscFree(startv); 
  PetscFree(nprocs);
  stash->svalues    = svalues;    stash->rvalues    = rvalues;
  stash->nsends     = nsends;     stash->nrecvs     = nreceives;
  stash->send_waits = send_waits; stash->recv_waits = recv_waits;
  stash->rmax       = nmax;

  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "StashScatterEnd_Private"
int StashScatterEnd_Private(Stash *stash)
{
  int         i,ierr,size,*nprocs;
  MPI_Status  *send_status,*recv_status;
  int         i1,i2,s1,s2,count,*rindices;

  PetscFunctionBegin;

  /* wait on receives */
  ierr = MPI_Comm_size(stash->comm,&size); CHKERRQ(ierr);
  if (stash->nrecvs) {
    recv_status = (MPI_Status *) PetscMalloc(2*(stash->nrecvs+1)*sizeof(MPI_Status));
    CHKPTRQ(recv_status);
    ierr = MPI_Waitall(2*stash->nrecvs,stash->recv_waits,recv_status);CHKERRQ(ierr);
    /* Now pack the received messages into a structure which is useable by others */
    stash->rdata = (Stash_rdata *) PetscMalloc(stash->nrecvs*sizeof(Stash_rdata)); 
    CHKPTRQ(stash->rdata);
    nprocs = (int *) PetscMalloc((2*size+1)*sizeof(int)); CHKPTRQ(nprocs);
    rindices = (int *) (stash->rvalues + stash->rmax*stash->nrecvs);
    for (i=0; i<2*size+1; i++ ) nprocs[i] = -1;
    for ( i=0; i<stash->nrecvs; i++ ){
      nprocs[2*recv_status[2*i].MPI_SOURCE] = i;
      nprocs[2*recv_status[2*i+1].MPI_SOURCE+1] = i;
    }
    for (i=0,count=0; i<size; i++) {
      i1 = nprocs[2*i];
      i2 = nprocs[2*i+1];
      if (i1 != -1) {
        if (i2 == -1) SETERRQ(1,0,"Internal Error");
        ierr = MPI_Get_count(recv_status+2*i1,MPIU_SCALAR,&s1);CHKERRQ(ierr);
        ierr = MPI_Get_count(recv_status+2*i2+1,MPI_INT,&s2);CHKERRQ(ierr);
        if (s1*2 != s2) SETERRQ(1,0,"Internal Error");
        stash->rdata[count].a = stash->rvalues + i1*stash->rmax;
        stash->rdata[count].i = rindices + 2*i2*stash->rmax;
        stash->rdata[count].j = stash->rdata[count].i + s1;
        stash->rdata[count].n = s1;
        count ++;
      }
    } 
    PetscFree(recv_status);
    PetscFree(nprocs);
  }
  /* wait on sends */
  if (stash->nsends) {
    send_status = (MPI_Status *)PetscMalloc(2*stash->nsends*sizeof(MPI_Status));
    CHKPTRQ(send_status);
    ierr        = MPI_Waitall(2*stash->nsends,stash->send_waits,send_status);CHKERRQ(ierr);
    PetscFree(send_status);
  }
  PetscFunctionReturn(0);
}
