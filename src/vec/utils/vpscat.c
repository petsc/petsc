/*
    Does the parallel vector scatter 
*/

#include "comm.h"
#include "is/isimpl.h"
#include "vecimpl.h"                     /*I "vec.h" I*/
#include "impls/dvecimpl.h"
#include "impls/mpi/pvecimpl.h"

int PrintPVecScatterCtx(VecScatterCtx ctx)
{
  VecScatterMPI *to = (VecScatterMPI *) ctx->todata;
  VecScatterMPI *from = (VecScatterMPI *) ctx->fromdata;
  int           i,mytid;

  MPI_Comm_rank(to->comm,&mytid);

  fprintf(stderr,"[%d]Number sends %d\n",mytid,to->n);
  fprintf(stderr,"[%d]Inital start should be zero %d\n",mytid,to->starts[0]);
  for ( i=0; i<to->n; i++ ){
    fprintf(stderr,"[%d]start %d %d\n",mytid,i,to->starts[i+1]);
    fprintf(stderr,"[%d]proc %d\n",mytid,to->procs[i]);
  }
  fprintf(stderr,"Now the indices\n");
  for ( i=0; i<to->starts[to->n]; i++ ){
    fprintf(stderr,"[%d]%d \n",mytid,to->indices[i]);
  }
  fprintf(stderr,"[%d]Number receives %d\n",mytid,from->n);
  fprintf(stderr,"[%d]Inital start should be zero %d\n",mytid,from->starts[0]);
  for ( i=0; i<from->n; i++ ){
    fprintf(stderr,"[%d]start %d %d\n",mytid,i,from->starts[i+1]);
    fprintf(stderr,"[%d]proc %d\n",mytid,from->procs[i]);
  }
  fprintf(stderr,"Now the indices\n");
  for ( i=0; i<from->starts[from->n]; i++ ){
    fprintf(stderr,"[%d]%d \n",mytid,from->indices[i]);
  }
  return 0;
}  
/*
     Even though the next routines are written with parallel 
  vectors, either xin or yin (but not both) may be sequential
  vectors, one for each processor.

     Note: since nsends, nrecvs and nx may be zero but they are used
  in mallocs, we always malloc the quantity plus one. This is not 
  an ideal solution, but it insures that we never try to malloc and 
  then free a zero size location.
  
     gen_from indices indicate where arriaving stuff is stashed
     gen_to   indices indicate where departing stuff came from. 
     the naming can be a little confusing.

*/
int PtoPScatterbegin(Vec xin,Vec yin,VecScatterCtx ctx,InsertMode addv)
{
  VecScatterMPI *gen_to = (VecScatterMPI *) ctx->todata;
  VecScatterMPI *gen_from = (VecScatterMPI *) ctx->fromdata;
  DvPVector     *x = (DvPVector *)xin->data;
  DvPVector     *y = (DvPVector *)yin->data;
  MPI_Comm      comm = gen_from->comm;
  Scalar        *rvalues = gen_from->values,*svalues = gen_to->values;
  int           nrecvs = gen_from->n, nsends = gen_to->n;
  MPI_Request   *rwaits = gen_from->requests, *swaits = gen_to->requests;
  Scalar        *xv = x->array,*val;
  int           tag = 23, i,j,*indices = gen_to->indices;
  int           *rstarts = gen_from->starts,*sstarts = gen_to->starts;
  int           *rprocs = gen_from->procs, *sprocs = gen_to->procs;

  /* post receives:   */
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv((void *)(rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
              MPI_SCALAR,rprocs[i],tag,comm,rwaits+i);
  }

  /* do sends:  */
  for ( i=0; i<nsends; i++ ) {
    val = svalues + sstarts[i];
    for ( j=0; j<sstarts[i+1]-sstarts[i]; j++ ) {
      val[j] = xv[*indices++];
    }
    MPI_Isend((void*)val,sstarts[i+1] - sstarts[i],
                MPI_SCALAR,sprocs[i],tag,comm,swaits+i);

  }
  return 0;
}

int PtoPScatterend(Vec xin,Vec yin,VecScatterCtx ctx,InsertMode addv)
{
  VecScatterMPI *gen_to = (VecScatterMPI *) ctx->todata;
  VecScatterMPI *gen_from = (VecScatterMPI *) ctx->fromdata;
  DvPVector     *x = (DvPVector *)xin->data;
  DvPVector     *y = (DvPVector *)yin->data;
  MPI_Comm      comm = gen_from->comm;
  Scalar        *rvalues = gen_from->values,*svalues = gen_to->values;
  int           nrecvs = gen_from->n, nsends = gen_to->n;
  MPI_Request   *rwaits = gen_from->requests, *swaits = gen_to->requests;
  Scalar        *yv = y->array,*val;
  int           tag = 23, i,j,*indices = gen_from->indices,count,index,n;
  MPI_Status    rstatus,*sstatus;
  int           *rstarts = gen_from->starts,*sstarts = gen_to->starts;
  int           *rprocs = gen_from->procs, *sprocs = gen_to->procs;

  /*  wait on receives */
  count = nrecvs;
  while (count) {
    MPI_Waitany(nrecvs,rwaits,&index,&rstatus);
    /* unpack receives into our local space */
    val = rvalues + rstarts[index];
    MPI_Get_count(&rstatus,MPI_SCALAR,&n);
    if (n != rstarts[index+1] - rstarts[index]) SETERR(1,"Bad message");

    if (addv == InsertValues) {
      for ( i=0; i<n; i++ ) {
        yv[indices[i+rstarts[index]]] = *val++;
      }
    }
     else {
      for ( i=0; i<n; i++ ) {
        yv[indices[i+rstarts[index]]] += *val++;
      }
    }
    count--;
  }
 
  /* wait on sends */
  if (nsends) {
    sstatus = (MPI_Status *) MALLOC(nsends*sizeof(MPI_Status));
    CHKPTR(sstatus);
    MPI_Waitall(nsends,swaits,sstatus);
    FREE(sstatus);
  }
  return 0;
}

int PtoPScatterDestroy(PetscObject obj)
{
  VecScatterCtx ctx = (VecScatterCtx) obj;
  VecScatterMPI *gen_to   = (VecScatterMPI *) ctx->todata;
  VecScatterMPI *gen_from = (VecScatterMPI *) ctx->fromdata;
  FREE(gen_to->values); FREE(gen_to);
  FREE(gen_from->values); FREE(gen_from);
  FREE(ctx);
  return 0;
}

/* --------------------------------------------------------------*/
int PtoSScatterCtxCreate(int nx,int *inidx,int ny,int *inidy,Vec xin,
                         VecScatterCtx ctx)
{
  DvPVector      *x = (DvPVector *)xin->data;
  int            to_first,to_step,from_first,from_step,*source;
  VecScatterMPI  *from,*to;
  int            *lens,mytid = x->mytid, *owners = x->ownership;
  int            numtids = x->numtids,*lowner,*start,found;
  int            *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work;
  int            *owner,*starts,count,tag = 25,rlen,slen;
  int            *rvalues,*svalues,base,index,nmax,*values,len;
  MPI_Comm       comm = x->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;

  /*  first count number of contributors to each processor */
  nprocs = (int *) MALLOC( 2*numtids*sizeof(int) ); CHKPTR(nprocs);
  MEMSET(nprocs,0,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) MALLOC((nx+1)*sizeof(int)); CHKPTR(owner); /* see note*/
  for ( i=0; i<nx; i++ ) {
    idx = inidx[i];
    found = 0;
    for ( j=0; j<numtids; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERR(1,"Index out of range");
  }
  nsends = 0;  for ( i=0; i<numtids; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) MALLOC( numtids*sizeof(int) ); CHKPTR(work);
  MPI_Allreduce((void *) procs,(void *) work,numtids,MPI_INT,MPI_SUM,comm);
  nrecvs = work[mytid]; 
  MPI_Allreduce((void *) nprocs,(void *) work,numtids,MPI_INT,MPI_MAX,comm);
  nmax = work[mytid];
  FREE(work);

  /* post receives:   */
  rvalues = (int *) MALLOC((nrecvs+1)*nmax*sizeof(int)); /*see note */
  CHKPTR(rvalues);
  recv_waits = (MPI_Request *) MALLOC((nrecvs+1)*sizeof(MPI_Request));
  CHKPTR(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv((void *)(rvalues+nmax*i),nmax,MPI_INT,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) MALLOC( (nx+1)*sizeof(int) ); CHKPTR(svalues);
  send_waits = (MPI_Request *) MALLOC( (nsends+1)*sizeof(MPI_Request));
  CHKPTR(send_waits);
  starts = (int *) MALLOC( (numtids+1)*sizeof(int) ); CHKPTR(starts);
  starts[0] = 0; 
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<nx; i++ ) {
    svalues[starts[owner[i]]++] = inidx[i];
  }

  starts[0] = 0;
  for ( i=1; i<numtids+1; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<numtids; i++ ) {
    if (procs[i]) {
      MPI_Isend((void*)(svalues+starts[i]),nprocs[i],MPI_INT,i,tag,
                comm,send_waits+count++);
    }
  }
  FREE(starts);

  base = owners[mytid];

  /*  wait on receives */
  lens = (int *) MALLOC( 2*(nrecvs+1)*sizeof(int) ); CHKPTR(lens);
  source = lens + nrecvs;
  count = nrecvs; slen = 0;
  while (count) {
    MPI_Waitany(nrecvs,recv_waits,&index,&recv_status);
    /* unpack receives into our local space */
    MPI_Get_count(&recv_status,MPI_INT,&n);
    source[index]  = recv_status.MPI_SOURCE;
    lens[index]  = n;
    slen += n;
    count--;
  }
  FREE(recv_waits); 
  
  /* allocate entire send scatter context */
  to = (VecScatterMPI *) MALLOC( sizeof(VecScatterMPI) ); CHKPTR(to);
  len = slen*(sizeof(int) + sizeof(Scalar)) + (nrecvs+1)*sizeof(int) +
        nrecvs*(sizeof(int) + sizeof(MPI_Request));
  to->n        = nrecvs; 
  to->comm     = comm;
  to->values   = (Scalar *) MALLOC( len ); CHKPTR(to->values);
  to->requests = (MPI_Request *) (to->values + slen);
  to->indices  = (int *) (to->requests + nrecvs); 
  to->starts   = (int *) (to->indices + slen);
  to->procs    = (int *) (to->starts + nrecvs + 1);
  ctx->todata  = (void *) to;

  /* move the data into the send scatter */
  to->starts[0] = 0;
  for ( i=0; i<nrecvs; i++ ) {
    to->starts[i+1] = to->starts[i] + lens[i];
    to->procs[i]    = source[i];
    values = rvalues + i*nmax;
    for ( j=0; j<lens[i]; j++ ) {
      to->indices[to->starts[i] + j] = values[j] - base;
    }
  }
  FREE(rvalues); FREE(lens);
 
  /* allocate entire receive scatter context */
  from = (VecScatterMPI *) MALLOC( sizeof(VecScatterMPI) ); CHKPTR(from);
  len = ny*(sizeof(int) + sizeof(Scalar)) + (nsends+1)*sizeof(int) +
        nsends*(sizeof(int) + sizeof(MPI_Request));
  from->n        = nsends; 
  from->comm     = comm;
  from->values   = (Scalar *) MALLOC( len );
  from->requests = (MPI_Request *) (from->values + ny);
  from->indices  = (int *) (from->requests + nsends); 
  from->starts   = (int *) (from->indices + ny);
  from->procs    = (int *) (from->starts + nsends + 1);
  ctx->fromdata  = (void *) from;

  /* move data into receive scatter */
  lowner = (int *) MALLOC( (numtids+nsends+1)*sizeof(int) ); CHKPTR(lowner);
  start = lowner + numtids;
  count = 0; from->starts[0] = start[0] = 0;
  for ( i=0; i<numtids; i++ ) {
    if (procs[i]) {
      lowner[i]            = count;
      from->procs[count++] = i;
      from->starts[count]  = start[count] = start[count-1] + nprocs[i];
    }
  }
  for ( i=0; i<nx; i++ ) {
    from->indices[start[lowner[owner[i]]]++] = inidy[i];
  }
  FREE(lowner); FREE(owner); FREE(nprocs);
    
  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) MALLOC( nsends*sizeof(MPI_Status) );
    CHKPTR(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    FREE(send_status);
  }
  FREE(send_waits); FREE(svalues);

  ctx->destroy = PtoPScatterDestroy;
  ctx->begin   = PtoPScatterbegin;
  ctx->end     = PtoPScatterend; 

  return 0;
}

/* ----------------------------------------------------------------*/
/*
     scatter from local sequential vectors to a parallel vector.
*/
int StoPScatterCtxCreate(int nx,int *inidx,int ny,int *inidy,Vec yin,
                         VecScatterCtx ctx)
{
  DvPVector      *y = (DvPVector *)yin->data;
  int            to_first,to_step,from_first,from_step,*source;
  VecScatterMPI  *from,*to;
  int            *lens,mytid = y->mytid, *owners = y->ownership;
  int            numtids = y->numtids,*lowner,*start;
  int            *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work;
  int            *owner,*starts,count,tag = 35,rlen,slen;
  int            *rvalues,*svalues,base,index,nmax,*values,len,found;
  MPI_Comm       comm = y->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;

  /*  first count number of contributors to each processor */
  nprocs = (int *) MALLOC( 2*numtids*sizeof(int) ); CHKPTR(nprocs);
  MEMSET(nprocs,0,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) MALLOC((nx+1)*sizeof(int)); CHKPTR(owner); /* see note*/
  for ( i=0; i<nx; i++ ) {
    idx = inidy[i];
    found = 0;
    for ( j=0; j<numtids; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERR(1,"Index out of range");
  }
  nsends = 0;  for ( i=0; i<numtids; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) MALLOC( numtids*sizeof(int) ); CHKPTR(work);
  MPI_Allreduce((void *) procs,(void *) work,numtids,MPI_INT,MPI_SUM,comm);
  nrecvs = work[mytid]; 
  MPI_Allreduce((void *) nprocs,(void *) work,numtids,MPI_INT,MPI_MAX,comm);
  nmax = work[mytid];
  FREE(work);

  /* post receives:   */
  rvalues = (int *) MALLOC((nrecvs+1)*nmax*sizeof(int)); /*see note */
  CHKPTR(rvalues);
  recv_waits = (MPI_Request *) MALLOC((nrecvs+1)*sizeof(MPI_Request));
  CHKPTR(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv((void *)(rvalues+nmax*i),nmax,MPI_INT,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) MALLOC( (nx+1)*sizeof(int) ); CHKPTR(svalues);
  send_waits = (MPI_Request *) MALLOC( (nsends+1)*sizeof(MPI_Request));
  CHKPTR(send_waits);
  starts = (int *) MALLOC( (numtids+1)*sizeof(int) ); CHKPTR(starts);
  starts[0] = 0; 
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<nx; i++ ) {
    svalues[starts[owner[i]]++] = inidy[i];
  }

  starts[0] = 0;
  for ( i=1; i<numtids+1; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<numtids; i++ ) {
    if (procs[i]) {
      MPI_Isend((void*)(svalues+starts[i]),nprocs[i],MPI_INT,i,tag,
                comm,send_waits+count++);
    }
  }
  FREE(starts);

  /* allocate entire send scatter context */
  to = (VecScatterMPI *) MALLOC( sizeof(VecScatterMPI) ); CHKPTR(to);
  len = ny*(sizeof(int) + sizeof(Scalar)) + (nsends+1)*sizeof(int) +
        nsends*(sizeof(int) + sizeof(MPI_Request));
  to->n        = nsends; 
  to->comm     = comm;
  to->values   = (Scalar *) MALLOC( len ); CHKPTR(to->values); 
  to->requests = (MPI_Request *) (to->values + ny);
  to->indices  = (int *) (to->requests + nsends); 
  to->starts   = (int *) (to->indices + ny);
  to->procs    = (int *) (to->starts + nsends + 1);
  ctx->todata  = (void *) to;

  /* move data into send scatter context */
  lowner = (int *) MALLOC( (numtids+nsends+1)*sizeof(int) ); CHKPTR(lowner);
  start = lowner + numtids;
  count = 0; to->starts[0] = start[0] = 0;
  for ( i=0; i<numtids; i++ ) {
    if (procs[i]) {
      lowner[i]            = count;
      to->procs[count++] = i;
      to->starts[count]  = start[count] = start[count-1] + nprocs[i];
    }
  }
  for ( i=0; i<nx; i++ ) {
    to->indices[start[lowner[owner[i]]]++] = inidx[i];
  }
  FREE(lowner); FREE(owner); FREE(nprocs);

  base = owners[mytid];

  /*  wait on receives */
  lens = (int *) MALLOC( 2*(nrecvs+1)*sizeof(int) ); CHKPTR(lens);
  source = lens + nrecvs;
  count = nrecvs; slen = 0;
  while (count) {
    MPI_Waitany(nrecvs,recv_waits,&index,&recv_status);
    /* unpack receives into our local space */
    MPI_Get_count(&recv_status,MPI_INT,&n);
    source[index]  = recv_status.MPI_SOURCE;
    lens[index]  = n;
    slen += n;
    count--;
  }
  FREE(recv_waits); 
 
  /* allocate entire receive scatter context */
  from = (VecScatterMPI *) MALLOC( sizeof(VecScatterMPI) ); CHKPTR(from);
  len = slen*(sizeof(int) + sizeof(Scalar)) + (nrecvs+1)*sizeof(int) +
        nrecvs*(sizeof(int) + sizeof(MPI_Request));
  from->n        = nrecvs; 
  from->comm     = comm;
  from->values   = (Scalar *) MALLOC( len );
  from->requests = (MPI_Request *) (from->values + slen);
  from->indices  = (int *) (from->requests + nrecvs); 
  from->starts   = (int *) (from->indices + slen);
  from->procs    = (int *) (from->starts + nrecvs + 1);
  ctx->fromdata  = (void *) from;

  /* move the data into the receive scatter context*/
  from->starts[0] = 0;
  for ( i=0; i<nrecvs; i++ ) {
    from->starts[i+1] = from->starts[i] + lens[i];
    from->procs[i]    = source[i];
    values = rvalues + i*nmax;
    for ( j=0; j<lens[i]; j++ ) {
      from->indices[from->starts[i] + j] = values[j] - base;
    }
  }
  FREE(rvalues); FREE(lens);
 
    
  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) MALLOC( nsends*sizeof(MPI_Status) );
    CHKPTR(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    FREE(send_status);
  }
  FREE(send_waits); FREE(svalues);

  ctx->destroy = PtoPScatterDestroy;
  ctx->begin   = PtoPScatterbegin;
  ctx->end     = PtoPScatterend; 

  /* PrintPVecScatterCtx(ctx); */
  return 0;
}


