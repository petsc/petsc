#ifndef lint
static char vcid[] = "$Id: vpscat.c,v 1.10 1995/03/25 01:25:18 bsmith Exp bsmith $";
#endif
/*
    Does the parallel vector scatter 
*/

#include "sys.h"
#include "is/isimpl.h"
#include "vecimpl.h"                     /*I "vec.h" I*/
#include "impls/dvecimpl.h"
#include "impls/mpi/pvecimpl.h"

int PrintPVecScatterCtx(VecScatterCtx ctx)
{
  VecScatterMPI *to = (VecScatterMPI *) ctx->todata;
  VecScatterMPI *from = (VecScatterMPI *) ctx->fromdata;
  int           i,mytid;

  MPE_Seq_begin(ctx->comm,1);
  MPI_Comm_rank(ctx->comm,&mytid);

  fprintf(stderr,"[%d]Number sends %d below %d self %d\n",mytid,to->n,
                 to->nbelow,to->nself);
  fprintf(stderr,"[%d]Inital start should be zero %d\n",mytid,to->starts[0]);
  for ( i=0; i<to->n; i++ ){
    fprintf(stderr,"[%d]start %d %d\n",mytid,i,to->starts[i+1]);
    fprintf(stderr,"[%d]proc %d\n",mytid,to->procs[i]);
  }
  fprintf(stderr,"Now the indices\n");
  for ( i=0; i<to->starts[to->n]; i++ ){
    fprintf(stderr,"[%d]%d \n",mytid,to->indices[i]);
  }
  fprintf(stderr,"[%d]Number receives %d below %d self %d\n",mytid,from->n,
                     from->nbelow,from->nself);
  fprintf(stderr,"[%d]Inital start should be zero %d\n",mytid,from->starts[0]);
  for ( i=0; i<from->n; i++ ){
    fprintf(stderr,"[%d]start %d %d\n",mytid,i,from->starts[i+1]);
    fprintf(stderr,"[%d]proc %d\n",mytid,from->procs[i]);
  }
  fprintf(stderr,"Now the indices\n");
  for ( i=0; i<from->starts[from->n]; i++ ){
    fprintf(stderr,"[%d]%d \n",mytid,from->indices[i]);
  }
  MPE_Seq_end(ctx->comm,1);
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
static int PtoPScatterbegin(Vec xin,Vec yin,VecScatterCtx ctx,InsertMode addv,
                     int mode)
{
  VecScatterMPI *gen_to, *gen_from;
  Vec_MPI     *x = (Vec_MPI *)xin->data;
  MPI_Comm      comm = ctx->comm;
  Scalar        *rvalues,*svalues;
  int           nrecvs, nsends;
  MPI_Request   *rwaits, *swaits;
  Scalar        *xv = x->array,*val;
  int           tag = 23, i,j,*indices;
  int           *rstarts,*sstarts;
  int           *rprocs, *sprocs;

  if (mode & ScatterReverse ){
    gen_to   = (VecScatterMPI *) ctx->fromdata;
    gen_from = (VecScatterMPI *) ctx->todata;
    mode -= ScatterReverse;
  }
  else {
    gen_to   = (VecScatterMPI *) ctx->todata;
    gen_from = (VecScatterMPI *) ctx->fromdata;
  }
  rvalues  = gen_from->values;
  svalues  = gen_to->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  rwaits   = gen_from->requests;
  swaits   = gen_to->requests;
  indices  = gen_to->indices;
  rstarts  = gen_from->starts;
  sstarts  = gen_to->starts;
  rprocs   = gen_from->procs;
  sprocs   = gen_to->procs;

  if (mode == ScatterAll) {
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
  }
  else if (mode == ScatterUp) {
    if (gen_to->nself || gen_from->nself) SETERR(1,"No scatterup to self");
    /* post receives:   */
    for ( i=gen_from->nbelow; i<nrecvs; i++ ) {
      MPI_Irecv((void *)(rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
                 MPI_SCALAR,rprocs[i],tag,comm,rwaits+i);
    }

    /* do sends:  */
    for ( i=0; i<gen_to->nbelow; i++ ) {
      val = svalues + sstarts[i];
      for ( j=0; j<sstarts[i+1]-sstarts[i]; j++ ) {
        val[j] = xv[*indices++];
      }
      MPI_Isend((void*)val,sstarts[i+1] - sstarts[i],
                 MPI_SCALAR,sprocs[i],tag,comm,swaits+i);
    }
  }
  else { 
    if (gen_to->nself || gen_from->nself) SETERR(1,"No scatterdown to self");
    /* post receives:   */
    for ( i=0; i<gen_from->nbelow; i++ ) {
      MPI_Irecv((void *)(rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
                 MPI_SCALAR,rprocs[i],tag,comm,rwaits+i);
    }

    /* do sends:  */
    indices += sstarts[gen_to->nbelow]; 
    for ( i=gen_to->nbelow; i<nsends; i++ ) {
      val = svalues + sstarts[i];
      for ( j=0; j<sstarts[i+1]-sstarts[i]; j++ ) {
        val[j] = xv[*indices++];
      }
      MPI_Isend((void*)val,sstarts[i+1] - sstarts[i],
                 MPI_SCALAR,sprocs[i],tag,comm,swaits+i-gen_to->nbelow);
    }
  }
  return 0;
}

static int PtoPScatterend(Vec xin,Vec yin,VecScatterCtx ctx,InsertMode addv,
                          int mode)
{
  VecScatterMPI *gen_to;
  VecScatterMPI *gen_from;
  Vec_MPI     *y = (Vec_MPI *)yin->data;
  Scalar        *rvalues,*svalues;
  int           nrecvs, nsends;
  MPI_Request   *rwaits, *swaits;
  Scalar        *yv = y->array,*val;
  int           i,*indices,count,imdex,n;
  MPI_Status    rstatus,*sstatus;
  int           *rstarts,*sstarts;
  int           *rprocs, *sprocs;

  if (mode & ScatterReverse ){
    gen_to   = (VecScatterMPI *) ctx->fromdata;
    gen_from = (VecScatterMPI *) ctx->todata;
    mode    -= ScatterReverse;
  }
  else {
    gen_to   = (VecScatterMPI *) ctx->todata;
    gen_from = (VecScatterMPI *) ctx->fromdata;
  }
  rvalues  = gen_from->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  rwaits   = gen_from->requests;
  swaits   = gen_to->requests;
  indices  = gen_from->indices;
  rstarts  = gen_from->starts;

  if (mode == ScatterAll) {
    /*  wait on receives */
    count = nrecvs;
    while (count) {
      MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus);
      /* unpack receives into our local space */
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPI_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) SETERR(1,"Bad message");

      if (addv == InsertValues) {
        for ( i=0; i<n; i++ ) {
          yv[indices[i+rstarts[imdex]]] = *val++;
        }
      }
       else {
        for ( i=0; i<n; i++ ) {
          yv[indices[i+rstarts[imdex]]] += *val++;
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
  }
  else if (mode == ScatterUp) {
    if (gen_to->nself || gen_from->nself) SETERR(1,"No scatterup to self");
    /*  wait on receives */
    count = nrecvs - gen_from->nbelow ;
    while (count) {
      MPI_Waitany(nrecvs-gen_from->nbelow,rwaits+gen_from->nbelow,&imdex,
                  &rstatus);
      imdex += gen_from->nbelow;
      /* unpack receives into our local space */
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPI_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) SETERR(1,"Bad message");
      if (addv == InsertValues) {
        for ( i=0; i<n; i++ ) {
          yv[indices[i+rstarts[imdex]]] = *val++;
        }
      }
       else {
        for ( i=0; i<n; i++ ) {
          yv[indices[i+rstarts[imdex]]] += *val++;
        }
      }
      count--;
    }
    /* wait on sends */
    if (gen_to->nbelow) {
      sstatus = (MPI_Status *) MALLOC(gen_to->nbelow*sizeof(MPI_Status));
      CHKPTR(sstatus);
      MPI_Waitall(gen_to->nbelow,swaits,sstatus);
      FREE(sstatus);
    }
  }
  else { 
    if (gen_to->nself || gen_from->nself) SETERR(1,"No scatterdown to self");
    /*  wait on receives */
    count = gen_from->nbelow;
    while (count) {
      MPI_Waitany(gen_from->nbelow,rwaits,&imdex,&rstatus);
      /* unpack receives into our local space */
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPI_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) SETERR(1,"Bad message");
      if (addv == InsertValues) {
        for ( i=0; i<n; i++ ) {
          yv[indices[i+rstarts[imdex]]] = *val++;
        }
      }
       else {
        for ( i=0; i<n; i++ ) {
          yv[indices[i+rstarts[imdex]]] += *val++;
        }
      }
      count--;
    }
    /* wait on sends */
    if (nsends - gen_to->nbelow > 0) {
      sstatus=(MPI_Status *) MALLOC((nsends-gen_to->nbelow)*sizeof(MPI_Status));
      CHKPTR(sstatus);
      MPI_Waitall(nsends-gen_to->nbelow,swaits,sstatus);
      FREE(sstatus);
    }
  }
  return 0;
}
/* --------------------------------------------------------------------*/
static int PtoPCopy(VecScatterCtx in,VecScatterCtx out)
{
  VecScatterMPI *in_to   = (VecScatterMPI *) in->todata;
  VecScatterMPI *in_from = (VecScatterMPI *) in->fromdata;
  VecScatterMPI *out_to,*out_from;
  int           len, ny;

  out->begin     = in->begin;
  out->end       = in->end;
  out->beginpipe = in->beginpipe;
  out->endpipe   = in->endpipe;
  out->copy      = in->copy;
  out->destroy   = in->destroy;
  out->view      = in->view;

  /* allocate entire send scatter context */
  out_to           = (VecScatterMPI *) MALLOC( sizeof(VecScatterMPI) );
  CHKPTR(out_to);
  ny               = in_to->starts[in_to->n];
  len              = ny*(sizeof(int) + sizeof(Scalar)) +
                     (in_to->n+1)*sizeof(int) +
                     (in_to->n)*(sizeof(int) + sizeof(MPI_Request));
  out_to->n        = in_to->n; 
  out_to->nbelow   = in_to->nbelow;
  out_to->nself    = in_to->nself;
  out_to->values   = (Scalar *) MALLOC( len ); CHKPTR(out_to->values); 
  out_to->requests = (MPI_Request *) (out_to->values + ny);
  out_to->indices  = (int *) (out_to->requests + out_to->n); 
  out_to->starts   = (int *) (out_to->indices + ny);
  out_to->procs    = (int *) (out_to->starts + out_to->n + 1);
  MEMCPY(out_to->indices,in_to->indices,ny*sizeof(int));
  MEMCPY(out_to->starts,in_to->starts,(out_to->n+1)*sizeof(int));
  MEMCPY(out_to->procs,in_to->procs,(out_to->n)*sizeof(int));
  out->todata      = (void *) out_to;

  /* allocate entire receive context */
  out_from           = (VecScatterMPI *) MALLOC( sizeof(VecScatterMPI) );
  CHKPTR(out_from);
  ny                 = in_from->starts[in_from->n];
  len                = ny*(sizeof(int) + sizeof(Scalar)) +
                       (in_from->n+1)*sizeof(int) +
                       (in_from->n)*(sizeof(int) + sizeof(MPI_Request));
  out_from->n        = in_from->n; 
  out_from->nbelow   = in_from->nbelow;
  out_from->nself    = in_from->nself;
  out_from->values   = (Scalar *) MALLOC( len ); CHKPTR(out_from->values); 
  out_from->requests = (MPI_Request *) (out_from->values + ny);
  out_from->indices  = (int *) (out_from->requests + out_from->n); 
  out_from->starts   = (int *) (out_from->indices + ny);
  out_from->procs    = (int *) (out_from->starts + out_from->n + 1);
  MEMCPY(out_from->indices,in_from->indices,ny*sizeof(int));
  MEMCPY(out_from->starts,in_from->starts,(out_from->n+1)*sizeof(int));
  MEMCPY(out_from->procs,in_from->procs,(out_from->n)*sizeof(int));
  out->fromdata      = (void *) out_from;
  return 0;
}
/* --------------------------------------------------------------------*/
static int PtoPPipelinebegin(Vec xin,Vec yin,VecScatterCtx ctx,InsertMode addv,
                      int mode)
{
  VecScatterMPI *gen_to = (VecScatterMPI *) ctx->todata;
  VecScatterMPI *gen_from = (VecScatterMPI *) ctx->fromdata;
  Vec_MPI     *y = (Vec_MPI *)yin->data;
  MPI_Comm      comm = ctx->comm;
  Scalar        *rvalues = gen_from->values;
  int           nrecvs = gen_from->nbelow;
  MPI_Request   *rwaits = gen_from->requests;
  int           tag = 33, i,*indices = gen_from->indices;
  int           *rstarts = gen_from->starts;
  int           *rprocs = gen_from->procs;
  int           count,imdex,n;
  MPI_Status    rstatus;
  Scalar        *yv = y->array,*val;

  if (gen_to->nself || gen_from->nself) SETERR(1,"No pipeline to self");

  if (mode == PipelineDown) {
    /* post receives:   */
    for ( i=0; i<nrecvs; i++ ) {
      MPI_Irecv((void *)(rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
                                MPI_SCALAR,rprocs[i],tag,comm,rwaits+i);
    }
    /*  wait on receives */
    count = nrecvs;
    while (count) {
      MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus);
      /* unpack receives into our local space */
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPI_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) SETERR(1,"Bad message");
      if (addv == InsertValues) {
        for ( i=0; i<n; i++ ) {
          yv[indices[i+rstarts[imdex]]] = *val++;
        }
      }
      else {
        for ( i=0; i<n; i++ ) {
          yv[indices[i+rstarts[imdex]]] += *val++;
        }
      }
      count--;
    }
  }
  else { /* Pipeline up */
    /* post receives:   */
    for ( i=nrecvs; i<gen_from->n; i++ ) {
      MPI_Irecv((void *)(rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
                                MPI_SCALAR,rprocs[i],tag,comm,rwaits+i);
    }
    /*  wait on receives */
    count = gen_from->n - nrecvs;
    while (count) {
      MPI_Waitany(gen_from->n-nrecvs,rwaits+nrecvs,&imdex,&rstatus);
      /* unpack receives into our local space */
      imdex += nrecvs;
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPI_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) SETERR(1,"Bad message");
      if (addv == InsertValues) {
        for ( i=0; i<n; i++ ) {
          yv[indices[i+rstarts[imdex]]] = *val++;
        }
      }
      else {
        for ( i=0; i<n; i++ ) {
          yv[indices[i+rstarts[imdex]]] += *val++;
        }
      }
      count--;
    }
  }
  return 0;
}

static int PtoPPipelineend(Vec xin,Vec yin,VecScatterCtx ctx,InsertMode addv,
                           int mode)
{
  VecScatterMPI *gen_to = (VecScatterMPI *) ctx->todata;
  Vec_MPI     *x = (Vec_MPI *)xin->data;
  MPI_Comm      comm = ctx->comm;
  Scalar        *svalues = gen_to->values;
  int           nsends = gen_to->n;
  MPI_Request   *swaits = gen_to->requests;
  int           tag = 33, i,j,*indices = gen_to->indices;
  MPI_Status    *sstatus;
  int           *sstarts = gen_to->starts;
  int           *sprocs = gen_to->procs;
  Scalar        *xv = x->array,*val;

  if (mode == PipelineDown) {
    /* do sends:  */
    indices += sstarts[gen_to->nbelow]; /* shift indices to match first i */
    for ( i=gen_to->nbelow; i<nsends; i++ ) {
      val = svalues + sstarts[i];
      for ( j=0; j<sstarts[i+1]-sstarts[i]; j++ ) {
        val[j] = xv[*indices++];
      }
      MPI_Isend((void*)val,sstarts[i+1] - sstarts[i],
                   MPI_SCALAR,sprocs[i],tag,comm,swaits+i-gen_to->nbelow);
    }
    /* wait on sends */
    if (nsends-gen_to->nbelow>0) {
      sstatus = (MPI_Status *)
                       MALLOC((nsends-gen_to->nbelow)*sizeof(MPI_Status));
      CHKPTR(sstatus);
      MPI_Waitall(nsends-gen_to->nbelow,swaits,sstatus);
      FREE(sstatus);
    }
  }
  else {
    /* do sends:  */
    for ( i=0; i<gen_to->nbelow; i++ ) {
      val = svalues + sstarts[i];
      for ( j=0; j<sstarts[i+1]-sstarts[i]; j++ ) {
        val[j] = xv[*indices++];
      }
      MPI_Isend((void*)val,sstarts[i+1] - sstarts[i],
                   MPI_SCALAR,sprocs[i],tag,comm,swaits+i);
    }
    /* wait on sends */
    if (gen_to->nbelow>0) {
      sstatus = (MPI_Status *)
                       MALLOC((gen_to->nbelow)*sizeof(MPI_Status));
      CHKPTR(sstatus);
      MPI_Waitall(gen_to->nbelow,swaits,sstatus);
      FREE(sstatus);
    }
  }
  return 0;
}

static int PtoPScatterDestroy(PetscObject obj)
{
  VecScatterCtx ctx = (VecScatterCtx) obj;
  VecScatterMPI *gen_to   = (VecScatterMPI *) ctx->todata;
  VecScatterMPI *gen_from = (VecScatterMPI *) ctx->fromdata;
  FREE(gen_to->values); FREE(gen_to);
  FREE(gen_from->values); FREE(gen_from);
  PETSCHEADERDESTROY(ctx);
  return 0;
}

/* --------------------------------------------------------------*/
int PtoSScatterCtxCreate(int nx,int *inidx,int ny,int *inidy,Vec xin,
                         VecScatterCtx ctx)
{
  Vec_MPI      *x = (Vec_MPI *)xin->data;
  int            *source;
  VecScatterMPI  *from,*to;
  int            *lens,mytid = x->mytid, *owners = x->ownership;
  int            numtids = x->numtids,*lowner,*start,found;
  int            *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work;
  int            *owner,*starts,count,tag = 25,slen;
  int            *rvalues,*svalues,base,imdex,nmax,*values,len;
  MPI_Comm       comm = x->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  int            *indx;

  ctx->comm = comm;
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
    if (!found) SETERR(1,"Imdex out of range");
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
  rvalues = (int *) MALLOC((nrecvs+1)*(nmax+1)*sizeof(int)); /*see note */
  CHKPTR(rvalues);
  recv_waits = (MPI_Request *) MALLOC((nrecvs+1)*sizeof(MPI_Request));
  CHKPTR(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv((void *)(rvalues+nmax*i),nmax,MPI_INT,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting imdex in svalues for stuff going to 
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
    MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    MPI_Get_count(&recv_status,MPI_INT,&n);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]  = n;
    slen += n;
    count--;
  }
  FREE(recv_waits); 
  
  /* allocate entire send scatter context */
  to = (VecScatterMPI *) MALLOC( sizeof(VecScatterMPI) ); CHKPTR(to);
  len = slen*(sizeof(int) + sizeof(Scalar)) + (nrecvs+1)*sizeof(int) +
        nrecvs*(sizeof(int) + sizeof(MPI_Request));
  to->n        = nrecvs; 
  to->nbelow   = 0;
  to->nself    = 0;
  to->values   = (Scalar *) MALLOC( len ); CHKPTR(to->values);
  to->requests = (MPI_Request *) (to->values + slen);
  to->indices  = (int *) (to->requests + nrecvs); 
  to->starts   = (int *) (to->indices + slen);
  to->procs    = (int *) (to->starts + nrecvs + 1);
  ctx->todata  = (void *) to;
  to->starts[0] = 0;


  if (nrecvs) {
    indx = (int *) MALLOC( nrecvs*sizeof(int) ); CHKPTR(indx);
    for ( i=0; i<nrecvs; i++ ) indx[i] = i;
    SYIsortperm(nrecvs,source,indx);

    /* move the data into the send scatter */
    for ( i=0; i<nrecvs; i++ ) {
      to->starts[i+1] = to->starts[i] + lens[indx[i]];
      to->procs[i]    = source[indx[i]];
      if (source[indx[i]] < mytid) to->nbelow++;
      if (source[indx[i]] == mytid) to->nself = 1;
      values = rvalues + indx[i]*nmax;
      for ( j=0; j<lens[indx[i]]; j++ ) {
        to->indices[to->starts[i] + j] = values[j] - base;
      }
    }
    FREE(indx);
  }
  FREE(rvalues); FREE(lens);
 
  /* allocate entire receive scatter context */
  from = (VecScatterMPI *) MALLOC( sizeof(VecScatterMPI) ); CHKPTR(from);
  len = ny*(sizeof(int) + sizeof(Scalar)) + (nsends+1)*sizeof(int) +
        nsends*(sizeof(int) + sizeof(MPI_Request));
  from->n        = nsends;
  from->nbelow   = 0; 
  from->nself    = 0; 
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
      if (i < mytid) from->nbelow++;
      if (i == mytid) from->nself = 1;
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

  ctx->destroy    = PtoPScatterDestroy;
  ctx->begin      = PtoPScatterbegin;
  ctx->end        = PtoPScatterend; 
  ctx->beginpipe  = PtoPPipelinebegin;
  ctx->endpipe    = PtoPPipelineend;
  ctx->copy       = PtoPCopy;
  return 0;
}

/* ----------------------------------------------------------------*/
/*
     scatter from local sequential vectors to a parallel vector.
*/
int StoPScatterCtxCreate(int nx,int *inidx,int ny,int *inidy,Vec yin,
                         VecScatterCtx ctx)
{
  Vec_MPI      *y = (Vec_MPI *)yin->data;
  int            *source;
  VecScatterMPI  *from,*to;
  int            *lens,mytid = y->mytid, *owners = y->ownership;
  int            numtids = y->numtids,*lowner,*start;
  int            *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work;
  int            *owner,*starts,count,tag = 35,slen;
  int            *rvalues,*svalues,base,imdex,nmax,*values,len,found;
  MPI_Comm       comm = y->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;

  ctx->comm = comm;
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
    if (!found) SETERR(1,"Imdex out of range");
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
  rvalues = (int *) MALLOC((nrecvs+1)*(nmax+1)*sizeof(int)); /*see note */
  CHKPTR(rvalues);
  recv_waits = (MPI_Request *) MALLOC((nrecvs+1)*sizeof(MPI_Request));
  CHKPTR(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv((void *)(rvalues+nmax*i),nmax,MPI_INT,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting imdex in svalues for stuff going to 
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
  to->nbelow   = 0;
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
      lowner[i]          = count;
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
    MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    MPI_Get_count(&recv_status,MPI_INT,&n);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]  = n;
    slen += n;
    count--;
  }
  FREE(recv_waits); 
 
  /* allocate entire receive scatter context */
  from = (VecScatterMPI *) MALLOC( sizeof(VecScatterMPI) ); CHKPTR(from);
  len = slen*(sizeof(int) + sizeof(Scalar)) + (nrecvs+1)*sizeof(int) +
        nrecvs*(sizeof(int) + sizeof(MPI_Request));
  from->n        = nrecvs; 
  from->nbelow   = 0;
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

  ctx->destroy    = PtoPScatterDestroy;
  ctx->begin      = PtoPScatterbegin;
  ctx->end        = PtoPScatterend; 
  ctx->beginpipe  = 0;
  ctx->endpipe    = 0;
  ctx->copy       = 0;

  /* PrintPVecScatterCtx(ctx); */
  return 0;
}

