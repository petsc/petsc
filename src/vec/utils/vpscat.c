#ifndef lint
static char vcid[] = "$Id: vpscat.c,v 1.28 1995/09/21 20:07:58 bsmith Exp bsmith $";
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

  MPIU_Seq_begin(ctx->comm,1);
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
  MPIU_Seq_end(ctx->comm,1);
  return 0;
}  
/*
     Even though the next routines are written with parallel 
  vectors, either xin or yin (but not both) may be Seq
  vectors, one for each processor.

     Note: since nsends, nrecvs and nx may be zero but they are used
  in mallocs, we always malloc the quantity plus one. This is not 
  an ideal solution, but it insures that we never try to malloc and 
  then free a zero size location.
  
     gen_from indices indicate where arriaving stuff is stashed
     gen_to   indices indicate where departing stuff came from. 
     the naming can be a little confusing.

*/
static int PtoPScatterbegin(Vec xin,Vec yin,InsertMode addv,
                     int mode,VecScatterCtx ctx)
{
  VecScatterMPI *gen_to, *gen_from;
  Vec_MPI       *x = (Vec_MPI *)xin->data,*y = (Vec_MPI*) yin->data;
  MPI_Comm      comm = ctx->comm;
  Scalar        *rvalues,*svalues;
  int           nrecvs, nsends;
  MPI_Request   *rwaits, *swaits;
  Scalar        *xv = x->array,*yv = y->array, *val;
  int           tag = ctx->tag, i,j,*indices;
  int           *rstarts,*sstarts;
  int           *rprocs, *sprocs;

  if (mode & SCATTERREVERSE ){
    gen_to   = (VecScatterMPI *) ctx->fromdata;
    gen_from = (VecScatterMPI *) ctx->todata;
    mode -= SCATTERREVERSE;
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

  if (mode == SCATTERALL) {
    /* post receives:   */
    for ( i=0; i<nrecvs; i++ ) {
      MPI_Irecv((void *)(rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
                 MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);
    }

    /* do sends:  */
    for ( i=0; i<nsends; i++ ) {
      val = svalues + sstarts[i];
      for ( j=0; j<sstarts[i+1]-sstarts[i]; j++ ) {
        val[j] = xv[*indices++];
      }
      MPI_Isend((void*)val,sstarts[i+1] - sstarts[i],
                 MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);
    }
  }
  else if (mode == SCATTERUP) {
    if (gen_to->nself || gen_from->nself) 
      SETERRQ(1,"PtoPScatterbegin:No SCATTERUP to self");
    /* post receives:   */
    for ( i=gen_from->nbelow; i<nrecvs; i++ ) {
      MPI_Irecv((void *)(rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
                 MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);
    }

    /* do sends:  */
    for ( i=0; i<gen_to->nbelow; i++ ) {
      val = svalues + sstarts[i];
      for ( j=0; j<sstarts[i+1]-sstarts[i]; j++ ) {
        val[j] = xv[*indices++];
      }
      MPI_Isend((void*)val,sstarts[i+1] - sstarts[i],
                 MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);
    }
  }
  else { 
    if (gen_to->nself || gen_from->nself) 
      SETERRQ(1,"PtoPScatterbegin:No SCATTERDOWN to self");
    /* post receives:   */
    for ( i=0; i<gen_from->nbelow; i++ ) {
      MPI_Irecv((void *)(rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
                 MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);
    }

    /* do sends:  */
    indices += sstarts[gen_to->nbelow]; 
    for ( i=gen_to->nbelow; i<nsends; i++ ) {
      val = svalues + sstarts[i];
      for ( j=0; j<sstarts[i+1]-sstarts[i]; j++ ) {
        val[j] = xv[*indices++];
      }
      MPI_Isend((void*)val,sstarts[i+1] - sstarts[i],
                 MPIU_SCALAR,sprocs[i],tag,comm,swaits+i-gen_to->nbelow);
    }
  }
  /* take care of local scatters */
  if (mode == SCATTERALL && addv == INSERT_VALUES) {
    int *tslots = gen_to->local.slots, *fslots = gen_from->local.slots;
    int n = gen_to->local.n;
    for ( i=0; i<n; i++ ) {yv[tslots[i]] = xv[fslots[i]];}
  }
  else if (mode == SCATTERALL) {
    int *tslots = gen_to->local.slots, *fslots = gen_from->local.slots;
    int n = gen_to->local.n;
    for ( i=0; i<n; i++ ) {yv[tslots[i]] += xv[fslots[i]];}
  }
  return 0;
}

static int PtoPScatterend(Vec xin,Vec yin,InsertMode addv,
                          int mode,VecScatterCtx ctx)
{
  VecScatterMPI *gen_to;
  VecScatterMPI *gen_from;
  Vec_MPI     *y = (Vec_MPI *)yin->data;
  Scalar        *rvalues;
  int           nrecvs, nsends;
  MPI_Request   *rwaits, *swaits;
  Scalar        *yv = y->array,*val;
  int           i,*indices,count,imdex,n;
  MPI_Status    rstatus,*sstatus;
  int           *rstarts;


  if (mode & SCATTERREVERSE ){
    gen_to   = (VecScatterMPI *) ctx->fromdata;
    gen_from = (VecScatterMPI *) ctx->todata;
    mode    -= SCATTERREVERSE;
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

  if (mode == SCATTERALL) {
    /*  wait on receives */
    count = nrecvs;
    while (count) {
      MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus);
      /* unpack receives into our local space */
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPIU_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) 
        SETERRQ(1,"PtoPScatterend:Bad message");

      if (addv == INSERT_VALUES) {
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
      sstatus = (MPI_Status *) PETSCMALLOC(nsends*sizeof(MPI_Status));
      CHKPTRQ(sstatus);
      MPI_Waitall(nsends,swaits,sstatus);
      PETSCFREE(sstatus);
    }
  }
  else if (mode == SCATTERUP) {
    if (gen_to->nself || gen_from->nself) 
      SETERRQ(1,"PtoPScatterend:No SCATTERUP to self");
    /*  wait on receives */
    count = nrecvs - gen_from->nbelow ;
    while (count) {
      MPI_Waitany(nrecvs-gen_from->nbelow,rwaits+gen_from->nbelow,&imdex,
                  &rstatus);
      imdex += gen_from->nbelow;
      /* unpack receives into our local space */
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPIU_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) 
        SETERRQ(1,"PtoPScatterend:Bad message");
      if (addv == INSERT_VALUES) {
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
      sstatus = (MPI_Status *) PETSCMALLOC(gen_to->nbelow*sizeof(MPI_Status));
      CHKPTRQ(sstatus);
      MPI_Waitall(gen_to->nbelow,swaits,sstatus);
      PETSCFREE(sstatus);
    }
  }
  else { 
    if (gen_to->nself || gen_from->nself) 
      SETERRQ(1,"PtoPScatterend:No SCATTERDOWN to self");
    /*  wait on receives */
    count = gen_from->nbelow;
    while (count) {
      MPI_Waitany(gen_from->nbelow,rwaits,&imdex,&rstatus);
      /* unpack receives into our local space */
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPIU_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) 
        SETERRQ(1,"PtoPScatterend:Bad message");
      if (addv == INSERT_VALUES) {
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
      sstatus=(MPI_Status *) PETSCMALLOC((nsends-gen_to->nbelow)*sizeof(MPI_Status));
      CHKPTRQ(sstatus);
      MPI_Waitall(nsends-gen_to->nbelow,swaits,sstatus);
      PETSCFREE(sstatus);
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

  out->scatterbegin     = in->scatterbegin;
  out->scatterend       = in->scatterend;
  out->pipelinebegin    = in->pipelinebegin;
  out->pipelineend      = in->pipelineend;
  out->copy      = in->copy;
  out->destroy   = in->destroy;
  out->view      = in->view;

  /* allocate entire send scatter context */
  out_to           = (VecScatterMPI *) PETSCMALLOC( sizeof(VecScatterMPI) );
  CHKPTRQ(out_to);
  PLogObjectMemory(out,sizeof(VecScatterMPI));
  ny               = in_to->starts[in_to->n];
  len              = ny*(sizeof(int) + sizeof(Scalar)) +
                     (in_to->n+1)*sizeof(int) +
                     (in_to->n)*(sizeof(int) + sizeof(MPI_Request));
  out_to->n        = in_to->n; 
  out_to->nbelow   = in_to->nbelow;
  out_to->nself    = in_to->nself;
  out_to->values   = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(out_to->values);
  PLogObjectMemory(out,len); 
  out_to->requests = (MPI_Request *) (out_to->values + ny);
  out_to->indices  = (int *) (out_to->requests + out_to->n); 
  out_to->starts   = (int *) (out_to->indices + ny);
  out_to->procs    = (int *) (out_to->starts + out_to->n + 1);
  PetscMemcpy(out_to->indices,in_to->indices,ny*sizeof(int));
  PetscMemcpy(out_to->starts,in_to->starts,(out_to->n+1)*sizeof(int));
  PetscMemcpy(out_to->procs,in_to->procs,(out_to->n)*sizeof(int));
  out->todata      = (void *) out_to;
  out_to->local.n  = in_to->local.n;
  if (in_to->local.n) {
    out_to->local.slots = (int *) PETSCMALLOC(in_to->local.n*sizeof(int));
    CHKPTRQ(out_to->local.slots);
    PetscMemcpy(out_to->local.slots,in_to->local.slots,
                                              in_to->local.n*sizeof(int));
    PLogObjectMemory(out,in_to->local.n*sizeof(int));
  }
  else {out_to->local.slots = 0;}

  /* allocate entire receive context */
  out_from           = (VecScatterMPI *) PETSCMALLOC( sizeof(VecScatterMPI) );
  CHKPTRQ(out_from);
  PLogObjectMemory(out,sizeof(VecScatterMPI));
  ny                 = in_from->starts[in_from->n];
  len                = ny*(sizeof(int) + sizeof(Scalar)) +
                       (in_from->n+1)*sizeof(int) +
                       (in_from->n)*(sizeof(int) + sizeof(MPI_Request));
  out_from->n        = in_from->n; 
  out_from->nbelow   = in_from->nbelow;
  out_from->nself    = in_from->nself;
  out_from->values   = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(out_from->values); 
  PLogObjectMemory(out,len);
  out_from->requests = (MPI_Request *) (out_from->values + ny);
  out_from->indices  = (int *) (out_from->requests + out_from->n); 
  out_from->starts   = (int *) (out_from->indices + ny);
  out_from->procs    = (int *) (out_from->starts + out_from->n + 1);
  PetscMemcpy(out_from->indices,in_from->indices,ny*sizeof(int));
  PetscMemcpy(out_from->starts,in_from->starts,(out_from->n+1)*sizeof(int));
  PetscMemcpy(out_from->procs,in_from->procs,(out_from->n)*sizeof(int));
  out->fromdata      = (void *) out_from;
  out_from->local.n  = in_from->local.n;
  if (in_from->local.n) {
    out_from->local.slots = (int *) PETSCMALLOC(in_from->local.n*sizeof(int));
    PLogObjectMemory(out,in_from->local.n*sizeof(int));
    CHKPTRQ(out_from->local.slots);
    PetscMemcpy(out_from->local.slots,in_from->local.slots,
                                              in_from->local.n*sizeof(int));
  }
  else {out_from->local.slots = 0;}
  return 0;
}
/* --------------------------------------------------------------------*/
static int PtoPPipelinebegin(Vec xin,Vec yin,
               InsertMode addv, PipelineMode mode,VecScatterCtx ctx)
{
  VecScatterMPI *gen_to = (VecScatterMPI *) ctx->todata;
  VecScatterMPI *gen_from = (VecScatterMPI *) ctx->fromdata;
  Vec_MPI     *y = (Vec_MPI *)yin->data;
  MPI_Comm      comm = ctx->comm;
  Scalar        *rvalues = gen_from->values;
  int           nrecvs = gen_from->nbelow;
  MPI_Request   *rwaits = gen_from->requests;
  int           tag = ctx->tag, i,*indices = gen_from->indices;
  int           *rstarts = gen_from->starts;
  int           *rprocs = gen_from->procs;
  int           count,imdex,n;
  MPI_Status    rstatus;
  Scalar        *yv = y->array,*val;

  if (gen_to->nself || gen_from->nself) 
    SETERRQ(1,"PtoPPipelinebegin:No pipeline to self");

  if (mode == PIPELINEDOWN) {
    /* post receives:   */
    for ( i=0; i<nrecvs; i++ ) {
      MPI_Irecv((void *)(rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
                                MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);
    }
    /*  wait on receives */
    count = nrecvs;
    while (count) {
      MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus);
      /* unpack receives into our local space */
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPIU_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) 
        SETERRQ(1,"PtoPPipelinebegin:Bad message");
      if (addv == INSERT_VALUES) {
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
                                MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);
    }
    /*  wait on receives */
    count = gen_from->n - nrecvs;
    while (count) {
      MPI_Waitany(gen_from->n-nrecvs,rwaits+nrecvs,&imdex,&rstatus);
      /* unpack receives into our local space */
      imdex += nrecvs;
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPIU_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) 
        SETERRQ(1,"PtoPPipelinebegin:Bad message");
      if (addv == INSERT_VALUES) {
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

static int PtoPPipelineend(Vec xin,Vec yin,
                 InsertMode addv, PipelineMode mode,VecScatterCtx ctx)
{
  VecScatterMPI *gen_to = (VecScatterMPI *) ctx->todata;
  Vec_MPI     *x = (Vec_MPI *)xin->data;
  MPI_Comm      comm = ctx->comm;
  Scalar        *svalues = gen_to->values;
  int           nsends = gen_to->n;
  MPI_Request   *swaits = gen_to->requests;
  int           tag = ctx->tag, i,j,*indices = gen_to->indices;
  MPI_Status    *sstatus;
  int           *sstarts = gen_to->starts;
  int           *sprocs = gen_to->procs;
  Scalar        *xv = x->array,*val;

  if (mode == PIPELINEDOWN) {
    /* do sends:  */
    indices += sstarts[gen_to->nbelow]; /* shift indices to match first i */
    for ( i=gen_to->nbelow; i<nsends; i++ ) {
      val = svalues + sstarts[i];
      for ( j=0; j<sstarts[i+1]-sstarts[i]; j++ ) {
        val[j] = xv[*indices++];
      }
      MPI_Isend((void*)val,sstarts[i+1] - sstarts[i],
                   MPIU_SCALAR,sprocs[i],tag,comm,swaits+i-gen_to->nbelow);
    }
    /* wait on sends */
    if (nsends-gen_to->nbelow>0) {
      sstatus = (MPI_Status *)
                       PETSCMALLOC((nsends-gen_to->nbelow)*sizeof(MPI_Status));
      CHKPTRQ(sstatus);
      MPI_Waitall(nsends-gen_to->nbelow,swaits,sstatus);
      PETSCFREE(sstatus);
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
                   MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);
    }
    /* wait on sends */
    if (gen_to->nbelow>0) {
      sstatus = (MPI_Status *)
                       PETSCMALLOC((gen_to->nbelow)*sizeof(MPI_Status));
      CHKPTRQ(sstatus);
      MPI_Waitall(gen_to->nbelow,swaits,sstatus);
      PETSCFREE(sstatus);
    }
  }
  return 0;
}

static int PtoPScatterDestroy(PetscObject obj)
{
  VecScatterCtx ctx = (VecScatterCtx) obj;
  VecScatterMPI *gen_to   = (VecScatterMPI *) ctx->todata;
  VecScatterMPI *gen_from = (VecScatterMPI *) ctx->fromdata;
  PETSCFREE(gen_to->values); PETSCFREE(gen_to);
  PETSCFREE(gen_from->values); PETSCFREE(gen_from);
  if (gen_to->local.slots) PETSCFREE(gen_to->local.slots);
  if (gen_from->local.slots) PETSCFREE(gen_from->local.slots);
  PLogObjectDestroy(ctx);
  PETSCHEADERDESTROY(ctx);
  return 0;
}

/* --------------------------------------------------------------*/
int PtoSScatterCtxCreate(int nx,int *inidx,int ny,int *inidy,Vec xin,
                         VecScatterCtx ctx)
{
  Vec_MPI        *x = (Vec_MPI *)xin->data;
  int            *source;
  VecScatterMPI  *from,*to;
  int            *lens,mytid = x->mytid, *owners = x->ownership;
  int            numtids = x->numtids,*lowner,*start,found;
  int            *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work;
  int            *owner,*starts,count,tag = xin->tag,slen;
  int            *rvalues,*svalues,base,imdex,nmax,*values,len;
  MPI_Comm       comm = xin->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  int            *indx,nprocslocal;

  /*  first count number of contributors to each processor */
  nprocs = (int *) PETSCMALLOC( 2*numtids*sizeof(int) ); CHKPTRQ(nprocs);
  PetscZero(nprocs,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) PETSCMALLOC((nx+1)*sizeof(int)); CHKPTRQ(owner);
  for ( i=0; i<nx; i++ ) {
    idx = inidx[i];
    found = 0;
    for ( j=0; j<numtids; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,"PtoSScatterCtxCreate:Index out of range");
  }
  nprocslocal = nprocs[mytid]; 
  nprocs[mytid] = procs[mytid] = 0; 
  nsends = 0;  for ( i=0; i<numtids; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PETSCMALLOC( numtids*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce((void *) procs,(void *) work,numtids,MPI_INT,MPI_SUM,comm);
  nrecvs = work[mytid]; 
  MPI_Allreduce((void *) nprocs,(void *) work,numtids,MPI_INT,MPI_MAX,comm);
  nmax = work[mytid];
  PETSCFREE(work);

  /* post receives:   */
  rvalues = (int *) PETSCMALLOC((nrecvs+1)*(nmax+1)*sizeof(int)); /*see note */
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PETSCMALLOC((nrecvs+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv((void *)(rvalues+nmax*i),nmax,MPI_INT,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) PETSCMALLOC( (nx+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PETSCMALLOC( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PETSCMALLOC( (numtids+1)*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<nx; i++ ) {
    if (owner[i] != mytid) {
      svalues[starts[owner[i]]++] = inidx[i];
    }
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
  PETSCFREE(starts);

  base = owners[mytid];

  /*  wait on receives */
  lens = (int *) PETSCMALLOC( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
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
  PETSCFREE(recv_waits); 
  
  /* allocate entire send scatter context */
  to = (VecScatterMPI *) PETSCMALLOC( sizeof(VecScatterMPI) ); CHKPTRQ(to);
  PLogObjectMemory(ctx,sizeof(VecScatterMPI));
  len = slen*(sizeof(int) + sizeof(Scalar)) + (nrecvs+1)*sizeof(int) +
        nrecvs*(sizeof(int) + sizeof(MPI_Request));
  to->n        = nrecvs; 
  to->nbelow   = 0;
  to->nself    = 0;
  to->values   = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(to->values);
  PLogObjectMemory(ctx,len);
  to->requests = (MPI_Request *) (to->values + slen);
  to->indices  = (int *) (to->requests + nrecvs); 
  to->starts   = (int *) (to->indices + slen);
  to->procs    = (int *) (to->starts + nrecvs + 1);
  ctx->todata  = (void *) to;
  to->starts[0] = 0;


  if (nrecvs) {
    indx = (int *) PETSCMALLOC( nrecvs*sizeof(int) ); CHKPTRQ(indx);
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
    PETSCFREE(indx);
  }
  PETSCFREE(rvalues); PETSCFREE(lens);
 
  /* allocate entire receive scatter context */
  from = (VecScatterMPI *) PETSCMALLOC( sizeof(VecScatterMPI) ); CHKPTRQ(from);
  PLogObjectMemory(ctx,sizeof(VecScatterMPI));
  len = ny*(sizeof(int) + sizeof(Scalar)) + (nsends+1)*sizeof(int) +
        nsends*(sizeof(int) + sizeof(MPI_Request));
  from->n        = nsends;
  from->nbelow   = 0; 
  from->nself    = 0; 
  from->values   = (Scalar *) PETSCMALLOC( len );
  PLogObjectMemory(ctx,len);
  from->requests = (MPI_Request *) (from->values + ny);
  from->indices  = (int *) (from->requests + nsends); 
  from->starts   = (int *) (from->indices + ny);
  from->procs    = (int *) (from->starts + nsends + 1);
  ctx->fromdata  = (void *) from;

  /* move data into receive scatter */
  lowner = (int *) PETSCMALLOC( (numtids+nsends+1)*sizeof(int) ); CHKPTRQ(lowner);
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
    if (owner[i] != mytid) {
      from->indices[start[lowner[owner[i]]]++] = inidy[i];
    }
  }
  PETSCFREE(lowner); PETSCFREE(owner); PETSCFREE(nprocs);
    
  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) PETSCMALLOC( nsends*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    PETSCFREE(send_status);
  }
  PETSCFREE(send_waits); PETSCFREE(svalues);

  if (nprocslocal) {
    int nt;
    /* we have a scatter to ourselfs */
    from->local.n = to->local.n = nt = nprocslocal;    
    from->local.slots = (int *) PETSCMALLOC(nt*sizeof(int));
    CHKPTRQ(from->local.slots);
    to->local.slots = (int *) PETSCMALLOC(nt*sizeof(int));
    CHKPTRQ(to->local.slots);
    PLogObjectMemory(ctx,2*nt*sizeof(int));
    nt = 0;
    for ( i=0; i<nx; i++ ) {
      idx = inidx[i];
      if (idx >= owners[mytid] && idx < owners[mytid+1]) {
        from->local.slots[nt] = idx - owners[mytid];        
        to->local.slots[nt++] = inidy[i];        
      }
    }
  }
  else { 
    from->local.n = 0; from->local.slots = 0;
    to->local.n = 0; to->local.slots = 0;
  } 

  ctx->destroy    = PtoPScatterDestroy;
  ctx->scatterbegin      = PtoPScatterbegin;
  ctx->scatterend        = PtoPScatterend; 
  ctx->pipelinebegin  = PtoPPipelinebegin;
  ctx->pipelineend    = PtoPPipelineend;
  ctx->copy       = PtoPCopy;
  return 0;
}

/* ----------------------------------------------------------------*/
/*
     scatter from local Seq vectors to a parallel vector.
*/
int StoPScatterCtxCreate(int nx,int *inidx,int ny,int *inidy,Vec yin,
                         VecScatterCtx ctx)
{
  Vec_MPI      *y = (Vec_MPI *)yin->data;
  int            *source,nprocslocal;
  VecScatterMPI  *from,*to;
  int            *lens,mytid = y->mytid, *owners = y->ownership;
  int            numtids = y->numtids,*lowner,*start;
  int            *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work;
  int            *owner,*starts,count,tag = yin->tag,slen;
  int            *rvalues,*svalues,base,imdex,nmax,*values,len,found;
  MPI_Comm       comm = yin->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;


  /*  first count number of contributors to each processor */
  nprocs = (int *) PETSCMALLOC( 2*numtids*sizeof(int) ); CHKPTRQ(nprocs);
  PetscZero(nprocs,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) PETSCMALLOC((nx+1)*sizeof(int)); CHKPTRQ(owner); /* see note*/
  for ( i=0; i<nx; i++ ) {
    idx = inidy[i];
    found = 0;
    for ( j=0; j<numtids; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,"StoPScatterCtxCreate:Index out of range");
  }
  nprocslocal = nprocs[mytid];
  nprocs[mytid] = procs[mytid] = 0; 
  nsends = 0;  for ( i=0; i<numtids; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PETSCMALLOC( numtids*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce((void *) procs,(void *) work,numtids,MPI_INT,MPI_SUM,comm);
  nrecvs = work[mytid]; 
  MPI_Allreduce((void *) nprocs,(void *) work,numtids,MPI_INT,MPI_MAX,comm);
  nmax = work[mytid];
  PETSCFREE(work);

  /* post receives:   */
  rvalues = (int *) PETSCMALLOC((nrecvs+1)*(nmax+1)*sizeof(int)); /*see note */
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PETSCMALLOC((nrecvs+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv((void *)(rvalues+nmax*i),nmax,MPI_INT,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) PETSCMALLOC( (nx+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PETSCMALLOC( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PETSCMALLOC( (numtids+1)*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<nx; i++ ) {
    if (owner[i] != mytid) {
      svalues[starts[owner[i]]++] = inidy[i];
    }
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
  PETSCFREE(starts);

  /* allocate entire send scatter context */
  to = (VecScatterMPI *) PETSCMALLOC( sizeof(VecScatterMPI) ); CHKPTRQ(to);
  PLogObjectMemory(ctx,sizeof(VecScatterMPI));
  len = ny*(sizeof(int) + sizeof(Scalar)) + (nsends+1)*sizeof(int) +
        nsends*(sizeof(int) + sizeof(MPI_Request));
  to->n        = nsends; 
  to->nbelow   = 0;
  to->values   = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(to->values); 
  PLogObjectMemory(ctx,len);
  to->requests = (MPI_Request *) (to->values + ny);
  to->indices  = (int *) (to->requests + nsends); 
  to->starts   = (int *) (to->indices + ny);
  to->procs    = (int *) (to->starts + nsends + 1);
  ctx->todata  = (void *) to;

  /* move data into send scatter context */
  lowner = (int *) PETSCMALLOC( (numtids+nsends+1)*sizeof(int) ); CHKPTRQ(lowner);
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
    if (owner[i] != mytid) {
      to->indices[start[lowner[owner[i]]]++] = inidx[i];
    }
  }
  PETSCFREE(lowner); PETSCFREE(owner); PETSCFREE(nprocs);

  base = owners[mytid];

  /*  wait on receives */
  lens = (int *) PETSCMALLOC( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
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
  PETSCFREE(recv_waits); 
 
  /* allocate entire receive scatter context */
  from = (VecScatterMPI *) PETSCMALLOC( sizeof(VecScatterMPI) ); CHKPTRQ(from);
  PLogObjectMemory(ctx,sizeof(VecScatterMPI));
  len = slen*(sizeof(int) + sizeof(Scalar)) + (nrecvs+1)*sizeof(int) +
        nrecvs*(sizeof(int) + sizeof(MPI_Request));
  from->n        = nrecvs; 
  from->nbelow   = 0;
  from->values   = (Scalar *) PETSCMALLOC( len );
  PLogObjectMemory(ctx,len);
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
  PETSCFREE(rvalues); PETSCFREE(lens);
 
    
  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) PETSCMALLOC( nsends*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    PETSCFREE(send_status);
  }
  PETSCFREE(send_waits); PETSCFREE(svalues);

  if (nprocslocal) {
    int nt;
    /* we have a scatter to ourselfs */
    from->local.n = to->local.n = nt = nprocslocal;    
    from->local.slots = (int *) PETSCMALLOC(nt*sizeof(int));
    CHKPTRQ(from->local.slots);
    to->local.slots = (int *) PETSCMALLOC(nt*sizeof(int));
    PLogObjectMemory(ctx,2*nt*sizeof(int));
    CHKPTRQ(to->local.slots);
    nt = 0;
    for ( i=0; i<ny; i++ ) {
      idx = inidy[i];
      if (idx >= owners[mytid] && idx < owners[mytid+1]) {
        to->local.slots[nt] = idx - owners[mytid];        
        from->local.slots[nt++] = inidx[i];        
      }
    }
  }
  else {
    from->local.n = 0; from->local.slots = 0;
    to->local.n = 0; to->local.slots = 0;
  }

  ctx->destroy    = PtoPScatterDestroy;
  ctx->scatterbegin      = PtoPScatterbegin;
  ctx->scatterend        = PtoPScatterend; 
  ctx->pipelinebegin  = 0;
  ctx->pipelineend    = 0;
  ctx->copy       = 0;

  /* PrintPVecScatterCtx(ctx); */
  return 0;
}

