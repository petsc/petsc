#ifndef lint
static char vcid[] = "$Id: vpscat.c,v 1.39 1995/12/31 21:09:48 curfman Exp curfman $";
#endif
/*
    Defines parallel vector scatters.
*/

#include "sys.h"
#include "is/isimpl.h"
#include "vecimpl.h"                     /*I "vec.h" I*/
#include "impls/dvecimpl.h"
#include "impls/mpi/pvecimpl.h"
#include "pinclude/pviewer.h"

int VecScatterView_MPI(PetscObject obj,Viewer viewer)
{
  VecScatter     ctx = (VecScatter) obj;
  VecScatter_MPI *to=(VecScatter_MPI *) ctx->todata, *from=(VecScatter_MPI *) ctx->fromdata;
  PetscObject    vobj = (PetscObject) viewer;
  int            i,rank,ierr;
  FILE           *fd;

  if (!viewer) { 
    viewer = STDOUT_VIEWER_SELF; vobj = (PetscObject) viewer;
  }
  if (vobj->cookie != VIEWER_COOKIE) return 0;
  if (vobj->type != ASCII_FILE_VIEWER && vobj->type != ASCII_FILES_VIEWER) return 0;

  MPI_Comm_rank(ctx->comm,&rank);
  ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
  MPIU_Seq_begin(ctx->comm,1);
  fprintf(fd,"[%d] Number sends %d below %d self %d\n",rank,to->n,to->nbelow,to->nself);
  for ( i=0; i<to->n; i++ ){
    fprintf(fd,"[%d]   %d length %d to whom %d\n",rank,i,to->starts[i+1]-to->starts[i],
            to->procs[i]);
  }
  /*
  fprintf(fd,"Now the indices\n");
  for ( i=0; i<to->starts[to->n]; i++ ){
    fprintf(fd,"[%d]%d \n",rank,to->indices[i]);
  }
  */
  fprintf(fd,"[%d]Number receives %d below %d self %d\n",rank,from->n,
          from->nbelow,from->nself);
  for ( i=0; i<from->n; i++ ){
    fprintf(fd,"[%d] %d length %d to whom %d\n",rank,i,from->starts[i+1]-from->starts[i],
            from->procs[i]);
  }
  /*
  fprintf(fd,"Now the indices\n");
  for ( i=0; i<from->starts[from->n]; i++ ){
    fprintf(fd,"[%d]%d \n",rank,from->indices[i]);
  }
  */
  fflush(fd);
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
  
     gen_from indices indicate where arriving stuff is stashed
     gen_to   indices indicate where departing stuff came from. 
     the naming can be a little confusing.

*/
static int PtoPScatterbegin(Vec xin,Vec yin,InsertMode addv,int mode,VecScatter ctx)
{
  VecScatter_MPI *gen_to, *gen_from;
  Vec_MPI        *x = (Vec_MPI *)xin->data,*y = (Vec_MPI*) yin->data;
  MPI_Comm       comm = ctx->comm;
  Scalar         *xv = x->array,*yv = y->array, *val, *rvalues,*svalues;
  MPI_Request    *rwaits, *swaits;
  int            tag = ctx->tag, i,j,*indices,*rstarts,*sstarts,*rprocs, *sprocs;
  int            nrecvs, nsends,iend;

  if (mode & SCATTER_REVERSE ){
    gen_to   = (VecScatter_MPI *) ctx->fromdata;
    gen_from = (VecScatter_MPI *) ctx->todata;
    mode -= SCATTER_REVERSE;
  }
  else {
    gen_to   = (VecScatter_MPI *) ctx->todata;
    gen_from = (VecScatter_MPI *) ctx->fromdata;
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

  if (mode == SCATTER_ALL) {
    /* post receives:   */
    for ( i=0; i<nrecvs; i++ ) {
      MPI_Irecv((rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
                 MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);
    }

    /* do sends:  */
    for ( i=0; i<nsends; i++ ) {
      val  = svalues + sstarts[i];
      iend = sstarts[i+1]-sstarts[i];
      for ( j=0; j<iend; j++ ) {
        val[j] = xv[*indices++];
      }
      MPI_Isend((void*)val,iend, MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);
    }
  }
  else if (mode == SCATTER_UP) {
    if (gen_to->nself || gen_from->nself) 
      SETERRQ(1,"PtoPScatterbegin:No SCATTER_UP to self");
    /* post receives:   */
    for ( i=gen_from->nbelow; i<nrecvs; i++ ) {
      MPI_Irecv((rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
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
      SETERRQ(1,"PtoPScatterbegin:No SCATTER_DOWN to self");
    /* post receives:   */
    for ( i=0; i<gen_from->nbelow; i++ ) {
      MPI_Irecv((rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
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
  if (mode == SCATTER_ALL && addv == INSERT_VALUES) {
    int *tslots = gen_to->local.slots, *fslots = gen_from->local.slots;
    int n = gen_to->local.n;
    for ( i=0; i<n; i++ ) {yv[tslots[i]] = xv[fslots[i]];}
  }
  else if (mode == SCATTER_ALL) {
    int *tslots = gen_to->local.slots, *fslots = gen_from->local.slots;
    int n = gen_to->local.n;
    for ( i=0; i<n; i++ ) {yv[tslots[i]] += xv[fslots[i]];}
  }
  return 0;
}

static int PtoPScatterend(Vec xin,Vec yin,InsertMode addv,int mode,VecScatter ctx)
{
  VecScatter_MPI *gen_to, *gen_from;
  Vec_MPI        *y = (Vec_MPI *)yin->data;
  Scalar         *rvalues, *yv = y->array,*val;
  int            nrecvs, nsends,i,*indices,count,imdex,n,*rstarts,*lindices;
  MPI_Request    *rwaits, *swaits;
  MPI_Status     rstatus,*sstatus;

  if (mode & SCATTER_REVERSE ){
    gen_to   = (VecScatter_MPI *) ctx->fromdata;
    gen_from = (VecScatter_MPI *) ctx->todata;
    mode    -= SCATTER_REVERSE;
  }
  else {
    gen_to   = (VecScatter_MPI *) ctx->todata;
    gen_from = (VecScatter_MPI *) ctx->fromdata;
  }
  rvalues  = gen_from->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  rwaits   = gen_from->requests;
  swaits   = gen_to->requests;
  indices  = gen_from->indices;
  rstarts  = gen_from->starts;

  if (mode == SCATTER_ALL) {
    /*  wait on receives */
    count = nrecvs;
    while (count) {
      MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus);
      /* unpack receives into our local space */
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPIU_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) SETERRQ(1,"PtoPScatterend:Bad message");

      lindices = indices + rstarts[imdex];
      if (addv == INSERT_VALUES) {
        for ( i=0; i<n; i++ ) {
          yv[lindices[i]] = *val++;
        }
      }
       else {
        for ( i=0; i<n; i++ ) {
          yv[lindices[i]] += *val++;
        }
      }
      count--;
    }
 
    /* wait on sends */
    if (nsends) {
      sstatus = gen_to->sstatus;
      MPI_Waitall(nsends,swaits,sstatus);
    }
  }
  else if (mode == SCATTER_UP) {
    if (gen_to->nself || gen_from->nself) SETERRQ(1,"PtoPScatterend:No SCATTER_UP to self");
    /*  wait on receives */
    count = nrecvs - gen_from->nbelow ;
    while (count) {
      MPI_Waitany(nrecvs-gen_from->nbelow,rwaits+gen_from->nbelow,&imdex,&rstatus);
      imdex += gen_from->nbelow;
      /* unpack receives into our local space */
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPIU_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) SETERRQ(1,"PtoPScatterend:Bad message");
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
      sstatus = (MPI_Status *)PetscMalloc(gen_to->nbelow*sizeof(MPI_Status));CHKPTRQ(sstatus);
      MPI_Waitall(gen_to->nbelow,swaits,sstatus);
      PetscFree(sstatus);
    }
  }
  else { 
    if (gen_to->nself || gen_from->nself)SETERRQ(1,"PtoPScatterend:No SCATTER_DOWN to self");
    /*  wait on receives */
    count = gen_from->nbelow;
    while (count) {
      MPI_Waitany(gen_from->nbelow,rwaits,&imdex,&rstatus);
      /* unpack receives into our local space */
      val = rvalues + rstarts[imdex];
      MPI_Get_count(&rstatus,MPIU_SCALAR,&n);
      if (n != rstarts[imdex+1] - rstarts[imdex]) SETERRQ(1,"PtoPScatterend:Bad message");
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
      sstatus=(MPI_Status *) PetscMalloc((nsends-gen_to->nbelow)*sizeof(MPI_Status));
      CHKPTRQ(sstatus);
      MPI_Waitall(nsends-gen_to->nbelow,swaits,sstatus);
      PetscFree(sstatus);
    }
  }
  return 0;
}
/* --------------------------------------------------------------------*/
static int PtoPCopy(VecScatter in,VecScatter out)
{
  VecScatter_MPI *in_to   = (VecScatter_MPI *) in->todata;
  VecScatter_MPI *in_from = (VecScatter_MPI *) in->fromdata,*out_to,*out_from;
  int            len, ny;

  out->scatterbegin     = in->scatterbegin;
  out->scatterend       = in->scatterend;
  out->pipelinebegin    = in->pipelinebegin;
  out->pipelineend      = in->pipelineend;
  out->copy             = in->copy;
  out->destroy          = in->destroy;
  out->view             = in->view;

  /* allocate entire send scatter context */
  out_to           = (VecScatter_MPI *) PetscMalloc(sizeof(VecScatter_MPI));CHKPTRQ(out_to);
  PLogObjectMemory(out,sizeof(VecScatter_MPI));
  ny               = in_to->starts[in_to->n];
  len              = ny*(sizeof(int) + sizeof(Scalar)) +
                     (in_to->n+1)*sizeof(int) +
                     (in_to->n)*(sizeof(int) + sizeof(MPI_Request));
  out_to->n        = in_to->n; 
  out_to->nbelow   = in_to->nbelow;
  out_to->nself    = in_to->nself;
  out_to->values   = (Scalar *) PetscMalloc( len ); CHKPTRQ(out_to->values);
  PLogObjectMemory(out,len); 
  out_to->requests = (MPI_Request *) (out_to->values + ny);
  out_to->indices  = (int *) (out_to->requests + out_to->n); 
  out_to->starts   = (int *) (out_to->indices + ny);
  out_to->procs    = (int *) (out_to->starts + out_to->n + 1);
  PetscMemcpy(out_to->indices,in_to->indices,ny*sizeof(int));
  PetscMemcpy(out_to->starts,in_to->starts,(out_to->n+1)*sizeof(int));
  PetscMemcpy(out_to->procs,in_to->procs,(out_to->n)*sizeof(int));
  out_to->sstatus  = (MPI_Status *) PetscMalloc((out_to->n+1)*sizeof(MPI_Status));
                     CHKPTRQ(out_to->sstatus);
  out->todata      = (void *) out_to;
  out_to->local.n  = in_to->local.n;
  if (in_to->local.n) {
    out_to->local.slots = (int *) PetscMalloc(in_to->local.n*sizeof(int));
    CHKPTRQ(out_to->local.slots);
    PetscMemcpy(out_to->local.slots,in_to->local.slots,in_to->local.n*sizeof(int));
    PLogObjectMemory(out,in_to->local.n*sizeof(int));
  }
  else {out_to->local.slots = 0;}

  /* allocate entire receive context */
  out_from           = (VecScatter_MPI *) PetscMalloc(sizeof(VecScatter_MPI));CHKPTRQ(out_from);
  PLogObjectMemory(out,sizeof(VecScatter_MPI));
  ny                 = in_from->starts[in_from->n];
  len                = ny*(sizeof(int) + sizeof(Scalar)) +
                       (in_from->n+1)*sizeof(int) +
                       (in_from->n)*(sizeof(int) + sizeof(MPI_Request));
  out_from->n        = in_from->n; 
  out_from->nbelow   = in_from->nbelow;
  out_from->nself    = in_from->nself;
  out_from->values   = (Scalar *) PetscMalloc( len ); CHKPTRQ(out_from->values); 
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
    out_from->local.slots = (int *) PetscMalloc(in_from->local.n*sizeof(int));
    PLogObjectMemory(out,in_from->local.n*sizeof(int));CHKPTRQ(out_from->local.slots);
    PetscMemcpy(out_from->local.slots,in_from->local.slots,in_from->local.n*sizeof(int));
  }
  else {out_from->local.slots = 0;}
  return 0;
}
/* --------------------------------------------------------------------*/
static int PtoPPipelinebegin(Vec xin,Vec yin,InsertMode addv,PipelineMode mode,
                             VecScatter ctx)
{
  VecScatter_MPI *gen_to = (VecScatter_MPI *) ctx->todata;
  VecScatter_MPI *gen_from = (VecScatter_MPI *) ctx->fromdata;
  Vec_MPI        *y = (Vec_MPI *)yin->data;
  MPI_Comm       comm = ctx->comm;
  MPI_Request    *rwaits = gen_from->requests;
  int            nrecvs = gen_from->nbelow,tag = ctx->tag, i,*indices = gen_from->indices;
  int            *rstarts = gen_from->starts,*rprocs = gen_from->procs,count,imdex,n;
  MPI_Status     rstatus;
  Scalar         *yv = y->array,*val,*rvalues = gen_from->values;

  if (gen_to->nself || gen_from->nself) SETERRQ(1,"PtoPPipelinebegin:No pipeline to self");

  if (mode == PIPELINE_DOWN) {
    /* post receives:   */
    for ( i=0; i<nrecvs; i++ ) {
      MPI_Irecv((rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
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
      MPI_Irecv((rvalues+rstarts[i]),rstarts[i+1] - rstarts[i],
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
      if (n != rstarts[imdex+1] - rstarts[imdex]) SETERRQ(1,"PtoPPipelinebegin:Bad message");
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

static int PtoPPipelineend(Vec xin,Vec yin,InsertMode addv, PipelineMode mode,
                           VecScatter ctx)
{
  VecScatter_MPI *gen_to = (VecScatter_MPI *) ctx->todata;
  Vec_MPI        *x = (Vec_MPI *)xin->data;
  MPI_Comm       comm = ctx->comm;
  MPI_Request    *swaits = gen_to->requests;
  MPI_Status     *sstatus;
  int            nsends = gen_to->n,tag = ctx->tag, i,j,*indices = gen_to->indices;
  int            *sstarts = gen_to->starts,*sprocs = gen_to->procs;
  Scalar         *xv = x->array,*val,*svalues = gen_to->values;

  if (mode == PIPELINE_DOWN) {
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
      sstatus = (MPI_Status *)PetscMalloc((nsends-gen_to->nbelow)*sizeof(MPI_Status));
      CHKPTRQ(sstatus);
      MPI_Waitall(nsends-gen_to->nbelow,swaits,sstatus);
      PetscFree(sstatus);
    }
  }
  else {
    /* do sends:  */
    for ( i=0; i<gen_to->nbelow; i++ ) {
      val = svalues + sstarts[i];
      for ( j=0; j<sstarts[i+1]-sstarts[i]; j++ ) {
        val[j] = xv[*indices++];
      }
      MPI_Isend((void*)val,sstarts[i+1]-sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);
    }
    /* wait on sends */
    if (gen_to->nbelow>0) {
      sstatus = (MPI_Status *)PetscMalloc((gen_to->nbelow)*sizeof(MPI_Status));
      CHKPTRQ(sstatus);
      MPI_Waitall(gen_to->nbelow,swaits,sstatus);
      PetscFree(sstatus);
    }
  }
  return 0;
}

static int PtoPScatterDestroy(PetscObject obj)
{
  VecScatter     ctx = (VecScatter) obj;
  VecScatter_MPI *gen_to   = (VecScatter_MPI *) ctx->todata;
  VecScatter_MPI *gen_from = (VecScatter_MPI *) ctx->fromdata;

  if (gen_to->local.slots) PetscFree(gen_to->local.slots);
  if (gen_from->local.slots) PetscFree(gen_from->local.slots);
  PetscFree(gen_to->sstatus);
  PetscFree(gen_to->values); PetscFree(gen_to);
  PetscFree(gen_from->values); PetscFree(gen_from);
  PLogObjectDestroy(ctx);
  PetscHeaderDestroy(ctx);
  return 0;
}

/* --------------------------------------------------------------*/
/* create parallel to sequential scatter context */
int PtoSScatterCreate(int nx,int *inidx,int ny,int *inidy,Vec xin,VecScatter ctx)
{
  Vec_MPI        *x = (Vec_MPI *)xin->data;
  VecScatter_MPI *from,*to;
  int            *source,*lens,rank = x->rank, *owners = x->ownership;
  int            size = x->size,*lowner,*start,found;
  int            *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work;
  int            *owner,*starts,count,tag = xin->tag,slen;
  int            *rvalues,*svalues,base,imdex,nmax,*values,len,*indx,nprocslocal;
  MPI_Comm       comm = xin->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc((nx+1)*sizeof(int)); CHKPTRQ(owner);
  for ( i=0; i<nx; i++ ) {
    idx = inidx[i];
    found = 0;
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,"PtoSScatterCreate:Index out of range");
  }
  nprocslocal = nprocs[rank]; 
  nprocs[rank] = procs[rank] = 0; 
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce( procs, work,size,MPI_INT,MPI_SUM,comm);
  nrecvs = work[rank]; 
  MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives:   */
  rvalues = (int *) PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int)); CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv((rvalues+nmax*i),nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) PetscMalloc( (nx+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *)PetscMalloc((nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( (size+1)*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<nx; i++ ) {
    if (owner[i] != rank) {
      svalues[starts[owner[i]]++] = inidx[i];
    }
  }

  starts[0] = 0;
  for ( i=1; i<size+1; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend((void*)(svalues+starts[i]),nprocs[i],MPI_INT,i,tag,
                comm,send_waits+count++);
    }
  }
  PetscFree(starts);

  base = owners[rank];

  /*  wait on receives */
  lens = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
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
  PetscFree(recv_waits); 
  
  /* allocate entire send scatter context */
  to = (VecScatter_MPI *) PetscMalloc( sizeof(VecScatter_MPI) ); CHKPTRQ(to);
  PLogObjectMemory(ctx,sizeof(VecScatter_MPI));
  len = slen*(sizeof(int) + sizeof(Scalar)) + (nrecvs+1)*sizeof(int) +
        nrecvs*(sizeof(int) + sizeof(MPI_Request));
  to->n        = nrecvs; 
  to->nbelow   = 0;
  to->nself    = 0;
  to->values   = (Scalar *) PetscMalloc( len ); CHKPTRQ(to->values);
  PLogObjectMemory(ctx,len);
  to->requests = (MPI_Request *) (to->values + slen);
  to->indices  = (int *) (to->requests + nrecvs); 
  to->starts   = (int *) (to->indices + slen);
  to->procs    = (int *) (to->starts + nrecvs + 1);
  to->sstatus  = (MPI_Status *) PetscMalloc((1+nrecvs)*sizeof(MPI_Status));
                 CHKPTRQ(to->sstatus);
  ctx->todata  = (void *) to;
  to->starts[0] = 0;

  if (nrecvs) {
    indx = (int *) PetscMalloc( nrecvs*sizeof(int) ); CHKPTRQ(indx);
    for ( i=0; i<nrecvs; i++ ) indx[i] = i;
    SYIsortperm(nrecvs,source,indx);

    /* move the data into the send scatter */
    for ( i=0; i<nrecvs; i++ ) {
      to->starts[i+1] = to->starts[i] + lens[indx[i]];
      to->procs[i]    = source[indx[i]];
      if (source[indx[i]] < rank) to->nbelow++;
      if (source[indx[i]] == rank) to->nself = 1;
      values = rvalues + indx[i]*nmax;
      for ( j=0; j<lens[indx[i]]; j++ ) {
        to->indices[to->starts[i] + j] = values[j] - base;
      }
    }
    PetscFree(indx);
  }
  PetscFree(rvalues); PetscFree(lens);
 
  /* allocate entire receive scatter context */
  from = (VecScatter_MPI *) PetscMalloc( sizeof(VecScatter_MPI) ); CHKPTRQ(from);
  PLogObjectMemory(ctx,sizeof(VecScatter_MPI));
  len = ny*(sizeof(int) + sizeof(Scalar)) + (nsends+1)*sizeof(int) +
        nsends*(sizeof(int) + sizeof(MPI_Request));
  from->n        = nsends;
  from->nbelow   = 0; 
  from->nself    = 0; 
  from->values   = (Scalar *) PetscMalloc( len );
  PLogObjectMemory(ctx,len);
  from->requests = (MPI_Request *) (from->values + ny);
  from->indices  = (int *) (from->requests + nsends); 
  from->starts   = (int *) (from->indices + ny);
  from->procs    = (int *) (from->starts + nsends + 1);
  ctx->fromdata  = (void *) from;

  /* move data into receive scatter */
  lowner = (int *) PetscMalloc( (size+nsends+1)*sizeof(int) ); CHKPTRQ(lowner);
  start = lowner + size;
  count = 0; from->starts[0] = start[0] = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      if (i < rank) from->nbelow++;
      if (i == rank) from->nself = 1;
      lowner[i]            = count;
      from->procs[count++] = i;
      from->starts[count]  = start[count] = start[count-1] + nprocs[i];
    }
  }
  for ( i=0; i<nx; i++ ) {
    if (owner[i] != rank) {
      from->indices[start[lowner[owner[i]]]++] = inidy[i];
    }
  }
  PetscFree(lowner); PetscFree(owner); PetscFree(nprocs);
    
  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *)PetscMalloc( nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(send_waits); PetscFree(svalues);

  if (nprocslocal) {
    int nt;
    /* we have a scatter to ourselves */
    from->local.n = to->local.n = nt = nprocslocal;    
    from->local.slots = (int *) PetscMalloc(nt*sizeof(int));CHKPTRQ(from->local.slots);
    to->local.slots = (int *) PetscMalloc(nt*sizeof(int));CHKPTRQ(to->local.slots);
    PLogObjectMemory(ctx,2*nt*sizeof(int));
    nt = 0;
    for ( i=0; i<nx; i++ ) {
      idx = inidx[i];
      if (idx >= owners[rank] && idx < owners[rank+1]) {
        from->local.slots[nt] = idx - owners[rank];        
        to->local.slots[nt++] = inidy[i];        
      }
    }
  }
  else { 
    from->local.n = 0; from->local.slots = 0;
    to->local.n   = 0; to->local.slots = 0;
  } 

  ctx->destroy        = PtoPScatterDestroy;
  ctx->scatterbegin   = PtoPScatterbegin;
  ctx->scatterend     = PtoPScatterend; 
  ctx->pipelinebegin  = PtoPPipelinebegin;
  ctx->pipelineend    = PtoPPipelineend;
  ctx->copy           = PtoPCopy;
  ctx->view           = VecScatterView_MPI;
  return 0;
}

/* ----------------------------------------------------------------*/
/*
    Scatter from local Seq vectors to a parallel vector. 
 */
int StoPScatterCreate(int nx,int *inidx,int ny,int *inidy,Vec yin,VecScatter ctx)
{
  Vec_MPI        *y = (Vec_MPI *)yin->data;
  VecScatter_MPI *from,*to;
  int            *source,nprocslocal,*lens,rank = y->rank, *owners = y->ownership;
  int            size = y->size,*lowner,*start;
  int            *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work;
  int            *owner,*starts,count,tag = yin->tag,slen;
  int            *rvalues,*svalues,base,imdex,nmax,*values,len,found;
  MPI_Comm       comm = yin->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc((nx+1)*sizeof(int)); CHKPTRQ(owner); /* see note*/
  for ( i=0; i<nx; i++ ) {
    idx = inidy[i];
    found = 0;
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,"StoPScatterCreate:Index out of range");
  }
  nprocslocal = nprocs[rank];
  nprocs[rank] = procs[rank] = 0; 
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce( procs, work,size,MPI_INT,MPI_SUM,comm);
  nrecvs = work[rank]; 
  MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives:   */
  rvalues = (int *) PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int)); CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) PetscMalloc( (nx+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *)PetscMalloc((nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( (size+1)*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<nx; i++ ) {
    if (owner[i] != rank) {
      svalues[starts[owner[i]]++] = inidy[i];
    }
  }

  starts[0] = 0;
  for ( i=1; i<size+1; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);
    }
  }
  PetscFree(starts);

  /* allocate entire send scatter context */
  to = (VecScatter_MPI *) PetscMalloc( sizeof(VecScatter_MPI) ); CHKPTRQ(to);
  PLogObjectMemory(ctx,sizeof(VecScatter_MPI));
  len = ny*(sizeof(int) + sizeof(Scalar)) + (nsends+1)*sizeof(int) +
        nsends*(sizeof(int) + sizeof(MPI_Request));
  to->n        = nsends; 
  to->nbelow   = 0;
  to->values   = (Scalar *) PetscMalloc( len ); CHKPTRQ(to->values); 
  PLogObjectMemory(ctx,len);
  to->requests = (MPI_Request *) (to->values + ny);
  to->indices  = (int *) (to->requests + nsends); 
  to->starts   = (int *) (to->indices + ny);
  to->procs    = (int *) (to->starts + nsends + 1);
  to->sstatus  = (MPI_Status *) PetscMalloc((1+nsends)*sizeof(MPI_Status));
                 CHKPTRQ(to->sstatus);
  ctx->todata  = (void *) to;

  /* move data into send scatter context */
  lowner = (int *) PetscMalloc( (size+nsends+1)*sizeof(int) ); CHKPTRQ(lowner);
  start = lowner + size;
  count = 0; to->starts[0] = start[0] = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      lowner[i]          = count;
      to->procs[count++] = i;
      to->starts[count]  = start[count] = start[count-1] + nprocs[i];
    }
  }
  for ( i=0; i<nx; i++ ) {
    if (owner[i] != rank) {
      to->indices[start[lowner[owner[i]]]++] = inidx[i];
    }
  }
  PetscFree(lowner); PetscFree(owner); PetscFree(nprocs);

  base = owners[rank];

  /*  wait on receives */
  lens = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
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
  PetscFree(recv_waits); 
 
  /* allocate entire receive scatter context */
  from = (VecScatter_MPI *) PetscMalloc( sizeof(VecScatter_MPI) ); CHKPTRQ(from);
  PLogObjectMemory(ctx,sizeof(VecScatter_MPI));
  len = slen*(sizeof(int) + sizeof(Scalar)) + (nrecvs+1)*sizeof(int) +
        nrecvs*(sizeof(int) + sizeof(MPI_Request));
  from->n        = nrecvs; 
  from->nbelow   = 0;
  from->values   = (Scalar *) PetscMalloc( len );
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
  PetscFree(rvalues); PetscFree(lens);
    
  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) PetscMalloc(nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(send_waits); PetscFree(svalues);

  if (nprocslocal) {
    int nt;
    /* we have a scatter to ourselves */
    from->local.n = to->local.n = nt = nprocslocal;    
    from->local.slots = (int *) PetscMalloc(nt*sizeof(int));CHKPTRQ(from->local.slots);
    to->local.slots = (int *) PetscMalloc(nt*sizeof(int));CHKPTRQ(to->local.slots);
    PLogObjectMemory(ctx,2*nt*sizeof(int));
    nt = 0;
    for ( i=0; i<ny; i++ ) {
      idx = inidy[i];
      if (idx >= owners[rank] && idx < owners[rank+1]) {
        to->local.slots[nt] = idx - owners[rank];        
        from->local.slots[nt++] = inidx[i];        
      }
    }
  }
  else {
    from->local.n = 0; from->local.slots = 0;
    to->local.n = 0; to->local.slots = 0;
  }

  ctx->destroy           = PtoPScatterDestroy;
  ctx->scatterbegin      = PtoPScatterbegin;
  ctx->scatterend        = PtoPScatterend; 
  ctx->pipelinebegin     = 0;
  ctx->pipelineend       = 0;
  ctx->copy              = 0;
  ctx->view              = VecScatterView_MPI;
  return 0;
}

/* ---------------------------------------------------------------------------------*/
int PtoPScatterCreate(int nx,int *inidx,int ny,int *inidy,Vec xin,Vec yin,VecScatter ctx)
{
  Vec_MPI        *x = (Vec_MPI *)xin->data;
  int            *lens,rank = x->rank, *owners = x->ownership,size = x->size,found;
  int            *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work,*local_inidx,*local_inidy;
  int            *owner,*starts,count,tag = xin->tag,slen,ierr;
  int            *rvalues,*svalues,base,imdex,nmax,*values;
  MPI_Comm       comm = xin->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status;

  /*
  Each processor ships off its inidx[j] and inidy[j] to the appropriate processor
  They then call the StoPScatterCreate()
  */
  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc((nx+1)*sizeof(int)); CHKPTRQ(owner);
  for ( i=0; i<nx; i++ ) {
    idx = inidx[i];
    found = 0;
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,"PtoPScatterCreate:Index out of range");
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce( procs, work,size,MPI_INT,MPI_SUM,comm);
  nrecvs = work[rank]; 
  MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives:   */
  rvalues = (int *) PetscMalloc(2*(nrecvs+1)*(nmax+1)*sizeof(int)); CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv(rvalues+2*nmax*i,2*nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) PetscMalloc( 2*(nx+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *)PetscMalloc((nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( (size+1)*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<nx; i++ ) {
    svalues[2*starts[owner[i]]]       = inidx[i];
    svalues[1 + 2*starts[owner[i]]++] = inidy[i];
  }
  PetscFree(owner);

  starts[0] = 0;
  for ( i=1; i<size+1; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend(svalues+2*starts[i],2*nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);
    }
  }
  PetscFree(starts);
  PetscFree(nprocs);

  base = owners[rank];

  /*  wait on receives */
  lens = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
  count = nrecvs; slen = 0;
  while (count) {
    MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    MPI_Get_count(&recv_status,MPI_INT,&n);
    lens[imdex]  =  n/2;
    slen         += n/2;
    count--;
  }
  PetscFree(recv_waits); 
  
  local_inidx = (int *) PetscMalloc( 2*(slen+1)*sizeof(int) ); CHKPTRQ(local_inidx);
  local_inidy = local_inidx + slen;

  count = 0;
  for ( i=0; i<nrecvs; i++ ) {
    values = rvalues + 2*i*nmax;
    for ( j=0; j<lens[i]; j++ ) {
      local_inidx[count]   = values[2*j] - base;
      local_inidy[count++] = values[2*j+1];
    }
  }
  PetscFree(rvalues); 
  PetscFree(lens);
 
  /* wait on sends */
  if (nsends) {
    MPI_Status *send_status;
    send_status = (MPI_Status *)PetscMalloc(nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(send_waits);
  PetscFree(svalues);

  /*
     should sort and remove duplicates from local_inidx,local_inidy 
  */
  ierr = StoPScatterCreate(slen,local_inidx,slen,local_inidy,yin,ctx); CHKERRQ(ierr);
  PetscFree(local_inidx);

  return 0;
}
