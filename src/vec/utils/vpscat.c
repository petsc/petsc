


#ifndef lint
static char vcid[] = "$Id: vpscat.c,v 1.63 1996/08/15 12:45:07 bsmith Exp bsmith $";
#endif
/*
    Defines parallel vector scatters.
*/

#include "sys.h"
#include "src/is/isimpl.h"
#include "src/vec/vecimpl.h"                     /*I "vec.h" I*/
#include "src/vec/impls/dvecimpl.h"
#include "src/vec/impls/mpi/pvecimpl.h"
#include "pinclude/pviewer.h"

int VecScatterView_MPI(PetscObject obj,Viewer viewer)
{
  VecScatter             ctx = (VecScatter) obj;
  VecScatter_MPI_General *to=(VecScatter_MPI_General *) ctx->todata;
  VecScatter_MPI_General *from=(VecScatter_MPI_General *) ctx->fromdata;
  int                    i,rank,ierr;
  FILE                   *fd;
  ViewerType             vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);

  if (vtype != ASCII_FILE_VIEWER && vtype != ASCII_FILES_VIEWER) return 0;

  MPI_Comm_rank(ctx->comm,&rank);
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  PetscSequentialPhaseBegin(ctx->comm,1);
  fprintf(fd,"[%d] Number sends %d self %d\n",rank,to->n,to->local.n);
  for ( i=0; i<to->n; i++ ){
    fprintf(fd,"[%d]   %d length %d to whom %d\n",rank,i,to->starts[i+1]-to->starts[i],to->procs[i]);
  }

  fprintf(fd,"Now the indices\n");
  for ( i=0; i<to->starts[to->n]; i++ ){
    fprintf(fd,"[%d]%d \n",rank,to->indices[i]);
  }

  fprintf(fd,"[%d]Number receives %d self %d\n",rank,from->n,from->local.n);
  for ( i=0; i<from->n; i++ ){
    fprintf(fd,"[%d] %d length %d to whom %d\n",rank,i,from->starts[i+1]-from->starts[i],
            from->procs[i]);
  }

  fprintf(fd,"Now the indices\n");
  for ( i=0; i<from->starts[from->n]; i++ ){
    fprintf(fd,"[%d]%d \n",rank,from->indices[i]);
  }

  fflush(fd);
  PetscSequentialPhaseEnd(ctx->comm,1);
  return 0;
}  

/*
    The next routine determines what part of  the local part of the scatter is an
exact copy of values into their current location. We check this here and
then know that we need not perform that portion of the scatter.
*/
static int VecScatterLocalOptimize_Private(VecScatter_Seq_General *gen_to,
                                           VecScatter_Seq_General *gen_from)
{
  int n = gen_to->n,n_nonmatching = 0,i,*to_slots = gen_to->slots,*from_slots = gen_from->slots;
  int *nto_slots, *nfrom_slots,j = 0;
  
  for ( i=0; i<n; i++ ) {
    if (to_slots[i] != from_slots[i]) n_nonmatching++;
  }

  if (!n_nonmatching) {
    gen_to->nonmatching_computed = 1;
    gen_to->n_nonmatching        = gen_from->n_nonmatching = 0;
    PLogInfo(0,"VecScatterLocalOptimize_Private:Reduced %d to 0\n");
  } else if (n_nonmatching == n) {
    gen_to->nonmatching_computed = -1;
    PLogInfo(0,"VecScatterLocalOptimize_Private:All values non-matching\n");
  } else {
    gen_to->nonmatching_computed = 1;
    gen_to->n_nonmatching        = gen_from->n_nonmatching = n_nonmatching;
    nto_slots                    = (int *) PetscMalloc(n_nonmatching*sizeof(int));CHKPTRQ(nto_slots);
    gen_to->slots_nonmatching    = nto_slots;
    nfrom_slots                  = (int *) PetscMalloc(n_nonmatching*sizeof(int));CHKPTRQ(nfrom_slots);
    gen_from->slots_nonmatching  = nfrom_slots;
    for ( i=0; i<n; i++ ) {
      if (to_slots[i] != from_slots[i]) {
        nto_slots[j]   = to_slots[i];
        nfrom_slots[j] = from_slots[i];
        j++;
      }
    }
    PLogInfo(0,"VecScatterLocalOptimize_Private:Reduced %d to %d\n",n,n_nonmatching);
  } 

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
static int VecScatterBegin_PtoP(Vec xin,Vec yin,InsertMode addv,int mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to, *gen_from;
  Vec_MPI                *x = (Vec_MPI *)xin->data,*y = (Vec_MPI*) yin->data;
  MPI_Comm               comm = ctx->comm;
  Scalar                 *xv = x->array,*yv = y->array, *val, *rvalues,*svalues;
  MPI_Request            *rwaits, *swaits;
  int                    tag = ctx->tag, i,j,*indices,*rstarts,*sstarts,*rprocs, *sprocs;
  int                    nrecvs, nsends,iend,ierr;

  if (mode & SCATTER_REVERSE ){
    gen_to   = (VecScatter_MPI_General *) ctx->fromdata;
    gen_from = (VecScatter_MPI_General *) ctx->todata;
  }
  else {
    gen_to   = (VecScatter_MPI_General *) ctx->todata;
    gen_from = (VecScatter_MPI_General *) ctx->fromdata;
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
  
  /* post receives:   */
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv(rvalues+rstarts[i],rstarts[i+1] - rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);
  }

  /* do sends:  */
  for ( i=0; i<nsends; i++ ) {
    val  = svalues + sstarts[i];
    iend = sstarts[i+1]-sstarts[i];

    for ( j=0; j<iend; j++ ) {
      val[j] = xv[*indices++];
      /* printf("[%d] sending idx %d val %g\n",PetscGlobalRank,indices[-1],val[j]); */
    } 
    MPI_Isend(val,iend, MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);
  }
  /* take care of local scatters */
  if (gen_to->local.n && addv == INSERT_VALUES) {
    if (yv == xv && !gen_to->local.nonmatching_computed) {
      ierr = VecScatterLocalOptimize_Private(&gen_to->local,&gen_from->local);CHKERRQ(ierr);
    }
    if (yv != xv || gen_to->local.nonmatching_computed == -1) {
      int *tslots = gen_to->local.slots, *fslots = gen_from->local.slots;
      int n       = gen_to->local.n;
      for ( i=0; i<n; i++ ) {yv[fslots[i]] = xv[tslots[i]];}
    } else {
      /* 
        In this case, it is copying the values into their old  locations, thus we can skip those  
      */
      int *tslots = gen_to->local.slots_nonmatching, *fslots = gen_from->local.slots_nonmatching;
      int n       = gen_to->local.n_nonmatching;
      for ( i=0; i<n; i++ ) {yv[fslots[i]] = xv[tslots[i]];}
    } 
  }
  else if (gen_to->local.n) {
    int *tslots = gen_to->local.slots, *fslots = gen_from->local.slots;
    int n = gen_to->local.n;
    for ( i=0; i<n; i++ ) {yv[fslots[i]] += xv[tslots[i]];}
  }

  return 0;
}

static int VecScatterEnd_PtoP(Vec xin,Vec yin,InsertMode addv,int mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to, *gen_from;
  Vec_MPI                *y = (Vec_MPI *)yin->data;
  Scalar                 *rvalues, *yv = y->array,*val;
  int                    nrecvs, nsends,i,*indices,count,imdex,n,*rstarts,*lindices;
  MPI_Request            *rwaits, *swaits;
  MPI_Status             rstatus, *sstatus;

  if (mode & SCATTER_REVERSE ){
    gen_to   = (VecScatter_MPI_General *) ctx->fromdata;
    gen_from = (VecScatter_MPI_General *) ctx->todata;
    sstatus  = gen_from->sstatus;
  }
  else {
    gen_to   = (VecScatter_MPI_General *) ctx->todata;
    gen_from = (VecScatter_MPI_General *) ctx->fromdata;
    sstatus  = gen_to->sstatus;
  }
  rvalues  = gen_from->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  rwaits   = gen_from->requests;
  swaits   = gen_to->requests;
  indices  = gen_from->indices;
  rstarts  = gen_from->starts;

  /*  wait on receives */
  count = nrecvs;
  while (count) {
    MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus);
    /* unpack receives into our local space */
    val      = rvalues + rstarts[imdex];
    n        = rstarts[imdex+1]-rstarts[imdex];
    lindices = indices + rstarts[imdex];
    if (addv == INSERT_VALUES) {
      for ( i=0; i<n; i++ ) {
        yv[lindices[i]] = *val++;
	/*    printf("[%d] recving idx %d val %g\n",PetscGlobalRank,indices[i],val[-1]); */
      }
    } else {
      for ( i=0; i<n; i++ ) {
       yv[lindices[i]] += *val++;
      }
    }
    count--;
  }
  /* wait on sends */
  if (nsends) {
    MPI_Waitall(nsends,swaits,sstatus);
  }
  return 0;
}
/* -------------------------------------------------------------------------------*/
/*
    Special scatters for fixed block sizes
*/
static int VecScatterBegin_PtoP_5(Vec xin,Vec yin,InsertMode addv,int mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to, *gen_from;
  Vec_MPI                *x = (Vec_MPI *)xin->data, *y = (Vec_MPI *)yin->data;
  Scalar                 *xv = x->array, *yv = y->array, *val, *rvalues,*svalues;
  MPI_Request            *rwaits, *swaits;
  int                    i,*indices,*sstarts,iend,j;
  int                    nrecvs, nsends,idx;

  if (mode & SCATTER_REVERSE ) SETERRQ(1,"VecScatterBegin_PtoP_5:No reverse currently");

  gen_to   = (VecScatter_MPI_General *) ctx->todata;
  gen_from = (VecScatter_MPI_General *) ctx->fromdata;

  rvalues  = gen_from->values;
  svalues  = gen_to->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  rwaits   = gen_from->requests;
  swaits   = gen_to->requests;
  indices  = gen_to->indices;
  sstarts  = gen_to->starts;
  
  /* post receives:   */
  MPI_Startall_irecv(gen_from->starts[nrecvs],nrecvs,rwaits); 

  /* this version packs all the messages together and sends */
  /*
  len  = 5*sstarts[nsends];
  val  = svalues;
  for ( i=0; i<len; i += 5 ) {
    idx     = *indices++;
    val[0] = xv[idx];
    val[1] = xv[idx+1];
    val[2] = xv[idx+2];
    val[3] = xv[idx+3];
    val[4] = xv[idx+4];
    val      += 5;
  }
  MPI_Startall_isend(len,nsends,swaits);
  */

  /* this version packs and sends one at a time */
  val  = svalues;
  for ( i=0; i<nsends; i++ ) {
    iend = sstarts[i+1]-sstarts[i];

    for ( j=0; j<iend; j++ ) {
      idx     = *indices++;
      val[0] = xv[idx];
      val[1] = xv[idx+1];
      val[2] = xv[idx+2];
      val[3] = xv[idx+3];
      val[4] = xv[idx+4];
      val    += 5;
    } 
    MPI_Start_isend(5*iend,swaits+i);
  }

  /* take care of local scatters */
  if (gen_to->local.n) {
    int *tslots = gen_to->local.slots, *fslots = gen_from->local.slots;
    int n       = gen_to->local.n, il,ir;
    if (addv == INSERT_VALUES) {
      for ( i=0; i<n; i++ ) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = xv[ir];
        yv[il+1] = xv[ir+1];
        yv[il+2] = xv[ir+2];
        yv[il+3] = xv[ir+3];
        yv[il+4] = xv[ir+4];
      }
    }  else {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
        yv[il+3] += xv[ir+3];
        yv[il+4] += xv[ir+4];
    }
  }

  return 0;
}

static int VecScatterEnd_PtoP_5(Vec xin,Vec yin,InsertMode addv,int mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to, *gen_from;
  Vec_MPI                *y = (Vec_MPI *)yin->data;
  Scalar                 *rvalues, *yv = y->array,*val;
  int                    nrecvs, nsends,i,*indices,count,imdex,n,*rstarts,*lindices;
  int                    idx;
  MPI_Request            *rwaits, *swaits;
  MPI_Status             rstatus, *sstatus;

  gen_to   = (VecScatter_MPI_General *) ctx->todata;
  gen_from = (VecScatter_MPI_General *) ctx->fromdata;
  sstatus  = gen_to->sstatus;
  rvalues  = gen_from->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  rwaits   = gen_from->requests;
  swaits   = gen_to->requests;
  indices  = gen_from->indices;
  rstarts  = gen_from->starts;

  /*  wait on receives */
  count = nrecvs;
  while (count) {
    MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus);
    /* unpack receives into our local space */
    val      = rvalues + 5*rstarts[imdex];
    lindices = indices + rstarts[imdex];
    n        = rstarts[imdex+1] - rstarts[imdex];
    if (addv == INSERT_VALUES) {
      for ( i=0; i<n; i++ ) {
        idx       = lindices[i];
        yv[idx]   = val[0];
        yv[idx+1] = val[1];
        yv[idx+2] = val[2];
        yv[idx+3] = val[3];
        yv[idx+4] = val[4];
	/*     printf("[%d]recving %d %g \n",PetscGlobalRank,idx,val[0]); */
        val      += 5;
      }
    } else {
      for ( i=0; i<n; i++ ) {
        idx       = lindices[i];
        yv[idx]   += val[0];
        yv[idx+1] += val[1];
        yv[idx+2] += val[2];
        yv[idx+3] += val[3];
        yv[idx+4] += val[4];
        val      += 5;
      }
    }
    count--;
  }
  /* wait on sends */
  if (nsends) {
    MPI_Waitall(nsends,swaits,sstatus);
  }
  return 0;
}

/* --------------------------------------------------------------------*/
static int VecScatterCopy_PtoP(VecScatter in,VecScatter out)
{
  VecScatter_MPI_General *in_to   = (VecScatter_MPI_General *) in->todata;
  VecScatter_MPI_General *in_from = (VecScatter_MPI_General *) in->fromdata,*out_to,*out_from;
  int                    len, ny;

  out->scatterbegin     = in->scatterbegin;
  out->scatterend       = in->scatterend;
  out->copy             = in->copy;
  out->destroy          = in->destroy;
  out->view             = in->view;

  /* allocate entire send scatter context */
  out_to           = PetscNew(VecScatter_MPI_General);CHKPTRQ(out_to);
  PetscMemzero(out_to,sizeof(VecScatter_MPI_General));
  PLogObjectMemory(out,sizeof(VecScatter_MPI_General));
  ny               = in_to->starts[in_to->n];
  len              = ny*(sizeof(int) + sizeof(Scalar)) + (in_to->n+1)*sizeof(int) +
                     (in_to->n)*(sizeof(int) + sizeof(MPI_Request));
  out_to->n        = in_to->n; 
  out_to->type     = in_to->type;

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
  out_to->local.nonmatching_computed = 0;
  out_to->local.n_nonmatching        = 0;
  out_to->local.slots_nonmatching    = 0;
  if (in_to->local.n) {
    out_to->local.slots = (int *) PetscMalloc(in_to->local.n*sizeof(int));
    CHKPTRQ(out_to->local.slots);
    PetscMemcpy(out_to->local.slots,in_to->local.slots,in_to->local.n*sizeof(int));
    PLogObjectMemory(out,in_to->local.n*sizeof(int));
  }
  else {out_to->local.slots = 0;}

  /* allocate entire receive context */
  out_from           = PetscNew(VecScatter_MPI_General);CHKPTRQ(out_from);
  PetscMemzero(out_from,sizeof(VecScatter_MPI_General));
  out_from->type     = in_from->type;
  PLogObjectMemory(out,sizeof(VecScatter_MPI_General));
  ny                 = in_from->starts[in_from->n];
  len                = ny*(sizeof(int) + sizeof(Scalar)) + (in_from->n+1)*sizeof(int) +
                       (in_from->n)*(sizeof(int) + sizeof(MPI_Request));
  out_from->n        = in_from->n; 
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
  out_from->local.nonmatching_computed = 0;
  out_from->local.n_nonmatching        = 0;
  out_from->local.slots_nonmatching    = 0;
  if (in_from->local.n) {
    out_from->local.slots = (int *) PetscMalloc(in_from->local.n*sizeof(int));
    PLogObjectMemory(out,in_from->local.n*sizeof(int));CHKPTRQ(out_from->local.slots);
    PetscMemcpy(out_from->local.slots,in_from->local.slots,in_from->local.n*sizeof(int));
  }
  else {out_from->local.slots = 0;}
  return 0;
}
/* --------------------------------------------------------------------*/

static int VecScatterDestroy_PtoP_5(PetscObject obj)
{
  VecScatter             ctx = (VecScatter) obj;
  VecScatter_MPI_General *gen_to   = (VecScatter_MPI_General *) ctx->todata;
  VecScatter_MPI_General *gen_from = (VecScatter_MPI_General *) ctx->fromdata;
  int                    i;

  if (gen_to->local.slots) PetscFree(gen_to->local.slots);
  if (gen_from->local.slots) PetscFree(gen_from->local.slots);
  if (gen_to->local.slots_nonmatching) PetscFree(gen_to->local.slots_nonmatching);
  if (gen_from->local.slots_nonmatching) PetscFree(gen_from->local.slots_nonmatching);

  /* release MPI resources obtained with MPI_Send_init() and MPI_Recv_init() */
  /* 
     IBM's PE version of MPI has a bug where freeing these guys will screw up later
     message passing.
  */
#if !defined(PARCH_rs6000)
  for (i=0; i<gen_to->n; i++) {
    MPI_Request_free(gen_to->requests + i);
  } 
  for (i=0; i<gen_from->n; i++) {
    MPI_Request_free(gen_from->requests + i);
  } 
#endif
 
  PetscFree(gen_to->sstatus);
  PetscFree(gen_to->values);
  PetscFree(gen_to);
  PetscFree(gen_from->values); 
  PetscFree(gen_from);
  PLogObjectDestroy(ctx);
  PetscHeaderDestroy(ctx);
  return 0;
}

static int VecScatterDestroy_PtoP(PetscObject obj)
{
  VecScatter             ctx = (VecScatter) obj;
  VecScatter_MPI_General *gen_to   = (VecScatter_MPI_General *) ctx->todata;
  VecScatter_MPI_General *gen_from = (VecScatter_MPI_General *) ctx->fromdata;

  if (gen_to->local.slots) PetscFree(gen_to->local.slots);
  if (gen_from->local.slots) PetscFree(gen_from->local.slots);
  if (gen_to->local.slots_nonmatching) PetscFree(gen_to->local.slots_nonmatching);
  if (gen_from->local.slots_nonmatching) PetscFree(gen_from->local.slots_nonmatching);
  PetscFree(gen_to->sstatus);
  PetscFree(gen_to->values); PetscFree(gen_to);
  PetscFree(gen_from->values); PetscFree(gen_from);
  PLogObjectDestroy(ctx);
  PetscHeaderDestroy(ctx);
  return 0;
}

/* --------------------------------------------------------------*/
/* create parallel to sequential scatter context                 */
/*
   bs indicates how many elements there are in each block. Normally
   this would be 1.
*/
int VecScatterCreate_PtoS(int nx,int *inidx,int ny,int *inidy,Vec xin,int bs,VecScatter ctx)
{
  Vec_MPI                *x = (Vec_MPI *)xin->data;
  VecScatter_MPI_General *from,*to;
  int                    *source,*lens,rank = x->rank, *owners = x->ownership;
  int                    size = x->size,*lowner,*start,found;
  int                    *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work;
  int                    *owner,*starts,count,tag = xin->tag,slen;
  int                    *rvalues,*svalues,base,imdex,nmax,*values,len,*indx,nprocslocal;
  MPI_Comm               comm = xin->comm;
  MPI_Request            *send_waits,*recv_waits;
  MPI_Status             recv_status,*send_status;
  
  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner  = (int *) PetscMalloc((nx+1)*sizeof(int)); CHKPTRQ(owner);
  for ( i=0; i<nx; i++ ) {
    idx = inidx[i];
    found = 0;
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,"VecScatterCreate_PtoS:Index out of range");
  }
  nprocslocal  = nprocs[rank]; 
  nprocs[rank] = procs[rank] = 0; 
  nsends       = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce( procs, work,size,MPI_INT,MPI_SUM,comm);
  nrecvs = work[rank]; 
  MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives:   */
  rvalues    = (int *) PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int)); CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv((rvalues+nmax*i),nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues    = (int *) PetscMalloc( (nx+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *)PetscMalloc((nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts     = (int *) PetscMalloc( (size+1)*sizeof(int) ); CHKPTRQ(starts);
  starts[0]  = 0; 
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
      MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);
    }
  }
  PetscFree(starts);

  base = owners[rank];

  /*  wait on receives */
  lens   = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
  source = lens + nrecvs;
  count  = nrecvs; slen = 0;
  while (count) {
    MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    MPI_Get_count(&recv_status,MPI_INT,&n);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen          += n;
    count--;
  }
  PetscFree(recv_waits); 
  
  /* allocate entire send scatter context */
  to = PetscNew(VecScatter_MPI_General); CHKPTRQ(to);
  PetscMemzero(to,sizeof(VecScatter_MPI_General));
  PLogObjectMemory(ctx,sizeof(VecScatter_MPI_General));
  len = slen*sizeof(int) + bs*slen*sizeof(Scalar) + (nrecvs+1)*sizeof(int) +
        nrecvs*(sizeof(int) + sizeof(MPI_Request));
  to->n         = nrecvs; 
  to->values    = (Scalar *) PetscMalloc( len ); CHKPTRQ(to->values);
  PLogObjectMemory(ctx,len);
  to->requests  = (MPI_Request *) (to->values + bs*slen);
  to->indices   = (int *) (to->requests + nrecvs); 
  to->starts    = (int *) (to->indices + slen);
  to->procs     = (int *) (to->starts + nrecvs + 1);
  to->sstatus   = (MPI_Status *) PetscMalloc((1+nrecvs)*sizeof(MPI_Status));CHKPTRQ(to->sstatus);
  ctx->todata   = (void *) to;
  to->starts[0] = 0;

  if (nrecvs) {
    indx = (int *) PetscMalloc( nrecvs*sizeof(int) ); CHKPTRQ(indx);
    for ( i=0; i<nrecvs; i++ ) indx[i] = i;
    PetscSortIntWithPermutation(nrecvs,source,indx);

    /* move the data into the send scatter */
    for ( i=0; i<nrecvs; i++ ) {
      to->starts[i+1] = to->starts[i] + lens[indx[i]];
      to->procs[i]    = source[indx[i]];
      values = rvalues + indx[i]*nmax;
      for ( j=0; j<lens[indx[i]]; j++ ) {
        to->indices[to->starts[i] + j] = values[j] - base;
      }
    }
    PetscFree(indx);
  }
  PetscFree(rvalues); PetscFree(lens);
 
  /* allocate entire receive scatter context */
  from = PetscNew(VecScatter_MPI_General);CHKPTRQ(from);
  PetscMemzero(from,sizeof(VecScatter_MPI_General));
  PLogObjectMemory(ctx,sizeof(VecScatter_MPI_General));
  len = ny*sizeof(int) + ny*bs*sizeof(Scalar) + (nsends+1)*sizeof(int) +
        nsends*(sizeof(int) + sizeof(MPI_Request));
  from->n        = nsends;
  from->values   = (Scalar *) PetscMalloc( len );
  PLogObjectMemory(ctx,len);
  from->requests = (MPI_Request *) (from->values + bs*ny);
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
    from->local.n     = to->local.n = nt = nprocslocal;    
    from->local.slots = (int *) PetscMalloc(nt*sizeof(int));CHKPTRQ(from->local.slots);
    to->local.slots   = (int *) PetscMalloc(nt*sizeof(int));CHKPTRQ(to->local.slots);
    PLogObjectMemory(ctx,2*nt*sizeof(int));
    nt = 0;
    for ( i=0; i<nx; i++ ) {
      idx = inidx[i];
      if (idx >= owners[rank] && idx < owners[rank+1]) {
        to->local.slots[nt]     = idx - owners[rank];        
        from->local.slots[nt++] = inidy[i];        
      }
    }
  }
  else { 
    from->local.n     = 0;
    from->local.slots = 0;
    to->local.n       = 0; 
    to->local.slots   = 0;
  } 
  from->local.nonmatching_computed = 0;
  from->local.n_nonmatching        = 0;
  from->local.slots_nonmatching    = 0;
  to->local.nonmatching_computed   = 0;
  to->local.n_nonmatching          = 0;
  to->local.slots_nonmatching      = 0;

  to->type   = VEC_SCATTER_MPI_GENERAL; 
  from->type = VEC_SCATTER_MPI_GENERAL;

  if (bs == 5) {
    int         *sstarts = to->starts,   *rstarts = from->starts;
    int         *sprocs  = to->procs,    *rprocs  = from->procs;
    MPI_Request *swaits  = to->requests, *rwaits  = from->requests;
    Scalar      *Ssvalues = to->values,  *Srvalues = from->values;

    ctx->destroy        = VecScatterDestroy_PtoP_5;
    ctx->scatterbegin   = VecScatterBegin_PtoP_5;
    ctx->scatterend     = VecScatterEnd_PtoP_5; 
    ctx->copy           = 0;
    ctx->view           = VecScatterView_MPI;
  
    tag     = ctx->tag;
    comm    = ctx->comm;

    /* Register the sends and receives that you will use later */
    for ( i=0; i<from->n; i++ ) {
      MPI_Recv_init(Srvalues + 5*rstarts[i],5*rstarts[i+1] - 5*rstarts[i],MPIU_SCALAR,rprocs[i],tag,
                    comm,rwaits+i);
    }

    for ( i=0; i<to->n; i++ ) {
      MPI_Send_init(Ssvalues + 5*sstarts[i],5*sstarts[i+1] - 5*sstarts[i],MPIU_SCALAR,sprocs[i],tag,
                    comm,swaits+i);
    } 

    PLogInfo(0,"Using blocksize 5 scatter\n");
  } else {
    ctx->destroy        = VecScatterDestroy_PtoP;
    ctx->scatterbegin   = VecScatterBegin_PtoP;
    ctx->scatterend     = VecScatterEnd_PtoP; 
    ctx->copy           = VecScatterCopy_PtoP;
    ctx->view           = VecScatterView_MPI;
  }

  return 0;
}

/* ----------------------------------------------------------------*/
/*
    Scatter from local Seq vectors to a parallel vector. 
 */
int VecScatterCreate_StoP(int nx,int *inidx,int ny,int *inidy,Vec yin,VecScatter ctx)
{
  Vec_MPI                *y = (Vec_MPI *)yin->data;
  VecScatter_MPI_General *from,*to;
  int                    *source,nprocslocal,*lens,rank = y->rank, *owners = y->ownership;
  int                    size = y->size,*lowner,*start;
  int                    *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work;
  int                    *owner,*starts,count,tag = yin->tag,slen;
  int                    *rvalues,*svalues,base,imdex,nmax,*values,len,found;
  MPI_Comm               comm = yin->comm;
  MPI_Request            *send_waits,*recv_waits;
  MPI_Status             recv_status,*send_status;

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc((nx+1)*sizeof(int)); CHKPTRQ(owner); 
  for ( i=0; i<nx; i++ ) {
    idx = inidy[i];
    found = 0;
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,"VecScatterCreate_StoP:Index out of range");
  }
  nprocslocal  = nprocs[rank];
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
  rvalues    = (int *) PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int)); CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues    = (int *) PetscMalloc( (nx+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *)PetscMalloc((nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts     = (int *) PetscMalloc( (size+1)*sizeof(int) ); CHKPTRQ(starts);
  starts[0]  = 0; 
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
  to = PetscNew(VecScatter_MPI_General); CHKPTRQ(to);
  PetscMemzero(to,sizeof(VecScatter_MPI_General));
  PLogObjectMemory(ctx,sizeof(VecScatter_MPI_General));
  len = ny*(sizeof(int) + sizeof(Scalar)) + (nsends+1)*sizeof(int) +
        nsends*(sizeof(int) + sizeof(MPI_Request));
  to->n        = nsends; 
  to->values   = (Scalar *) PetscMalloc( len ); CHKPTRQ(to->values); 
  PLogObjectMemory(ctx,len);
  to->requests = (MPI_Request *) (to->values + ny);
  to->indices  = (int *) (to->requests + nsends); 
  to->starts   = (int *) (to->indices + ny);
  to->procs    = (int *) (to->starts + nsends + 1);
  to->sstatus  = (MPI_Status *) PetscMalloc((1+nsends)*sizeof(MPI_Status));CHKPTRQ(to->sstatus);
  ctx->todata  = (void *) to;

  /* move data into send scatter context */
  lowner        = (int *) PetscMalloc( (size+nsends+1)*sizeof(int) ); CHKPTRQ(lowner);
  start         = lowner + size;
  count         = 0;
  to->starts[0] = start[0] = 0;
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
  lens   = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
  source = lens + nrecvs;
  count  = nrecvs; slen = 0;
  while (count) {
    MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    MPI_Get_count(&recv_status,MPI_INT,&n);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen          += n;
    count--;
  }
  PetscFree(recv_waits); 
 
  /* allocate entire receive scatter context */
  from = PetscNew( VecScatter_MPI_General ); CHKPTRQ(from);
  PetscMemzero(from,sizeof(VecScatter_MPI_General));
  PLogObjectMemory(ctx,sizeof(VecScatter_MPI_General));
  len = slen*(sizeof(int) + sizeof(Scalar)) + (nrecvs+1)*sizeof(int) +
        nrecvs*(sizeof(int) + sizeof(MPI_Request));
  from->n        = nrecvs; 
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
    values            = rvalues + i*nmax;
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
    from->local.n     = to->local.n = nt = nprocslocal;    
    from->local.slots = (int *) PetscMalloc(nt*sizeof(int));CHKPTRQ(from->local.slots);
    to->local.slots   = (int *) PetscMalloc(nt*sizeof(int));CHKPTRQ(to->local.slots);
    PLogObjectMemory(ctx,2*nt*sizeof(int));
    nt = 0;
    for ( i=0; i<ny; i++ ) {
      idx = inidy[i];
      if (idx >= owners[rank] && idx < owners[rank+1]) {
        from->local.slots[nt] = idx - owners[rank];        
        to->local.slots[nt++] = inidx[i];        
      }
    }
  }
  else {
    from->local.n     = 0; 
    from->local.slots = 0;
    to->local.n       = 0;
    to->local.slots   = 0;

  }
  from->local.nonmatching_computed = 0;
  from->local.n_nonmatching        = 0;
  from->local.slots_nonmatching    = 0;
  to->local.nonmatching_computed   = 0;
  to->local.n_nonmatching          = 0;
  to->local.slots_nonmatching      = 0;

  to->type   = VEC_SCATTER_MPI_GENERAL; 
  from->type = VEC_SCATTER_MPI_GENERAL;

  ctx->destroy           = VecScatterDestroy_PtoP;
  ctx->scatterbegin      = VecScatterBegin_PtoP;
  ctx->scatterend        = VecScatterEnd_PtoP; 
  ctx->copy              = 0;
  ctx->view              = VecScatterView_MPI;

  return 0;
}

/* ---------------------------------------------------------------------------------*/
int VecScatterCreate_PtoP(int nx,int *inidx,int ny,int *inidy,Vec xin,Vec yin,VecScatter ctx)
{
  Vec_MPI     *x = (Vec_MPI *)xin->data;
  int         *lens,rank = x->rank, *owners = x->ownership,size = x->size,found;
  int         *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work,*local_inidx,*local_inidy;
  int         *owner,*starts,count,tag = xin->tag,slen,ierr;
  int         *rvalues,*svalues,base,imdex,nmax,*values;
  MPI_Comm    comm = xin->comm;
  MPI_Request *send_waits,*recv_waits;
  MPI_Status  recv_status;

  /*
  Each processor ships off its inidx[j] and inidy[j] to the appropriate processor
  They then call the StoPScatterCreate()
  */
  /*  first count number of contributors to each processor */
  nprocs  = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner   = (int *) PetscMalloc((nx+1)*sizeof(int)); CHKPTRQ(owner);
  for ( i=0; i<nx; i++ ) {
    idx   = inidx[i];
    found = 0;
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,"VecScatterCreate_PtoP:Index out of range");
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
  rvalues    = (int *) PetscMalloc(2*(nrecvs+1)*(nmax+1)*sizeof(int)); CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv(rvalues+2*nmax*i,2*nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues    = (int *) PetscMalloc( 2*(nx+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *)PetscMalloc((nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts     = (int *) PetscMalloc( (size+1)*sizeof(int) ); CHKPTRQ(starts);
  starts[0]  = 0; 
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
  lens  = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
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
  ierr = VecScatterCreate_StoP(slen,local_inidx,slen,local_inidy,yin,ctx); CHKERRQ(ierr);
  PetscFree(local_inidx);

  return 0;
}





