#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pipeline.c,v 1.10 1998/12/03 03:56:41 bsmith Exp balay $";
#endif

/*
       Vector pipeline routines. These routines have all been contributed
    by Victor Eijkhout while working at UCLA and UTK. 

       Note: these are not completely PETScified. For example the PETSCHEADER in the 
   pipeline object is not completely filled in so that the pipeline object cannot 
   be used as a TRUE PETSc object. Also there are some memory leaks. This code 
   attempts to reuse some of the VecScatter internal code and thus is kind of tricky.

*/

#include "src/vec/vecimpl.h" /*I "vec.h" I*/
#include "sys.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"

typedef int (*PipelineFunction)(int,PetscObject);

struct _p_VecPipeline {
  PETSCHEADER(int)
  VecScatter             scatter;
  PipelineType           pipe_type; /* duplicated in the subdomain structure */
  VecScatter_MPI_General *upto,*upfrom,*dnto,*dnfrom;
  VecScatter_MPI_General *scatterto,*scatterfrom;
  PipelineFunction       upfn,dnfn;
  PetscObject            aux_data;
  PetscObject            custom_pipe_data;
  int                    setupcalled;
  int                    (*setup)(VecPipeline,PetscObject,PetscObject*);
};

#undef __FUNC__
#define __FUNC__ "VecPipelineCreateUpDown"
static int VecPipelineCreateUpDown(VecScatter scatter,VecScatter_MPI_General **to,VecScatter_MPI_General **from)
{
  VecScatter_MPI_General *gen_to,*gen_from, *pipe_to,*pipe_from;

  gen_to    = (VecScatter_MPI_General *) scatter->todata;
  gen_from  = (VecScatter_MPI_General *) scatter->fromdata;

  pipe_to   = (VecScatter_MPI_General *)PetscMalloc(sizeof(VecScatter_MPI_General));CHKPTRQ(pipe_to);
  PetscMemzero(pipe_to,sizeof(VecScatter_MPI_General));
  pipe_from = (VecScatter_MPI_General *)PetscMalloc(sizeof(VecScatter_MPI_General));CHKPTRQ(pipe_from);
  PetscMemzero(pipe_from,sizeof(VecScatter_MPI_General));

  pipe_to->requests   = gen_to->requests;
  pipe_from->requests = gen_from->requests;
  pipe_to->local      = gen_to->local;
  pipe_from->local    = gen_from->local;
  pipe_to->values     = gen_to->values;
  pipe_from->values   = gen_from->values;
  pipe_to->sstatus    = (MPI_Status *)PetscMalloc((gen_to->n+1)*sizeof(MPI_Status));CHKPTRQ(pipe_to->sstatus);
  pipe_from->sstatus  = (MPI_Status *)PetscMalloc((gen_from->n+1)*sizeof(MPI_Status));CHKPTRQ(pipe_to->sstatus);
  
  *to   = pipe_to; 
  *from = pipe_from;

  return 0;
}

/* --------------------------------------------------------------*/
/*C
   VecPipelineCreate - Creates a vector pipeline context.
*/
#undef __FUNC__
#define __FUNC__ "VecPipelineCreate"
int VecPipelineCreate(MPI_Comm comm,Vec xin,IS ix,Vec yin,IS iy,VecPipeline *newctx)
{
  VecPipeline ctx;
  int ierr;

  ctx       = (VecPipeline) PetscMalloc(sizeof(struct _p_VecPipeline));CHKPTRQ(ctx);
  PetscMemzero(ctx,sizeof(struct _p_VecPipeline));
  ctx->comm = comm;
  ierr      = VecScatterCreate(xin,ix,yin,iy,&(ctx->scatter)); CHKERRQ(ierr);
  ierr      = VecPipelineSetType(ctx,PIPELINE_SEQUENTIAL,PETSC_NULL); CHKERRQ(ierr);
  ctx->setupcalled = 0;
  ctx->upfn        = 0;
  ctx->dnfn        = 0;

  /*
      Keep a copy of the original scatter fields so we can destroy them later
     when the pipeline is destroyed
  */
  ctx->scatterto   = (VecScatter_MPI_General *)ctx->scatter->todata;
  ctx->scatterfrom = (VecScatter_MPI_General *)ctx->scatter->fromdata;

  ierr = VecPipelineCreateUpDown(ctx->scatter,&(ctx->upto),&(ctx->upfrom));CHKERRQ(ierr);
  ierr = VecPipelineCreateUpDown(ctx->scatter,&(ctx->dnto),&(ctx->dnfrom));CHKERRQ(ierr);

  *newctx = ctx;

  return 0;
}

#undef __FUNC__
#define __FUNC__ "VecPipelineSetupSelect"
static int VecPipelineSetupSelect(VecScatter_MPI_General *gen,VecScatter_MPI_General *pipe,
                                  int (*test)(int,PetscObject),PetscObject pipe_data)
{
  int i;

  pipe->n = 0;
  for (i=0; i<gen->n; i++) {
    if ((*test)(gen->procs[i],pipe_data)) {
      pipe->n++;
    }
  }
  
  pipe->procs = (int *) PetscMalloc((pipe->n+1)*sizeof(int));CHKPTRQ(pipe->procs);
  pipe->starts = (int *) PetscMalloc((pipe->n+1)*sizeof(int));CHKPTRQ(pipe->starts);
  {
    int pipe_size = 1;
    if (gen->n) pipe_size = gen->starts[gen->n]+1;
    pipe->indices = (int *) PetscMalloc(pipe_size*sizeof(int));CHKPTRQ(pipe->indices); 
  }
  {
    int *starts = gen->starts, *pstarts = pipe->starts;
    int *procs = gen->procs, *pprocs = pipe->procs;
    int *indices = gen->indices, *pindices = pipe->indices;
    int n = 0;
    
    pstarts[0]=0;
    for (i=0; i<gen->n; i++) {
      if ((*test)(gen->procs[i],pipe_data)) {
	int j;
	pprocs[n] = procs[i];
	pstarts[n+1] = pstarts[n]+ starts[i+1]-starts[i];
	for (j=0; j<pstarts[n+1]-pstarts[n]; j++) {
	  pindices[pstarts[n]+j] = indices[starts[i]+j];
	}
	n++;
      }
    }
  }

  return 0;
}

/*C
   VecPipelineSetup - Sets up a vector pipeline context.
   This call is done implicitly in VecPipelineBegin, but
   since it is a bit costly, you may want to do it explicitly
   when timing.
*/
#undef __FUNC__
#define __FUNC__ "VecPipelineSetup"
int VecPipelineSetup(VecPipeline ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  int                    ierr;

  if (ctx->setupcalled) return 0;

  ierr = (ctx->setup)(ctx,ctx->aux_data,&(ctx->custom_pipe_data));CHKERRQ(ierr);

  gen_to    = (VecScatter_MPI_General *) ctx->scatter->todata;
  gen_from  = (VecScatter_MPI_General *) ctx->scatter->fromdata;

  /* data for PIPELINE_UP */
  ierr = VecPipelineSetupSelect(gen_to,ctx->upto,ctx->upfn,ctx->custom_pipe_data);CHKERRQ(ierr);
  ierr = VecPipelineSetupSelect(gen_from,ctx->upfrom,ctx->dnfn,ctx->custom_pipe_data);CHKERRQ(ierr);

  /* data for PIPELINE_DOWN */
  ierr = VecPipelineSetupSelect(gen_to,ctx->dnto,ctx->dnfn,ctx->custom_pipe_data);CHKERRQ(ierr);
  ierr = VecPipelineSetupSelect(gen_from,ctx->dnfrom,ctx->upfn,ctx->custom_pipe_data);CHKERRQ(ierr);

  ctx->setupcalled = 1;
  
  return 0;
}

/*
   VecPipelineSetType
*/
extern int ProcYes(int proc,PetscObject pipe_info);
extern int ProcUp(int proc,PetscObject pipe_info);
extern int ProcDown(int proc,PetscObject pipe_info);
extern int ProcColorUp(int proc,PetscObject pipe_info);
extern int ProcColorDown(int proc,PetscObject pipe_info);
extern int PipelineSequentialSetup(VecPipeline,PetscObject,PetscObject*);
extern int PipelineRedblackSetup(VecPipeline,PetscObject,PetscObject*);
extern int PipelineMulticolorSetup(VecPipeline,PetscObject,PetscObject*);

int ProcNo(int proc,PetscObject pipe_info);

#undef __FUNC__
#define __FUNC__ "VecPipelineSetType"
/*C
   VecPipelineSetType - Sets the type of a vector pipeline. Vector
   pipelines are to be used as

   VecPipelineBegin(<see below for parameters>)
   <do useful work with incoming data>
   VecPipelineEnd(<see below for paramteres>)

   Input Parameters:
+  ctx - vector pipeline object
+  type - vector pipeline type
   Choices currently allowed are 
   -- PIPELINE_NONE all processors send and receive simultaneously
   -- PIPELINE_SEQUENTIAL processors send and receive in ascending
   order of MPI rank
   -- PIPELINE_RED_BLACK even numbered processors only send; odd numbered
   processors only receive
   -- PIPELINE_MULTICOLOR processors are given a color and send/receive
   according to ascending color
+  x - auxiliary data; for PIPELINE_MULTICOLOR this should be
   <(PetscObject) pmat> where pmat is the matrix on which the coloring
   is to be based.

.seealso: VecPipelineCreate(), VecPipelineBegin(), VecPipelineEnd().
*/
int VecPipelineSetType(VecPipeline ctx,PipelineType type,PetscObject x)
{
  ctx->pipe_type = type;
  ctx->aux_data = x;
  if (type==PIPELINE_NONE) {
    ctx->upfn = &ProcYes;
    ctx->dnfn = &ProcYes;
    ctx->setup = &PipelineSequentialSetup;
  } else if (type==PIPELINE_SEQUENTIAL) {
    ctx->upfn = &ProcUp;
    ctx->dnfn = &ProcDown;
    ctx->setup = &PipelineSequentialSetup;
  } else if (type == PIPELINE_REDBLACK) {
    ctx->upfn = &ProcColorUp;
    ctx->dnfn = &ProcColorDown;
    ctx->setup = &PipelineRedblackSetup;
  } else if (type == PIPELINE_MULTICOLOR) {
    ctx->upfn = &ProcColorUp;
    ctx->dnfn = &ProcColorDown;
    ctx->setup = &PipelineMulticolorSetup;
  } else {
    SETERRQ(1,(int)type,"VecPipelineSetType: unknown or not implemented type\n");
  }

  return 0;
}

#undef __FUNC__
#define __FUNC__ "VecPipelineBegin"
/*
   VecPipelineBegin - Receive data from processor earlier in
   a processor pipeline from one vector to another. 

.seealso: VecPipelineEnd.
*/
int VecPipelineBegin(Vec x,Vec y,InsertMode addv,ScatterMode smode,PipelineDirection pmode,VecPipeline ctx)
{
  int ierr;

  if (!ctx->setupcalled) {
    ierr = VecPipelineSetup(ctx); CHKERRQ(ierr);
  }

  if (pmode==PIPELINE_UP) {
    ctx->scatter->todata = ctx->upto;
    ctx->scatter->fromdata = ctx->upfrom;
  } else if (pmode==PIPELINE_DOWN) {
    ctx->scatter->todata = ctx->dnto;
    ctx->scatter->fromdata = ctx->dnfrom;
  } else SETERRQ(1,pmode,"VecPipelineBegin: unknown or not implemented pipeline mode");

  {
    VecScatter             scat = ctx->scatter;
    VecScatter_MPI_General *gen_to;
    int                    nsends=0;

    if (smode & SCATTER_REVERSE ){
      gen_to   = (VecScatter_MPI_General *) scat->fromdata;
    }
    else {
      gen_to   = (VecScatter_MPI_General *) scat->todata;
    }
    if (ctx->pipe_type != PIPELINE_NONE) {
      nsends   = gen_to->n;
      gen_to->n = 0;
    }
    ierr = VecScatterBegin(x,y,addv,smode,scat); CHKERRQ(ierr);
    ierr = VecScatterEnd(x,y,addv,smode,scat); CHKERRQ(ierr);
    if (ctx->pipe_type != PIPELINE_NONE) gen_to->n = nsends;
  }
  return 0;
}

#undef __FUNC__
#define __FUNC__ "VecPipelineEnd"
/*
   VecPipelineEnd - Send data to processors later in
   a processor pipeline from one vector to another.
 
.seealso: VecPipelineBegin.
*/
int VecPipelineEnd(Vec x,Vec y,InsertMode addv,ScatterMode smode,PipelineDirection pmode, VecPipeline ctx)
{
  VecScatter             scat = ctx->scatter;
  VecScatter_MPI_General *gen_from,*gen_to;
  int                    nsends=0,nrecvs,ierr;
  
  if (smode & SCATTER_REVERSE ){
    gen_to   = (VecScatter_MPI_General *) scat->fromdata;
    gen_from = (VecScatter_MPI_General *) scat->todata;
  } else {
    gen_to   = (VecScatter_MPI_General *) scat->todata;
    gen_from = (VecScatter_MPI_General *) scat->fromdata;
  }
  if (ctx->pipe_type == PIPELINE_NONE) {
      nsends   = gen_to->n;
      gen_to->n = 0;
  }
  nrecvs      = gen_from->n;
  gen_from->n = 0;
  ierr = VecScatterBegin(x,y,addv,smode,scat); CHKERRQ(ierr);
  ierr = VecScatterEnd(x,y,addv,smode,scat); CHKERRQ(ierr);
  if (ctx->pipe_type == PIPELINE_NONE) gen_to->n = nsends;
  gen_from->n = nrecvs;

  return 0;
}

/*
      This destroys the material that is allocated inside the 
   VecScatter_MPI_General datastructure. Mimics the 
   VecScatterDestroy_PtoP() object except it does not destroy 
   the wrapper VecScatter object.
*/

#undef __FUNC__
#define __FUNC__ "VecPipelineDestroy_MPI_General"
static int VecPipelineDestroy_MPI_General(VecScatter_MPI_General *gen)
{
  PetscFunctionBegin;
  if (gen->sstatus) {PetscFree(gen->sstatus);}
  PetscFree(gen);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "VecPipelineDestroy"
/*C
   VecPipelineDestroy - Destroys a pipeline context created by 
   VecPipelineCreate().

*/
int VecPipelineDestroy( VecPipeline ctx )
{
  int ierr;

  /* free the VecScatter_MPI_General data structures */
  if (ctx->upto) {
    PetscFree(ctx->upto->procs);
    PetscFree(ctx->upto->starts);
    PetscFree(ctx->upto->indices);
    ierr = VecPipelineDestroy_MPI_General(ctx->upto);CHKERRQ(ierr);
  }
  if (ctx->upfrom) {
    PetscFree(ctx->upfrom->procs);
    PetscFree(ctx->upfrom->starts);
    PetscFree(ctx->upfrom->indices);
    ierr = VecPipelineDestroy_MPI_General(ctx->upfrom);CHKERRQ(ierr);
  }
  if (ctx->dnto) {
    PetscFree(ctx->dnto->procs);
    PetscFree(ctx->dnto->starts);
    PetscFree(ctx->dnto->indices);
    ierr = VecPipelineDestroy_MPI_General(ctx->dnto);CHKERRQ(ierr);
  }
  if (ctx->dnfrom) {
    PetscFree(ctx->dnfrom->procs);
    PetscFree(ctx->dnfrom->starts);
    PetscFree(ctx->dnfrom->indices);
    ierr = VecPipelineDestroy_MPI_General(ctx->dnfrom);CHKERRQ(ierr);
  }
  if (ctx->scatterto) {
    PetscFree(ctx->scatterto->values);
    if (ctx->scatterto->local.slots) PetscFree(ctx->scatterto->local.slots);
    ierr = VecPipelineDestroy_MPI_General(ctx->scatterto);CHKERRQ(ierr);
  }
  if (ctx->scatterfrom) {
    PetscFree(ctx->scatterfrom->values);
    if (ctx->scatterfrom->local.slots) PetscFree(ctx->scatterfrom->local.slots);
    ierr = VecPipelineDestroy_MPI_General(ctx->scatterfrom);CHKERRQ(ierr);
  }

  if (ctx->custom_pipe_data) PetscFree(ctx->custom_pipe_data);
  PetscHeaderDestroy(ctx->scatter);
  PetscFree(ctx);

  return 0;
}

/* >>>> Routines for sequential ordering of processors <<<< */

typedef struct {int rank;} Pipeline_sequential_info;

#undef __FUNC__
#define __FUNC__ "ProcYes"
int ProcYes(int proc,PetscObject pipe_info)
{
  return 1;
}
#undef __FUNC__
#define __FUNC__ "IsProcYes"
int IsProcYes(long fun)
{
  /* 
     Pass the actual function pointer, instead of typecasting it
     into a long?
  */
  return (fun==(long)&ProcYes);
}
#undef __FUNC__
#define __FUNC__ "ProcNo"
int ProcNo(int proc,PetscObject pipe_info)
{
  return 0;
}

#undef __FUNC__
#define __FUNC__ "ProcUp"
int ProcUp(int proc,PetscObject pipe_info)
{
  int rank = ((Pipeline_sequential_info *)pipe_info)->rank;

  if (rank<proc) {
    return 1;
  } else {
    return 0;
  }
}
int ProcDown(int proc,PetscObject pipe_info)
{ 
  int rank = ((Pipeline_sequential_info *)pipe_info)->rank;

  if (rank>proc) {
    return 1;
  } else {
    return 0;
  }
}

#undef __FUNC__
#define __FUNC__ "PipelineSequentialSetup"
int PipelineSequentialSetup(VecPipeline vs,PetscObject x,PetscObject *obj)
{
  Pipeline_sequential_info *info;

  info = PetscNew(Pipeline_sequential_info);
  MPI_Comm_rank(vs->scatter->comm,&(info->rank));
  *obj = (PetscObject) info;

  return 0;
}

/* >>>> Routines for multicolor ordering of processors <<<< */

typedef struct {
  int rank,size,*proc_colors;
} Pipeline_colored_info;

int ProcColorUp(int proc,PetscObject pipe_info)
{
  Pipeline_colored_info* comm_info = (Pipeline_colored_info *) pipe_info;
  int                     rank = comm_info->rank;

  if (comm_info->proc_colors[rank]<comm_info->proc_colors[proc]) {
    return 1;
  } else {
    return 0;
  }
}
int ProcColorDown(int proc,PetscObject pipe_info)
{ 
  Pipeline_colored_info* comm_info = (Pipeline_colored_info *) pipe_info;
  int rank = comm_info->rank;

  if (comm_info->proc_colors[rank]>comm_info->proc_colors[proc]) {
    return 1;
  } else {
    return 0;
  }
}

#undef __FUNC__
#define __FUNC__ "PipelineRedblackSetup"
int PipelineRedblackSetup(VecPipeline vs,PetscObject x,PetscObject *obj)
{
  Pipeline_colored_info *info;
  int                    size,i;

  info = PetscNew(Pipeline_colored_info);
  MPI_Comm_rank(vs->scatter->comm,&(info->rank));
  MPI_Comm_size(vs->scatter->comm,&size);
  info->proc_colors = (int*)PetscMalloc(size*sizeof(int));CHKPTRQ(info->proc_colors);
  for (i=0; i<size; i++) {info->proc_colors[i] = i%2;}
  *obj = (PetscObject) info;

  return 0;
}

#undef __FUNC__
#define __FUNC__ "PipelineMulticolorSetup"
int PipelineMulticolorSetup(VecPipeline vs,PetscObject x,PetscObject *obj)
{
  Pipeline_colored_info *info;
  Mat                    mat = (Mat) x;
  int                    size;

  info = PetscNew(Pipeline_colored_info);
  MPI_Comm_rank(mat->comm,&(info->rank));
  MPI_Comm_size(mat->comm,&size);
  info->proc_colors = (int*)PetscMalloc(size*sizeof(int));CHKPTRQ(info->proc_colors);
  PetscMemzero(info->proc_colors,size*sizeof(int));

  /* coloring */
  {
    Mat_MPIAIJ  *Aij = (Mat_MPIAIJ *) mat->data;
    int *owners = Aij->rowners, *touch = Aij->garray;
    int ntouch = ((Mat_SeqAIJ *)Aij->B->data)->n;
    int *conn,*colr;
    int *colors = info->proc_colors, base = info->rank*size;
    int p,e;

    /* allocate connectivity matrix */
    conn = (int *) PetscMalloc(size*size*sizeof(int)); CHKPTRQ(conn);
    colr = (int *) PetscMalloc(size*sizeof(int)); CHKPTRQ(colr);
    PetscMemzero(conn,size*size*sizeof(int));

    /* fill in local row of connectivity matrix */
    p = 0; e = 0;
  loop:
    while (touch[e]>=owners[p+1]) {
      p++;
#if defined(PETSC_DEBUG)
      if (p>=size) SETERRQ(1,p,"Processor overflow");
#endif
    }
    conn[base+p] = 1;
    if (p==size-1) ;
    else {
      while (touch[e]<owners[p+1]) {
	e++;
	if (e>=ntouch) goto exit;
      }
      goto loop;
    }
  exit:
    /* distribute to establish local copies of full connectivity matrix */
    MPI_Allgather(conn+base,size,MPI_INT,conn,size,MPI_INT,mat->comm);

    base = size;
    /*PetscPrintf(mat->comm,"Coloring: 0->0");*/
    for (p=1; p<size; p++) {
      int q,hi=-1,nc=0;
      PetscMemzero(colr,size*sizeof(int));
      for (q=0; q<p; q++) { /* inspect colors of all connect previous procs */
	if (conn[base+q] /* should be tranposed! */) {
	  if (!colr[colors[q]]) {
	    nc++;
	    colr[colors[q]] = 1;
	    if (colors[q]>hi) hi = colors[q];
	  }
	}
      } 
      if (hi+1!=nc) {
	nc = 0;
	while (colr[nc]) nc++;
      }
      colors[p] = nc;
      /*PetscPrintf(mat->comm,", %d->%d",p,colors[p]);*/
      base = base+size;
    }
    /*PetscPrintf(mat->comm,"\n");*/
    PetscFree(conn);
    PetscFree(colr);
  }
  *obj = (PetscObject) info;

  return 0;
}

#undef __FUNC__
#define __FUNC__ "VecPipelineView"
int VecPipelineView(VecPipeline pipe,Viewer viewer)
{
  MPI_Comm comm = pipe->comm;

  PetscPrintf(comm,">> Vector Pipeline<<\n");
  if (!pipe->setupcalled) PetscPrintf(comm,"Not yet set up\n");
  PetscPrintf(comm,"Pipelinetype: %d\n",(int)pipe->pipe_type);
  PetscPrintf(comm,"based on scatter:\n");
  /*  ierr = VecScatterView(pipe->scatter,viewer); CHKERRQ(ierr);*/
  PetscPrintf(comm,"Up function %p\n",pipe->upfn);
  PetscPrintf(comm,"Dn function %p\n",pipe->dnfn);

  return 0;
}









