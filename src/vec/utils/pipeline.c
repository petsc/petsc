/*
       Vector pipeline routines. These routines have all been contributed
    by Victor Eijkhout while working at UCLA and UTK. 

       Note: these are not completely PETScified. For example the PETSCHEADER in the 
   pipeline object is not completely filled in so that the pipeline object cannot 
   be used as a TRUE PETSc object. Also there are some memory leaks. This code 
   attempts to reuse some of the VecScatter internal code and thus is kind of tricky.

*/

#include "vecimpl.h" /*I "petscvec.h" I*/
#include "petscsys.h"
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

#undef __FUNCT__
#define __FUNCT__ "VecPipelineCreateUpDown"
static int VecPipelineCreateUpDown(VecScatter scatter,VecScatter_MPI_General **to,VecScatter_MPI_General **from)
{
  VecScatter_MPI_General *gen_to,*gen_from,*pipe_to,*pipe_from;
  int                    ierr;

  PetscFunctionBegin;
  gen_to    = (VecScatter_MPI_General*)scatter->todata;
  gen_from  = (VecScatter_MPI_General*)scatter->fromdata;

  ierr = PetscNew(VecScatter_MPI_General,&pipe_to);CHKERRQ(ierr);
  ierr = PetscMemzero(pipe_to,sizeof(VecScatter_MPI_General));CHKERRQ(ierr);
  ierr = PetscNew(VecScatter_MPI_General,&pipe_from);CHKERRQ(ierr);
  ierr = PetscMemzero(pipe_from,sizeof(VecScatter_MPI_General));CHKERRQ(ierr);

  pipe_to->requests   = gen_to->requests;
  pipe_from->requests = gen_from->requests;
  pipe_to->local      = gen_to->local;
  pipe_from->local    = gen_from->local;
  pipe_to->values     = gen_to->values;
  pipe_from->values   = gen_from->values;
  ierr                = PetscMalloc((gen_to->n+1)*sizeof(MPI_Status),&pipe_to->sstatus);CHKERRQ(ierr);
  ierr                = PetscMalloc((gen_from->n+1)*sizeof(MPI_Status),&pipe_from->sstatus);CHKERRQ(ierr);
  
  *to   = pipe_to; 
  *from = pipe_from;

  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------*/
/*C
   VecPipelineCreate - Creates a vector pipeline context.
*/
#undef __FUNCT__
#define __FUNCT__ "VecPipelineCreate"
int VecPipelineCreate(MPI_Comm comm,Vec xin,IS ix,Vec yin,IS iy,VecPipeline *newctx)
{
  VecPipeline ctx;
  int         ierr;

  PetscFunctionBegin;
  ierr      = PetscNew(struct _p_VecPipeline,&ctx);CHKERRQ(ierr);
  ierr      = PetscMemzero(ctx,sizeof(struct _p_VecPipeline));CHKERRQ(ierr);
  ctx->comm = comm;
  ierr      = VecScatterCreate(xin,ix,yin,iy,&(ctx->scatter));CHKERRQ(ierr);
  ierr      = VecPipelineSetType(ctx,PIPELINE_SEQUENTIAL,PETSC_NULL);CHKERRQ(ierr);
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

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPipelineSetupSelect"
static int VecPipelineSetupSelect(VecScatter_MPI_General *gen,VecScatter_MPI_General *pipe,
                                  int (*test)(int,PetscObject),PetscObject pipe_data)
{
  int ierr,i;

  PetscFunctionBegin;
  pipe->n = 0;
  for (i=0; i<gen->n; i++) {
    if ((*test)(gen->procs[i],pipe_data)) {
      pipe->n++;
    }
  }
  
  ierr = PetscMalloc((pipe->n+1)*sizeof(int),&pipe->procs);CHKERRQ(ierr);
  ierr = PetscMalloc((pipe->n+1)*sizeof(int),&pipe->starts);CHKERRQ(ierr);
  {
    int pipe_size = 1;
    if (gen->n) pipe_size = gen->starts[gen->n]+1;
    ierr = PetscMalloc(pipe_size*sizeof(int),&pipe->indices);CHKERRQ(ierr); 
  }
  {
    int *starts = gen->starts,*pstarts = pipe->starts;
    int *procs = gen->procs,*pprocs = pipe->procs;
    int *indices = gen->indices,*pindices = pipe->indices;
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

  PetscFunctionReturn(0);
}

/*C
   VecPipelineSetup - Sets up a vector pipeline context.
   This call is done implicitly in VecPipelineBegin, but
   since it is a bit costly, you may want to do it explicitly
   when timing.
*/
#undef __FUNCT__
#define __FUNCT__ "VecPipelineSetup"
int VecPipelineSetup(VecPipeline ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  int                    ierr;

  PetscFunctionBegin;
  if (ctx->setupcalled) PetscFunctionReturn(0);

  ierr = (ctx->setup)(ctx,ctx->aux_data,&(ctx->custom_pipe_data));CHKERRQ(ierr);

  gen_to    = (VecScatter_MPI_General*)ctx->scatter->todata;
  gen_from  = (VecScatter_MPI_General*)ctx->scatter->fromdata;

  /* data for PIPELINE_UP */
  ierr = VecPipelineSetupSelect(gen_to,ctx->upto,ctx->upfn,ctx->custom_pipe_data);CHKERRQ(ierr);
  ierr = VecPipelineSetupSelect(gen_from,ctx->upfrom,ctx->dnfn,ctx->custom_pipe_data);CHKERRQ(ierr);

  /* data for PIPELINE_DOWN */
  ierr = VecPipelineSetupSelect(gen_to,ctx->dnto,ctx->dnfn,ctx->custom_pipe_data);CHKERRQ(ierr);
  ierr = VecPipelineSetupSelect(gen_from,ctx->dnfrom,ctx->upfn,ctx->custom_pipe_data);CHKERRQ(ierr);

  ctx->setupcalled = 1;
  
  PetscFunctionReturn(0);
}

/*
   VecPipelineSetType
*/
EXTERN int ProcYes(int proc,PetscObject pipe_info);
EXTERN int ProcUp(int proc,PetscObject pipe_info);
EXTERN int ProcDown(int proc,PetscObject pipe_info);
EXTERN int ProcColorUp(int proc,PetscObject pipe_info);
EXTERN int ProcColorDown(int proc,PetscObject pipe_info);
EXTERN int PipelineSequentialSetup(VecPipeline,PetscObject,PetscObject*);
EXTERN int PipelineRedblackSetup(VecPipeline,PetscObject,PetscObject*);
EXTERN int PipelineMulticolorSetup(VecPipeline,PetscObject,PetscObject*);

int ProcNo(int proc,PetscObject pipe_info);

#undef __FUNCT__
#define __FUNCT__ "VecPipelineSetType"
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
   <(PetscObject)pmat> where pmat is the matrix on which the coloring
   is to be based.

.seealso: VecPipelineCreate(), VecPipelineBegin(), VecPipelineEnd().
*/
int VecPipelineSetType(VecPipeline ctx,PipelineType type,PetscObject x)
{
  PetscFunctionBegin;
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
    SETERRQ1(1,"VecPipelineSetType: unknown or not implemented type %d",(int)type);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPipelineBegin"
/*
   VecPipelineBegin - Receive data from processor earlier in
   a processor pipeline from one vector to another. 

.seealso: VecPipelineEnd.
*/
int VecPipelineBegin(Vec x,Vec y,InsertMode addv,ScatterMode smode,PipelineDirection pmode,VecPipeline ctx)
{
  int ierr;

  PetscFunctionBegin;
  if (!ctx->setupcalled) {
    ierr = VecPipelineSetup(ctx);CHKERRQ(ierr);
  }

  if (pmode==PIPELINE_UP) {
    ctx->scatter->todata = ctx->upto;
    ctx->scatter->fromdata = ctx->upfrom;
  } else if (pmode==PIPELINE_DOWN) {
    ctx->scatter->todata = ctx->dnto;
    ctx->scatter->fromdata = ctx->dnfrom;
  } else SETERRQ1(1,"VecPipelineBegin: unknown or not implemented pipeline mode %d",pmode);

  {
    VecScatter             scat = ctx->scatter;
    VecScatter_MPI_General *gen_to;
    int                    nsends=0;

    if (smode & SCATTER_REVERSE){
      gen_to   = (VecScatter_MPI_General*)scat->fromdata;
    } else {
      gen_to   = (VecScatter_MPI_General*)scat->todata;
    }
    if (ctx->pipe_type != PIPELINE_NONE) {
      nsends   = gen_to->n;
      gen_to->n = 0;
    }
    ierr = VecScatterBegin(x,y,addv,smode,scat);CHKERRQ(ierr);
    ierr = VecScatterEnd(x,y,addv,smode,scat);CHKERRQ(ierr);
    if (ctx->pipe_type != PIPELINE_NONE) gen_to->n = nsends;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPipelineEnd"
/*
   VecPipelineEnd - Send data to processors later in
   a processor pipeline from one vector to another.
 
.seealso: VecPipelineBegin.
*/
int VecPipelineEnd(Vec x,Vec y,InsertMode addv,ScatterMode smode,PipelineDirection pmode,VecPipeline ctx)
{
  VecScatter             scat = ctx->scatter;
  VecScatter_MPI_General *gen_from,*gen_to;
  int                    nsends=0,nrecvs,ierr;
  
  PetscFunctionBegin;
  if (smode & SCATTER_REVERSE){
    gen_to   = (VecScatter_MPI_General*)scat->fromdata;
    gen_from = (VecScatter_MPI_General*)scat->todata;
  } else {
    gen_to   = (VecScatter_MPI_General*)scat->todata;
    gen_from = (VecScatter_MPI_General*)scat->fromdata;
  }
  if (ctx->pipe_type == PIPELINE_NONE) {
      nsends   = gen_to->n;
      gen_to->n = 0;
  }
  nrecvs      = gen_from->n;
  gen_from->n = 0;
  ierr = VecScatterBegin(x,y,addv,smode,scat);CHKERRQ(ierr);
  ierr = VecScatterEnd(x,y,addv,smode,scat);CHKERRQ(ierr);
  if (ctx->pipe_type == PIPELINE_NONE) gen_to->n = nsends;
  gen_from->n = nrecvs;

  PetscFunctionReturn(0);
}

/*
      This destroys the material that is allocated inside the 
   VecScatter_MPI_General datastructure. Mimics the 
   VecScatterDestroy_PtoP() object except it does not destroy 
   the wrapper VecScatter object.
*/

#undef __FUNCT__
#define __FUNCT__ "VecPipelineDestroy_MPI_General"
static int VecPipelineDestroy_MPI_General(VecScatter_MPI_General *gen)
{
  int ierr;

  PetscFunctionBegin;
  if (gen->sstatus) {ierr = PetscFree(gen->sstatus);CHKERRQ(ierr);}
  ierr = PetscFree(gen);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPipelineDestroy"
/*C
   VecPipelineDestroy - Destroys a pipeline context created by 
   VecPipelineCreate().

*/
int VecPipelineDestroy(VecPipeline ctx)
{
  int ierr;

  PetscFunctionBegin;
  /* free the VecScatter_MPI_General data structures */
  if (ctx->upto) {
    ierr = PetscFree(ctx->upto->procs);CHKERRQ(ierr);
    ierr = PetscFree(ctx->upto->starts);CHKERRQ(ierr);
    ierr = PetscFree(ctx->upto->indices);CHKERRQ(ierr);
    ierr = VecPipelineDestroy_MPI_General(ctx->upto);CHKERRQ(ierr);
  }
  if (ctx->upfrom) {
    ierr = PetscFree(ctx->upfrom->procs);CHKERRQ(ierr);
    ierr = PetscFree(ctx->upfrom->starts);CHKERRQ(ierr);
    ierr = PetscFree(ctx->upfrom->indices);CHKERRQ(ierr);
    ierr = VecPipelineDestroy_MPI_General(ctx->upfrom);CHKERRQ(ierr);
  }
  if (ctx->dnto) {
    ierr = PetscFree(ctx->dnto->procs);CHKERRQ(ierr);
    ierr = PetscFree(ctx->dnto->starts);CHKERRQ(ierr);
    ierr = PetscFree(ctx->dnto->indices);CHKERRQ(ierr);
    ierr = VecPipelineDestroy_MPI_General(ctx->dnto);CHKERRQ(ierr);
  }
  if (ctx->dnfrom) {
    ierr = PetscFree(ctx->dnfrom->procs);CHKERRQ(ierr);
    ierr = PetscFree(ctx->dnfrom->starts);CHKERRQ(ierr);
    ierr = PetscFree(ctx->dnfrom->indices);CHKERRQ(ierr);
    ierr = VecPipelineDestroy_MPI_General(ctx->dnfrom);CHKERRQ(ierr);
  }
  if (ctx->scatterto) {
    ierr = PetscFree(ctx->scatterto->values);CHKERRQ(ierr);
    if (ctx->scatterto->local.slots) {ierr = PetscFree(ctx->scatterto->local.slots);CHKERRQ(ierr);}
    ierr = VecPipelineDestroy_MPI_General(ctx->scatterto);CHKERRQ(ierr);
  }
  if (ctx->scatterfrom) {
    ierr = PetscFree(ctx->scatterfrom->values);CHKERRQ(ierr);
    if (ctx->scatterfrom->local.slots) {ierr = PetscFree(ctx->scatterfrom->local.slots);CHKERRQ(ierr);}
    ierr = VecPipelineDestroy_MPI_General(ctx->scatterfrom);CHKERRQ(ierr);
  }

  if (ctx->custom_pipe_data) {ierr = PetscFree(ctx->custom_pipe_data);CHKERRQ(ierr);}
  PetscHeaderDestroy(ctx->scatter);
  ierr = PetscFree(ctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* >>>> Routines for sequential ordering of processors <<<< */

typedef struct {int rank;} Pipeline_sequential_info;

#undef __FUNCT__
#define __FUNCT__ "ProcYes"
int ProcYes(int proc,PetscObject pipe_info)
{
  PetscFunctionBegin;
  PetscFunctionReturn(1);
}
#undef __FUNCT__
#define __FUNCT__ "ProcNo"
int ProcNo(int proc,PetscObject pipe_info)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcUp"
int ProcUp(int proc,PetscObject pipe_info)
{
  int rank = ((Pipeline_sequential_info *)pipe_info)->rank;

  PetscFunctionBegin;
  if (rank<proc) {
    PetscFunctionReturn(1);
  } else {
    PetscFunctionReturn(0);
  }
}

#undef __FUNCT__
#define __FUNCT__ "ProcDown"
int ProcDown(int proc,PetscObject pipe_info)
{ 
  int rank = ((Pipeline_sequential_info *)pipe_info)->rank;

  PetscFunctionBegin;
  if (rank>proc) {
    PetscFunctionReturn(1);
  } else {
    PetscFunctionReturn(0);
  }
}

#undef __FUNCT__
#define __FUNCT__ "PipelineSequentialSetup"
int PipelineSequentialSetup(VecPipeline vs,PetscObject x,PetscObject *obj)
{
  Pipeline_sequential_info *info;
  int                      ierr;

  PetscFunctionBegin;
  ierr = PetscNew(Pipeline_sequential_info,&info);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(vs->scatter->comm,&(info->rank));CHKERRQ(ierr);
  *obj = (PetscObject)info;

  PetscFunctionReturn(0);
}

/* >>>> Routines for multicolor ordering of processors <<<< */

typedef struct {
  int rank,size,*proc_colors;
} Pipeline_colored_info;

#undef __FUNCT__
#define __FUNCT__ "ProcColorUp"
int ProcColorUp(int proc,PetscObject pipe_info)
{
  Pipeline_colored_info* comm_info = (Pipeline_colored_info*)pipe_info;
  int                    rank = comm_info->rank;

  PetscFunctionBegin;
  if (comm_info->proc_colors[rank]<comm_info->proc_colors[proc]) {
    PetscFunctionReturn(1);
  } else {
    PetscFunctionReturn(0);
  }
}
#undef __FUNCT__
#define __FUNCT__ "ProcColorDown"
int ProcColorDown(int proc,PetscObject pipe_info)
{ 
  Pipeline_colored_info* comm_info = (Pipeline_colored_info*)pipe_info;
  int                    rank = comm_info->rank;

  PetscFunctionBegin;
  if (comm_info->proc_colors[rank]>comm_info->proc_colors[proc]) {
    PetscFunctionReturn(1);
  } else {
    PetscFunctionReturn(0);
  }
}

#undef __FUNCT__
#define __FUNCT__ "PipelineRedblackSetup"
int PipelineRedblackSetup(VecPipeline vs,PetscObject x,PetscObject *obj)
{
  Pipeline_colored_info *info;
  int                    size,i,ierr;

  PetscFunctionBegin;
  ierr = PetscNew(Pipeline_colored_info,&info);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(vs->scatter->comm,&(info->rank));CHKERRQ(ierr);
  ierr = MPI_Comm_size(vs->scatter->comm,&size);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(int),&info->proc_colors);CHKERRQ(ierr);
  for (i=0; i<size; i++) {info->proc_colors[i] = i%2;}
  *obj = (PetscObject)info;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PipelineMulticolorSetup"
int PipelineMulticolorSetup(VecPipeline vs,PetscObject x,PetscObject *obj)
{
  Pipeline_colored_info *info;
  Mat                    mat = (Mat) x;
  int                    size,ierr;

  PetscFunctionBegin;
  ierr = PetscNew(Pipeline_colored_info,&info);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(mat->comm,&(info->rank));CHKERRQ(ierr);
  ierr = MPI_Comm_size(mat->comm,&size);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(int),&info->proc_colors);CHKERRQ(ierr);
  ierr = PetscMemzero(info->proc_colors,size*sizeof(int));CHKERRQ(ierr);

  /* coloring */
  {
    Mat_MPIAIJ  *Aij = (Mat_MPIAIJ*)mat->data;
    int *owners = Aij->rowners,*touch = Aij->garray;
    int ntouch = Aij->B->n;
    int *conn,*colr;
    int *colors = info->proc_colors,base = info->rank*size;
    int p,e;

    /* allocate connectivity matrix */
    ierr = PetscMalloc(size*size*sizeof(int),&conn);CHKERRQ(ierr);
    ierr = PetscMalloc(size*sizeof(int),&colr);CHKERRQ(ierr);
    ierr = PetscMemzero(conn,size*size*sizeof(int));CHKERRQ(ierr);

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
    ierr = MPI_Allgather(conn+base,size,MPI_INT,conn,size,MPI_INT,mat->comm);CHKERRQ(ierr);

    base = size;
    /* PetscPrintf(mat->comm,"Coloring: 0->0"); */
    for (p=1; p<size; p++) {
      int q,hi=-1,nc=0;
      ierr = PetscMemzero(colr,size*sizeof(int));CHKERRQ(ierr);
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
      /* PetscPrintf(mat->comm,",%d->%d",p,colors[p]); */
      base = base+size;
    }
    /* PetscPrintf(mat->comm,"\n"); */
    ierr = PetscFree(conn);CHKERRQ(ierr);
    ierr = PetscFree(colr);CHKERRQ(ierr);
  }
  *obj = (PetscObject)info;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPipelineView"
int VecPipelineView(VecPipeline pipe,PetscViewer viewer)
{
  MPI_Comm comm = pipe->comm;
  int      ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(comm,">> Vector Pipeline<<\n");CHKERRQ(ierr);
  if (!pipe->setupcalled) {ierr = PetscPrintf(comm,"Not yet set up\n");CHKERRQ(ierr);}
  ierr = PetscPrintf(comm,"Pipelinetype: %d\n",(int)pipe->pipe_type);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"based on scatter:\n");CHKERRQ(ierr);
  /*  ierr = VecScatterView(pipe->scatter,viewer);CHKERRQ(ierr);*/
  ierr = PetscPrintf(comm,"Up function %p\n",pipe->upfn);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Dn function %p\n",pipe->dnfn);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}









