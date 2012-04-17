#include <petscsys.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <pami.h>
#if defined(PETSC_HAVE_PTHREAD_H)
#  include <pthread.h>
#elif defined(PETSC_HAVE_WINPTHREADS_H)
#  include "winpthreads.h"       /* http://locklessinc.com/downloads/winpthreads.h */
#else
#  error Need pthreads to use this PAMI interface
#endif

/* a useful link for PPC memory ordering issues:
 *   http://www.rdrop.com/users/paulmck/scalability/paper/N2745r.2009.02.22a.html
 *
 * lwsync: orders L-L, S-S, L-S, but *not* S-L (i.e. gives x86-ish ordering)
 * eieio: orders S-S (but only for cacheable memory, not for MMIO)
 * sync: totally orders memops
 * isync: force all preceeding insns to appear complete before starting
 *        subsequent insns, but w/o cumulativity (very confusing)
 */
#define PetscReadOnce(type,val) (*(volatile type *)&val)
#define PetscCompilerBarrier()   __asm__ __volatile__  ( ""  ::: "memory" )
#define PetscMemoryBarrierWrite() __asm__ __volatile__  ( "eieio"  ::: "memory" )
#define PetscMemoryBarrierReadWrite() __asm__ __volatile__  ( "sync"  ::: "memory" )
#define PetscMemoryBarrierRead() __asm__ __volatile__  ( "lwsync" ::: "memory" )

typedef int PetscPAMIInt;
/* The context for the MPI generalized request and the PAMI callback */
typedef struct {
  pami_context_t pamictx;       /* only valid if not using comm threads, in which case the Grequest_poll_function will advance the context */
  MPI_Request request;
  PetscBool done;
} PetscPAMIReqCtx;
typedef PetscErrorCode (*PetscThreadFunction)(pami_context_t,PetscPAMIReqCtx*);
typedef struct {
  PetscThreadFunction func;
  pami_context_t pamictx;
  PetscPAMIReqCtx *reqctx;
  PetscBool active;               /* FALSE when available to compute thread to add a task */
} PetscPAMIThreadContext;

static PetscBool pami_initialized;
static pami_client_t pami_client;
static pami_geometry_t pami_geom_world;
struct PetscPAMI {
  PetscPAMIInt num_contexts;
  pami_context_t *contexts;
  pthread_t thread;
  PetscPAMIThreadContext threadctx;
  pami_algorithm_t allreduce_alg;
} pami;

static MPIX_Grequest_class grequest_class; /* no way to free an MPIX_Grequest_class */

static PetscErrorCode PetscPAMITypeFromMPI(MPI_Datatype,pami_type_t*);
static PetscErrorCode PetscPAMIOpFromMPI(MPI_Op,pami_data_function*);
static PetscErrorCode PetscPAMIInitialize(void);
static PetscErrorCode PetscPAMIFinalize(void);


#define PAMICHK(perr,func) do {                 \
    if ((perr) != PAMI_SUCCESS) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_LIB,"%s failed with error code %d",#func,(int)(perr)); \
  } while (0)

static void Spin(void) { return; }

#undef __FUNCT__
#define __FUNCT__ "PetscPAMIGetAlgorithm"
static PetscErrorCode PetscPAMIGetAlgorithm(pami_geometry_t world,pami_xfer_type_t xfertype,pami_algorithm_t *alg,pami_metadata_t *meta)
{
  PetscErrorCode ierr;
  pami_result_t perr;
  size_t numalgs[2];
  pami_algorithm_t safealgs[3],fastalgs[1];
  pami_metadata_t safemeta[3],fastmeta[1];

  PetscFunctionBegin;
  perr = PAMI_Geometry_algorithms_num(world,xfertype,numalgs);PAMICHK(perr,PAMI_Geometry_algorithms_num);
  numalgs[0] = PetscMin(3,numalgs[0]); /* Query the first few safe algorithms */
  numalgs[1] = PetscMin(1,numalgs[1]); /* I don't actually care about unsafe algorithms, but query one anyway just for giggles */
  perr = PAMI_Geometry_algorithms_query(world,xfertype,safealgs,safemeta,numalgs[0],fastalgs,fastmeta,numalgs[1]);PAMICHK(perr,PAMI_Geometry_algorithms_query);
  if (alg) *alg = safealgs[0];
  if (meta) *meta = safemeta[0];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPAMIThreadExit(pami_context_t pctx,PetscPAMIReqCtx *reqctx)
{
  pthread_exit(0);
  return 0;                     /* not actually reached */
}
static PetscErrorCode PetscPAMIThreadPoll(pami_context_t pctx,PetscPAMIReqCtx *reqctx)
{
  pami_result_t perr;
  while (!PetscReadOnce(PetscBool,reqctx->done)) {
    perr = PAMI_Context_advance(pctx,1);PAMICHK(perr,PAMI_Context_advance);
  }
  return 0;
}

static void *PetscPAMIPthread(void *args)
{
  PetscPAMIThreadContext *threadctx = (PetscPAMIThreadContext*)args;
  while (1) {
    PetscErrorCode ierr;
    while (!PetscReadOnce(PetscBool,threadctx->active)) Spin();
    ierr = threadctx->func(threadctx->pamictx,threadctx->reqctx);CHKERRABORT(PETSC_COMM_SELF,ierr);
    threadctx->active = PETSC_FALSE;
  }
}
#undef __FUNCT__
#define __FUNCT__ "PetscPAMIThreadSend"
static PetscErrorCode PetscPAMIThreadSend(PetscPAMIThreadContext *threadctx,PetscThreadFunction func,pami_context_t pamictx,PetscPAMIReqCtx *reqctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (PetscReadOnce(PetscBool,threadctx->active)) Spin();
  threadctx->func = func;
  threadctx->pamictx = pamictx;
  threadctx->reqctx = reqctx;
  PetscMemoryBarrierWrite();
  threadctx->active = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPAMIInitialize"
static PetscErrorCode PetscPAMIInitialize(void)
{
  pami_result_t        perr;
  PetscErrorCode       ierr;
  pami_configuration_t config;
  PetscBool            thread;
  const char           *clientname = "PETSC"; /* --env PAMI_NUMCLIENTS=2:PAMI_CLIENTNAMES=MPI,PETSC */

  PetscFunctionBegin;
  if (pami_initialized) PetscFunctionReturn(0);
  perr = PAMI_Client_create(clientname,&pami_client,0,0);PAMICHK(perr,PAMI_Client_create);

  config.name = PAMI_CLIENT_NUM_CONTEXTS;
  perr = PAMI_Client_query(pami_client,&config,1);PAMICHK(perr,PAMI_Client_query);
  pami.num_contexts = PetscMin(10,config.value.intval); /* Only need one or perhaps a few contexts */

  ierr = PetscMalloc(pami.num_contexts*sizeof(pami_context_t),&pami.contexts);CHKERRQ(ierr);
  perr = PAMI_Context_createv(pami_client,&config,0,pami.contexts,pami.num_contexts);PAMICHK(perr,PAMI_Context_createv);

  perr = PAMI_Geometry_world(pami_client,&pami_geom_world);PAMICHK(perr,PAMI_Geometry_world);
  /* Jeff says that I have to query the barrier before I can query something else */
  ierr = PetscPAMIGetAlgorithm(pami_geom_world,PAMI_XFER_BARRIER,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPAMIGetAlgorithm(pami_geom_world,PAMI_XFER_ALLREDUCE,&pami.allreduce_alg,PETSC_NULL);CHKERRQ(ierr);

  thread = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-pami_thread",&thread,PETSC_NULL);CHKERRQ(ierr);
  if (thread) {
    ierr = pthread_create(&pami.thread,0,PetscPAMIPthread,&pami.threadctx);CHKERRQ(ierr);
    ierr = PetscInfo(0,"PAMI initialized with communication thread\n");CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(0,"PAMI initialized without communication thread\n");CHKERRQ(ierr);
  }

  pami_initialized = PETSC_TRUE;
  ierr = PetscRegisterFinalize(PetscPAMIFinalize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPAMIFinalize"
static PetscErrorCode PetscPAMIFinalize(void)
{
  PetscErrorCode ierr;
  pami_result_t perr;

  PetscFunctionBegin;
  ierr = PetscPAMIThreadSend(&pami.threadctx,PetscPAMIThreadExit,PAMI_CONTEXT_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = pthread_join(pami.thread,PETSC_NULL);CHKERRQ(ierr);
  pami.thread = PETSC_NULL;

  perr = PAMI_Context_destroyv(pami.contexts,pami.num_contexts);PAMICHK(perr,PAMI_Context_destroyv);
  ierr = PetscFree(pami.contexts);CHKERRQ(ierr);
  perr = PAMI_Client_destroy(&pami_client);PAMICHK(perr,PAMI_Client_destroy);

  pami.num_contexts = 0;
  pami.contexts = PETSC_NULL;
  pami_client = 0;
  pami_initialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* PAMI calls this from a different thread when the task is done */
static void PetscPAMICallbackDone(void *vctx,void *clientdata,pami_result_t err)
{
  PetscPAMIReqCtx *ctx = (PetscPAMIReqCtx*)vctx;
  ctx->done = PETSC_TRUE;
  MPI_Grequest_complete(ctx->request);
}

/* MPI_Grequest_query_function */
static PetscMPIInt PetscMPIGrequestQuery_Default(void *state,MPI_Status *status)
{
  PetscPAMIReqCtx *ctx = (PetscPAMIReqCtx*)state;

  if (ctx) {                    /* We could put meaningful values here */
    MPI_Status_set_elements(status,MPI_INT,0);
    MPI_Status_set_cancelled(status,0);
    status->MPI_SOURCE = MPI_UNDEFINED;
    status->MPI_TAG = MPI_UNDEFINED;
  } else {
    MPI_Status_set_elements(status,MPI_INT,0);
    MPI_Status_set_cancelled(status,0);
    status->MPI_SOURCE = MPI_UNDEFINED;
    status->MPI_TAG = MPI_UNDEFINED;
  }
  return MPI_SUCCESS;
}
/* MPI_Grequest_free_function */
static PetscMPIInt PetscMPIGrequestFree_Default(void *state)
{
  return PetscFree(state);
}
/* MPI_Grequest_cancel_function */
static PetscMPIInt PetscMPIGrequestCancel_Nothing(void *state,int complete)
{
  if (!complete) MPI_ERR_UNSUPPORTED_OPERATION;
  return MPI_SUCCESS;
}
/* MPIX_Grequest_poll_function */
static PetscMPIInt PetscMPIGrequestPoll_PAMI(void *state,MPI_Status *status)
{
  PetscPAMIReqCtx *ctx = (PetscPAMIReqCtx*)state;
  if (ctx->pamictx == PAMI_CONTEXT_NULL) {
    /* separate comm thread, so nothing for th poll function to do */
  } else {                      /* no comm thread, so advance the context */
    PetscPAMIInt ierr;
    ierr = PAMI_Context_advance(ctx->pamictx,1);
    if (ierr != PAMI_SUCCESS) return MPI_ERR_OTHER;
  }
  return MPI_SUCCESS;
}
/* MPIX_Grequest_wait_function */
#define PetscMPIGrequestWait_PAMI ((MPIX_Grequest_wait_function*)0)

#undef __FUNCT__
#define __FUNCT__ "MPIPetsc_Iallreduce_PAMI"
PetscErrorCode MPIPetsc_Iallreduce_PAMI(void *sendbuf,void *recvbuf,PetscMPIInt count,MPI_Datatype datatype,MPI_Op op,MPI_Comm comm,MPI_Request *request)
{
  PetscErrorCode ierr;
  PetscMPIInt match;

  PetscFunctionBegin;
  ierr = MPI_Comm_compare(comm,MPI_COMM_WORLD,&match);CHKERRQ(ierr);
  if (match == MPI_UNEQUAL) {   /* safe mode, just use MPI */
    ierr = PetscInfo(0,"Communicators do not match, using synchronous MPI_Allreduce\n");CHKERRQ(ierr);
    ierr = MPI_Allreduce(sendbuf,recvbuf,count,datatype,op,comm);CHKERRQ(ierr);
    /* create a dummy request so the external interface does not need to know */
    ierr = MPI_Grequest_start(PetscMPIGrequestQuery_Default,PetscMPIGrequestFree_Default,PetscMPIGrequestCancel_Nothing,0,request);CHKERRQ(ierr);
    ierr = MPI_Grequest_complete(*request);CHKERRQ(ierr);
  } else {                      /* set up a PAMI request */
    pami_xfer_t allreduce;
    PetscBool *done;
    pami_type_t pamitype;
    pami_data_function pamiop;
    PetscPAMIReqCtx *reqctx;
    pami_result_t perr;

    ierr = PetscInfo(0,"Using PAMI Iallreduce\n");CHKERRQ(ierr);
    ierr = PetscPAMIInitialize();CHKERRQ(ierr);

    ierr = PetscPAMITypeFromMPI(datatype,&pamitype);CHKERRQ(ierr);
    ierr = PetscPAMIOpFromMPI(op,&pamiop);CHKERRQ(ierr);

    if (!grequest_class) {
      ierr = MPIX_Grequest_class_create(PetscMPIGrequestQuery_Default,
                                        PetscMPIGrequestFree_Default,
                                        PetscMPIGrequestCancel_Nothing,
                                        PetscMPIGrequestPoll_PAMI,
                                        PetscMPIGrequestWait_PAMI,
                                        &grequest_class);CHKERRQ(ierr);
    }
    ierr = PetscMalloc(sizeof(PetscPAMIReqCtx),&reqctx);CHKERRQ(ierr);

    /* Create a generalized request to wait/poke PAMI */
    ierr = MPIX_Grequest_class_allocate(grequest_class,reqctx,request);CHKERRQ(ierr);
    reqctx->done = PETSC_FALSE;
    reqctx->request = *request;  /* The PAMI callback will call MPI_Grequest_complete() */
    if (pami.thread) {           /* The PAMI thread will advance progress */
      reqctx->pamictx = PAMI_CONTEXT_NULL;
    } else {                    /* The MPI Grequest_poll_function will advance progress */
      reqctx->pamictx = pami.contexts[0];
    }

    allreduce.cb_done = PetscPAMICallbackDone;
    allreduce.cookie = (void*)reqctx;
    allreduce.algorithm = pami.allreduce_alg;
    allreduce.cmd.xfer_allreduce.op         = pamiop;
    allreduce.cmd.xfer_allreduce.sndbuf     = sendbuf;
    allreduce.cmd.xfer_allreduce.stype      = pamitype;
    allreduce.cmd.xfer_allreduce.stypecount = count;
    allreduce.cmd.xfer_allreduce.rcvbuf     = recvbuf;
    allreduce.cmd.xfer_allreduce.rtype      = pamitype;
    allreduce.cmd.xfer_allreduce.rtypecount = count;

    /* Start the collective on the context, should confirm that the context is available */
    perr = PAMI_Collective(pami.contexts[0],&allreduce);PAMICHK(perr,PAMI_Collective);

    if (pami.thread) {
      ierr = PetscPAMIThreadSend(&pami.threadctx,PetscPAMIThreadPoll,pami.contexts[0],reqctx);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPAMITypeFromMPI"
static PetscErrorCode PetscPAMITypeFromMPI(MPI_Datatype datatype,pami_type_t *pamitype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (datatype == MPI_INT) *pamitype = PAMI_TYPE_SIGNED_INT;
  else if (datatype == MPI_LONG) *pamitype = PAMI_TYPE_SIGNED_LONG;
  else if (datatype == MPI_LONG_LONG_INT) *pamitype = PAMI_TYPE_SIGNED_LONG_LONG;
  else if (datatype == MPI_DOUBLE) *pamitype = PAMI_TYPE_DOUBLE;
  else if (datatype == MPI_FLOAT)  *pamitype = PAMI_TYPE_FLOAT;
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported MPI_Datatype");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPAMIOpFromMPI"
static PetscErrorCode PetscPAMIOpFromMPI(MPI_Op op,pami_data_function *pamiop)
{

  PetscFunctionBegin;
  if (op == MPI_SUM) *pamiop = PAMI_DATA_SUM;
  else if (op == MPI_MAX) *pamiop = PAMI_DATA_MAX;
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported MPI_Op");
  PetscFunctionReturn(0);
}
