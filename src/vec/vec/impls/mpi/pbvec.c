
/*
   This file contains routines for Parallel vector operations.
 */
#include <petscsys.h>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/

PetscErrorCode VecDot_MPI(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;

  PetscFunctionBegin;
  PetscCall(VecDot_Seq(xin,yin,&work));
  PetscCall(MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_MPI(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;

  PetscFunctionBegin;
  PetscCall(VecTDot_Seq(xin,yin,&work));
  PetscCall(MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  *z   = sum;
  PetscFunctionReturn(0);
}

extern PetscErrorCode VecView_MPI_Draw(Vec,PetscViewer);

static PetscErrorCode VecPlaceArray_MPI(Vec vin,const PetscScalar *a)
{
  Vec_MPI        *v = (Vec_MPI*)vin->data;

  PetscFunctionBegin;
  PetscCheck(!v->unplacedarray,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecPlaceArray() was already called on this vector, without a call to VecResetArray()");
  v->unplacedarray = v->array;  /* save previous array so reset can bring it back */
  v->array         = (PetscScalar*)a;
  if (v->localrep) {
    PetscCall(VecPlaceArray(v->localrep,a));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDuplicate_MPI(Vec win,Vec *v)
{
  Vec_MPI        *vw,*w = (Vec_MPI*)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  PetscCall(VecCreate(PetscObjectComm((PetscObject)win),v));
  PetscCall(PetscLayoutReference(win->map,&(*v)->map));

  PetscCall(VecCreate_MPI_Private(*v,PETSC_TRUE,w->nghost,NULL));
  vw   = (Vec_MPI*)(*v)->data;
  PetscCall(PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps)));

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    PetscCall(VecGetArray(*v,&array));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,PetscAbs(win->map->bs),win->map->n+w->nghost,array,&vw->localrep));
    PetscCall(PetscMemcpy(vw->localrep->ops,w->localrep->ops,sizeof(struct _VecOps)));
    PetscCall(VecRestoreArray(*v,&array));
    PetscCall(PetscLogObjectParent((PetscObject)*v,(PetscObject)vw->localrep));

    vw->localupdate = w->localupdate;
    if (vw->localupdate) {
      PetscCall(PetscObjectReference((PetscObject)vw->localupdate));
    }
  }

  /* New vector should inherit stashing property of parent */
  (*v)->stash.donotstash   = win->stash.donotstash;
  (*v)->stash.ignorenegidx = win->stash.ignorenegidx;

  PetscCall(PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist));
  PetscCall(PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist));

  (*v)->map->bs   = PetscAbs(win->map->bs);
  (*v)->bstash.bs = win->bstash.bs;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetOption_MPI(Vec V,VecOption op,PetscBool flag)
{
  Vec_MPI        *v = (Vec_MPI*)V->data;

  PetscFunctionBegin;
  switch (op) {
  case VEC_IGNORE_OFF_PROC_ENTRIES: V->stash.donotstash = flag;
    break;
  case VEC_IGNORE_NEGATIVE_INDICES: V->stash.ignorenegidx = flag;
    break;
  case VEC_SUBSET_OFF_PROC_ENTRIES:
    v->assembly_subset = flag; /* See the same logic in MatAssembly wrt MAT_SUBSET_OFF_PROC_ENTRIES */
    if (!v->assembly_subset) { /* User indicates "do not reuse the communication pattern" */
      PetscCall(VecAssemblyReset_MPI(V)); /* Reset existing pattern to free memory */
      v->first_assembly_done = PETSC_FALSE; /* Mark the first assembly is not done */
    }
    break;
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode VecResetArray_MPI(Vec vin)
{
  Vec_MPI        *v = (Vec_MPI*)vin->data;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = NULL;
  if (v->localrep) {
    PetscCall(VecResetArray(v->localrep));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAssemblySend_MPI_Private(MPI_Comm comm,const PetscMPIInt tag[],PetscMPIInt rankid,PetscMPIInt rank,void *sdata,MPI_Request req[],void *ctx)
{
  Vec X = (Vec)ctx;
  Vec_MPI *x = (Vec_MPI*)X->data;
  VecAssemblyHeader *hdr = (VecAssemblyHeader*)sdata;
  PetscInt bs = X->map->bs;

  PetscFunctionBegin;
  /* x->first_assembly_done indicates we are reusing a communication network. In that case, some
     messages can be empty, but we have to send them this time if we sent them before because the
     receiver is expecting them.
   */
  if (hdr->count || (x->first_assembly_done && x->sendptrs[rankid].ints)) {
    PetscCallMPI(MPI_Isend(x->sendptrs[rankid].ints,hdr->count,MPIU_INT,rank,tag[0],comm,&req[0]));
    PetscCallMPI(MPI_Isend(x->sendptrs[rankid].scalars,hdr->count,MPIU_SCALAR,rank,tag[1],comm,&req[1]));
  }
  if (hdr->bcount || (x->first_assembly_done && x->sendptrs[rankid].intb)) {
    PetscCallMPI(MPI_Isend(x->sendptrs[rankid].intb,hdr->bcount,MPIU_INT,rank,tag[2],comm,&req[2]));
    PetscCallMPI(MPI_Isend(x->sendptrs[rankid].scalarb,hdr->bcount*bs,MPIU_SCALAR,rank,tag[3],comm,&req[3]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAssemblyRecv_MPI_Private(MPI_Comm comm,const PetscMPIInt tag[],PetscMPIInt rank,void *rdata,MPI_Request req[],void *ctx)
{
  Vec X = (Vec)ctx;
  Vec_MPI *x = (Vec_MPI*)X->data;
  VecAssemblyHeader *hdr = (VecAssemblyHeader*)rdata;
  PetscInt bs = X->map->bs;
  VecAssemblyFrame *frame;

  PetscFunctionBegin;
  PetscCall(PetscSegBufferGet(x->segrecvframe,1,&frame));

  if (hdr->count) {
    PetscCall(PetscSegBufferGet(x->segrecvint,hdr->count,&frame->ints));
    PetscCallMPI(MPI_Irecv(frame->ints,hdr->count,MPIU_INT,rank,tag[0],comm,&req[0]));
    PetscCall(PetscSegBufferGet(x->segrecvscalar,hdr->count,&frame->scalars));
    PetscCallMPI(MPI_Irecv(frame->scalars,hdr->count,MPIU_SCALAR,rank,tag[1],comm,&req[1]));
    frame->pendings = 2;
  } else {
    frame->ints = NULL;
    frame->scalars = NULL;
    frame->pendings = 0;
  }

  if (hdr->bcount) {
    PetscCall(PetscSegBufferGet(x->segrecvint,hdr->bcount,&frame->intb));
    PetscCallMPI(MPI_Irecv(frame->intb,hdr->bcount,MPIU_INT,rank,tag[2],comm,&req[2]));
    PetscCall(PetscSegBufferGet(x->segrecvscalar,hdr->bcount*bs,&frame->scalarb));
    PetscCallMPI(MPI_Irecv(frame->scalarb,hdr->bcount*bs,MPIU_SCALAR,rank,tag[3],comm,&req[3]));
    frame->pendingb = 2;
  } else {
    frame->intb = NULL;
    frame->scalarb = NULL;
    frame->pendingb = 0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAssemblyBegin_MPI_BTS(Vec X)
{
  Vec_MPI        *x = (Vec_MPI*)X->data;
  MPI_Comm       comm;
  PetscInt       i,j,jb,bs;

  PetscFunctionBegin;
  if (X->stash.donotstash) PetscFunctionReturn(0);

  PetscCall(PetscObjectGetComm((PetscObject)X,&comm));
  PetscCall(VecGetBlockSize(X,&bs));
  if (PetscDefined(USE_DEBUG)) {
    InsertMode addv;
    PetscCall(MPIU_Allreduce((PetscEnum*)&X->stash.insertmode,(PetscEnum*)&addv,1,MPIU_ENUM,MPI_BOR,comm));
    PetscCheck(addv != (ADD_VALUES|INSERT_VALUES),comm,PETSC_ERR_ARG_NOTSAMETYPE,"Some processors inserted values while others added");
  }
  X->bstash.insertmode = X->stash.insertmode; /* Block stash implicitly tracks InsertMode of scalar stash */

  PetscCall(VecStashSortCompress_Private(&X->stash));
  PetscCall(VecStashSortCompress_Private(&X->bstash));

  if (!x->sendranks) {
    PetscMPIInt nowners,bnowners,*owners,*bowners;
    PetscInt ntmp;
    PetscCall(VecStashGetOwnerList_Private(&X->stash,X->map,&nowners,&owners));
    PetscCall(VecStashGetOwnerList_Private(&X->bstash,X->map,&bnowners,&bowners));
    PetscCall(PetscMergeMPIIntArray(nowners,owners,bnowners,bowners,&ntmp,&x->sendranks));
    x->nsendranks = ntmp;
    PetscCall(PetscFree(owners));
    PetscCall(PetscFree(bowners));
    PetscCall(PetscMalloc1(x->nsendranks,&x->sendhdr));
    PetscCall(PetscCalloc1(x->nsendranks,&x->sendptrs));
  }
  for (i=0,j=0,jb=0; i<x->nsendranks; i++) {
    PetscMPIInt rank = x->sendranks[i];
    x->sendhdr[i].insertmode = X->stash.insertmode;
    /* Initialize pointers for non-empty stashes the first time around.  Subsequent assemblies with
     * VEC_SUBSET_OFF_PROC_ENTRIES will leave the old pointers (dangling because the stash has been collected) when
     * there is nothing new to send, so that size-zero messages get sent instead. */
    x->sendhdr[i].count = 0;
    if (X->stash.n) {
      x->sendptrs[i].ints    = &X->stash.idx[j];
      x->sendptrs[i].scalars = &X->stash.array[j];
      for (; j<X->stash.n && X->stash.idx[j] < X->map->range[rank+1]; j++) x->sendhdr[i].count++;
    }
    x->sendhdr[i].bcount = 0;
    if (X->bstash.n) {
      x->sendptrs[i].intb    = &X->bstash.idx[jb];
      x->sendptrs[i].scalarb = &X->bstash.array[jb*bs];
      for (; jb<X->bstash.n && X->bstash.idx[jb]*bs < X->map->range[rank+1]; jb++) x->sendhdr[i].bcount++;
    }
  }

  if (!x->segrecvint) PetscCall(PetscSegBufferCreate(sizeof(PetscInt),1000,&x->segrecvint));
  if (!x->segrecvscalar) PetscCall(PetscSegBufferCreate(sizeof(PetscScalar),1000,&x->segrecvscalar));
  if (!x->segrecvframe) PetscCall(PetscSegBufferCreate(sizeof(VecAssemblyFrame),50,&x->segrecvframe));
  if (x->first_assembly_done) { /* this is not the first assembly */
    PetscMPIInt tag[4];
    for (i=0; i<4; i++) PetscCall(PetscCommGetNewTag(comm,&tag[i]));
    for (i=0; i<x->nsendranks; i++) {
      PetscCall(VecAssemblySend_MPI_Private(comm,tag,i,x->sendranks[i],x->sendhdr+i,x->sendreqs+4*i,X));
    }
    for (i=0; i<x->nrecvranks; i++) {
      PetscCall(VecAssemblyRecv_MPI_Private(comm,tag,x->recvranks[i],x->recvhdr+i,x->recvreqs+4*i,X));
    }
    x->use_status = PETSC_TRUE;
  } else { /* First time assembly */
    PetscCall(PetscCommBuildTwoSidedFReq(comm,3,MPIU_INT,x->nsendranks,x->sendranks,(PetscInt*)x->sendhdr,&x->nrecvranks,&x->recvranks,&x->recvhdr,4,&x->sendreqs,&x->recvreqs,VecAssemblySend_MPI_Private,VecAssemblyRecv_MPI_Private,X));
    x->use_status = PETSC_FALSE;
  }

  /* The first_assembly_done flag is only meaningful when x->assembly_subset is set.
     This line says when assembly_subset is set, then we mark that the first assembly is done.
   */
  x->first_assembly_done = x->assembly_subset;

  {
    PetscInt nstash,reallocs;
    PetscCall(VecStashGetInfo_Private(&X->stash,&nstash,&reallocs));
    PetscCall(PetscInfo(X,"Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs));
    PetscCall(VecStashGetInfo_Private(&X->bstash,&nstash,&reallocs));
    PetscCall(PetscInfo(X,"Block-Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAssemblyEnd_MPI_BTS(Vec X)
{
  Vec_MPI *x = (Vec_MPI*)X->data;
  PetscInt bs = X->map->bs;
  PetscMPIInt npending,*some_indices,r;
  MPI_Status  *some_statuses;
  PetscScalar *xarray;
  VecAssemblyFrame *frame;

  PetscFunctionBegin;
  if (X->stash.donotstash) {
    X->stash.insertmode = NOT_SET_VALUES;
    X->bstash.insertmode = NOT_SET_VALUES;
    PetscFunctionReturn(0);
  }

  PetscCheck(x->segrecvframe,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing segrecvframe! Probably you forgot to call VecAssemblyBegin first");
  PetscCall(VecGetArray(X,&xarray));
  PetscCall(PetscSegBufferExtractInPlace(x->segrecvframe,&frame));
  PetscCall(PetscMalloc2(4*x->nrecvranks,&some_indices,x->use_status?4*x->nrecvranks:0,&some_statuses));
  for (r=0,npending=0; r<x->nrecvranks; r++) npending += frame[r].pendings + frame[r].pendingb;
  while (npending>0) {
    PetscMPIInt ndone=0,ii;
    /* Filling MPI_Status fields requires some resources from the MPI library.  We skip it on the first assembly, or
     * when VEC_SUBSET_OFF_PROC_ENTRIES has not been set, because we could exchange exact sizes in the initial
     * rendezvous.  When the rendezvous is elided, however, we use MPI_Status to get actual message lengths, so that
     * subsequent assembly can set a proper subset of the values. */
    PetscCallMPI(MPI_Waitsome(4*x->nrecvranks,x->recvreqs,&ndone,some_indices,x->use_status?some_statuses:MPI_STATUSES_IGNORE));
    for (ii=0; ii<ndone; ii++) {
      PetscInt i = some_indices[ii]/4,j,k;
      InsertMode imode = (InsertMode)x->recvhdr[i].insertmode;
      PetscInt *recvint;
      PetscScalar *recvscalar;
      PetscBool intmsg = (PetscBool)(some_indices[ii]%2 == 0);
      PetscBool blockmsg = (PetscBool)((some_indices[ii]%4)/2 == 1);
      npending--;
      if (!blockmsg) { /* Scalar stash */
        PetscMPIInt count;
        if (--frame[i].pendings > 0) continue;
        if (x->use_status) {
          PetscCallMPI(MPI_Get_count(&some_statuses[ii],intmsg ? MPIU_INT : MPIU_SCALAR,&count));
        } else count = x->recvhdr[i].count;
        for (j=0,recvint=frame[i].ints,recvscalar=frame[i].scalars; j<count; j++,recvint++) {
          PetscInt loc = *recvint - X->map->rstart;
          PetscCheckFalse(*recvint < X->map->rstart || X->map->rend <= *recvint,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Received vector entry %" PetscInt_FMT " out of local range [%" PetscInt_FMT ",%" PetscInt_FMT ")]",*recvint,X->map->rstart,X->map->rend);
          switch (imode) {
          case ADD_VALUES:
            xarray[loc] += *recvscalar++;
            break;
          case INSERT_VALUES:
            xarray[loc] = *recvscalar++;
            break;
          default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Insert mode not supported 0x%x",imode);
          }
        }
      } else {                  /* Block stash */
        PetscMPIInt count;
        if (--frame[i].pendingb > 0) continue;
        if (x->use_status) {
          PetscCallMPI(MPI_Get_count(&some_statuses[ii],intmsg ? MPIU_INT : MPIU_SCALAR,&count));
          if (!intmsg) count /= bs; /* Convert from number of scalars to number of blocks */
        } else count = x->recvhdr[i].bcount;
        for (j=0,recvint=frame[i].intb,recvscalar=frame[i].scalarb; j<count; j++,recvint++) {
          PetscInt loc = (*recvint)*bs - X->map->rstart;
          switch (imode) {
          case ADD_VALUES:
            for (k=loc; k<loc+bs; k++) xarray[k] += *recvscalar++;
            break;
          case INSERT_VALUES:
            for (k=loc; k<loc+bs; k++) xarray[k] = *recvscalar++;
            break;
          default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Insert mode not supported 0x%x",imode);
          }
        }
      }
    }
  }
  PetscCall(VecRestoreArray(X,&xarray));
  PetscCallMPI(MPI_Waitall(4*x->nsendranks,x->sendreqs,MPI_STATUSES_IGNORE));
  PetscCall(PetscFree2(some_indices,some_statuses));
  if (x->assembly_subset) {
    PetscCall(PetscSegBufferExtractInPlace(x->segrecvint,NULL));
    PetscCall(PetscSegBufferExtractInPlace(x->segrecvscalar,NULL));
  } else {
    PetscCall(VecAssemblyReset_MPI(X));
  }

  X->stash.insertmode = NOT_SET_VALUES;
  X->bstash.insertmode = NOT_SET_VALUES;
  PetscCall(VecStashScatterEnd_Private(&X->stash));
  PetscCall(VecStashScatterEnd_Private(&X->bstash));
  PetscFunctionReturn(0);
}

PetscErrorCode VecAssemblyReset_MPI(Vec X)
{
  Vec_MPI *x = (Vec_MPI*)X->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(x->sendreqs));
  PetscCall(PetscFree(x->recvreqs));
  PetscCall(PetscFree(x->sendranks));
  PetscCall(PetscFree(x->recvranks));
  PetscCall(PetscFree(x->sendhdr));
  PetscCall(PetscFree(x->recvhdr));
  PetscCall(PetscFree(x->sendptrs));
  PetscCall(PetscSegBufferDestroy(&x->segrecvint));
  PetscCall(PetscSegBufferDestroy(&x->segrecvscalar));
  PetscCall(PetscSegBufferDestroy(&x->segrecvframe));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetFromOptions_MPI(PetscOptionItems *PetscOptionsObject,Vec X)
{
#if !defined(PETSC_HAVE_MPIUNI)
  PetscBool      flg = PETSC_FALSE,set;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"VecMPI Options");
  PetscCall(PetscOptionsBool("-vec_assembly_legacy","Use MPI 1 version of assembly","",flg,&flg,&set));
  if (set) {
    X->ops->assemblybegin = flg ? VecAssemblyBegin_MPI : VecAssemblyBegin_MPI_BTS;
    X->ops->assemblyend   = flg ? VecAssemblyEnd_MPI   : VecAssemblyEnd_MPI_BTS;
  }
  PetscOptionsHeadEnd();
#else
  PetscFunctionBegin;
  X->ops->assemblybegin = VecAssemblyBegin_MPI;
  X->ops->assemblyend   = VecAssemblyEnd_MPI;
#endif
  PetscFunctionReturn(0);
}

static struct _VecOps DvOps = {
  PetscDesignatedInitializer(duplicate,VecDuplicate_MPI), /* 1 */
  PetscDesignatedInitializer(duplicatevecs,VecDuplicateVecs_Default),
  PetscDesignatedInitializer(destroyvecs,VecDestroyVecs_Default),
  PetscDesignatedInitializer(dot,VecDot_MPI),
  PetscDesignatedInitializer(mdot,VecMDot_MPI),
  PetscDesignatedInitializer(norm,VecNorm_MPI),
  PetscDesignatedInitializer(tdot,VecTDot_MPI),
  PetscDesignatedInitializer(mtdot,VecMTDot_MPI),
  PetscDesignatedInitializer(scale,VecScale_Seq),
  PetscDesignatedInitializer(copy,VecCopy_Seq), /* 10 */
  PetscDesignatedInitializer(set,VecSet_Seq),
  PetscDesignatedInitializer(swap,VecSwap_Seq),
  PetscDesignatedInitializer(axpy,VecAXPY_Seq),
  PetscDesignatedInitializer(axpby,VecAXPBY_Seq),
  PetscDesignatedInitializer(maxpy,VecMAXPY_Seq),
  PetscDesignatedInitializer(aypx,VecAYPX_Seq),
  PetscDesignatedInitializer(waxpy,VecWAXPY_Seq),
  PetscDesignatedInitializer(axpbypcz,VecAXPBYPCZ_Seq),
  PetscDesignatedInitializer(pointwisemult,VecPointwiseMult_Seq),
  PetscDesignatedInitializer(pointwisedivide,VecPointwiseDivide_Seq),
  PetscDesignatedInitializer(setvalues,VecSetValues_MPI), /* 20 */
  PetscDesignatedInitializer(assemblybegin,VecAssemblyBegin_MPI_BTS),
  PetscDesignatedInitializer(assemblyend,VecAssemblyEnd_MPI_BTS),
  PetscDesignatedInitializer(getarray,NULL),
  PetscDesignatedInitializer(getsize,VecGetSize_MPI),
  PetscDesignatedInitializer(getlocalsize,VecGetSize_Seq),
  PetscDesignatedInitializer(restorearray,NULL),
  PetscDesignatedInitializer(max,VecMax_MPI),
  PetscDesignatedInitializer(min,VecMin_MPI),
  PetscDesignatedInitializer(setrandom,VecSetRandom_Seq),
  PetscDesignatedInitializer(setoption,VecSetOption_MPI),
  PetscDesignatedInitializer(setvaluesblocked,VecSetValuesBlocked_MPI),
  PetscDesignatedInitializer(destroy,VecDestroy_MPI),
  PetscDesignatedInitializer(view,VecView_MPI),
  PetscDesignatedInitializer(placearray,VecPlaceArray_MPI),
  PetscDesignatedInitializer(replacearray,VecReplaceArray_Seq),
  PetscDesignatedInitializer(dot_local,VecDot_Seq),
  PetscDesignatedInitializer(tdot_local,VecTDot_Seq),
  PetscDesignatedInitializer(norm_local,VecNorm_Seq),
  PetscDesignatedInitializer(mdot_local,VecMDot_Seq),
  PetscDesignatedInitializer(mtdot_local,VecMTDot_Seq),
  PetscDesignatedInitializer(load,VecLoad_Default),
  PetscDesignatedInitializer(reciprocal,VecReciprocal_Default),
  PetscDesignatedInitializer(conjugate,VecConjugate_Seq),
  PetscDesignatedInitializer(setlocaltoglobalmapping,NULL),
  PetscDesignatedInitializer(setvalueslocal,NULL),
  PetscDesignatedInitializer(resetarray,VecResetArray_MPI),
  PetscDesignatedInitializer(setfromoptions,VecSetFromOptions_MPI),/*set from options */
  PetscDesignatedInitializer(maxpointwisedivide,VecMaxPointwiseDivide_Seq),
  PetscDesignatedInitializer(pointwisemax,VecPointwiseMax_Seq),
  PetscDesignatedInitializer(pointwisemaxabs,VecPointwiseMaxAbs_Seq),
  PetscDesignatedInitializer(pointwisemin,VecPointwiseMin_Seq),
  PetscDesignatedInitializer(getvalues,VecGetValues_MPI),
  PetscDesignatedInitializer(sqrt,NULL),
  PetscDesignatedInitializer(abs,NULL),
  PetscDesignatedInitializer(exp,NULL),
  PetscDesignatedInitializer(log,NULL),
  PetscDesignatedInitializer(shift,NULL),
  PetscDesignatedInitializer(create,NULL), /* really? */
  PetscDesignatedInitializer(stridegather,VecStrideGather_Default),
  PetscDesignatedInitializer(stridescatter,VecStrideScatter_Default),
  PetscDesignatedInitializer(dotnorm2,NULL),
  PetscDesignatedInitializer(getsubvector,NULL),
  PetscDesignatedInitializer(restoresubvector,NULL),
  PetscDesignatedInitializer(getarrayread,NULL),
  PetscDesignatedInitializer(restorearrayread,NULL),
  PetscDesignatedInitializer(stridesubsetgather,VecStrideSubSetGather_Default),
  PetscDesignatedInitializer(stridesubsetscatter,VecStrideSubSetScatter_Default),
  PetscDesignatedInitializer(viewnative,VecView_MPI),
  PetscDesignatedInitializer(loadnative,NULL),
  PetscDesignatedInitializer(getlocalvector,NULL),
};

/*
    VecCreate_MPI_Private - Basic create routine called by VecCreate_MPI() (i.e. VecCreateMPI()),
    VecCreateMPIWithArray(), VecCreate_Shared() (i.e. VecCreateShared()), VecCreateGhost(),
    VecDuplicate_MPI(), VecCreateGhostWithArray(), VecDuplicate_MPI(), and VecDuplicate_Shared()

    If alloc is true and array is NULL then this routine allocates the space, otherwise
    no space is allocated.
*/
PetscErrorCode VecCreate_MPI_Private(Vec v,PetscBool alloc,PetscInt nghost,const PetscScalar array[])
{
  Vec_MPI        *s;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(v,&s));
  v->data        = (void*)s;
  PetscCall(PetscMemcpy(v->ops,&DvOps,sizeof(DvOps)));
  s->nghost      = nghost;
  v->petscnative = PETSC_TRUE;
  if (array) v->offloadmask = PETSC_OFFLOAD_CPU;

  PetscCall(PetscLayoutSetUp(v->map));

  s->array           = (PetscScalar*)array;
  s->array_allocated = NULL;
  if (alloc && !array) {
    PetscInt n = v->map->n+nghost;
    PetscCall(PetscCalloc1(n,&s->array));
    PetscCall(PetscLogObjectMemory((PetscObject)v,n*sizeof(PetscScalar)));
    s->array_allocated = s->array;
  }

  /* By default parallel vectors do not have local representation */
  s->localrep    = NULL;
  s->localupdate = NULL;

  v->stash.insertmode = NOT_SET_VALUES;
  v->bstash.insertmode = NOT_SET_VALUES;
  /* create the stashes. The block-size for bstash is set later when
     VecSetValuesBlocked is called.
  */
  PetscCall(VecStashCreate_Private(PetscObjectComm((PetscObject)v),1,&v->stash));
  PetscCall(VecStashCreate_Private(PetscObjectComm((PetscObject)v),PetscAbs(v->map->bs),&v->bstash));

#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscCall(PetscObjectComposeFunction((PetscObject)v,"PetscMatlabEnginePut_C",VecMatlabEnginePut_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)v,"PetscMatlabEngineGet_C",VecMatlabEngineGet_Default));
#endif
  PetscCall(PetscObjectChangeTypeName((PetscObject)v,VECMPI));
  PetscFunctionReturn(0);
}

/*MC
   VECMPI - VECMPI = "mpi" - The basic parallel vector

   Options Database Keys:
. -vec_type mpi - sets the vector type to VECMPI during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateMPI()
M*/

PetscErrorCode VecCreate_MPI(Vec vv)
{
  PetscFunctionBegin;
  PetscCall(VecCreate_MPI_Private(vv,PETSC_TRUE,0,NULL));
  PetscFunctionReturn(0);
}

/*MC
   VECSTANDARD = "standard" - A VECSEQ on one process and VECMPI on more than one process

   Options Database Keys:
. -vec_type standard - sets a vector type to standard on calls to VecSetFromOptions()

  Level: beginner

.seealso: VecCreateSeq(), VecCreateMPI()
M*/

PETSC_EXTERN PetscErrorCode VecCreate_Standard(Vec v)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)v),&size));
  if (size == 1) {
    PetscCall(VecSetType(v,VECSEQ));
  } else {
    PetscCall(VecSetType(v,VECMPI));
  }
  PetscFunctionReturn(0);
}

/*@C
   VecCreateMPIWithArray - Creates a parallel, array-style vector,
   where the user provides the array space to store the vector values.

   Collective

   Input Parameters:
+  comm  - the MPI communicator to use
.  bs    - block size, same meaning as VecSetBlockSize()
.  n     - local vector length, cannot be PETSC_DECIDE
.  N     - global vector length (or PETSC_DECIDE to have calculated)
-  array - the user provided array to store the vector values

   Output Parameter:
.  vv - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is NULL, then VecPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateSeqWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()

@*/
PetscErrorCode  VecCreateMPIWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar array[],Vec *vv)
{
  PetscFunctionBegin;
  PetscCheck(n != PETSC_DECIDE,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  PetscCall(PetscSplitOwnership(comm,&n,&N));
  PetscCall(VecCreate(comm,vv));
  PetscCall(VecSetSizes(*vv,n,N));
  PetscCall(VecSetBlockSize(*vv,bs));
  PetscCall(VecCreate_MPI_Private(*vv,PETSC_FALSE,0,array));
  PetscFunctionReturn(0);
}

/*@C
   VecCreateGhostWithArray - Creates a parallel vector with ghost padding on each processor;
   the caller allocates the array space.

   Collective

   Input Parameters:
+  comm - the MPI communicator to use
.  n - local vector length
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)
.  nghost - number of local ghost points
.  ghosts - global indices of ghost points (or NULL if not needed), these do not need to be in increasing order (sorted)
-  array - the space to store the vector values (as long as n + nghost)

   Output Parameter:
.  vv - the global vector representation (without ghost points as part of vector)

   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation
   of the vector.

   This also automatically sets the ISLocalToGlobalMapping() for this vector.

   Level: advanced

.seealso: VecCreate(), VecGhostGetLocalForm(), VecGhostRestoreLocalForm(),
          VecCreateGhost(), VecCreateSeqWithArray(), VecCreateMPIWithArray(),
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray(), VecMPISetGhost()

@*/
PetscErrorCode  VecCreateGhostWithArray(MPI_Comm comm,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],const PetscScalar array[],Vec *vv)
{
  Vec_MPI                *w;
  PetscScalar            *larray;
  IS                     from,to;
  ISLocalToGlobalMapping ltog;
  PetscInt               rstart,i,*indices;

  PetscFunctionBegin;
  *vv = NULL;

  PetscCheck(n != PETSC_DECIDE,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size");
  PetscCheck(nghost != PETSC_DECIDE,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local ghost size");
  PetscCheck(nghost >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Ghost length must be >= 0");
  PetscCall(PetscSplitOwnership(comm,&n,&N));
  /* Create global representation */
  PetscCall(VecCreate(comm,vv));
  PetscCall(VecSetSizes(*vv,n,N));
  PetscCall(VecCreate_MPI_Private(*vv,PETSC_TRUE,nghost,array));
  w    = (Vec_MPI*)(*vv)->data;
  /* Create local representation */
  PetscCall(VecGetArray(*vv,&larray));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,n+nghost,larray,&w->localrep));
  PetscCall(PetscLogObjectParent((PetscObject)*vv,(PetscObject)w->localrep));
  PetscCall(VecRestoreArray(*vv,&larray));

  /*
       Create scatter context for scattering (updating) ghost values
  */
  PetscCall(ISCreateGeneral(comm,nghost,ghosts,PETSC_COPY_VALUES,&from));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,nghost,n,1,&to));
  PetscCall(VecScatterCreate(*vv,from,w->localrep,to,&w->localupdate));
  PetscCall(PetscLogObjectParent((PetscObject)*vv,(PetscObject)w->localupdate));
  PetscCall(ISDestroy(&to));
  PetscCall(ISDestroy(&from));

  /* set local to global mapping for ghosted vector */
  PetscCall(PetscMalloc1(n+nghost,&indices));
  PetscCall(VecGetOwnershipRange(*vv,&rstart,NULL));
  for (i=0; i<n; i++) {
    indices[i] = rstart + i;
  }
  for (i=0; i<nghost; i++) {
    indices[n+i] = ghosts[i];
  }
  PetscCall(ISLocalToGlobalMappingCreate(comm,1,n+nghost,indices,PETSC_OWN_POINTER,&ltog));
  PetscCall(VecSetLocalToGlobalMapping(*vv,ltog));
  PetscCall(ISLocalToGlobalMappingDestroy(&ltog));
  PetscFunctionReturn(0);
}

/*@
   VecCreateGhost - Creates a parallel vector with ghost padding on each processor.

   Collective

   Input Parameters:
+  comm - the MPI communicator to use
.  n - local vector length
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)
.  nghost - number of local ghost points
-  ghosts - global indices of ghost points, these do not need to be in increasing order (sorted)

   Output Parameter:
.  vv - the global vector representation (without ghost points as part of vector)

   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation
   of the vector.

   This also automatically sets the ISLocalToGlobalMapping() for this vector.

   Level: advanced

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), VecGhostUpdateBegin(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray(), VecGhostUpdateEnd(),
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray(), VecMPISetGhost()

@*/
PetscErrorCode  VecCreateGhost(MPI_Comm comm,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],Vec *vv)
{
  PetscFunctionBegin;
  PetscCall(VecCreateGhostWithArray(comm,n,N,nghost,ghosts,NULL,vv));
  PetscFunctionReturn(0);
}

/*@
   VecMPISetGhost - Sets the ghost points for an MPI ghost vector

   Collective on Vec

   Input Parameters:
+  vv - the MPI vector
.  nghost - number of local ghost points
-  ghosts - global indices of ghost points, these do not need to be in increasing order (sorted)

   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation
   of the vector.

   This also automatically sets the ISLocalToGlobalMapping() for this vector.

   You must call this AFTER you have set the type of the vector (with VecSetType()) and the size (with VecSetSizes()).

   Level: advanced

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), VecGhostUpdateBegin(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray(), VecGhostUpdateEnd(),
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray()

@*/
PetscErrorCode  VecMPISetGhost(Vec vv,PetscInt nghost,const PetscInt ghosts[])
{
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)vv,VECMPI,&flg));
  /* if already fully existant VECMPI then basically destroy it and rebuild with ghosting */
  if (flg) {
    PetscInt               n,N;
    Vec_MPI                *w;
    PetscScalar            *larray;
    IS                     from,to;
    ISLocalToGlobalMapping ltog;
    PetscInt               rstart,i,*indices;
    MPI_Comm               comm;

    PetscCall(PetscObjectGetComm((PetscObject)vv,&comm));
    n    = vv->map->n;
    N    = vv->map->N;
    PetscCall((*vv->ops->destroy)(vv));
    PetscCall(VecSetSizes(vv,n,N));
    PetscCall(VecCreate_MPI_Private(vv,PETSC_TRUE,nghost,NULL));
    w    = (Vec_MPI*)(vv)->data;
    /* Create local representation */
    PetscCall(VecGetArray(vv,&larray));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,n+nghost,larray,&w->localrep));
    PetscCall(PetscLogObjectParent((PetscObject)vv,(PetscObject)w->localrep));
    PetscCall(VecRestoreArray(vv,&larray));

    /*
     Create scatter context for scattering (updating) ghost values
     */
    PetscCall(ISCreateGeneral(comm,nghost,ghosts,PETSC_COPY_VALUES,&from));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,nghost,n,1,&to));
    PetscCall(VecScatterCreate(vv,from,w->localrep,to,&w->localupdate));
    PetscCall(PetscLogObjectParent((PetscObject)vv,(PetscObject)w->localupdate));
    PetscCall(ISDestroy(&to));
    PetscCall(ISDestroy(&from));

    /* set local to global mapping for ghosted vector */
    PetscCall(PetscMalloc1(n+nghost,&indices));
    PetscCall(VecGetOwnershipRange(vv,&rstart,NULL));

    for (i=0; i<n; i++)      indices[i]   = rstart + i;
    for (i=0; i<nghost; i++) indices[n+i] = ghosts[i];

    PetscCall(ISLocalToGlobalMappingCreate(comm,1,n+nghost,indices,PETSC_OWN_POINTER,&ltog));
    PetscCall(VecSetLocalToGlobalMapping(vv,ltog));
    PetscCall(ISLocalToGlobalMappingDestroy(&ltog));
  } else PetscCheck(vv->ops->create != VecCreate_MPI,PetscObjectComm((PetscObject)vv),PETSC_ERR_ARG_WRONGSTATE,"Must set local or global size before setting ghosting");
  else PetscCheckFalse(!((PetscObject)vv)->type_name,PetscObjectComm((PetscObject)vv),PETSC_ERR_ARG_WRONGSTATE,"Must set type to VECMPI before ghosting");
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------------*/
/*@C
   VecCreateGhostBlockWithArray - Creates a parallel vector with ghost padding on each processor;
   the caller allocates the array space. Indices in the ghost region are based on blocks.

   Collective

   Input Parameters:
+  comm - the MPI communicator to use
.  bs - block size
.  n - local vector length
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)
.  nghost - number of local ghost blocks
.  ghosts - global indices of ghost blocks (or NULL if not needed), counts are by block not by index, these do not need to be in increasing order (sorted)
-  array - the space to store the vector values (as long as n + nghost*bs)

   Output Parameter:
.  vv - the global vector representation (without ghost points as part of vector)

   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation
   of the vector.

   n is the local vector size (total local size not the number of blocks) while nghost
   is the number of blocks in the ghost portion, i.e. the number of elements in the ghost
   portion is bs*nghost

   Level: advanced

.seealso: VecCreate(), VecGhostGetLocalForm(), VecGhostRestoreLocalForm(),
          VecCreateGhost(), VecCreateSeqWithArray(), VecCreateMPIWithArray(),
          VecCreateGhostWithArray(), VecCreateGhostBlock()

@*/
PetscErrorCode  VecCreateGhostBlockWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],const PetscScalar array[],Vec *vv)
{
  Vec_MPI                *w;
  PetscScalar            *larray;
  IS                     from,to;
  ISLocalToGlobalMapping ltog;
  PetscInt               rstart,i,nb,*indices;

  PetscFunctionBegin;
  *vv = NULL;

  PetscCheck(n != PETSC_DECIDE,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size");
  PetscCheck(nghost != PETSC_DECIDE,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local ghost size");
  PetscCheck(nghost >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Ghost length must be >= 0");
  PetscCheck(n % bs == 0,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local size must be a multiple of block size");
  PetscCall(PetscSplitOwnership(comm,&n,&N));
  /* Create global representation */
  PetscCall(VecCreate(comm,vv));
  PetscCall(VecSetSizes(*vv,n,N));
  PetscCall(VecSetBlockSize(*vv,bs));
  PetscCall(VecCreate_MPI_Private(*vv,PETSC_TRUE,nghost*bs,array));
  w    = (Vec_MPI*)(*vv)->data;
  /* Create local representation */
  PetscCall(VecGetArray(*vv,&larray));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n+bs*nghost,larray,&w->localrep));
  PetscCall(PetscLogObjectParent((PetscObject)*vv,(PetscObject)w->localrep));
  PetscCall(VecRestoreArray(*vv,&larray));

  /*
       Create scatter context for scattering (updating) ghost values
  */
  PetscCall(ISCreateBlock(comm,bs,nghost,ghosts,PETSC_COPY_VALUES,&from));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,bs*nghost,n,1,&to));
  PetscCall(VecScatterCreate(*vv,from,w->localrep,to,&w->localupdate));
  PetscCall(PetscLogObjectParent((PetscObject)*vv,(PetscObject)w->localupdate));
  PetscCall(ISDestroy(&to));
  PetscCall(ISDestroy(&from));

  /* set local to global mapping for ghosted vector */
  nb     = n/bs;
  PetscCall(PetscMalloc1(nb+nghost,&indices));
  PetscCall(VecGetOwnershipRange(*vv,&rstart,NULL));
  rstart = rstart/bs;

  for (i=0; i<nb; i++)      indices[i]    = rstart + i;
  for (i=0; i<nghost; i++)  indices[nb+i] = ghosts[i];

  PetscCall(ISLocalToGlobalMappingCreate(comm,bs,nb+nghost,indices,PETSC_OWN_POINTER,&ltog));
  PetscCall(VecSetLocalToGlobalMapping(*vv,ltog));
  PetscCall(ISLocalToGlobalMappingDestroy(&ltog));
  PetscFunctionReturn(0);
}

/*@
   VecCreateGhostBlock - Creates a parallel vector with ghost padding on each processor.
        The indicing of the ghost points is done with blocks.

   Collective

   Input Parameters:
+  comm - the MPI communicator to use
.  bs - the block size
.  n - local vector length
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)
.  nghost - number of local ghost blocks
-  ghosts - global indices of ghost blocks, counts are by block, not by individual index, these do not need to be in increasing order (sorted)

   Output Parameter:
.  vv - the global vector representation (without ghost points as part of vector)

   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation
   of the vector.

   n is the local vector size (total local size not the number of blocks) while nghost
   is the number of blocks in the ghost portion, i.e. the number of elements in the ghost
   portion is bs*nghost

   Level: advanced

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray(), VecCreateGhostBlockWithArray()

@*/
PetscErrorCode  VecCreateGhostBlock(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],Vec *vv)
{
  PetscFunctionBegin;
  PetscCall(VecCreateGhostBlockWithArray(comm,bs,n,N,nghost,ghosts,NULL,vv));
  PetscFunctionReturn(0);
}
