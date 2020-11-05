
#include <../src/vec/vec/impls/node/vecnodeimpl.h>   /*I  "petscvec.h"   I*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/

#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)

PetscErrorCode VecSetValues_Node(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode addv)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"Not implemented yet");
}

/* check all blocks are filled */
static PetscErrorCode VecAssemblyBegin_Node(Vec v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAssemblyEnd_Node(Vec v)
{
  Vec_Node       *s = (Vec_Node*)v->data;

  PetscFunctionBegin;
  s->array[-1] += 1.0; /* update local object state counter if this routine changes values of v */
  /* printf("VecAssemblyEnd_Node s->array[-1] %g\n",s->array[-1]); */
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScale_Node(Vec v, PetscScalar alpha)
{
  PetscErrorCode ierr;
  Vec_Node       *s = (Vec_Node*)v->data;

  PetscFunctionBegin;
  ierr = VecScale_Seq(v,alpha);CHKERRQ(ierr);
  s->array[-1] += 1.0; /* update local object state counter if this routine changes values of v */
  /* printf("VecScale_Node s->array[-1] %g\n",s->array[-1]); */
  PetscFunctionReturn(0);
}

static PetscErrorCode VecCopy_Node(Vec v,Vec y)
{
  PetscErrorCode ierr;
  Vec_Node       *s = (Vec_Node*)y->data;

  PetscFunctionBegin;
  ierr = VecCopy_Seq(v,y);CHKERRQ(ierr);
  s->array[-1] += 1.0; /* update local object state counter if this routine changes values of y */
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSet_Node(Vec v,PetscScalar alpha)
{
  PetscErrorCode ierr;
  Vec_Node       *s = (Vec_Node*)v->data;

  PetscFunctionBegin;
  ierr = VecSet_Seq(v,alpha);CHKERRQ(ierr);
  s->array[-1] += 1.0; /* update local object state counter if this routine changes values of v */
  /* printf("VecSet_Node s->array[-1] %g\n",s->array[-1]); */
  PetscFunctionReturn(0);
}

static PetscErrorCode VecDestroy_Node(Vec v)
{
  Vec_Node       *vs = (Vec_Node*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Win_free(&vs->win);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&vs->shmcomm);CHKERRQ(ierr);
  ierr = PetscFree(vs->winarray);CHKERRQ(ierr);
  ierr = PetscFree(vs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecDuplicate_Node(Vec x,Vec *y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(PetscObjectComm((PetscObject)x),y);CHKERRQ(ierr);
  ierr = VecSetSizes(*y,x->map->n,x->map->N);CHKERRQ(ierr);
  ierr = VecSetType(*y,((PetscObject)x)->type_name);CHKERRQ(ierr);
  ierr = PetscLayoutReference(x->map,&(*y)->map);CHKERRQ(ierr);
  ierr = PetscObjectListDuplicate(((PetscObject)x)->olist,&((PetscObject)(*y))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)x)->qlist,&((PetscObject)(*y))->qlist);CHKERRQ(ierr);

  ierr = PetscMemcpy((*y)->ops,x->ops,sizeof(struct _VecOps));CHKERRQ(ierr);

  /* New vector should inherit stashing property of parent */
  (*y)->stash.donotstash   = x->stash.donotstash;
  (*y)->stash.ignorenegidx = x->stash.ignorenegidx;

  (*y)->map->bs   = PetscAbs(x->map->bs);
  (*y)->bstash.bs = x->bstash.bs;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAYPX_Node(Vec y,PetscScalar alpha,Vec x)
{
  PetscErrorCode ierr;
  Vec_Node       *s = (Vec_Node*)y->data;

  PetscFunctionBegin;
  ierr = VecAYPX_Seq(y,alpha,x);CHKERRQ(ierr);
  s->array[-1] += 1.0;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAXPBY_Node(Vec y,PetscScalar alpha,PetscScalar beta,Vec x)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Not implemented yet");
}

static PetscErrorCode VecAXPBYPCZ_Node(Vec z,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Vec_Node       *s = (Vec_Node*)z->data;

  PetscFunctionBegin;
  ierr = VecAXPBYPCZ_Seq(z,alpha,beta,gamma,x,y);CHKERRQ(ierr);
  s->array[-1] += 1.0;
  PetscFunctionReturn(0);
}


static PetscErrorCode VecConjugate_Node(Vec x)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Not implemented yet");
}

static PetscErrorCode VecWAXPY_Node(Vec w,PetscScalar alpha,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Vec_Node       *s = (Vec_Node*)w->data;

  PetscFunctionBegin;
  ierr = VecWAXPY_Seq(w,alpha,x,y);CHKERRQ(ierr);
  s->array[-1] += 1.0;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMax_Node(Vec x,PetscInt *p,PetscReal *max)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Not implemented yet");
}

static PetscErrorCode VecMin_Node(Vec x,PetscInt *p,PetscReal *min)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Not implemented yet");
}

/* supports nested blocks */
static PetscErrorCode VecView_Node(Vec x,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecView_MPI(x,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecGetArray_Node(Vec x,PetscScalar **a)
{
  Vec_Node       *s = (Vec_Node*)x->data;
  PetscFunctionBegin;
  *a = s->array;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecRestoreArray_Node(Vec x,PetscScalar **a)
{
  Vec_Node       *s = (Vec_Node*)x->data;

  PetscFunctionBegin;
  s->array[-1] += 1.0; /* update local object state counter if this routine changes values of v */
  /* printf("VecRestoreArray_Node s->array[-1] %g\n",s->array[-1]); */
  PetscFunctionReturn(0);
}

static PetscErrorCode VecGetArrayRead_Node(Vec x,const PetscScalar **a)
{
  Vec_Node       *s = (Vec_Node*)x->data;

  PetscFunctionBegin;
  *a = s->array;
  PetscFunctionReturn(0);
}

/* This routine prevents VecRestoreArrayRead() calls VecRestoreArray_Node(), which increaments s->array[-1] */
static PetscErrorCode VecRestoreArrayRead_Node(Vec x,const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static struct _VecOps DvOps = { VecDuplicate_Node, /* 1 */
                                VecDuplicateVecs_Default,
                                VecDestroyVecs_Default,
                                VecDot_MPI,
                                VecMDot_MPI,
                                VecNorm_MPI,
                                VecTDot_MPI,
                                VecMTDot_MPI,
                                VecScale_Node,
                                VecCopy_Node, /* 10 */
                                VecSet_Node,
                                VecSwap_Seq,
                                VecAXPY_Seq,
                                VecAXPBY_Node,
                                VecMAXPY_Seq,
                                VecAYPX_Node,
                                VecWAXPY_Node,
                                VecAXPBYPCZ_Node,
                                NULL,
                                NULL,
                                VecSetValues_Node, /* 20 */
                                VecAssemblyBegin_Node,
                                VecAssemblyEnd_Node,
                                VecGetArray_Node,
                                VecGetSize_MPI,
                                VecGetSize_Seq,
                                VecRestoreArray_Node,
                                VecMax_Node,
                                VecMin_Node,
                                VecSetRandom_Seq,
                                NULL,
                                VecSetValuesBlocked_Seq,
                                VecDestroy_Node,
                                VecView_Node,
                                VecPlaceArray_Seq,
                                VecReplaceArray_Seq,
                                VecDot_Seq,
                                VecTDot_Seq,
                                VecNorm_Seq,
                                VecMDot_Seq,
                                VecMTDot_Seq,
                                VecLoad_Default,
                                VecReciprocal_Default,
                                VecConjugate_Node,
                                NULL,
                                NULL,
                                VecResetArray_Seq,
                                NULL,/*set from options */
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                VecGetArrayRead_Node,
                                VecRestoreArrayRead_Node,
                                VecStrideSubSetGather_Default,
                                VecStrideSubSetScatter_Default,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL
};

/*@C
   VecCreateNode - Creates a new parallel vector whose arrays are stored in shared memory

   Collective on Vec

   Input Parameter:
+  comm  - Communicator for the new Vec
.  n - local vector length (or PETSC_DECIDE to have calculated if N is given)
-  N - global vector length (or PETSC_DETERMINE to have calculated if n is given)

   Output Parameter:
.  v - new vector

   Level: advanced

.seealso: VecCreate(), VecType(), VecCreateMPIWithArray(), VECNODE
@*/
PetscErrorCode VecCreateNode(MPI_Comm comm,PetscInt n,PetscInt N,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,N);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECNODE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  VECNODE - VECNODE = "node" - Vector type uses on-node shared memory.

  Level: intermediate

  Notes:
  This vector type uses on-node shared memory.

.seealso: VecCreate(), VecType.
M*/

PETSC_EXTERN PetscErrorCode VecCreate_Node(Vec v)
{
  PetscErrorCode ierr;
  Vec_Node       *s;
  PetscBool      alloc=PETSC_TRUE;
  PetscScalar    *array=NULL;
  MPI_Comm       shmcomm;
  MPI_Win        win;

  PetscFunctionBegin;
  ierr           = PetscNewLog(v,&s);CHKERRQ(ierr);
  v->data        = (void*)s;
  ierr           = PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  v->petscnative = PETSC_FALSE;

  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);

  s->array           = (PetscScalar*)array;
  s->array_allocated = NULL;

  if (alloc && !array) {
    PetscInt n = v->map->n;
    PetscMPIInt msize,mrank,disp_unit;
    PetscInt    i;
    MPI_Aint    sz;

    ierr = MPI_Comm_split_type(PetscObjectComm((PetscObject)v),MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&shmcomm);CHKERRQ(ierr);
    ierr = MPIU_Win_allocate_shared((n+1)*sizeof(PetscScalar),sizeof(PetscScalar),MPI_INFO_NULL,shmcomm,&s->array,&win);CHKERRQ(ierr);
    ierr               = PetscLogObjectMemory((PetscObject)v,(n+1)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr               = PetscArrayzero(s->array,n+1);CHKERRQ(ierr);
    s->array++;    /* create initial space for object state counter */

    ierr = MPI_Comm_size(shmcomm,&msize);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(shmcomm,&mrank);CHKERRQ(ierr);
    ierr = PetscMalloc1(msize,&s->winarray);CHKERRQ(ierr);
    for (i=0; i<msize; i++) {
      if (i != mrank) {
        MPIU_Win_shared_query(win,i,&sz,&disp_unit,&s->winarray[i]);
        s->winarray[i]++;
      }
    }
    s->win             = win;
    s->shmcomm         = shmcomm;
  }

  ierr = PetscObjectChangeTypeName((PetscObject)v,VECNODE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif
