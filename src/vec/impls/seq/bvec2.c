
#ifndef lint
static char vcid[] = "$Id: bvec2.c,v 1.11 1995/03/21 23:18:11 bsmith Exp bsmith $";
#endif
/*
   Defines the sequential BLAS based vectors
*/

#include "sys/flog.h"
#include "inline/dot.h"
#include "inline/vmult.h"
#include "inline/setval.h"
#include "inline/copy.h"
#include "inline/axpy.h"
#include <math.h>
#include "vecimpl.h" 
#include "dvecimpl.h" 

#include "../bvec1.c"
#include "../dvec2.c"

static int VecCreate_Blas(Vec,Vec*);

static struct _VeOps DvOps = {VecCreate_Blas, 
            Veiobtain_vectors, Veirelease_vectors, VecDot_Blas, VecMDot_Seq,
            VecNorm_Blas, VecMax_Seq, VecAsum_Blas, VecDot_Blas, VecMDot_Seq,
            VecScale_Blas, VecCopy_Blas,
            VecSet_Seq, VecSwap_Blas, VecAXPY_Blas, VecMAXPY_Seq, VecAYPX_Seq,
            VecWAXPY_Seq, VecPMult_Seq,
            VecPDiv_Seq,  
            VecSetValues_Seq,0,0,
            VecGetArray_Seq, VecGetSize_Seq,VecGetSize_Seq ,
            VecGetOwnershipRange_Seq};

int VecCreateSequentialBLAS(int n,Vec *V)
{
  int      size = sizeof(Vec_Seq)+n*sizeof(Scalar);
  Vec      v;
  Vec_Seq *s;
  *V             = 0;
  PETSCHEADERCREATE(v,_Vec,VEC_COOKIE,SEQVECTOR,MPI_COMM_SELF);
  PLogObjectCreate(v);
  v->destroy     = VecDestroy_Seq;
  v->view        = VecView_Seq;
  s              = (Vec_Seq *) MALLOC(size); CHKPTR(s);
  v->ops         = &DvOps;
  v->data        = (void *) s;
  s->n           = n;
  s->array       = (Scalar *)(s + 1);
  *V = v; return 0;
}

static int VecCreate_Blas(Vec win,Vec *V)
{
  Vec_Seq *w = (Vec_Seq *)win->data;
  return VecCreateSequentialBLAS(w->n,V);
}

