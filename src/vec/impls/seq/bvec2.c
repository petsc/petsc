#ifndef lint
static char vcid[] = "$Id: bvec2.c,v 1.24 1995/05/18 22:43:46 bsmith Exp bsmith $";
#endif
/*
   Defines the sequential BLAS based vectors
*/

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

static int VecNorm_Blas(Vec xin,double* z )
{
  Vec_Seq * x = (Vec_Seq *) xin->data;
  int  one = 1;
  *z = BLnrm2_( &x->n, x->array, &one );
  PLogFlops(2*x->n-1);
  return 0;
}

static int VecGetOwnershipRange_Seq(Vec xin, int *low,int *high )
{
  Vec_Seq *x = (Vec_Seq *) xin->data;
  *low = 0; *high = x->n;
  return 0;
}
#include "viewer.h"

static int VecView_Seq(PetscObject obj,Viewer ptr)
{
  Vec         xin = (Vec) obj;
  Vec_Seq    *x = (Vec_Seq *)xin->data;
  PetscObject vobj = (PetscObject) ptr;
  int         i, n = x->n, ierr;
  FILE        *fd;

  if (!ptr) { /* so that viewers may be used from debuggers */
    ptr = STDOUT_VIEWER; vobj = (PetscObject) ptr;
  }
  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;

  if (vobj->cookie == VIEWER_COOKIE && ((vobj->type == FILE_VIEWER) ||
                                       (vobj->type == FILES_VIEWER)))  {
    fd = ViewerFileGetPointer_Private(ptr);
    for (i=0; i<n; i++ ) {
#if defined(PETSC_COMPLEX)
      fprintf(fd,"%g + %gi\n",real(x->array[i]),imag(x->array[i]));
#else
      fprintf(fd,"%g\n",x->array[i]);
#endif
    }
  }
#if !defined(PETSC_COMPLEX)
  else if (vobj->cookie == LG_COOKIE){
    DrawLGCtx lg = (DrawLGCtx) ptr;
    DrawCtx   win;
    double    *xx;
    DrawLGGetDrawCtx(lg,&win);
    DrawLGReset(lg);
    xx = (double *) MALLOC( (n+1)*sizeof(double) ); CHKPTR(xx);
    for ( i=0; i<n; i++ ) {
      xx[i] = (double) i;
    }
    DrawLGAddPoints(lg,n,&xx,&x->array);
    FREE(xx);
    DrawLG(lg);
    DrawSyncFlush(win);
  }
  else if (vobj->cookie == DRAW_COOKIE) {
    DrawCtx   win = (DrawCtx) ptr;
    DrawLGCtx lg;
    ierr = DrawLGCreate(win,1,&lg); CHKERR(ierr);
    ierr = VecView(xin,(Viewer) lg); CHKERR(ierr);
    DrawLGDestroy(lg);
  }
  else if (vobj->cookie == VIEWER_COOKIE && vobj->type == MATLAB_VIEWER) {
    return ViewerMatlabPutArray_Private(ptr,x->n,1,x->array); 
  }
#endif
  return 0;
}

static int VecSetValues_Seq(Vec xin, int ni, int *ix,Scalar* y,InsertMode m)
{
  Vec_Seq *x = (Vec_Seq *)xin->data;
  Scalar   *xx = x->array;
  int      i;

  if (m == INSERTVALUES) {
    for ( i=0; i<ni; i++ ) {
#if defined(PETSC_DEBUG)
      if (ix[i] < 0 || ix[i] >= x->n) SETERR(1,"Index out of range");
#endif
      xx[ix[i]] = y[i];
    }
  }
  else {
    for ( i=0; i<ni; i++ ) {
#if defined(PETSC_DEBUG)
      if (ix[i] < 0 || ix[i] >= x->n) SETERR(1,"Index out of range");
#endif
      xx[ix[i]] += y[i];
    }  
  }  
  return 0;
}

static int VecDestroy_Seq(PetscObject obj )
{
  Vec      v = (Vec ) obj;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows %d",((Vec_Seq *)v->data)->n);
#endif
  FREE(v->data);
  PLogObjectDestroy(v);
  PETSCHEADERDESTROY(v); 
  return 0;
}

static int VecDuplicate_Blas(Vec,Vec*);

static struct _VeOps DvOps = {VecDuplicate_Blas, 
            Veiobtain_vectors, Veirelease_vectors, VecDot_Blas, VecMDot_Seq,
            VecNorm_Blas, VecAMax_Seq, VecAsum_Blas, VecDot_Blas, VecMDot_Seq,
            VecScale_Blas, VecCopy_Blas,
            VecSet_Seq, VecSwap_Blas, VecAXPY_Blas, VecMAXPY_Seq, VecAYPX_Seq,
            VecWAXPY_Seq, VecPMult_Seq,
            VecPDiv_Seq,  
            VecSetValues_Seq,0,0,
            VecGetArray_Seq, VecGetSize_Seq,VecGetSize_Seq ,
            VecGetOwnershipRange_Seq,0,VecMax_Seq,VecMin_Seq};

/*@
   VecCreateSequential - Creates a standard, array-style vector.

   Input Parameter:
.  comm - the communicator, should be MPI_COMM_SELF
.  n - the vector length 

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecGetVecs() to form additional vectors of the
   same type as an existing vector.

.keywords: vector, sequential, create, BLAS

.seealso: VecCreateMPI(), VecCreate(), VecDuplicate(), VecGetVecs()
@*/
int VecCreateSequential(MPI_Comm comm,int n,Vec *V)
{
  int      size = sizeof(Vec_Seq)+n*sizeof(Scalar),flag;
  Vec      v;
  Vec_Seq *s;
  *V             = 0;
  MPI_Comm_compare(MPI_COMM_SELF,comm,&flag);
  if (flag == MPI_UNEQUAL) SETERR(1,"Must call with MPI_COMM_SELF");
  PETSCHEADERCREATE(v,_Vec,VEC_COOKIE,SEQVECTOR,comm);
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

static int VecDuplicate_Blas(Vec win,Vec *V)
{
  Vec_Seq *w = (Vec_Seq *)win->data;
  return VecCreateSequential(win->comm,w->n,V);
}

