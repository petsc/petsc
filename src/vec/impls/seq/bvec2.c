#ifndef lint
static char vcid[] = "$Id: bvec2.c,v 1.52 1995/11/01 19:08:35 bsmith Exp bsmith $";
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
#include "vecimpl.h"          /*I  "vec.h"   I*/
#include "dvecimpl.h" 

#include "../bvec1.c"
#include "../dvec2.c"

int VecNorm_Seq(Vec xin,NormType type,double* z )
{
  Vec_Seq * x = (Vec_Seq *) xin->data;
  int     one = 1;

  if (type == NORM_2) {
    /*
      This is because the Fortran BLAS 1 Norm is very slow! 
    */
#if defined(PARCH_sun4) && !defined(PETSC_COMPLEX)
    *z = BLdot_( &x->n, x->array, &one, x->array, &one );
    *z = sqrt(*z);
#else
    *z = BLnrm2_( &x->n, x->array, &one );
#endif
    PLogFlops(2*x->n-1);
  }
  else if (type == NORM_INFINITY) {
    register int    i, n = x->n;
    register double max = 0.0, tmp;
    Scalar          *xx = x->array;

    for (i=0; i<n; i++) {
      if ((tmp = PetscAbsScalar(*xx)) > max) max = tmp;
      xx++;
    }
    *z   = max;
  }
  else if (type == NORM_1) {
    *z = BLasum_( &x->n, x->array, &one );
    PLogFlops(x->n-1);
  }
  return 0;
}

static int VecGetOwnershipRange_Seq(Vec xin, int *low,int *high )
{
  Vec_Seq *x = (Vec_Seq *) xin->data;
  *low = 0; *high = x->n;
  return 0;
}
#include "viewer.h"
#include "sysio.h"

static int VecView_Seq_File(Vec xin,Viewer ptr)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  int      i, n = x->n,ierr;
  FILE     *fd;
  ierr = ViewerFileGetPointer_Private(ptr,&fd); CHKERRQ(ierr);

  for (i=0; i<n; i++ ) {
#if defined(PETSC_COMPLEX)
    if (imag(x->array[i]) != 0.0) {
      fprintf(fd,"%g + %gi\n",real(x->array[i]),imag(x->array[i]));
    }
    else {
      fprintf(fd,"%g\n",real(x->array[i]));
    }
#else
    fprintf(fd,"%g\n",x->array[i]);
#endif
  }
  return 0;
}

#if !defined(PETSC_COMPLEX)
static int VecView_Seq_LG(Vec xin,DrawLGCtx lg)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  int      i, n = x->n;
  DrawCtx  win;
  double   *xx;
  DrawLGGetDrawCtx(lg,&win);
  DrawLGReset(lg);
  xx = (double *) PetscMalloc( (n+1)*sizeof(double) ); CHKPTRQ(xx);
  for ( i=0; i<n; i++ ) {
    xx[i] = (double) i;
  }
  DrawLGAddPoints(lg,n,&xx,&x->array);
  PetscFree(xx);
  DrawLG(lg);
  DrawSyncFlush(win);
  DrawPause(win);
  return 0;
}

static int VecView_Seq_DrawCtx(Vec xin,DrawCtx win)
{
  int       ierr;
  DrawLGCtx lg;
  ierr = DrawLGCreate(win,1,&lg); CHKERRQ(ierr);
  PLogObjectParent(win,lg);
  ierr = VecView(xin,(Viewer) lg); CHKERRQ(ierr);
  DrawLGDestroy(lg);
  return 0;
}
#endif

static int VecView_Seq_Binary(Vec xin,Viewer ptr)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  int      ierr,fdes,n = x->n;

  ierr  = ViewerFileGetDescriptor_Private(ptr,&fdes); CHKERRQ(ierr);
  /* Write vector header */
  ierr = SYWrite(fdes,&xin->cookie,1,SYINT,0);CHKERRQ(ierr);
  ierr = SYWrite(fdes,&n,1,SYINT,0); CHKERRQ(ierr);

  /* Write vector contents */
  ierr = SYWrite(fdes,x->array,n,SYSCALAR,0);
  CHKERRQ(ierr);
  return 0;
}


static int VecView_Seq(PetscObject obj,Viewer ptr)
{
  Vec         xin = (Vec) obj;
  Vec_Seq    *x = (Vec_Seq *)xin->data;
  PetscObject vobj = (PetscObject) ptr;

  if (!ptr) { 
    ptr = STDOUT_VIEWER_SELF; vobj = (PetscObject) ptr;
  }

  if (vobj->cookie == VIEWER_COOKIE) {
    if ((vobj->type == ASCII_FILE_VIEWER) || (vobj->type == ASCII_FILES_VIEWER)){
      return VecView_Seq_File(xin,ptr);
    }
    else if (vobj->type == MATLAB_VIEWER) {
      return ViewerMatlabPutArray_Private(ptr,x->n,1,x->array);
    } 
    else if (vobj->type==BINARY_FILE_VIEWER) {
      return VecView_Seq_Binary(xin,ptr);
    }
  }
#if !defined(PETSC_COMPLEX)
  else if (vobj->cookie == LG_COOKIE){
    return VecView_Seq_LG(xin,(DrawLGCtx) ptr);
  }
  else if (vobj->cookie == DRAW_COOKIE) {
    if (vobj->type == NULLWINDOW){ 
      return 0;
    }
    else {
      return VecView_Seq_DrawCtx(xin,(DrawCtx) ptr);
    }
  }
#endif
  return 0;
}

static int VecSetValues_Seq(Vec xin, int ni, int *ix,Scalar* y,InsertMode m)
{
  Vec_Seq *x = (Vec_Seq *)xin->data;
  Scalar   *xx = x->array;
  int      i;

  if (m == INSERT_VALUES) {
    for ( i=0; i<ni; i++ ) {
      if (ix[i] < 0 || ix[i] >= x->n) SETERRQ(1,"VecSetValues_Seq:Out of range");
      xx[ix[i]] = y[i];
    }
  }
  else {
    for ( i=0; i<ni; i++ ) {
      if (ix[i] < 0 || ix[i] >= x->n) SETERRQ(1,"VecSetValues_Seq:Out of range");
      xx[ix[i]] += y[i];
    }  
  }  
  return 0;
}

static int VecDestroy_Seq(PetscObject obj )
{
  Vec      v = (Vec ) obj;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Length=%d",((Vec_Seq *)v->data)->n);
#endif
  PetscFree(v->data);
  PLogObjectDestroy(v);
  PetscHeaderDestroy(v); 
  return 0;
}

static int VecDuplicate_Seq(Vec,Vec*);

static struct _VeOps DvOps = {VecDuplicate_Seq, 
            Veiobtain_vectors, Veirelease_vectors, VecDot_Seq, VecMDot_Seq,
            VecNorm_Seq,  VecDot_Seq, VecMDot_Seq,
            VecScale_Seq, VecCopy_Seq,
            VecSet_Seq, VecSwap_Seq, VecAXPY_Seq, VecMAXPY_Seq, VecAYPX_Seq,
            VecWAXPY_Seq, VecPMult_Seq,
            VecPDiv_Seq,  
            VecSetValues_Seq,0,0,
            VecGetArray_Seq, VecGetSize_Seq,VecGetSize_Seq ,
            VecGetOwnershipRange_Seq,0,VecMax_Seq,VecMin_Seq};

/*@C
   VecCreateSeq - Creates a standard, array-style vector.

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
int VecCreateSeq(MPI_Comm comm,int n,Vec *V)
{
  int      size = sizeof(Vec_Seq)+n*sizeof(Scalar),flag;
  Vec      v;
  Vec_Seq *s;
  *V             = 0;
  MPI_Comm_compare(MPI_COMM_SELF,comm,&flag);
  if (flag == MPI_UNEQUAL) SETERRQ(1,"VecCreateSeq:Must call with MPI_COMM_SELF");
  PetscHeaderCreate(v,_Vec,VEC_COOKIE,VECSEQ,comm);
  PLogObjectCreate(v);
  PLogObjectMemory(v,sizeof(struct _Vec)+size);
  v->destroy     = VecDestroy_Seq;
  v->view        = VecView_Seq;
  s              = (Vec_Seq *) PetscMalloc(size); CHKPTRQ(s);
  PetscMemcpy(&v->ops,&DvOps,sizeof(DvOps));
  v->data        = (void *) s;
  s->n           = n;
  s->array       = (Scalar *)(s + 1);
  *V = v; return 0;
}

static int VecDuplicate_Seq(Vec win,Vec *V)
{
  Vec_Seq *w = (Vec_Seq *)win->data;
  return VecCreateSeq(win->comm,w->n,V);
}

