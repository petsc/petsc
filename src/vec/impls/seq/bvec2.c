

#ifndef lint
static char vcid[] = "$Id: bvec2.c,v 1.73 1996/07/08 22:16:35 bsmith Exp bsmith $";
#endif
/*
   Implements the sequential vectors.
*/

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
#include "sys.h"

static int VecView_Seq_File(Vec xin,Viewer viewer)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  int      i, n = x->n,ierr,format;
  FILE     *fd;
  char     *outputname;

  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);

  ierr = ViewerGetFormat(viewer,&format);
  if (format == ASCII_FORMAT_MATLAB) {
    ierr = ViewerFileGetOutputname_Private(viewer,&outputname); CHKERRQ(ierr);
    fprintf(fd,"%s = [\n",outputname);
    for (i=0; i<n; i++ ) {
#if defined(PETSC_COMPLEX)
      if (imag(x->array[i]) != 0.0) {
        fprintf(fd,"%18.16e + %18.16e\n",real(x->array[i]),imag(x->array[i]));
      }
      else {
        fprintf(fd,"%18.16e\n",real(x->array[i]));
      }
#else
      fprintf(fd,"%18.16e\n",x->array[i]);
#endif
    }
    fprintf(fd,"];\n");
  } else {
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
  }
  fflush(fd);
  return 0;
}

static int VecView_Seq_Draw_LG(Vec xin,Viewer v)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  int      i, n = x->n,ierr;
  Draw     win;
  double   *xx;
  DrawLG   lg;

  ierr = ViewerDrawGetDrawLG(v,&lg); CHKERRQ(ierr);
  ierr = DrawLGGetDraw(lg,&win); CHKERRQ(ierr);
  ierr = DrawLGReset(lg); CHKERRQ(ierr);
  xx = (double *) PetscMalloc( (n+1)*sizeof(double) ); CHKPTRQ(xx);
  for ( i=0; i<n; i++ ) {
    xx[i] = (double) i;
  }
#if !defined(PETSC_COMPLEX)
  DrawLGAddPoints(lg,n,&xx,&x->array);
#else 
  {
    double *yy;
    yy = (double *) PetscMalloc( (n+1)*sizeof(double) ); CHKPTRQ(yy);    
    for ( i=0; i<n; i++ ) {
      yy[i] = real(x->array[i]);
    }
    DrawLGAddPoints(lg,n,&xx,&yy);
    PetscFree(yy);
  }
#endif
  PetscFree(xx);
  DrawLGDraw(lg);
  DrawSyncFlush(win);
  DrawPause(win);
  return 0;
}

static int VecView_Seq_Draw(Vec xin,Viewer v)
{
  int        ierr;
  Draw       draw;
  PetscTruth isnull;
  int        format;

  ierr = ViewerDrawGetDraw(v,&draw); CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;
  
  ierr = ViewerGetFormat(v,&format); CHKERRQ(ierr);
  /*
     Currently it only supports drawing to a line graph */
  if (format != VIEWER_FORMAT_DRAW_LG) {
    ViewerPushFormat(v,VIEWER_FORMAT_DRAW_LG,PETSC_NULL);
  } 
  ierr = VecView_Seq_Draw_LG(xin,v); CHKERRQ(ierr);
  if (format != VIEWER_FORMAT_DRAW_LG) {
    ViewerPopFormat(v);
  } 

  return 0;
}

static int VecView_Seq_Binary(Vec xin,Viewer viewer)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  int      ierr,fdes,n = x->n;

  ierr  = ViewerBinaryGetDescriptor(viewer,&fdes); CHKERRQ(ierr);
  /* Write vector header */
  ierr = PetscBinaryWrite(fdes,&xin->cookie,1,BINARY_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(fdes,&n,1,BINARY_INT,0); CHKERRQ(ierr);

  /* Write vector contents */
  ierr = PetscBinaryWrite(fdes,x->array,n,BINARY_SCALAR,0);
  CHKERRQ(ierr);
  return 0;
}


static int VecView_Seq(PetscObject obj,Viewer viewer)
{
  Vec         xin = (Vec) obj;
  Vec_Seq     *x = (Vec_Seq *)xin->data;
  ViewerType  vtype;
  int         ierr;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == DRAW_VIEWER){ 
    return VecView_Seq_Draw(xin,viewer);
  }
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER){
    return VecView_Seq_File(xin,viewer);
  }
  else if (vtype == MATLAB_VIEWER) {
    return ViewerMatlabPutArray_Private(viewer,x->n,1,x->array);
  } 
  else if (vtype == BINARY_FILE_VIEWER) {
    return VecView_Seq_Binary(xin,viewer);
  }
  return 0;
}

static int VecSetValues_Seq(Vec xin, int ni, int *ix,Scalar* y,InsertMode m)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
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
  Vec      v  = (Vec ) obj;
  Vec_Seq *vs = (Vec_Seq*) v->data;

#if defined(PETSC_LOG)
  PLogObjectState(obj,"Length=%d",((Vec_Seq *)v->data)->n);
#endif
  PetscFree(vs->array);
  PetscFree(vs);
  PLogObjectDestroy(v);
  PetscHeaderDestroy(v); 
  return 0;
}

static int VecDuplicate_Seq(Vec,Vec*);

static struct _VeOps DvOps = {VecDuplicate_Seq, 
            Veiobtain_vectors, Veirelease_vectors, VecDot_Seq, VecMDot_Seq,
            VecNorm_Seq,  VecDot_Seq, VecMDot_Seq,
            VecScale_Seq, VecCopy_Seq,
            VecSet_Seq, VecSwap_Seq, VecAXPY_Seq, VecAXPBY_Seq,
            VecMAXPY_Seq, VecAYPX_Seq,
            VecWAXPY_Seq, VecPointwiseMult_Seq,
            VecPointwiseDivide_Seq,  
            VecSetValues_Seq,0,0,
            VecGetArray_Seq, VecGetSize_Seq,VecGetSize_Seq,
            VecGetOwnershipRange_Seq,0,VecMax_Seq,VecMin_Seq,
            VecSetRandom_Seq};

/*@C
   VecCreateSeq - Creates a standard, sequential array-style vector.

   Input Parameter:
.  comm - the communicator, should be MPI_COMM_SELF
.  n - the vector length 

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

.keywords: vector, sequential, create, BLAS

.seealso: VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs()
@*/
int VecCreateSeq(MPI_Comm comm,int n,Vec *V)
{
  Vec      v;
  Vec_Seq *s;
  int     flag;

  *V             = 0;
  MPI_Comm_compare(MPI_COMM_SELF,comm,&flag);
  if (flag == MPI_UNEQUAL) SETERRQ(1,"VecCreateSeq:Must call with MPI_COMM_SELF");
  PetscHeaderCreate(v,_Vec,VEC_COOKIE,VECSEQ,comm);
  PLogObjectCreate(v);
  PLogObjectMemory(v,sizeof(struct _Vec)+n*sizeof(Scalar));
  v->destroy     = VecDestroy_Seq;
  v->view        = VecView_Seq;
  s              = (Vec_Seq *) PetscMalloc(sizeof(Vec_Seq)); CHKPTRQ(s);
  PetscMemcpy(&v->ops,&DvOps,sizeof(DvOps));
  v->data        = (void *) s;
  s->n           = n;
  v->n           = n; 
  v->N           = n;
  s->array       = (Scalar *) PetscMalloc((n+1)*sizeof(Scalar));
  PetscMemzero(s->array,n*sizeof(Scalar));
  *V = v; return 0;
}

static int VecDuplicate_Seq(Vec win,Vec *V)
{
  int     ierr;
  Vec_Seq *w = (Vec_Seq *)win->data;
  ierr = VecCreateSeq(win->comm,w->n,V);
  (*V)->childcopy = win->childcopy;
  if (win->child) return (*win->childcopy)(win->child,&(*V)->child);
  return 0;
}

