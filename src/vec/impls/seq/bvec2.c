/*$Id: bvec2.c,v 1.202 2001/09/12 03:26:24 bsmith Exp $*/
/*
   Implements the sequential vectors.
*/

#include "src/vec/vecimpl.h"          /*I "petscvec.h" I*/
#include "src/vec/impls/dvecimpl.h" 
#include "petscblaslapack.h"
#if defined(PETSC_HAVE_AMS)
EXTERN int PetscViewerAMSGetAMSComm(PetscViewer,AMS_Comm *);
#endif

#undef __FUNCT__  
#define __FUNCT__ "VecNorm_Seq"
int VecNorm_Seq(Vec xin,NormType type,PetscReal* z)
{
  PetscScalar *xa;
  int         ierr,one = 1;

  PetscFunctionBegin;
  if (type == NORM_2) {
    ierr = VecGetArrayFast(xin,&xa);CHKERRQ(ierr);
    /*
      This is because the Fortran BLAS 1 Norm is very slow! 
    */
#if defined(HAVE_SLOW_NRM2)
    {
      int         i;
      PetscScalar sum=0.0;
      for (i=0; i<xin->n; i++) {
        sum += (xa[i])*(PetscConj(xa[i]));
      }
      *z = sqrt(PetscRealPart(sum));
    }
#else
    *z = BLnrm2_(&xin->n,xa,&one);
#endif
    ierr = VecRestoreArrayFast(xin,&xa);CHKERRQ(ierr);
    PetscLogFlops(2*xin->n-1);
  } else if (type == NORM_INFINITY) {
    int          i,n = xin->n;
    PetscReal    max = 0.0,tmp;

    ierr = VecGetArrayFast(xin,&xa);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if ((tmp = PetscAbsScalar(*xa)) > max) max = tmp;
      /* check special case of tmp == NaN */
      if (tmp != tmp) {max = tmp; break;}
      xa++;
    }
    ierr = VecRestoreArrayFast(xin,&xa);CHKERRQ(ierr);
    *z   = max;
  } else if (type == NORM_1) {
    ierr = VecGetArrayFast(xin,&xa);CHKERRQ(ierr);
    *z = BLasum_(&xin->n,xa,&one);
    ierr = VecRestoreArrayFast(xin,&xa);CHKERRQ(ierr);
    PetscLogFlops(xin->n-1);
  } else if (type == NORM_1_AND_2) {
    ierr = VecNorm_Seq(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_Seq(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include "petscviewer.h"
#include "petscsys.h"

#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq_File"
int VecView_Seq_File(Vec xin,PetscViewer viewer)
{
  Vec_Seq           *x = (Vec_Seq *)xin->data;
  int               i,n = xin->n,ierr;
  char              *name;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_MATLAB) {
    ierr = PetscObjectGetName((PetscObject)xin,&name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%s = [\n",name);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
      if (PetscImaginaryPart(x->array[i]) > 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei\n",PetscRealPart(x->array[i]),PetscImaginaryPart(x->array[i]));CHKERRQ(ierr);
      } else if (PetscImaginaryPart(x->array[i]) < 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e - %18.16ei\n",PetscRealPart(x->array[i]),-PetscImaginaryPart(x->array[i]));CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",PetscRealPart(x->array[i]));CHKERRQ(ierr);
      }
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",x->array[i]);CHKERRQ(ierr);
#endif
    }
    ierr = PetscViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_SYMMODU) {
    for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e\n",PetscRealPart(x->array[i]),PetscImaginaryPart(x->array[i]));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",x->array[i]);CHKERRQ(ierr);
#endif
    }
  } else {
    for (i=0; i<n; i++) {
      if (format == PETSC_VIEWER_ASCII_INDEX) {
        ierr = PetscViewerASCIIPrintf(viewer,"%d: ",i);CHKERRQ(ierr);
      }
#if defined(PETSC_USE_COMPLEX)
      if (PetscImaginaryPart(x->array[i]) > 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%g + %g i\n",PetscRealPart(x->array[i]),PetscImaginaryPart(x->array[i]));CHKERRQ(ierr);
      } else if (PetscImaginaryPart(x->array[i]) < 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%g - %g i\n",PetscRealPart(x->array[i]),-PetscImaginaryPart(x->array[i]));CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"%g\n",PetscRealPart(x->array[i]));CHKERRQ(ierr);
      }
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%g\n",x->array[i]);CHKERRQ(ierr);
#endif
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq_Draw_LG"
static int VecView_Seq_Draw_LG(Vec xin,PetscViewer v)
{
  Vec_Seq     *x = (Vec_Seq *)xin->data;
  int         i,n = xin->n,ierr;
  PetscDraw   win;
  PetscReal   *xx;
  PetscDrawLG lg;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDrawLG(v,0,&lg);CHKERRQ(ierr);
  ierr = PetscDrawLGGetDraw(lg,&win);CHKERRQ(ierr);
  ierr = PetscDrawCheckResizedWindow(win);CHKERRQ(ierr);
  ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
  ierr = PetscMalloc((n+1)*sizeof(PetscReal),&xx);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    xx[i] = (PetscReal) i;
  }
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscDrawLGAddPoints(lg,n,&xx,&x->array);CHKERRQ(ierr);
#else 
  {
    PetscReal *yy;
    ierr = PetscMalloc((n+1)*sizeof(PetscReal),&yy);CHKERRQ(ierr);    
    for (i=0; i<n; i++) {
      yy[i] = PetscRealPart(x->array[i]);
    }
    ierr = PetscDrawLGAddPoints(lg,n,&xx,&yy);CHKERRQ(ierr);
    ierr = PetscFree(yy);CHKERRQ(ierr);
  }
#endif
  ierr = PetscFree(xx);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  ierr = PetscDrawSynchronizedFlush(win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq_Draw"
static int VecView_Seq_Draw(Vec xin,PetscViewer v)
{
  int               ierr;
  PetscDraw         draw;
  PetscTruth        isnull;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(v,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  
  ierr = PetscViewerGetFormat(v,&format);CHKERRQ(ierr);
  /*
     Currently it only supports drawing to a line graph */
  if (format != PETSC_VIEWER_DRAW_LG) {
    ierr = PetscViewerPushFormat(v,PETSC_VIEWER_DRAW_LG);CHKERRQ(ierr);
  } 
  ierr = VecView_Seq_Draw_LG(xin,v);CHKERRQ(ierr);
  if (format != PETSC_VIEWER_DRAW_LG) {
    ierr = PetscViewerPopFormat(v);CHKERRQ(ierr);
  } 

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq_Binary"
static int VecView_Seq_Binary(Vec xin,PetscViewer viewer)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  int      ierr,fdes,n = xin->n;
  FILE     *file;

  PetscFunctionBegin;
  ierr  = PetscViewerBinaryGetDescriptor(viewer,&fdes);CHKERRQ(ierr);
  /* Write vector header */
  ierr = PetscBinaryWrite(fdes,&xin->cookie,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(fdes,&n,1,PETSC_INT,0);CHKERRQ(ierr);

  /* Write vector contents */
  ierr = PetscBinaryWrite(fdes,x->array,n,PETSC_SCALAR,0);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
  if (file && xin->bs > 1) {
    fprintf(file,"-vecload_block_size %d\n",xin->bs);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq"
int VecView_Seq(Vec xin,PetscViewer viewer)
{
  Vec_Seq     *x = (Vec_Seq *)xin->data;
  int         ierr;
  PetscTruth  isdraw,isascii,issocket,isbinary;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (isdraw){ 
    ierr = VecView_Seq_Draw(xin,viewer);CHKERRQ(ierr);
  } else if (isascii){
    ierr = VecView_Seq_File(xin,viewer);CHKERRQ(ierr);
  } else if (issocket) {
    ierr = PetscViewerSocketPutScalar(viewer,xin->n,1,x->array);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = VecView_Seq_Binary(xin,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported by this vector object",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetValues_Seq"
int VecSetValues_Seq(Vec xin,int ni,const int ix[],const PetscScalar y[],InsertMode m)
{
  Vec_Seq     *x = (Vec_Seq *)xin->data;
  PetscScalar *xx = x->array;
  int         i;

  PetscFunctionBegin;
  if (m == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      if (ix[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
      if (ix[i] >= xin->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %d maximum %d",ix[i],xin->n);
#endif
      xx[ix[i]] = y[i];
    }
  } else {
    for (i=0; i<ni; i++) {
      if (ix[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
      if (ix[i] >= xin->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %d maximum %d",ix[i],xin->n);
#endif
      xx[ix[i]] += y[i];
    }  
  }  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetValuesBlocked_Seq"
int VecSetValuesBlocked_Seq(Vec xin,int ni,const int ix[],const PetscScalar yin[],InsertMode m)
{
  Vec_Seq     *x = (Vec_Seq *)xin->data;
  PetscScalar *xx = x->array,*y = (PetscScalar*)yin;
  int         i,bs = xin->bs,start,j;

  /*
       For optimization could treat bs = 2, 3, 4, 5 as special cases with loop unrolling
  */
  PetscFunctionBegin;
  if (m == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      start = bs*ix[i];
      if (start < 0) continue;
#if defined(PETSC_USE_BOPT_g)
      if (start >= xin->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %d maximum %d",start,xin->n);
#endif
      for (j=0; j<bs; j++) {
        xx[start+j] = y[j];
      }
      y += bs;
    }
  } else {
    for (i=0; i<ni; i++) {
      start = bs*ix[i];
      if (start < 0) continue;
#if defined(PETSC_USE_BOPT_g)
      if (start >= xin->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %d maximum %d",start,xin->n);
#endif
      for (j=0; j<bs; j++) {
        xx[start+j] += y[j];
      }
      y += bs;
    }  
  }  
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecDestroy_Seq"
int VecDestroy_Seq(Vec v)
{
  Vec_Seq *vs = (Vec_Seq*)v->data;
  int     ierr;

  PetscFunctionBegin;

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(v);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%d",v->n);
#endif
  if (vs->array_allocated) {ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr);}
  ierr = PetscFree(vs);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPublish_Seq"
static int VecPublish_Seq(PetscObject obj)
{
#if defined(PETSC_HAVE_AMS)
  Vec          v = (Vec) obj;
  Vec_Seq      *s = (Vec_Seq*)v->data;
  int          ierr,(*f)(AMS_Memory,char *,Vec);
#endif

  PetscFunctionBegin;

#if defined(PETSC_HAVE_AMS)
  /* if it is already published then return */
  if (v->amem >=0) PetscFunctionReturn(0);

  /* if array in vector was not allocated (for example PCSetUp_BJacobi_Singleblock()) then
     cannot AMS publish the object*/
  if (!s->array) PetscFunctionReturn(0);

  ierr = PetscObjectPublishBaseBegin(obj);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"values",s->array,v->n,AMS_DOUBLE,AMS_READ,
                                AMS_DISTRIBUTED,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

  /* if the vector knows its "layout" let it set it*/
  ierr = PetscObjectQueryFunction(obj,"AMSSetFieldBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)((AMS_Memory)v->amem,"values",v);CHKERRQ(ierr);
  }
  ierr = PetscObjectPublishBaseEnd(obj);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

static struct _VecOps DvOps = {VecDuplicate_Seq,
            VecDuplicateVecs_Default,
            VecDestroyVecs_Default,
            VecDot_Seq,
            VecMDot_Seq,
            VecNorm_Seq, 
            VecTDot_Seq,
            VecMTDot_Seq,
            VecScale_Seq,
            VecCopy_Seq,
            VecSet_Seq,
            VecSwap_Seq,
            VecAXPY_Seq,
            VecAXPBY_Seq,
            VecMAXPY_Seq,
            VecAYPX_Seq,
            VecWAXPY_Seq,
            VecPointwiseMult_Seq,
            VecPointwiseDivide_Seq, 
            VecSetValues_Seq,0,0,
            VecGetArray_Seq,
            VecGetSize_Seq,
            VecGetSize_Seq,
            VecRestoreArray_Seq,
            VecMax_Seq,
            VecMin_Seq,
            VecSetRandom_Seq,0,
            VecSetValuesBlocked_Seq,
            VecDestroy_Seq,
            VecView_Seq,
            VecPlaceArray_Seq,
            VecReplaceArray_Seq,
            VecDot_Seq,
            VecTDot_Seq,
            VecNorm_Seq,
            VecLoadIntoVector_Default,
            VecReciprocal_Default,
            0, /* VecViewNative */
            VecConjugate_Seq,
	    0,
	    0,
            VecResetArray_Seq};


/*
      This is called by VecCreate_Seq() (i.e. VecCreateSeq()) and VecCreateSeqWithArray()
*/
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_Seq_Private"
static int VecCreate_Seq_Private(Vec v,const PetscScalar array[])
{
  Vec_Seq *s;
  int     ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  ierr = PetscNew(Vec_Seq,&s);CHKERRQ(ierr);
  ierr = PetscMemzero(s,sizeof(Vec_Seq));CHKERRQ(ierr);
  v->data            = (void*)s;
  v->bops->publish   = VecPublish_Seq;
  v->n               = PetscMax(v->n,v->N); 
  v->N               = PetscMax(v->n,v->N); 
  v->bs              = -1;
  v->petscnative     = PETSC_TRUE;
  s->array           = (PetscScalar *)array;
  s->array_allocated = 0;
  if (!v->map) {
    ierr = PetscMapCreateMPI(v->comm,v->n,v->N,&v->map);CHKERRQ(ierr);
  }
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECSEQ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscMatlabEnginePut_C","VecMatlabEnginePut_Default",VecMatlabEnginePut_Default);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscMatlabEngineGet_C","VecMatlabEngineGet_Default",VecMatlabEngineGet_Default);CHKERRQ(ierr);
#endif
  ierr = PetscPublishAll(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCreateSeqWithArray"
/*@C
   VecCreateSeqWithArray - Creates a standard,sequential array-style vector,
   where the user provides the array space to store the vector values.

   Collective on MPI_Comm

   Input Parameter:
+  comm - the communicator, should be PETSC_COMM_SELF
.  n - the vector length 
-  array - memory where the vector elements are to be stored.

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is PETSC_NULL, then VecPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

   Concepts: vectors^creating with array

.seealso: VecCreateMPIWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), 
          VecCreateGhost(), VecCreateSeq(), VecPlaceArray()
@*/
int VecCreateSeqWithArray(MPI_Comm comm,int n,const PetscScalar array[],Vec *V)
{
  int  ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,V);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,n,n);CHKERRQ(ierr);
  ierr = VecCreate_Seq_Private(*V,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_Seq"
int VecCreate_Seq(Vec V)
{
  Vec_Seq      *s;
  PetscScalar  *array;
  int          ierr,n = PetscMax(V->n,V->N);

  PetscFunctionBegin;
  ierr = PetscMalloc((n+1)*sizeof(PetscScalar),&array);CHKERRQ(ierr);
  ierr = PetscMemzero(array,n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecCreate_Seq_Private(V,array);CHKERRQ(ierr);
  s    = (Vec_Seq*)V->data;
  s->array_allocated = array;
  ierr = VecSetSerializeType(V,VEC_SER_SEQ_BINARY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSerialize_Seq"
int VecSerialize_Seq(MPI_Comm comm, Vec *vec, PetscViewer viewer, PetscTruth store)
{
  Vec          v;
  Vec_Seq     *x;
  PetscScalar *array;
  int          fd;
  int          vars;
  int          ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryGetDescriptor(viewer, &fd);                                                     CHKERRQ(ierr);
  if (store) {
    v    = *vec;
    x    = (Vec_Seq *) v->data;
    ierr = PetscBinaryWrite(fd, &v->n,     1,    PETSC_INT,    0);                                        CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,  x->array, v->n, PETSC_SCALAR, 0);                                        CHKERRQ(ierr);
  } else {
    ierr = PetscBinaryRead(fd, &vars,  1,    PETSC_INT);                                                  CHKERRQ(ierr);
    ierr = VecCreate(comm, &v);                                                                           CHKERRQ(ierr);
    ierr = VecSetSizes(v, vars, vars);                                                                     CHKERRQ(ierr);
    ierr = PetscMalloc(vars * sizeof(PetscScalar), &array);                                               CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,  array, vars, PETSC_SCALAR);                                               CHKERRQ(ierr);
    ierr = VecCreate_Seq_Private(v, array);                                                               CHKERRQ(ierr);
    ((Vec_Seq *) v->data)->array_allocated = array;

    ierr = VecAssemblyBegin(v);                                                                           CHKERRQ(ierr);
    ierr = VecAssemblyEnd(v);                                                                             CHKERRQ(ierr);
    *vec = v;
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_Seq"
int VecDuplicate_Seq(Vec win,Vec *V)
{
  int     ierr;

  PetscFunctionBegin;
  ierr = VecCreateSeq(win->comm,win->n,V);CHKERRQ(ierr);
  if (win->mapping) {
    (*V)->mapping = win->mapping;
    ierr = PetscObjectReference((PetscObject)win->mapping);CHKERRQ(ierr);
  }
  if (win->bmapping) {
    (*V)->bmapping = win->bmapping;
    ierr = PetscObjectReference((PetscObject)win->bmapping);CHKERRQ(ierr);
  }
  (*V)->bs = win->bs;
  ierr = PetscOListDuplicate(win->olist,&(*V)->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(win->qlist,&(*V)->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

