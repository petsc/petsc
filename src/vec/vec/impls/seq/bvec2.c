
/*
   Implements the sequential vectors.
*/

#include <../src/vec/vec/impls/dvecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>      /* For VecView_MPI_HDF5 */
#include <petsc/private/glvisviewerimpl.h>
#include <petsc/private/glvisvecimpl.h>
#include <petscblaslapack.h>

#if defined(PETSC_HAVE_HDF5)
extern PetscErrorCode VecView_MPI_HDF5(Vec,PetscViewer);
#endif

PetscErrorCode VecPointwiseMax_Seq(Vec win,Vec xin,Vec yin)
{
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,(const PetscScalar**)&xx));
  PetscCall(VecGetArrayRead(yin,(const PetscScalar**)&yy));
  PetscCall(VecGetArray(win,&ww));

  for (i=0; i<n; i++) ww[i] = PetscMax(PetscRealPart(xx[i]),PetscRealPart(yy[i]));

  PetscCall(VecRestoreArrayRead(xin,(const PetscScalar**)&xx));
  PetscCall(VecRestoreArrayRead(yin,(const PetscScalar**)&yy));
  PetscCall(VecRestoreArray(win,&ww));
  PetscCall(PetscLogFlops(n));
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMin_Seq(Vec win,Vec xin,Vec yin)
{
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,(const PetscScalar**)&xx));
  PetscCall(VecGetArrayRead(yin,(const PetscScalar**)&yy));
  PetscCall(VecGetArray(win,&ww));

  for (i=0; i<n; i++) ww[i] = PetscMin(PetscRealPart(xx[i]),PetscRealPart(yy[i]));

  PetscCall(VecRestoreArrayRead(xin,(const PetscScalar**)&xx));
  PetscCall(VecRestoreArrayRead(yin,(const PetscScalar**)&yy));
  PetscCall(VecRestoreArray(win,&ww));
  PetscCall(PetscLogFlops(n));
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMaxAbs_Seq(Vec win,Vec xin,Vec yin)
{
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,(const PetscScalar**)&xx));
  PetscCall(VecGetArrayRead(yin,(const PetscScalar**)&yy));
  PetscCall(VecGetArray(win,&ww));

  for (i=0; i<n; i++) ww[i] = PetscMax(PetscAbsScalar(xx[i]),PetscAbsScalar(yy[i]));

  PetscCall(PetscLogFlops(n));
  PetscCall(VecRestoreArrayRead(xin,(const PetscScalar**)&xx));
  PetscCall(VecRestoreArrayRead(yin,(const PetscScalar**)&yy));
  PetscCall(VecRestoreArray(win,&ww));
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fxtimesy.h>

PetscErrorCode VecPointwiseMult_Seq(Vec win,Vec xin,Vec yin)
{
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,(const PetscScalar**)&xx));
  PetscCall(VecGetArrayRead(yin,(const PetscScalar**)&yy));
  PetscCall(VecGetArray(win,&ww));
  if (ww == xx) {
    for (i=0; i<n; i++) ww[i] *= yy[i];
  } else if (ww == yy) {
    for (i=0; i<n; i++) ww[i] *= xx[i];
  } else {
#if defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
    fortranxtimesy_(xx,yy,ww,&n);
#else
    for (i=0; i<n; i++) ww[i] = xx[i] * yy[i];
#endif
  }
  PetscCall(VecRestoreArrayRead(xin,(const PetscScalar**)&xx));
  PetscCall(VecRestoreArrayRead(yin,(const PetscScalar**)&yy));
  PetscCall(VecRestoreArray(win,&ww));
  PetscCall(PetscLogFlops(n));
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseDivide_Seq(Vec win,Vec xin,Vec yin)
{
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,(const PetscScalar**)&xx));
  PetscCall(VecGetArrayRead(yin,(const PetscScalar**)&yy));
  PetscCall(VecGetArray(win,&ww));

  for (i=0; i<n; i++) {
    if (yy[i] != 0.0) ww[i] = xx[i] / yy[i];
    else ww[i] = 0.0;
  }

  PetscCall(PetscLogFlops(n));
  PetscCall(VecRestoreArrayRead(xin,(const PetscScalar**)&xx));
  PetscCall(VecRestoreArrayRead(yin,(const PetscScalar**)&yy));
  PetscCall(VecRestoreArray(win,&ww));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetRandom_Seq(Vec xin,PetscRandom r)
{
  PetscInt       n = xin->map->n,i;
  PetscScalar    *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayWrite(xin,&xx));
  for (i=0; i<n; i++) PetscCall(PetscRandomGetValue(r,&xx[i]));
  PetscCall(VecRestoreArrayWrite(xin,&xx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetSize_Seq(Vec vin,PetscInt *size)
{
  PetscFunctionBegin;
  *size = vin->map->n;
  PetscFunctionReturn(0);
}

PetscErrorCode VecConjugate_Seq(Vec xin)
{
  PetscScalar    *x;
  PetscInt       n = xin->map->n;

  PetscFunctionBegin;
  PetscCall(VecGetArray(xin,&x));
  while (n-->0) {
    *x = PetscConj(*x);
    x++;
  }
  PetscCall(VecRestoreArray(xin,&x));
  PetscFunctionReturn(0);
}

PetscErrorCode VecResetArray_Seq(Vec vin)
{
  Vec_Seq *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_Seq(Vec xin,Vec yin)
{
  PetscScalar       *ya;
  const PetscScalar *xa;

  PetscFunctionBegin;
  if (xin != yin) {
    PetscCall(VecGetArrayRead(xin,&xa));
    PetscCall(VecGetArray(yin,&ya));
    PetscCall(PetscArraycpy(ya,xa,xin->map->n));
    PetscCall(VecRestoreArrayRead(xin,&xa));
    PetscCall(VecRestoreArray(yin,&ya));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSwap_Seq(Vec xin,Vec yin)
{
  PetscScalar    *ya, *xa;
  PetscBLASInt   one = 1,bn;

  PetscFunctionBegin;
  if (xin != yin) {
    PetscCall(PetscBLASIntCast(xin->map->n,&bn));
    PetscCall(VecGetArray(xin,&xa));
    PetscCall(VecGetArray(yin,&ya));
    PetscStackCallBLAS("BLASswap",BLASswap_(&bn,xa,&one,ya,&one));
    PetscCall(VecRestoreArray(xin,&xa));
    PetscCall(VecRestoreArray(yin,&ya));
  }
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fnorm.h>

PetscErrorCode VecNorm_Seq(Vec xin,NormType type,PetscReal *z)
{
  const PetscScalar *xx;
  PetscInt          n = xin->map->n;
  PetscBLASInt      one = 1, bn = 0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(n,&bn));
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    PetscCall(VecGetArrayRead(xin,&xx));
#if defined(PETSC_USE_REAL___FP16)
    PetscStackCallBLAS("BLASnrm2",*z = BLASnrm2_(&bn,xx,&one));
#else
    PetscStackCallBLAS("BLASdot",*z   = PetscRealPart(BLASdot_(&bn,xx,&one,xx,&one)));
    *z   = PetscSqrtReal(*z);
#endif
    PetscCall(VecRestoreArrayRead(xin,&xx));
    PetscCall(PetscLogFlops(PetscMax(2.0*n-1,0.0)));
  } else if (type == NORM_INFINITY) {
    PetscInt  i;
    PetscReal max = 0.0,tmp;

    PetscCall(VecGetArrayRead(xin,&xx));
    for (i=0; i<n; i++) {
      if ((tmp = PetscAbsScalar(*xx)) > max) max = tmp;
      /* check special case of tmp == NaN */
      if (tmp != tmp) {max = tmp; break;}
      xx++;
    }
    PetscCall(VecRestoreArrayRead(xin,&xx));
    *z   = max;
  } else if (type == NORM_1) {
#if defined(PETSC_USE_COMPLEX)
    PetscReal tmp = 0.0;
    PetscInt    i;
#endif
    PetscCall(VecGetArrayRead(xin,&xx));
#if defined(PETSC_USE_COMPLEX)
    /* BLASasum() returns the nonstandard 1 norm of the 1 norm of the complex entries so we provide a custom loop instead */
    for (i=0; i<n; i++) {
      tmp += PetscAbsScalar(xx[i]);
    }
    *z = tmp;
#else
    PetscStackCallBLAS("BLASasum",*z   = BLASasum_(&bn,xx,&one));
#endif
    PetscCall(VecRestoreArrayRead(xin,&xx));
    PetscCall(PetscLogFlops(PetscMax(n-1.0,0.0)));
  } else if (type == NORM_1_AND_2) {
    PetscCall(VecNorm_Seq(xin,NORM_1,z));
    PetscCall(VecNorm_Seq(xin,NORM_2,z+1));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Seq_ASCII(Vec xin,PetscViewer viewer)
{
  PetscInt          i,n = xin->map->n;
  const char        *name;
  PetscViewerFormat format;
  const PetscScalar *xv;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,&xv));
  PetscCall(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_ASCII_MATLAB) {
    PetscCall(PetscObjectGetName((PetscObject)xin,&name));
    PetscCall(PetscViewerASCIIPrintf(viewer,"%s = [\n",name));
    for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
      if (PetscImaginaryPart(xv[i]) > 0.0) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei\n",(double)PetscRealPart(xv[i]),(double)PetscImaginaryPart(xv[i])));
      } else if (PetscImaginaryPart(xv[i]) < 0.0) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e - %18.16ei\n",(double)PetscRealPart(xv[i]),-(double)PetscImaginaryPart(xv[i])));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)PetscRealPart(xv[i])));
      }
#else
      PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double) xv[i]));
#endif
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"];\n"));
  } else if (format == PETSC_VIEWER_ASCII_SYMMODU) {
    for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e\n",(double)PetscRealPart(xv[i]),(double)PetscImaginaryPart(xv[i])));
#else
      PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)xv[i]));
#endif
    }
  } else if (format == PETSC_VIEWER_ASCII_VTK_DEPRECATED || format == PETSC_VIEWER_ASCII_VTK_CELL_DEPRECATED) {
    /*
       state 0: No header has been output
       state 1: Only POINT_DATA has been output
       state 2: Only CELL_DATA has been output
       state 3: Output both, POINT_DATA last
       state 4: Output both, CELL_DATA last
    */
    static PetscInt stateId = -1;
    int outputState = 0;
    PetscBool  hasState;
    int doOutput = 0;
    PetscInt bs, b;

    if (stateId < 0) {
      PetscCall(PetscObjectComposedDataRegister(&stateId));
    }
    PetscCall(PetscObjectComposedDataGetInt((PetscObject) viewer, stateId, outputState, hasState));
    if (!hasState) outputState = 0;
    PetscCall(PetscObjectGetName((PetscObject) xin, &name));
    PetscCall(VecGetBlockSize(xin, &bs));
    PetscCheck(bs >= 1 && bs <= 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %" PetscInt_FMT, bs);
    if (format == PETSC_VIEWER_ASCII_VTK_DEPRECATED) {
      if (outputState == 0) {
        outputState = 1;
        doOutput = 1;
      } else if (outputState == 1) doOutput = 0;
      else if (outputState == 2) {
        outputState = 3;
        doOutput = 1;
      } else if (outputState == 3) doOutput = 0;
      else PetscCheck(outputState != 4,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output POINT_DATA again after intervening CELL_DATA");

      if (doOutput) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "POINT_DATA %" PetscInt_FMT "\n", n/bs));
      }
    } else {
      if (outputState == 0) {
        outputState = 2;
        doOutput = 1;
      } else if (outputState == 1) {
        outputState = 4;
        doOutput = 1;
      } else if (outputState == 2) doOutput = 0;
      else PetscCheck(outputState != 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output CELL_DATA again after intervening POINT_DATA");
      else if (outputState == 4) doOutput = 0;

      if (doOutput) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "CELL_DATA %" PetscInt_FMT "\n", n));
      }
    }
    PetscCall(PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, outputState));
    if (name) {
      if (bs == 3) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "VECTORS %s double\n", name));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "SCALARS %s double %" PetscInt_FMT "\n", name, bs));
      }
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "SCALARS scalars double %" PetscInt_FMT "\n", bs));
    }
    if (bs != 3) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n"));
    }
    for (i=0; i<n/bs; i++) {
      for (b=0; b<bs; b++) {
        if (b > 0) {
          PetscCall(PetscViewerASCIIPrintf(viewer," "));
        }
#if !defined(PETSC_USE_COMPLEX)
        PetscCall(PetscViewerASCIIPrintf(viewer,"%g",(double)xv[i*bs+b]));
#endif
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
    }
  } else if (format == PETSC_VIEWER_ASCII_VTK_COORDS_DEPRECATED) {
    PetscInt bs, b;

    PetscCall(VecGetBlockSize(xin, &bs));
    PetscCheck(bs >= 1 && bs <= 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %" PetscInt_FMT, bs);
    for (i=0; i<n/bs; i++) {
      for (b=0; b<bs; b++) {
        if (b > 0) {
          PetscCall(PetscViewerASCIIPrintf(viewer," "));
        }
#if !defined(PETSC_USE_COMPLEX)
        PetscCall(PetscViewerASCIIPrintf(viewer,"%g",(double)xv[i*bs+b]));
#endif
      }
      for (b=bs; b<3; b++) {
        PetscCall(PetscViewerASCIIPrintf(viewer," 0.0"));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
    }
  } else if (format == PETSC_VIEWER_ASCII_PCICE) {
    PetscInt bs, b;

    PetscCall(VecGetBlockSize(xin, &bs));
    PetscCheck(bs >= 1 && bs <= 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "PCICE can only handle up to 3D objects, but vector dimension is %" PetscInt_FMT, bs);
    PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT "\n", xin->map->N/bs));
    for (i=0; i<n/bs; i++) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"%7" PetscInt_FMT "   ", i+1));
      for (b=0; b<bs; b++) {
        if (b > 0) {
          PetscCall(PetscViewerASCIIPrintf(viewer," "));
        }
#if !defined(PETSC_USE_COMPLEX)
        PetscCall(PetscViewerASCIIPrintf(viewer,"% 12.5E",(double)xv[i*bs+b]));
#endif
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
    }
  } else if (format == PETSC_VIEWER_ASCII_GLVIS) {
    /* GLVis ASCII visualization/dump: this function mimics mfem::GridFunction::Save() */
    const PetscScalar       *array;
    PetscInt                i,n,vdim, ordering = 1; /* mfem::FiniteElementSpace::Ordering::byVDIM */
    PetscContainer          glvis_container;
    PetscViewerGLVisVecInfo glvis_vec_info;
    PetscViewerGLVisInfo    glvis_info;

    /* mfem::FiniteElementSpace::Save() */
    PetscCall(VecGetBlockSize(xin,&vdim));
    PetscCall(PetscViewerASCIIPrintf(viewer,"FiniteElementSpace\n"));
    PetscCall(PetscObjectQuery((PetscObject)xin,"_glvis_info_container",(PetscObject*)&glvis_container));
    PetscCheck(glvis_container,PetscObjectComm((PetscObject)xin),PETSC_ERR_PLIB,"Missing GLVis container");
    PetscCall(PetscContainerGetPointer(glvis_container,(void**)&glvis_vec_info));
    PetscCall(PetscViewerASCIIPrintf(viewer,"%s\n",glvis_vec_info->fec_type));
    PetscCall(PetscViewerASCIIPrintf(viewer,"VDim: %" PetscInt_FMT "\n",vdim));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Ordering: %" PetscInt_FMT "\n",ordering));
    PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
    /* mfem::Vector::Print() */
    PetscCall(PetscObjectQuery((PetscObject)viewer,"_glvis_info_container",(PetscObject*)&glvis_container));
    PetscCheck(glvis_container,PetscObjectComm((PetscObject)viewer),PETSC_ERR_PLIB,"Missing GLVis container");
    PetscCall(PetscContainerGetPointer(glvis_container,(void**)&glvis_info));
    if (glvis_info->enabled) {
      PetscCall(VecGetLocalSize(xin,&n));
      PetscCall(VecGetArrayRead(xin,&array));
      for (i=0;i<n;i++) {
        PetscCall(PetscViewerASCIIPrintf(viewer,glvis_info->fmt,(double)PetscRealPart(array[i])));
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      }
      PetscCall(VecRestoreArrayRead(xin,&array));
    }
  } else if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    /* No info */
  } else {
    for (i=0; i<n; i++) {
      if (format == PETSC_VIEWER_ASCII_INDEX) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT ": ",i));
      }
#if defined(PETSC_USE_COMPLEX)
      if (PetscImaginaryPart(xv[i]) > 0.0) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"%g + %g i\n",(double)PetscRealPart(xv[i]),(double)PetscImaginaryPart(xv[i])));
      } else if (PetscImaginaryPart(xv[i]) < 0.0) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"%g - %g i\n",(double)PetscRealPart(xv[i]),-(double)PetscImaginaryPart(xv[i])));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"%g\n",(double)PetscRealPart(xv[i])));
      }
#else
      PetscCall(PetscViewerASCIIPrintf(viewer,"%g\n",(double)xv[i]));
#endif
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(VecRestoreArrayRead(xin,&xv));
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
PetscErrorCode VecView_Seq_Draw_LG(Vec xin,PetscViewer v)
{
  PetscDraw         draw;
  PetscBool         isnull;
  PetscDrawLG       lg;
  PetscInt          i,c,bs = PetscAbs(xin->map->bs),n = xin->map->n/bs;
  const PetscScalar *xv;
  PetscReal         *xx,*yy,xmin,xmax,h;
  int               colors[] = {PETSC_DRAW_RED};
  PetscViewerFormat format;
  PetscDrawAxis     axis;

  PetscFunctionBegin;
  PetscCall(PetscViewerDrawGetDraw(v,0,&draw));
  PetscCall(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);

  PetscCall(PetscViewerGetFormat(v,&format));
  PetscCall(PetscMalloc2(n,&xx,n,&yy));
  PetscCall(VecGetArrayRead(xin,&xv));
  for (c=0; c<bs; c++) {
    PetscCall(PetscViewerDrawGetDrawLG(v,c,&lg));
    PetscCall(PetscDrawLGReset(lg));
    PetscCall(PetscDrawLGSetDimension(lg,1));
    PetscCall(PetscDrawLGSetColors(lg,colors));
    if (format == PETSC_VIEWER_DRAW_LG_XRANGE) {
      PetscCall(PetscDrawLGGetAxis(lg,&axis));
      PetscCall(PetscDrawAxisGetLimits(axis,&xmin,&xmax,NULL,NULL));
      h = (xmax - xmin)/n;
      for (i=0; i<n; i++) xx[i] = i*h + 0.5*h; /* cell center */
    } else {
      for (i=0; i<n; i++) xx[i] = (PetscReal)i;
    }
    for (i=0; i<n; i++) yy[i] = PetscRealPart(xv[c + i*bs]);

    PetscCall(PetscDrawLGAddPoints(lg,n,&xx,&yy));
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }
  PetscCall(VecRestoreArrayRead(xin,&xv));
  PetscCall(PetscFree2(xx,yy));
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Seq_Draw(Vec xin,PetscViewer v)
{
  PetscDraw         draw;
  PetscBool         isnull;

  PetscFunctionBegin;
  PetscCall(PetscViewerDrawGetDraw(v,0,&draw));
  PetscCall(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);

  PetscCall(VecView_Seq_Draw_LG(xin,v));
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Seq_Binary(Vec xin,PetscViewer viewer)
{
  return VecView_Binary(xin,viewer);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <petscmatlab.h>
#include <mat.h>   /* MATLAB include file */
PetscErrorCode VecView_Seq_Matlab(Vec vec,PetscViewer viewer)
{
  PetscInt          n;
  const PetscScalar *array;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(vec,&n));
  PetscCall(PetscObjectName((PetscObject)vec));
  PetscCall(VecGetArrayRead(vec,&array));
  PetscCall(PetscViewerMatlabPutArray(viewer,n,1,array,((PetscObject)vec)->name));
  PetscCall(VecRestoreArrayRead(vec,&array));
  PetscFunctionReturn(0);
}
#endif

PETSC_EXTERN PetscErrorCode VecView_Seq(Vec xin,PetscViewer viewer)
{
  PetscBool      isdraw,iascii,issocket,isbinary;
#if defined(PETSC_HAVE_MATHEMATICA)
  PetscBool      ismathematica;
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscBool      ismatlab;
#endif
#if defined(PETSC_HAVE_HDF5)
  PetscBool      ishdf5;
#endif
  PetscBool      isglvis;
#if defined(PETSC_HAVE_ADIOS)
  PetscBool      isadios;
#endif

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSOCKET,&issocket));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
#if defined(PETSC_HAVE_MATHEMATICA)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATHEMATICA,&ismathematica));
#endif
#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATLAB,&ismatlab));
#endif
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&isglvis));
#if defined(PETSC_HAVE_ADIOS)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS,&isadios));
#endif

  if (isdraw) {
    PetscCall(VecView_Seq_Draw(xin,viewer));
  } else if (iascii) {
    PetscCall(VecView_Seq_ASCII(xin,viewer));
  } else if (isbinary) {
    PetscCall(VecView_Seq_Binary(xin,viewer));
#if defined(PETSC_HAVE_MATHEMATICA)
  } else if (ismathematica) {
    PetscCall(PetscViewerMathematicaPutVector(viewer,xin));
#endif
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    PetscCall(VecView_MPI_HDF5(xin,viewer)); /* Reusing VecView_MPI_HDF5 ... don't want code duplication*/
#endif
#if defined(PETSC_HAVE_ADIOS)
  } else if (isadios) {
    PetscCall(VecView_MPI_ADIOS(xin,viewer)); /* Reusing VecView_MPI_ADIOS ... don't want code duplication*/
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  } else if (ismatlab) {
    PetscCall(VecView_Seq_Matlab(xin,viewer));
#endif
  } else if (isglvis) {
    PetscCall(VecView_GLVis(xin,viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetValues_Seq(Vec xin,PetscInt ni,const PetscInt ix[],PetscScalar y[])
{
  const PetscScalar *xx;
  PetscInt          i;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,&xx));
  for (i=0; i<ni; i++) {
    if (xin->stash.ignorenegidx && ix[i] < 0) continue;
    if (PetscDefined(USE_DEBUG)) {
      PetscCheck(ix[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " cannot be negative",ix[i]);
      PetscCheck(ix[i] < xin->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " to large maximum allowed %" PetscInt_FMT,ix[i],xin->map->n);
    }
    y[i] = xx[ix[i]];
  }
  PetscCall(VecRestoreArrayRead(xin,&xx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetValues_Seq(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode m)
{
  PetscScalar    *xx;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCall(VecGetArray(xin,&xx));
  if (m == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      if (xin->stash.ignorenegidx && ix[i] < 0) continue;
      if (PetscDefined(USE_DEBUG)) {
        PetscCheck(ix[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " cannot be negative",ix[i]);
        PetscCheck(ix[i] < xin->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " maximum %" PetscInt_FMT,ix[i],xin->map->n);
      }
      xx[ix[i]] = y[i];
    }
  } else {
    for (i=0; i<ni; i++) {
      if (xin->stash.ignorenegidx && ix[i] < 0) continue;
      if (PetscDefined(USE_DEBUG)) {
        PetscCheck(ix[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " cannot be negative",ix[i]);
        PetscCheck(ix[i] < xin->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " maximum %" PetscInt_FMT,ix[i],xin->map->n);
      }
      xx[ix[i]] += y[i];
    }
  }
  PetscCall(VecRestoreArray(xin,&xx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetValuesBlocked_Seq(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar yin[],InsertMode m)
{
  PetscScalar    *xx,*y = (PetscScalar*)yin;
  PetscInt       i,bs,start,j;

  /*
       For optimization could treat bs = 2, 3, 4, 5 as special cases with loop unrolling
  */
  PetscFunctionBegin;
  PetscCall(VecGetBlockSize(xin,&bs));
  PetscCall(VecGetArray(xin,&xx));
  if (m == INSERT_VALUES) {
    for (i=0; i<ni; i++, y+=bs) {
      start = bs*ix[i];
      if (start < 0) continue;
      PetscCheck(start < xin->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " maximum %" PetscInt_FMT,start,xin->map->n);
      for (j=0; j<bs; j++) xx[start+j] = y[j];
    }
  } else {
    for (i=0; i<ni; i++, y+=bs) {
      start = bs*ix[i];
      if (start < 0) continue;
      PetscCheck(start < xin->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " maximum %" PetscInt_FMT,start,xin->map->n);
      for (j=0; j<bs; j++) xx[start+j] += y[j];
    }
  }
  PetscCall(VecRestoreArray(xin,&xx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_Seq(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%" PetscInt_FMT,v->map->n);
#endif
  if (vs) PetscCall(PetscFree(vs->array_allocated));
  PetscCall(PetscFree(v->data));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetOption_Seq(Vec v,VecOption op,PetscBool flag)
{
  PetscFunctionBegin;
  if (op == VEC_IGNORE_NEGATIVE_INDICES) v->stash.ignorenegidx = flag;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDuplicate_Seq(Vec win,Vec *V)
{
  PetscFunctionBegin;
  PetscCall(VecCreate(PetscObjectComm((PetscObject)win),V));
  PetscCall(VecSetSizes(*V,win->map->n,win->map->n));
  PetscCall(VecSetType(*V,((PetscObject)win)->type_name));
  PetscCall(PetscLayoutReference(win->map,&(*V)->map));
  PetscCall(PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist));
  PetscCall(PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist));

  (*V)->ops->view          = win->ops->view;
  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;
  PetscFunctionReturn(0);
}

static struct _VecOps DvOps = {
  PetscDesignatedInitializer(duplicate,VecDuplicate_Seq), /* 1 */
  PetscDesignatedInitializer(duplicatevecs,VecDuplicateVecs_Default),
  PetscDesignatedInitializer(destroyvecs,VecDestroyVecs_Default),
  PetscDesignatedInitializer(dot,VecDot_Seq),
  PetscDesignatedInitializer(mdot,VecMDot_Seq),
  PetscDesignatedInitializer(norm,VecNorm_Seq),
  PetscDesignatedInitializer(tdot,VecTDot_Seq),
  PetscDesignatedInitializer(mtdot,VecMTDot_Seq),
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
  PetscDesignatedInitializer(setvalues,VecSetValues_Seq), /* 20 */
  PetscDesignatedInitializer(assemblybegin,NULL),
  PetscDesignatedInitializer(assemblyend,NULL),
  PetscDesignatedInitializer(getarray,NULL),
  PetscDesignatedInitializer(getsize,VecGetSize_Seq),
  PetscDesignatedInitializer(getlocalsize,VecGetSize_Seq),
  PetscDesignatedInitializer(restorearray,NULL),
  PetscDesignatedInitializer(max,VecMax_Seq),
  PetscDesignatedInitializer(min,VecMin_Seq),
  PetscDesignatedInitializer(setrandom,VecSetRandom_Seq),
  PetscDesignatedInitializer(setoption,VecSetOption_Seq), /* 30 */
  PetscDesignatedInitializer(setvaluesblocked,VecSetValuesBlocked_Seq),
  PetscDesignatedInitializer(destroy,VecDestroy_Seq),
  PetscDesignatedInitializer(view,VecView_Seq),
  PetscDesignatedInitializer(placearray,VecPlaceArray_Seq),
  PetscDesignatedInitializer(replacearray,VecReplaceArray_Seq),
  PetscDesignatedInitializer(dot_local,VecDot_Seq),
  PetscDesignatedInitializer(tdot_local,VecTDot_Seq),
  PetscDesignatedInitializer(norm_local,VecNorm_Seq),
  PetscDesignatedInitializer(mdot_local,VecMDot_Seq),
  PetscDesignatedInitializer(mtdot_local,VecMTDot_Seq), /* 40 */
  PetscDesignatedInitializer(load,VecLoad_Default),
  PetscDesignatedInitializer(reciprocal,VecReciprocal_Default),
  PetscDesignatedInitializer(conjugate,VecConjugate_Seq),
  PetscDesignatedInitializer(setlocaltoglobalmapping,NULL),
  PetscDesignatedInitializer(setvalueslocal,NULL),
  PetscDesignatedInitializer(resetarray,VecResetArray_Seq),
  PetscDesignatedInitializer(setfromoptions,NULL),
  PetscDesignatedInitializer(maxpointwisedivide,VecMaxPointwiseDivide_Seq),
  PetscDesignatedInitializer(pointwisemax,VecPointwiseMax_Seq),
  PetscDesignatedInitializer(pointwisemaxabs,VecPointwiseMaxAbs_Seq),
  PetscDesignatedInitializer(pointwisemin,VecPointwiseMin_Seq),
  PetscDesignatedInitializer(getvalues,VecGetValues_Seq),
  PetscDesignatedInitializer(sqrt,NULL),
  PetscDesignatedInitializer(abs,NULL),
  PetscDesignatedInitializer(exp,NULL),
  PetscDesignatedInitializer(log,NULL),
  PetscDesignatedInitializer(shift,NULL),
  PetscDesignatedInitializer(create,NULL),
  PetscDesignatedInitializer(stridegather,VecStrideGather_Default),
  PetscDesignatedInitializer(stridescatter,VecStrideScatter_Default),
  PetscDesignatedInitializer(dotnorm2,NULL),
  PetscDesignatedInitializer(getsubvector,NULL),
  PetscDesignatedInitializer(restoresubvector,NULL),
  PetscDesignatedInitializer(getarrayread,NULL),
  PetscDesignatedInitializer(restorearrayread,NULL),
  PetscDesignatedInitializer(stridesubsetgather,VecStrideSubSetGather_Default),
  PetscDesignatedInitializer(stridesubsetscatter,VecStrideSubSetScatter_Default),
  PetscDesignatedInitializer(viewnative,VecView_Seq),
  PetscDesignatedInitializer(loadnative,NULL),
  PetscDesignatedInitializer(getlocalvector,NULL),
};

/*
      This is called by VecCreate_Seq() (i.e. VecCreateSeq()) and VecCreateSeqWithArray()
*/
PetscErrorCode VecCreate_Seq_Private(Vec v,const PetscScalar array[])
{
  Vec_Seq        *s;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(v,&s));
  PetscCall(PetscMemcpy(v->ops,&DvOps,sizeof(DvOps)));

  v->data            = (void*)s;
  v->petscnative     = PETSC_TRUE;
  s->array           = (PetscScalar*)array;
  s->array_allocated = NULL;
  if (array) v->offloadmask = PETSC_OFFLOAD_CPU;

  PetscCall(PetscLayoutSetUp(v->map));
  PetscCall(PetscObjectChangeTypeName((PetscObject)v,VECSEQ));
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscCall(PetscObjectComposeFunction((PetscObject)v,"PetscMatlabEnginePut_C",VecMatlabEnginePut_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)v,"PetscMatlabEngineGet_C",VecMatlabEngineGet_Default));
#endif
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqWithArray - Creates a standard,sequential array-style vector,
   where the user provides the array space to store the vector values.

   Collective

   Input Parameters:
+  comm - the communicator, should be PETSC_COMM_SELF
.  bs - the block size
.  n - the vector length
-  array - memory where the vector elements are to be stored.

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is NULL, then VecPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateMPIWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(),
          VecCreateGhost(), VecCreateSeq(), VecPlaceArray()
@*/
PetscErrorCode  VecCreateSeqWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar array[],Vec *V)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCall(VecCreate(comm,V));
  PetscCall(VecSetSizes(*V,n,n));
  PetscCall(VecSetBlockSize(*V,bs));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCheck(size <= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQ on more than one process");
  PetscCall(VecCreate_Seq_Private(*V,array));
  PetscFunctionReturn(0);
}
