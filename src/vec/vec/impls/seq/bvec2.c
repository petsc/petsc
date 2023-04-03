
/*
   Implements the sequential vectors.
*/

#include <../src/vec/vec/impls/dvecimpl.h>     /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h> /* For VecView_MPI_HDF5 */
#include <petsc/private/glvisviewerimpl.h>
#include <petsc/private/glvisvecimpl.h>
#include <petscblaslapack.h>

#if defined(PETSC_HAVE_HDF5)
extern PetscErrorCode VecView_MPI_HDF5(Vec, PetscViewer);
#endif

static PetscErrorCode VecPointwiseApply_Seq(Vec win, Vec xin, Vec yin, PetscScalar (*const func)(PetscScalar, PetscScalar))
{
  const PetscInt n = win->map->n;
  PetscScalar   *ww, *xx, *yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin, (const PetscScalar **)&xx));
  PetscCall(VecGetArrayRead(yin, (const PetscScalar **)&yy));
  PetscCall(VecGetArray(win, &ww));
  for (PetscInt i = 0; i < n; ++i) ww[i] = func(xx[i], yy[i]);
  PetscCall(VecRestoreArrayRead(xin, (const PetscScalar **)&xx));
  PetscCall(VecRestoreArrayRead(yin, (const PetscScalar **)&yy));
  PetscCall(VecRestoreArray(win, &ww));
  PetscCall(PetscLogFlops(n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscScalar MaxRealPart(PetscScalar x, PetscScalar y)
{
  // use temporaries to avoid reevaluating side-effects
  const PetscReal rx = PetscRealPart(x), ry = PetscRealPart(y);

  return PetscMax(rx, ry);
}

PetscErrorCode VecPointwiseMax_Seq(Vec win, Vec xin, Vec yin)
{
  PetscFunctionBegin;
  PetscCall(VecPointwiseApply_Seq(win, xin, yin, MaxRealPart));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscScalar MinRealPart(PetscScalar x, PetscScalar y)
{
  // use temporaries to avoid reevaluating side-effects
  const PetscReal rx = PetscRealPart(x), ry = PetscRealPart(y);

  return PetscMin(rx, ry);
}

PetscErrorCode VecPointwiseMin_Seq(Vec win, Vec xin, Vec yin)
{
  PetscFunctionBegin;
  PetscCall(VecPointwiseApply_Seq(win, xin, yin, MinRealPart));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscScalar MaxAbs(PetscScalar x, PetscScalar y)
{
  return (PetscScalar)PetscMax(PetscAbsScalar(x), PetscAbsScalar(y));
}

PetscErrorCode VecPointwiseMaxAbs_Seq(Vec win, Vec xin, Vec yin)
{
  PetscFunctionBegin;
  PetscCall(VecPointwiseApply_Seq(win, xin, yin, MaxAbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fxtimesy.h>

PetscErrorCode VecPointwiseMult_Seq(Vec win, Vec xin, Vec yin)
{
  PetscInt     n = win->map->n, i;
  PetscScalar *ww, *xx, *yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin, (const PetscScalar **)&xx));
  PetscCall(VecGetArrayRead(yin, (const PetscScalar **)&yy));
  PetscCall(VecGetArray(win, &ww));
  if (ww == xx) {
    for (i = 0; i < n; i++) ww[i] *= yy[i];
  } else if (ww == yy) {
    for (i = 0; i < n; i++) ww[i] *= xx[i];
  } else {
#if defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
    fortranxtimesy_(xx, yy, ww, &n);
#else
    for (i = 0; i < n; i++) ww[i] = xx[i] * yy[i];
#endif
  }
  PetscCall(VecRestoreArrayRead(xin, (const PetscScalar **)&xx));
  PetscCall(VecRestoreArrayRead(yin, (const PetscScalar **)&yy));
  PetscCall(VecRestoreArray(win, &ww));
  PetscCall(PetscLogFlops(n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscScalar ScalDiv(PetscScalar x, PetscScalar y)
{
  return y == 0.0 ? 0.0 : x / y;
}

PetscErrorCode VecPointwiseDivide_Seq(Vec win, Vec xin, Vec yin)
{
  PetscFunctionBegin;
  PetscCall(VecPointwiseApply_Seq(win, xin, yin, ScalDiv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSetRandom_Seq(Vec xin, PetscRandom r)
{
  PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayWrite(xin, &xx));
  PetscCall(PetscRandomGetValues(r, xin->map->n, xx));
  PetscCall(VecRestoreArrayWrite(xin, &xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecGetSize_Seq(Vec vin, PetscInt *size)
{
  PetscFunctionBegin;
  *size = vin->map->n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecConjugate_Seq(Vec xin)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    const PetscInt n = xin->map->n;
    PetscScalar   *x;

    PetscCall(VecGetArray(xin, &x));
    for (PetscInt i = 0; i < n; ++i) x[i] = PetscConj(x[i]);
    PetscCall(VecRestoreArray(xin, &x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecResetArray_Seq(Vec vin)
{
  Vec_Seq *v = (Vec_Seq *)vin->data;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecCopy_Seq(Vec xin, Vec yin)
{
  PetscFunctionBegin;
  if (xin != yin) {
    const PetscScalar *xa;
    PetscScalar       *ya;

    PetscCall(VecGetArrayRead(xin, &xa));
    PetscCall(VecGetArray(yin, &ya));
    PetscCall(PetscArraycpy(ya, xa, xin->map->n));
    PetscCall(VecRestoreArrayRead(xin, &xa));
    PetscCall(VecRestoreArray(yin, &ya));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSwap_Seq(Vec xin, Vec yin)
{
  PetscFunctionBegin;
  if (xin != yin) {
    const PetscBLASInt one = 1;
    PetscScalar       *ya, *xa;
    PetscBLASInt       bn;

    PetscCall(PetscBLASIntCast(xin->map->n, &bn));
    PetscCall(VecGetArray(xin, &xa));
    PetscCall(VecGetArray(yin, &ya));
    PetscCallBLAS("BLASswap", BLASswap_(&bn, xa, &one, ya, &one));
    PetscCall(VecRestoreArray(xin, &xa));
    PetscCall(VecRestoreArray(yin, &ya));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fnorm.h>

PetscErrorCode VecNorm_Seq(Vec xin, NormType type, PetscReal *z)
{
  // use a local variable to ensure compiler doesn't think z aliases any of the other arrays
  PetscReal      ztmp[] = {0.0, 0.0};
  const PetscInt n      = xin->map->n;

  PetscFunctionBegin;
  if (n) {
    const PetscScalar *xx;
    const PetscBLASInt one = 1;
    PetscBLASInt       bn  = 0;

    PetscCall(PetscBLASIntCast(n, &bn));
    PetscCall(VecGetArrayRead(xin, &xx));
    if (type == NORM_2 || type == NORM_FROBENIUS) {
    NORM_1_AND_2_DOING_NORM_2:
      if (PetscDefined(USE_REAL___FP16)) {
        PetscCallBLAS("BLASnrm2", ztmp[type == NORM_1_AND_2] = BLASnrm2_(&bn, xx, &one));
      } else {
        PetscCallBLAS("BLASdot", ztmp[type == NORM_1_AND_2] = PetscSqrtReal(PetscRealPart(BLASdot_(&bn, xx, &one, xx, &one))));
      }
      PetscCall(PetscLogFlops(2.0 * n - 1));
    } else if (type == NORM_INFINITY) {
      for (PetscInt i = 0; i < n; ++i) {
        const PetscReal tmp = PetscAbsScalar(xx[i]);

        /* check special case of tmp == NaN */
        if ((tmp > ztmp[0]) || (tmp != tmp)) {
          ztmp[0] = tmp;
          if (tmp != tmp) break;
        }
      }
    } else if (type == NORM_1 || type == NORM_1_AND_2) {
      if (PetscDefined(USE_COMPLEX)) {
        // BLASasum() returns the nonstandard 1 norm of the 1 norm of the complex entries so we
        // provide a custom loop instead
        for (PetscInt i = 0; i < n; ++i) ztmp[0] += PetscAbsScalar(xx[i]);
      } else {
        PetscCallBLAS("BLASasum", ztmp[0] = BLASasum_(&bn, xx, &one));
      }
      PetscCall(PetscLogFlops(n - 1.0));
      /* slight reshuffle so we can skip getting the array again (but still log the flops) if we
         do norm2 after this */
      if (type == NORM_1_AND_2) goto NORM_1_AND_2_DOING_NORM_2;
    }
    PetscCall(VecRestoreArrayRead(xin, &xx));
  }
  z[0] = ztmp[0];
  if (type == NORM_1_AND_2) z[1] = ztmp[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecView_Seq_ASCII(Vec xin, PetscViewer viewer)
{
  PetscInt           i, n = xin->map->n;
  const char        *name;
  PetscViewerFormat  format;
  const PetscScalar *xv;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin, &xv));
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_ASCII_MATLAB) {
    PetscCall(PetscObjectGetName((PetscObject)xin, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s = [\n", name));
    for (i = 0; i < n; i++) {
#if defined(PETSC_USE_COMPLEX)
      if (PetscImaginaryPart(xv[i]) > 0.0) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "%18.16e + %18.16ei\n", (double)PetscRealPart(xv[i]), (double)PetscImaginaryPart(xv[i])));
      } else if (PetscImaginaryPart(xv[i]) < 0.0) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "%18.16e - %18.16ei\n", (double)PetscRealPart(xv[i]), -(double)PetscImaginaryPart(xv[i])));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "%18.16e\n", (double)PetscRealPart(xv[i])));
      }
#else
      PetscCall(PetscViewerASCIIPrintf(viewer, "%18.16e\n", (double)xv[i]));
#endif
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "];\n"));
  } else if (format == PETSC_VIEWER_ASCII_SYMMODU) {
    for (i = 0; i < n; i++) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscViewerASCIIPrintf(viewer, "%18.16e %18.16e\n", (double)PetscRealPart(xv[i]), (double)PetscImaginaryPart(xv[i])));
#else
      PetscCall(PetscViewerASCIIPrintf(viewer, "%18.16e\n", (double)xv[i]));
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
    static PetscInt stateId     = -1;
    int             outputState = 0;
    PetscBool       hasState;
    int             doOutput = 0;
    PetscInt        bs, b;

    if (stateId < 0) PetscCall(PetscObjectComposedDataRegister(&stateId));
    PetscCall(PetscObjectComposedDataGetInt((PetscObject)viewer, stateId, outputState, hasState));
    if (!hasState) outputState = 0;
    PetscCall(PetscObjectGetName((PetscObject)xin, &name));
    PetscCall(VecGetBlockSize(xin, &bs));
    PetscCheck(bs >= 1 && bs <= 3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %" PetscInt_FMT, bs);
    if (format == PETSC_VIEWER_ASCII_VTK_DEPRECATED) {
      if (outputState == 0) {
        outputState = 1;
        doOutput    = 1;
      } else if (outputState == 1) doOutput = 0;
      else if (outputState == 2) {
        outputState = 3;
        doOutput    = 1;
      } else if (outputState == 3) doOutput = 0;
      else PetscCheck(outputState != 4, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Tried to output POINT_DATA again after intervening CELL_DATA");

      if (doOutput) PetscCall(PetscViewerASCIIPrintf(viewer, "POINT_DATA %" PetscInt_FMT "\n", n / bs));
    } else {
      if (outputState == 0) {
        outputState = 2;
        doOutput    = 1;
      } else if (outputState == 1) {
        outputState = 4;
        doOutput    = 1;
      } else if (outputState == 2) {
        doOutput = 0;
      } else {
        PetscCheck(outputState != 3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Tried to output CELL_DATA again after intervening POINT_DATA");
        if (outputState == 4) doOutput = 0;
      }

      if (doOutput) PetscCall(PetscViewerASCIIPrintf(viewer, "CELL_DATA %" PetscInt_FMT "\n", n));
    }
    PetscCall(PetscObjectComposedDataSetInt((PetscObject)viewer, stateId, outputState));
    if (name) {
      if (bs == 3) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "VECTORS %s double\n", name));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "SCALARS %s double %" PetscInt_FMT "\n", name, bs));
      }
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "SCALARS scalars double %" PetscInt_FMT "\n", bs));
    }
    if (bs != 3) PetscCall(PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n"));
    for (i = 0; i < n / bs; i++) {
      for (b = 0; b < bs; b++) {
        if (b > 0) PetscCall(PetscViewerASCIIPrintf(viewer, " "));
#if !defined(PETSC_USE_COMPLEX)
        PetscCall(PetscViewerASCIIPrintf(viewer, "%g", (double)xv[i * bs + b]));
#endif
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    }
  } else if (format == PETSC_VIEWER_ASCII_VTK_COORDS_DEPRECATED) {
    PetscInt bs, b;

    PetscCall(VecGetBlockSize(xin, &bs));
    PetscCheck(bs >= 1 && bs <= 3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %" PetscInt_FMT, bs);
    for (i = 0; i < n / bs; i++) {
      for (b = 0; b < bs; b++) {
        if (b > 0) PetscCall(PetscViewerASCIIPrintf(viewer, " "));
#if !defined(PETSC_USE_COMPLEX)
        PetscCall(PetscViewerASCIIPrintf(viewer, "%g", (double)xv[i * bs + b]));
#endif
      }
      for (b = bs; b < 3; b++) PetscCall(PetscViewerASCIIPrintf(viewer, " 0.0"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    }
  } else if (format == PETSC_VIEWER_ASCII_PCICE) {
    PetscInt bs, b;

    PetscCall(VecGetBlockSize(xin, &bs));
    PetscCheck(bs >= 1 && bs <= 3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PCICE can only handle up to 3D objects, but vector dimension is %" PetscInt_FMT, bs);
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "\n", xin->map->N / bs));
    for (i = 0; i < n / bs; i++) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "%7" PetscInt_FMT "   ", i + 1));
      for (b = 0; b < bs; b++) {
        if (b > 0) PetscCall(PetscViewerASCIIPrintf(viewer, " "));
#if !defined(PETSC_USE_COMPLEX)
        PetscCall(PetscViewerASCIIPrintf(viewer, "% 12.5E", (double)xv[i * bs + b]));
#endif
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    }
  } else if (format == PETSC_VIEWER_ASCII_GLVIS) {
    /* GLVis ASCII visualization/dump: this function mimics mfem::GridFunction::Save() */
    const PetscScalar      *array;
    PetscInt                i, n, vdim, ordering = 1; /* mfem::FiniteElementSpace::Ordering::byVDIM */
    PetscContainer          glvis_container;
    PetscViewerGLVisVecInfo glvis_vec_info;
    PetscViewerGLVisInfo    glvis_info;

    /* mfem::FiniteElementSpace::Save() */
    PetscCall(VecGetBlockSize(xin, &vdim));
    PetscCall(PetscViewerASCIIPrintf(viewer, "FiniteElementSpace\n"));
    PetscCall(PetscObjectQuery((PetscObject)xin, "_glvis_info_container", (PetscObject *)&glvis_container));
    PetscCheck(glvis_container, PetscObjectComm((PetscObject)xin), PETSC_ERR_PLIB, "Missing GLVis container");
    PetscCall(PetscContainerGetPointer(glvis_container, (void **)&glvis_vec_info));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s\n", glvis_vec_info->fec_type));
    PetscCall(PetscViewerASCIIPrintf(viewer, "VDim: %" PetscInt_FMT "\n", vdim));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Ordering: %" PetscInt_FMT "\n", ordering));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    /* mfem::Vector::Print() */
    PetscCall(PetscObjectQuery((PetscObject)viewer, "_glvis_info_container", (PetscObject *)&glvis_container));
    PetscCheck(glvis_container, PetscObjectComm((PetscObject)viewer), PETSC_ERR_PLIB, "Missing GLVis container");
    PetscCall(PetscContainerGetPointer(glvis_container, (void **)&glvis_info));
    if (glvis_info->enabled) {
      PetscCall(VecGetLocalSize(xin, &n));
      PetscCall(VecGetArrayRead(xin, &array));
      for (i = 0; i < n; i++) {
        PetscCall(PetscViewerASCIIPrintf(viewer, glvis_info->fmt, (double)PetscRealPart(array[i])));
        PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      }
      PetscCall(VecRestoreArrayRead(xin, &array));
    }
  } else if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    /* No info */
  } else {
    for (i = 0; i < n; i++) {
      if (format == PETSC_VIEWER_ASCII_INDEX) PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT ": ", i));
#if defined(PETSC_USE_COMPLEX)
      if (PetscImaginaryPart(xv[i]) > 0.0) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "%g + %g i\n", (double)PetscRealPart(xv[i]), (double)PetscImaginaryPart(xv[i])));
      } else if (PetscImaginaryPart(xv[i]) < 0.0) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "%g - %g i\n", (double)PetscRealPart(xv[i]), -(double)PetscImaginaryPart(xv[i])));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "%g\n", (double)PetscRealPart(xv[i])));
      }
#else
      PetscCall(PetscViewerASCIIPrintf(viewer, "%g\n", (double)xv[i]));
#endif
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(VecRestoreArrayRead(xin, &xv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscdraw.h>
PetscErrorCode VecView_Seq_Draw_LG(Vec xin, PetscViewer v)
{
  PetscDraw          draw;
  PetscBool          isnull;
  PetscDrawLG        lg;
  PetscInt           i, c, bs = PetscAbs(xin->map->bs), n = xin->map->n / bs;
  const PetscScalar *xv;
  PetscReal         *xx, *yy, xmin, xmax, h;
  int                colors[] = {PETSC_DRAW_RED};
  PetscViewerFormat  format;
  PetscDrawAxis      axis;

  PetscFunctionBegin;
  PetscCall(PetscViewerDrawGetDraw(v, 0, &draw));
  PetscCall(PetscDrawIsNull(draw, &isnull));
  if (isnull) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscViewerGetFormat(v, &format));
  PetscCall(PetscMalloc2(n, &xx, n, &yy));
  PetscCall(VecGetArrayRead(xin, &xv));
  for (c = 0; c < bs; c++) {
    PetscCall(PetscViewerDrawGetDrawLG(v, c, &lg));
    PetscCall(PetscDrawLGReset(lg));
    PetscCall(PetscDrawLGSetDimension(lg, 1));
    PetscCall(PetscDrawLGSetColors(lg, colors));
    if (format == PETSC_VIEWER_DRAW_LG_XRANGE) {
      PetscCall(PetscDrawLGGetAxis(lg, &axis));
      PetscCall(PetscDrawAxisGetLimits(axis, &xmin, &xmax, NULL, NULL));
      h = (xmax - xmin) / n;
      for (i = 0; i < n; i++) xx[i] = i * h + 0.5 * h; /* cell center */
    } else {
      for (i = 0; i < n; i++) xx[i] = (PetscReal)i;
    }
    for (i = 0; i < n; i++) yy[i] = PetscRealPart(xv[c + i * bs]);

    PetscCall(PetscDrawLGAddPoints(lg, n, &xx, &yy));
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }
  PetscCall(VecRestoreArrayRead(xin, &xv));
  PetscCall(PetscFree2(xx, yy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecView_Seq_Draw(Vec xin, PetscViewer v)
{
  PetscDraw draw;
  PetscBool isnull;

  PetscFunctionBegin;
  PetscCall(PetscViewerDrawGetDraw(v, 0, &draw));
  PetscCall(PetscDrawIsNull(draw, &isnull));
  if (isnull) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(VecView_Seq_Draw_LG(xin, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecView_Seq_Binary(Vec xin, PetscViewer viewer)
{
  return VecView_Binary(xin, viewer);
}

#if defined(PETSC_HAVE_MATLAB)
  #include <petscmatlab.h>
  #include <mat.h> /* MATLAB include file */
PetscErrorCode VecView_Seq_Matlab(Vec vec, PetscViewer viewer)
{
  PetscInt           n;
  const PetscScalar *array;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(vec, &n));
  PetscCall(PetscObjectName((PetscObject)vec));
  PetscCall(VecGetArrayRead(vec, &array));
  PetscCall(PetscViewerMatlabPutArray(viewer, n, 1, array, ((PetscObject)vec)->name));
  PetscCall(VecRestoreArrayRead(vec, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PETSC_EXTERN PetscErrorCode VecView_Seq(Vec xin, PetscViewer viewer)
{
  PetscBool isdraw, iascii, issocket, isbinary;
#if defined(PETSC_HAVE_MATHEMATICA)
  PetscBool ismathematica;
#endif
#if defined(PETSC_HAVE_MATLAB)
  PetscBool ismatlab;
#endif
#if defined(PETSC_HAVE_HDF5)
  PetscBool ishdf5;
#endif
  PetscBool isglvis;
#if defined(PETSC_HAVE_ADIOS)
  PetscBool isadios;
#endif

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSOCKET, &issocket));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
#if defined(PETSC_HAVE_MATHEMATICA)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERMATHEMATICA, &ismathematica));
#endif
#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5));
#endif
#if defined(PETSC_HAVE_MATLAB)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERMATLAB, &ismatlab));
#endif
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERGLVIS, &isglvis));
#if defined(PETSC_HAVE_ADIOS)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERADIOS, &isadios));
#endif

  if (isdraw) {
    PetscCall(VecView_Seq_Draw(xin, viewer));
  } else if (iascii) {
    PetscCall(VecView_Seq_ASCII(xin, viewer));
  } else if (isbinary) {
    PetscCall(VecView_Seq_Binary(xin, viewer));
#if defined(PETSC_HAVE_MATHEMATICA)
  } else if (ismathematica) {
    PetscCall(PetscViewerMathematicaPutVector(viewer, xin));
#endif
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    PetscCall(VecView_MPI_HDF5(xin, viewer)); /* Reusing VecView_MPI_HDF5 ... don't want code duplication*/
#endif
#if defined(PETSC_HAVE_ADIOS)
  } else if (isadios) {
    PetscCall(VecView_MPI_ADIOS(xin, viewer)); /* Reusing VecView_MPI_ADIOS ... don't want code duplication*/
#endif
#if defined(PETSC_HAVE_MATLAB)
  } else if (ismatlab) {
    PetscCall(VecView_Seq_Matlab(xin, viewer));
#endif
  } else if (isglvis) PetscCall(VecView_GLVis(xin, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecGetValues_Seq(Vec xin, PetscInt ni, const PetscInt ix[], PetscScalar y[])
{
  const PetscBool    ignorenegidx = xin->stash.ignorenegidx;
  const PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin, &xx));
  for (PetscInt i = 0; i < ni; ++i) {
    if (ignorenegidx && (ix[i] < 0)) continue;
    if (PetscDefined(USE_DEBUG)) {
      PetscCheck(ix[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Out of range index value %" PetscInt_FMT " cannot be negative", ix[i]);
      PetscCheck(ix[i] < xin->map->n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Out of range index value %" PetscInt_FMT " to large maximum allowed %" PetscInt_FMT, ix[i], xin->map->n);
    }
    y[i] = xx[ix[i]];
  }
  PetscCall(VecRestoreArrayRead(xin, &xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSetValues_Seq(Vec xin, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode m)
{
  const PetscBool ignorenegidx = xin->stash.ignorenegidx;
  PetscScalar    *xx;

  PetscFunctionBegin;
  // call to getarray (not e.g. getarraywrite() if m is INSERT_VALUES) is deliberate! If this
  // is secretly a VECSEQCUDA it may have values currently on the device, in which case --
  // unless we are replacing the entire array -- we need to copy them up
  PetscCall(VecGetArray(xin, &xx));
  for (PetscInt i = 0; i < ni; i++) {
    if (ignorenegidx && (ix[i] < 0)) continue;
    if (PetscDefined(USE_DEBUG)) {
      PetscCheck(ix[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Out of range index value %" PetscInt_FMT " cannot be negative", ix[i]);
      PetscCheck(ix[i] < xin->map->n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Out of range index value %" PetscInt_FMT " maximum %" PetscInt_FMT, ix[i], xin->map->n);
    }
    if (m == INSERT_VALUES) {
      xx[ix[i]] = y[i];
    } else {
      xx[ix[i]] += y[i];
    }
  }
  PetscCall(VecRestoreArray(xin, &xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSetValuesBlocked_Seq(Vec xin, PetscInt ni, const PetscInt ix[], const PetscScalar yin[], InsertMode m)
{
  PetscScalar *xx;
  PetscInt     bs;

  /* For optimization could treat bs = 2, 3, 4, 5 as special cases with loop unrolling */
  PetscFunctionBegin;
  PetscCall(VecGetBlockSize(xin, &bs));
  PetscCall(VecGetArray(xin, &xx));
  for (PetscInt i = 0; i < ni; ++i, yin += bs) {
    const PetscInt start = bs * ix[i];

    if (start < 0) continue;
    PetscCheck(start < xin->map->n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Out of range index value %" PetscInt_FMT " maximum %" PetscInt_FMT, start, xin->map->n);
    for (PetscInt j = 0; j < bs; j++) {
      if (m == INSERT_VALUES) {
        xx[start + j] = yin[j];
      } else {
        xx[start + j] += yin[j];
      }
    }
  }
  PetscCall(VecRestoreArray(xin, &xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecResetPreallocationCOO_Seq(Vec x)
{
  Vec_Seq *vs = (Vec_Seq *)x->data;

  PetscFunctionBegin;
  if (vs) {
    PetscCall(PetscFree(vs->jmap1)); /* Destroy old stuff */
    PetscCall(PetscFree(vs->perm1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSetPreallocationCOO_Seq(Vec x, PetscCount coo_n, const PetscInt coo_i[])
{
  PetscInt    m, *i;
  PetscCount  k, nneg;
  PetscCount *perm1, *jmap1;
  Vec_Seq    *vs = (Vec_Seq *)x->data;

  PetscFunctionBegin;
  PetscCall(VecResetPreallocationCOO_Seq(x)); /* Destroy old stuff */
  PetscCall(PetscMalloc1(coo_n, &i));
  PetscCall(PetscArraycpy(i, coo_i, coo_n)); /* Make a copy since we'll modify it */
  PetscCall(PetscMalloc1(coo_n, &perm1));
  for (k = 0; k < coo_n; k++) perm1[k] = k;
  PetscCall(PetscSortIntWithCountArray(coo_n, i, perm1));
  for (k = 0; k < coo_n; k++) {
    if (i[k] >= 0) break;
  } /* Advance k to the first entry with a non-negative index */
  nneg = k;

  PetscCall(VecGetLocalSize(x, &m));
  PetscCheck(!nneg || x->stash.ignorenegidx, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Found a negative index in VecSetPreallocateCOO() but VEC_IGNORE_NEGATIVE_INDICES was not set");
  PetscCheck(!coo_n || i[coo_n - 1] < m, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Found index (%" PetscInt_FMT ") greater than the size of the vector (%" PetscInt_FMT ") in VecSetPreallocateCOO()", i[coo_n - 1], m);

  PetscCall(PetscCalloc1(m + 1, &jmap1));
  for (; k < coo_n; k++) jmap1[i[k] + 1]++;         /* Count repeats of each entry */
  for (k = 0; k < m; k++) jmap1[k + 1] += jmap1[k]; /* Transform jmap[] to CSR-like data structure */
  PetscCall(PetscFree(i));

  if (nneg) { /* Discard leading negative indices */
    PetscCount *perm1_new;
    PetscCall(PetscMalloc1(coo_n - nneg, &perm1_new));
    PetscCall(PetscArraycpy(perm1_new, perm1 + nneg, coo_n - nneg));
    PetscCall(PetscFree(perm1));
    perm1 = perm1_new;
  }

  /* Record COO fields */
  vs->coo_n = coo_n;
  vs->tot1  = coo_n - nneg;
  vs->jmap1 = jmap1; /* [m+1] */
  vs->perm1 = perm1; /* [tot] */
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSetValuesCOO_Seq(Vec x, const PetscScalar coo_v[], InsertMode imode)
{
  Vec_Seq          *vs    = (Vec_Seq *)x->data;
  const PetscCount *perm1 = vs->perm1, *jmap1 = vs->jmap1;
  PetscScalar      *xv;
  PetscInt          m;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(x, &m));
  PetscCall(VecGetArray(x, &xv));
  for (PetscInt i = 0; i < m; i++) {
    PetscScalar sum = 0.0;
    for (PetscCount j = jmap1[i]; j < jmap1[i + 1]; j++) sum += coo_v[perm1[j]];
    xv[i] = (imode == INSERT_VALUES ? 0.0 : xv[i]) + sum;
  }
  PetscCall(VecRestoreArray(x, &xv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecDestroy_Seq(Vec v)
{
  Vec_Seq *vs = (Vec_Seq *)v->data;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscCall(PetscLogObjectState((PetscObject)v, "Length=%" PetscInt_FMT, v->map->n));
#endif
  if (vs) PetscCall(PetscFree(vs->array_allocated));
  PetscCall(VecResetPreallocationCOO_Seq(v));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscMatlabEnginePut_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscMatlabEngineGet_C", NULL));
  PetscCall(PetscFree(v->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSetOption_Seq(Vec v, VecOption op, PetscBool flag)
{
  PetscFunctionBegin;
  if (op == VEC_IGNORE_NEGATIVE_INDICES) v->stash.ignorenegidx = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecDuplicate_Seq(Vec win, Vec *V)
{
  PetscFunctionBegin;
  PetscCall(VecCreate(PetscObjectComm((PetscObject)win), V));
  PetscCall(VecSetSizes(*V, win->map->n, win->map->n));
  PetscCall(VecSetType(*V, ((PetscObject)win)->type_name));
  PetscCall(PetscLayoutReference(win->map, &(*V)->map));
  PetscCall(PetscObjectListDuplicate(((PetscObject)win)->olist, &((PetscObject)(*V))->olist));
  PetscCall(PetscFunctionListDuplicate(((PetscObject)win)->qlist, &((PetscObject)(*V))->qlist));

  (*V)->ops->view          = win->ops->view;
  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static struct _VecOps DvOps = {
  PetscDesignatedInitializer(duplicate, VecDuplicate_Seq), /* 1 */
  PetscDesignatedInitializer(duplicatevecs, VecDuplicateVecs_Default),
  PetscDesignatedInitializer(destroyvecs, VecDestroyVecs_Default),
  PetscDesignatedInitializer(dot, VecDot_Seq),
  PetscDesignatedInitializer(mdot, VecMDot_Seq),
  PetscDesignatedInitializer(norm, VecNorm_Seq),
  PetscDesignatedInitializer(tdot, VecTDot_Seq),
  PetscDesignatedInitializer(mtdot, VecMTDot_Seq),
  PetscDesignatedInitializer(scale, VecScale_Seq),
  PetscDesignatedInitializer(copy, VecCopy_Seq), /* 10 */
  PetscDesignatedInitializer(set, VecSet_Seq),
  PetscDesignatedInitializer(swap, VecSwap_Seq),
  PetscDesignatedInitializer(axpy, VecAXPY_Seq),
  PetscDesignatedInitializer(axpby, VecAXPBY_Seq),
  PetscDesignatedInitializer(maxpy, VecMAXPY_Seq),
  PetscDesignatedInitializer(aypx, VecAYPX_Seq),
  PetscDesignatedInitializer(waxpy, VecWAXPY_Seq),
  PetscDesignatedInitializer(axpbypcz, VecAXPBYPCZ_Seq),
  PetscDesignatedInitializer(pointwisemult, VecPointwiseMult_Seq),
  PetscDesignatedInitializer(pointwisedivide, VecPointwiseDivide_Seq),
  PetscDesignatedInitializer(setvalues, VecSetValues_Seq), /* 20 */
  PetscDesignatedInitializer(assemblybegin, NULL),
  PetscDesignatedInitializer(assemblyend, NULL),
  PetscDesignatedInitializer(getarray, NULL),
  PetscDesignatedInitializer(getsize, VecGetSize_Seq),
  PetscDesignatedInitializer(getlocalsize, VecGetSize_Seq),
  PetscDesignatedInitializer(restorearray, NULL),
  PetscDesignatedInitializer(max, VecMax_Seq),
  PetscDesignatedInitializer(min, VecMin_Seq),
  PetscDesignatedInitializer(setrandom, VecSetRandom_Seq),
  PetscDesignatedInitializer(setoption, VecSetOption_Seq), /* 30 */
  PetscDesignatedInitializer(setvaluesblocked, VecSetValuesBlocked_Seq),
  PetscDesignatedInitializer(destroy, VecDestroy_Seq),
  PetscDesignatedInitializer(view, VecView_Seq),
  PetscDesignatedInitializer(placearray, VecPlaceArray_Seq),
  PetscDesignatedInitializer(replacearray, VecReplaceArray_Seq),
  PetscDesignatedInitializer(dot_local, VecDot_Seq),
  PetscDesignatedInitializer(tdot_local, VecTDot_Seq),
  PetscDesignatedInitializer(norm_local, VecNorm_Seq),
  PetscDesignatedInitializer(mdot_local, VecMDot_Seq),
  PetscDesignatedInitializer(mtdot_local, VecMTDot_Seq), /* 40 */
  PetscDesignatedInitializer(load, VecLoad_Default),
  PetscDesignatedInitializer(reciprocal, VecReciprocal_Default),
  PetscDesignatedInitializer(conjugate, VecConjugate_Seq),
  PetscDesignatedInitializer(setlocaltoglobalmapping, NULL),
  PetscDesignatedInitializer(setvalueslocal, NULL),
  PetscDesignatedInitializer(resetarray, VecResetArray_Seq),
  PetscDesignatedInitializer(setfromoptions, NULL),
  PetscDesignatedInitializer(maxpointwisedivide, VecMaxPointwiseDivide_Seq),
  PetscDesignatedInitializer(pointwisemax, VecPointwiseMax_Seq),
  PetscDesignatedInitializer(pointwisemaxabs, VecPointwiseMaxAbs_Seq),
  PetscDesignatedInitializer(pointwisemin, VecPointwiseMin_Seq),
  PetscDesignatedInitializer(getvalues, VecGetValues_Seq),
  PetscDesignatedInitializer(sqrt, NULL),
  PetscDesignatedInitializer(abs, NULL),
  PetscDesignatedInitializer(exp, NULL),
  PetscDesignatedInitializer(log, NULL),
  PetscDesignatedInitializer(shift, NULL),
  PetscDesignatedInitializer(create, NULL),
  PetscDesignatedInitializer(stridegather, VecStrideGather_Default),
  PetscDesignatedInitializer(stridescatter, VecStrideScatter_Default),
  PetscDesignatedInitializer(dotnorm2, NULL),
  PetscDesignatedInitializer(getsubvector, NULL),
  PetscDesignatedInitializer(restoresubvector, NULL),
  PetscDesignatedInitializer(getarrayread, NULL),
  PetscDesignatedInitializer(restorearrayread, NULL),
  PetscDesignatedInitializer(stridesubsetgather, VecStrideSubSetGather_Default),
  PetscDesignatedInitializer(stridesubsetscatter, VecStrideSubSetScatter_Default),
  PetscDesignatedInitializer(viewnative, VecView_Seq),
  PetscDesignatedInitializer(loadnative, NULL),
  PetscDesignatedInitializer(createlocalvector, NULL),
  PetscDesignatedInitializer(getlocalvector, NULL),
  PetscDesignatedInitializer(restorelocalvector, NULL),
  PetscDesignatedInitializer(getlocalvectorread, NULL),
  PetscDesignatedInitializer(restorelocalvectorread, NULL),
  PetscDesignatedInitializer(bindtocpu, NULL),
  PetscDesignatedInitializer(getarraywrite, NULL),
  PetscDesignatedInitializer(restorearraywrite, NULL),
  PetscDesignatedInitializer(getarrayandmemtype, NULL),
  PetscDesignatedInitializer(restorearrayandmemtype, NULL),
  PetscDesignatedInitializer(getarrayreadandmemtype, NULL),
  PetscDesignatedInitializer(restorearrayreadandmemtype, NULL),
  PetscDesignatedInitializer(getarraywriteandmemtype, NULL),
  PetscDesignatedInitializer(restorearraywriteandmemtype, NULL),
  PetscDesignatedInitializer(concatenate, NULL),
  PetscDesignatedInitializer(sum, NULL),
  PetscDesignatedInitializer(setpreallocationcoo, VecSetPreallocationCOO_Seq),
  PetscDesignatedInitializer(setvaluescoo, VecSetValuesCOO_Seq),
};

/*
      This is called by VecCreate_Seq() (i.e. VecCreateSeq()) and VecCreateSeqWithArray()
*/
PetscErrorCode VecCreate_Seq_Private(Vec v, const PetscScalar array[])
{
  Vec_Seq *s;

  PetscFunctionBegin;
  PetscCall(PetscNew(&s));
  PetscCall(PetscMemcpy(v->ops, &DvOps, sizeof(DvOps)));

  v->data            = (void *)s;
  v->petscnative     = PETSC_TRUE;
  s->array           = (PetscScalar *)array;
  s->array_allocated = NULL;
  if (array) v->offloadmask = PETSC_OFFLOAD_CPU;

  PetscCall(PetscLayoutSetUp(v->map));
  PetscCall(PetscObjectChangeTypeName((PetscObject)v, VECSEQ));
#if defined(PETSC_HAVE_MATLAB)
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscMatlabEnginePut_C", VecMatlabEnginePut_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscMatlabEngineGet_C", VecMatlabEngineGet_Default));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecCreateSeqWithArray - Creates a standard,sequential array-style vector,
   where the user provides the array space to store the vector values.

   Collective

   Input Parameters:
+  comm - the communicator, should be `PETSC_COMM_SELF`
.  bs - the block size
.  n - the vector length
-  array - memory where the vector elements are to be stored.

   Output Parameter:
.  V - the vector

   Level: intermediate

   Notes:
   Use `VecDuplicate()` or `VecDuplicateVecs(`) to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is` NULL`, then `VecPlaceArray()` can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via `VecDestroy()`.
   The user should not free the array until the vector is destroyed.

.seealso: `VecCreateMPIWithArray()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`,
          `VecCreateGhost()`, `VecCreateSeq()`, `VecPlaceArray()`
@*/
PetscErrorCode VecCreateSeqWithArray(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar array[], Vec *V)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(VecCreate(comm, V));
  PetscCall(VecSetSizes(*V, n, n));
  PetscCall(VecSetBlockSize(*V, bs));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot create VECSEQ on more than one process");
  PetscCall(VecCreate_Seq_Private(*V, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}
