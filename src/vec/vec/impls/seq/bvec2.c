
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
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);

  for (i=0; i<n; i++) ww[i] = PetscMax(PetscRealPart(xx[i]),PetscRealPart(yy[i]));

  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMin_Seq(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);

  for (i=0; i<n; i++) ww[i] = PetscMin(PetscRealPart(xx[i]),PetscRealPart(yy[i]));

  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMaxAbs_Seq(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);

  for (i=0; i<n; i++) ww[i] = PetscMax(PetscAbsScalar(xx[i]),PetscAbsScalar(yy[i]));

  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fxtimesy.h>

PetscErrorCode VecPointwiseMult_Seq(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);
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
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseDivide_Seq(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    if (yy[i] != 0.0) ww[i] = xx[i] / yy[i];
    else ww[i] = 0.0;
  }

  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetRandom_Seq(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n,i;
  PetscScalar    *xx;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  for (i=0; i<n; i++) {ierr = PetscRandomGetValue(r,&xx[i]);CHKERRQ(ierr);}
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&x);CHKERRQ(ierr);
  while (n-->0) {
    *x = PetscConj(*x);
    x++;
  }
  ierr = VecRestoreArray(xin,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecResetArray_Seq(Vec vin)
{
  Vec_Seq *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_Seq(Vec xin,Vec yin)
{
  PetscScalar       *ya;
  const PetscScalar *xa;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    ierr = PetscArraycpy(ya,xa,xin->map->n);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSwap_Seq(Vec xin,Vec yin)
{
  PetscScalar    *ya, *xa;
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASswap",BLASswap_(&bn,xa,&one,ya,&one));
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fnorm.h>

PetscErrorCode VecNorm_Seq(Vec xin,NormType type,PetscReal *z)
{
  const PetscScalar *xx;
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n;
  PetscBLASInt      one = 1, bn;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
#if defined(PETSC_USE_REAL___FP16)
    *z   = BLASnrm2_(&bn,xx,&one);
#else
    *z   = PetscRealPart(BLASdot_(&bn,xx,&one,xx,&one));
    *z   = PetscSqrtReal(*z);
#endif
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = PetscLogFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    PetscInt  i;
    PetscReal max = 0.0,tmp;

    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if ((tmp = PetscAbsScalar(*xx)) > max) max = tmp;
      /* check special case of tmp == NaN */
      if (tmp != tmp) {max = tmp; break;}
      xx++;
    }
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    *z   = max;
  } else if (type == NORM_1) {
#if defined(PETSC_USE_COMPLEX)
    PetscReal tmp = 0.0;
    PetscInt    i;
#endif
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    /* BLASasum() returns the nonstandard 1 norm of the 1 norm of the complex entries so we provide a custom loop instead */
    for (i=0; i<n; i++) {
      tmp += PetscAbsScalar(xx[i]);
    }
    *z = tmp;
#else
    PetscStackCallBLAS("BLASasum",*z   = BLASasum_(&bn,xx,&one));
#endif
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = PetscLogFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    ierr = VecNorm_Seq(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_Seq(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Seq_ASCII(Vec xin,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscInt          i,n = xin->map->n;
  const char        *name;
  PetscViewerFormat format;
  const PetscScalar *xv;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xv);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_MATLAB) {
    ierr = PetscObjectGetName((PetscObject)xin,&name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%s = [\n",name);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
      if (PetscImaginaryPart(xv[i]) > 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei\n",(double)PetscRealPart(xv[i]),(double)PetscImaginaryPart(xv[i]));CHKERRQ(ierr);
      } else if (PetscImaginaryPart(xv[i]) < 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e - %18.16ei\n",(double)PetscRealPart(xv[i]),-(double)PetscImaginaryPart(xv[i]));CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)PetscRealPart(xv[i]));CHKERRQ(ierr);
      }
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double) xv[i]);CHKERRQ(ierr);
#endif
    }
    ierr = PetscViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_SYMMODU) {
    for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e\n",(double)PetscRealPart(xv[i]),(double)PetscImaginaryPart(xv[i]));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)xv[i]);CHKERRQ(ierr);
#endif
    }
  } else if (format == PETSC_VIEWER_ASCII_VTK || format == PETSC_VIEWER_ASCII_VTK_CELL) {
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
      ierr = PetscObjectComposedDataRegister(&stateId);CHKERRQ(ierr);
    }
    ierr = PetscObjectComposedDataGetInt((PetscObject) viewer, stateId, outputState, hasState);CHKERRQ(ierr);
    if (!hasState) outputState = 0;
    ierr = PetscObjectGetName((PetscObject) xin, &name);CHKERRQ(ierr);
    ierr = VecGetBlockSize(xin, &bs);CHKERRQ(ierr);
    if ((bs < 1) || (bs > 3)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %d", bs);
    if (format == PETSC_VIEWER_ASCII_VTK) {
      if (outputState == 0) {
        outputState = 1;
        doOutput = 1;
      } else if (outputState == 1) doOutput = 0;
      else if (outputState == 2) {
        outputState = 3;
        doOutput = 1;
      } else if (outputState == 3) doOutput = 0;
      else if (outputState == 4) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output POINT_DATA again after intervening CELL_DATA");

      if (doOutput) {
        ierr = PetscViewerASCIIPrintf(viewer, "POINT_DATA %d\n", n/bs);CHKERRQ(ierr);
      }
    } else {
      if (outputState == 0) {
        outputState = 2;
        doOutput = 1;
      } else if (outputState == 1) {
        outputState = 4;
        doOutput = 1;
      } else if (outputState == 2) doOutput = 0;
      else if (outputState == 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output CELL_DATA again after intervening POINT_DATA");
      else if (outputState == 4) doOutput = 0;

      if (doOutput) {
        ierr = PetscViewerASCIIPrintf(viewer, "CELL_DATA %d\n", n);CHKERRQ(ierr);
      }
    }
    ierr = PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, outputState);CHKERRQ(ierr);
    if (name) {
      if (bs == 3) {
        ierr = PetscViewerASCIIPrintf(viewer, "VECTORS %s double\n", name);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer, "SCALARS %s double %d\n", name, bs);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "SCALARS scalars double %d\n", bs);CHKERRQ(ierr);
    }
    if (bs != 3) {
      ierr = PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n");CHKERRQ(ierr);
    }
    for (i=0; i<n/bs; i++) {
      for (b=0; b<bs; b++) {
        if (b > 0) {
          ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
        }
#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%g",(double)xv[i*bs+b]);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_VTK_COORDS) {
    PetscInt bs, b;

    ierr = VecGetBlockSize(xin, &bs);CHKERRQ(ierr);
    if ((bs < 1) || (bs > 3)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %d", bs);
    for (i=0; i<n/bs; i++) {
      for (b=0; b<bs; b++) {
        if (b > 0) {
          ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
        }
#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%g",(double)xv[i*bs+b]);CHKERRQ(ierr);
#endif
      }
      for (b=bs; b<3; b++) {
        ierr = PetscViewerASCIIPrintf(viewer," 0.0");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_PCICE) {
    PetscInt bs, b;

    ierr = VecGetBlockSize(xin, &bs);CHKERRQ(ierr);
    if ((bs < 1) || (bs > 3)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "PCICE can only handle up to 3D objects, but vector dimension is %d", bs);
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n", xin->map->N/bs);CHKERRQ(ierr);
    for (i=0; i<n/bs; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"%7D   ", i+1);CHKERRQ(ierr);
      for (b=0; b<bs; b++) {
        if (b > 0) {
          ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
        }
#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"% 12.5E",(double)xv[i*bs+b]);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_GLVIS) {
    /* GLVis ASCII visualization/dump: this function mimicks mfem::GridFunction::Save() */
    const PetscScalar       *array;
    PetscInt                i,n,vdim, ordering = 1; /* mfem::FiniteElementSpace::Ordering::byVDIM */
    PetscContainer          glvis_container;
    PetscViewerGLVisVecInfo glvis_vec_info;
    PetscViewerGLVisInfo    glvis_info;
    PetscErrorCode          ierr;

    /* mfem::FiniteElementSpace::Save() */
    ierr = VecGetBlockSize(xin,&vdim);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"FiniteElementSpace\n");CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)xin,"_glvis_info_container",(PetscObject*)&glvis_container);CHKERRQ(ierr);
    if (!glvis_container) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_PLIB,"Missing GLVis container");
    ierr = PetscContainerGetPointer(glvis_container,(void**)&glvis_vec_info);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%s\n",glvis_vec_info->fec_type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"VDim: %d\n",vdim);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Ordering: %d\n",ordering);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    /* mfem::Vector::Print() */
    ierr = PetscObjectQuery((PetscObject)viewer,"_glvis_info_container",(PetscObject*)&glvis_container);CHKERRQ(ierr);
    if (!glvis_container) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_PLIB,"Missing GLVis container");
    ierr = PetscContainerGetPointer(glvis_container,(void**)&glvis_info);CHKERRQ(ierr);
    if (glvis_info->enabled) {
      ierr = VecGetLocalSize(xin,&n);CHKERRQ(ierr);
      ierr = VecGetArrayRead(xin,&array);CHKERRQ(ierr);
      for (i=0;i<n;i++) {
        ierr = PetscViewerASCIIPrintf(viewer,glvis_info->fmt,(double)PetscRealPart(array[i]));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
      ierr = VecRestoreArrayRead(xin,&array);CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    /* No info */
  } else {
    for (i=0; i<n; i++) {
      if (format == PETSC_VIEWER_ASCII_INDEX) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D: ",i);CHKERRQ(ierr);
      }
#if defined(PETSC_USE_COMPLEX)
      if (PetscImaginaryPart(xv[i]) > 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%g + %g i\n",(double)PetscRealPart(xv[i]),(double)PetscImaginaryPart(xv[i]));CHKERRQ(ierr);
      } else if (PetscImaginaryPart(xv[i]) < 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%g - %g i\n",(double)PetscRealPart(xv[i]),-(double)PetscImaginaryPart(xv[i]));CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"%g\n",(double)PetscRealPart(xv[i]));CHKERRQ(ierr);
      }
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%g\n",(double)xv[i]);CHKERRQ(ierr);
#endif
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xin,&xv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
PetscErrorCode VecView_Seq_Draw_LG(Vec xin,PetscViewer v)
{
  PetscDraw         draw;
  PetscBool         isnull;
  PetscDrawLG       lg;
  PetscErrorCode    ierr;
  PetscInt          i,c,bs = PetscAbs(xin->map->bs),n = xin->map->n/bs;
  const PetscScalar *xv;
  PetscReal         *xx,*yy,xmin,xmax,h;
  int               colors[] = {PETSC_DRAW_RED};
  PetscViewerFormat format;
  PetscDrawAxis     axis;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(v,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = PetscViewerGetFormat(v,&format);CHKERRQ(ierr);
  ierr = PetscMalloc2(n,&xx,n,&yy);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xin,&xv);CHKERRQ(ierr);
  for (c=0; c<bs; c++) {
    ierr = PetscViewerDrawGetDrawLG(v,c,&lg);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,1);CHKERRQ(ierr);
    ierr = PetscDrawLGSetColors(lg,colors);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_DRAW_LG_XRANGE) {
      ierr = PetscDrawLGGetAxis(lg,&axis);CHKERRQ(ierr);
      ierr = PetscDrawAxisGetLimits(axis,&xmin,&xmax,NULL,NULL);CHKERRQ(ierr);
      h = (xmax - xmin)/n;
      for (i=0; i<n; i++) xx[i] = i*h + 0.5*h; /* cell center */
    } else {
      for (i=0; i<n; i++) xx[i] = (PetscReal)i;
    }
    for (i=0; i<n; i++) yy[i] = PetscRealPart(xv[c + i*bs]);

    ierr = PetscDrawLGAddPoints(lg,n,&xx,&yy);CHKERRQ(ierr);
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xin,&xv);CHKERRQ(ierr);
  ierr = PetscFree2(xx,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Seq_Draw(Vec xin,PetscViewer v)
{
  PetscErrorCode    ierr;
  PetscDraw         draw;
  PetscBool         isnull;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(v,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = VecView_Seq_Draw_LG(xin,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Seq_Binary(Vec xin,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  int               fdes;
  PetscInt          n = xin->map->n,classid=VEC_FILE_CLASSID;
  FILE              *file;
  const PetscScalar *xv;
#if defined(PETSC_HAVE_MPIIO)
  PetscBool         isMPIIO;
#endif
  PetscBool         skipHeader;
  PetscViewerFormat format;

  PetscFunctionBegin;
  /* Write vector header */
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipHeader);CHKERRQ(ierr);
  if (!skipHeader) {
    ierr = PetscViewerBinaryWrite(viewer,&classid,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&n,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  }

  /* Write vector contents */
#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscViewerBinaryGetUseMPIIO(viewer,&isMPIIO);CHKERRQ(ierr);
  if (!isMPIIO) {
#endif
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fdes);CHKERRQ(ierr);
    ierr = VecGetArrayRead(xin,&xv);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fdes,(void*)xv,n,PETSC_SCALAR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xin,&xv);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_BINARY_MATLAB) {
      MPI_Comm   comm;
      FILE       *info;
      const char *name;

      ierr = PetscObjectGetName((PetscObject)xin,&name);CHKERRQ(ierr);
      ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
      ierr = PetscViewerBinaryGetInfoPointer(viewer,&info);CHKERRQ(ierr);
      ierr = PetscFPrintf(comm,info,"#--- begin code written by PetscViewerBinary for MATLAB format ---#\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(comm,info,"#$$ Set.%s = PetscBinaryRead(fd);\n",name);CHKERRQ(ierr);
      ierr = PetscFPrintf(comm,info,"#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n");CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_MPIIO)
  } else {
    MPI_Offset   off;
    MPI_File     mfdes;
    PetscMPIInt  lsize;

    ierr = PetscMPIIntCast(n,&lsize);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetMPIIODescriptor(viewer,&mfdes);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetMPIIOOffset(viewer,&off);CHKERRQ(ierr);
    ierr = VecGetArrayRead(xin,&xv);CHKERRQ(ierr);
    ierr = MPIU_File_write_at_all(mfdes,off,(void*)xv,lsize,MPIU_SCALAR,MPI_STATUS_IGNORE);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xin,&xv);CHKERRQ(ierr);
    ierr = PetscViewerBinaryAddMPIIOOffset(viewer,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
#endif

  ierr = PetscViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
  if (file) {
    if (((PetscObject)xin)->prefix) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,file,"-%svecload_block_size %D\n",((PetscObject)xin)->prefix,PetscAbs(xin->map->bs));CHKERRQ(ierr);
    } else {
      ierr = PetscFPrintf(PETSC_COMM_SELF,file,"-vecload_block_size %D\n",PetscAbs(xin->map->bs));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <petscmatlab.h>
#include <mat.h>   /* MATLAB include file */
PetscErrorCode VecView_Seq_Matlab(Vec vec,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscInt          n;
  const PetscScalar *array;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  ierr = PetscObjectName((PetscObject)vec);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vec,&array);CHKERRQ(ierr);
  ierr = PetscViewerMatlabPutArray(viewer,n,1,array,((PetscObject)vec)->name);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PETSC_EXTERN PetscErrorCode VecView_Seq(Vec xin,PetscViewer viewer)
{
  PetscErrorCode ierr;
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
#if defined(PETSC_HAVE_ADIOS2)
  PetscBool      isadios2;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATHEMATICA)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATHEMATICA,&ismathematica);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERMATLAB,&ismatlab);CHKERRQ(ierr);
#endif
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&isglvis);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ADIOS)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS,&isadios);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ADIOS2)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS2,&isadios2);CHKERRQ(ierr);
#endif

  if (isdraw) {
    ierr = VecView_Seq_Draw(xin,viewer);CHKERRQ(ierr);
  } else if (iascii) {
    ierr = VecView_Seq_ASCII(xin,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = VecView_Seq_Binary(xin,viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATHEMATICA)
  } else if (ismathematica) {
    ierr = PetscViewerMathematicaPutVector(viewer,xin);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    ierr = VecView_MPI_HDF5(xin,viewer);CHKERRQ(ierr); /* Reusing VecView_MPI_HDF5 ... don't want code duplication*/
#endif
#if defined(PETSC_HAVE_ADIOS)
  } else if (isadios) {
    ierr = VecView_MPI_ADIOS(xin,viewer);CHKERRQ(ierr); /* Reusing VecView_MPI_ADIOS ... don't want code duplication*/
#endif
#if defined(PETSC_HAVE_ADIOS2)
  } else if (isadios2) {
    ierr = VecView_MPI_ADIOS2(xin,viewer);CHKERRQ(ierr); /* Reusing VecView_MPI_ADIOS2 ... don't want code duplication*/
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  } else if (ismatlab) {
    ierr = VecView_Seq_Matlab(xin,viewer);CHKERRQ(ierr);
#endif
  } else if (isglvis) {
    ierr = VecView_GLVis(xin,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetValues_Seq(Vec xin,PetscInt ni,const PetscInt ix[],PetscScalar y[])
{
  const PetscScalar *xx;
  PetscInt          i;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  for (i=0; i<ni; i++) {
    if (xin->stash.ignorenegidx && ix[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
    if (ix[i] < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D cannot be negative",ix[i]);
    if (ix[i] >= xin->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D to large maximum allowed %D",ix[i],xin->map->n);
#endif
    y[i] = xx[ix[i]];
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetValues_Seq(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode m)
{
  PetscScalar    *xx;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  if (m == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      if (xin->stash.ignorenegidx && ix[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
      if (ix[i] < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D cannot be negative",ix[i]);
      if (ix[i] >= xin->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",ix[i],xin->map->n);
#endif
      xx[ix[i]] = y[i];
    }
  } else {
    for (i=0; i<ni; i++) {
      if (xin->stash.ignorenegidx && ix[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
      if (ix[i] < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D cannot be negative",ix[i]);
      if (ix[i] >= xin->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",ix[i],xin->map->n);
#endif
      xx[ix[i]] += y[i];
    }
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetValuesBlocked_Seq(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar yin[],InsertMode m)
{
  PetscScalar    *xx,*y = (PetscScalar*)yin;
  PetscInt       i,bs,start,j;
  PetscErrorCode ierr;

  /*
       For optimization could treat bs = 2, 3, 4, 5 as special cases with loop unrolling
  */
  PetscFunctionBegin;
  ierr = VecGetBlockSize(xin,&bs);CHKERRQ(ierr);
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  if (m == INSERT_VALUES) {
    for (i=0; i<ni; i++, y+=bs) {
      start = bs*ix[i];
      if (start < 0) continue;
#if defined(PETSC_USE_DEBUG)
      if (start >= xin->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",start,xin->map->n);
#endif
      for (j=0; j<bs; j++) xx[start+j] = y[j];
    }
  } else {
    for (i=0; i<ni; i++, y+=bs) {
      start = bs*ix[i];
      if (start < 0) continue;
#if defined(PETSC_USE_DEBUG)
      if (start >= xin->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",start,xin->map->n);
#endif
      for (j=0; j<bs; j++) xx[start+j] += y[j];
    }
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_Seq(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr);
  ierr = PetscFree(v->data);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(PetscObjectComm((PetscObject)win),V);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,win->map->n,win->map->n);CHKERRQ(ierr);
  ierr = VecSetType(*V,((PetscObject)win)->type_name);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*V)->map);CHKERRQ(ierr);
  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);

  (*V)->ops->view          = win->ops->view;
  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;
  PetscFunctionReturn(0);
}

static struct _VecOps DvOps = {VecDuplicate_Seq, /* 1 */
                               VecDuplicateVecs_Default,
                               VecDestroyVecs_Default,
                               VecDot_Seq,
                               VecMDot_Seq,
                               VecNorm_Seq,
                               VecTDot_Seq,
                               VecMTDot_Seq,
                               VecScale_Seq,
                               VecCopy_Seq, /* 10 */
                               VecSet_Seq,
                               VecSwap_Seq,
                               VecAXPY_Seq,
                               VecAXPBY_Seq,
                               VecMAXPY_Seq,
                               VecAYPX_Seq,
                               VecWAXPY_Seq,
                               VecAXPBYPCZ_Seq,
                               VecPointwiseMult_Seq,
                               VecPointwiseDivide_Seq,
                               VecSetValues_Seq, /* 20 */
                               0,0,
                               0,
                               VecGetSize_Seq,
                               VecGetSize_Seq,
                               0,
                               VecMax_Seq,
                               VecMin_Seq,
                               VecSetRandom_Seq,
                               VecSetOption_Seq, /* 30 */
                               VecSetValuesBlocked_Seq,
                               VecDestroy_Seq,
                               VecView_Seq,
                               VecPlaceArray_Seq,
                               VecReplaceArray_Seq,
                               VecDot_Seq,
                               VecTDot_Seq,
                               VecNorm_Seq,
                               VecMDot_Seq,
                               VecMTDot_Seq, /* 40 */
                               VecLoad_Default,
                               VecReciprocal_Default,
                               VecConjugate_Seq,
                               0,
                               0,
                               VecResetArray_Seq,
                               0,
                               VecMaxPointwiseDivide_Seq,
                               VecPointwiseMax_Seq,
                               VecPointwiseMaxAbs_Seq,
                               VecPointwiseMin_Seq,
                               VecGetValues_Seq,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               VecStrideGather_Default,
                               VecStrideScatter_Default,
                               0,
                               0,
                               0,
                               0,
                               0,
                               VecStrideSubSetGather_Default,
                               VecStrideSubSetScatter_Default,
                               0,
                               0
};


/*
      This is called by VecCreate_Seq() (i.e. VecCreateSeq()) and VecCreateSeqWithArray()
*/
PetscErrorCode VecCreate_Seq_Private(Vec v,const PetscScalar array[])
{
  Vec_Seq        *s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(v,&s);CHKERRQ(ierr);
  ierr = PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);

  v->data            = (void*)s;
  v->petscnative     = PETSC_TRUE;
  s->array           = (PetscScalar*)array;
  s->array_allocated = 0;

  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECSEQ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscMatlabEnginePut_C",VecMatlabEnginePut_Default);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscMatlabEngineGet_C",VecMatlabEngineGet_Default);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqWithArray - Creates a standard,sequential array-style vector,
   where the user provides the array space to store the vector values.

   Collective

   Input Parameter:
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
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = VecCreate(comm,V);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,n,n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*V,bs);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQ on more than one process");
  ierr = VecCreate_Seq_Private(*V,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
