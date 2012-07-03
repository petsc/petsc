
/*
   Implements the sequential vectors.
*/

#include <petsc-private/vecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/mpi/pvecimpl.h> /* For VecView_MPI_HDF5 */
#include <petscblaslapack.h>

#if defined(PETSC_HAVE_HDF5)
extern PetscErrorCode VecView_MPI_HDF5(Vec,PetscViewer);
#endif

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMax_Seq"
PetscErrorCode VecPointwiseMax_Seq(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ww[i] = PetscMax(PetscRealPart(xx[i]),PetscRealPart(yy[i]));
  }
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMin_Seq"
PetscErrorCode VecPointwiseMin_Seq(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ww[i] = PetscMin(PetscRealPart(xx[i]),PetscRealPart(yy[i]));
  }
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMaxAbs_Seq"
PetscErrorCode VecPointwiseMaxAbs_Seq(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ww[i] = PetscMax(PetscAbsScalar(xx[i]),PetscAbsScalar(yy[i]));
  }
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fxtimesy.h>
#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMult_Seq"
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

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide_Seq"
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
    ww[i] = xx[i] / yy[i];
  }
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,(const PetscScalar**)&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetRandom_Seq"
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

#undef __FUNCT__  
#define __FUNCT__ "VecGetSize_Seq"
PetscErrorCode VecGetSize_Seq(Vec vin,PetscInt *size)
{
  PetscFunctionBegin;
  *size = vin->map->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecConjugate_Seq"
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

#undef __FUNCT__  
#define __FUNCT__ "VecResetArray_Seq"
PetscErrorCode VecResetArray_Seq(Vec vin)
{
  Vec_Seq *v = (Vec_Seq *)vin->data;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCopy_Seq"
PetscErrorCode VecCopy_Seq(Vec xin,Vec yin)
{
  PetscScalar       *ya;
  const PetscScalar *xa;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    ierr = PetscMemcpy(ya,xa,xin->map->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSwap_Seq"
PetscErrorCode VecSwap_Seq(Vec xin,Vec yin)
{
  PetscScalar    *ya, *xa;
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(xin->map->n);

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecGetArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    BLASswap_(&bn,xa,&one,ya,&one);
    ierr = VecRestoreArray(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fnorm.h>
#undef __FUNCT__  
#define __FUNCT__ "VecNorm_Seq"
PetscErrorCode VecNorm_Seq(Vec xin,NormType type,PetscReal* z)
{
  const PetscScalar *xx;
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n;
  PetscBLASInt      one = 1, bn = PetscBLASIntCast(n);

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    *z = PetscRealPart(BLASdot_(&bn,xx,&one,xx,&one));
    *z = PetscSqrtReal(*z);
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = PetscLogFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    PetscInt     i;
    PetscReal    max = 0.0,tmp;

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
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    *z = BLASasum_(&bn,xx,&one);
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = PetscLogFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    ierr = VecNorm_Seq(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_Seq(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq_ASCII"
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
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei\n",PetscRealPart(xv[i]),PetscImaginaryPart(xv[i]));CHKERRQ(ierr);
      } else if (PetscImaginaryPart(xv[i]) < 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e - %18.16ei\n",PetscRealPart(xv[i]),-PetscImaginaryPart(xv[i]));CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",PetscRealPart(xv[i]));CHKERRQ(ierr);
      }
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double) xv[i]);CHKERRQ(ierr);
#endif
    }
    ierr = PetscViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_SYMMODU) {
    for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e\n",PetscRealPart(xv[i]),PetscImaginaryPart(xv[i]));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",xv[i]);CHKERRQ(ierr);
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
    if (!hasState) {
      outputState = 0;
    }
    ierr = PetscObjectGetName((PetscObject) xin, &name);CHKERRQ(ierr);
    ierr = VecGetBlockSize(xin, &bs);CHKERRQ(ierr);
    if ((bs < 1) || (bs > 3)) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %d", bs);
    }
    if (format == PETSC_VIEWER_ASCII_VTK) {
      if (outputState == 0) {
        outputState = 1;
        doOutput = 1;
      } else if (outputState == 1) {
        doOutput = 0;
      } else if (outputState == 2) {
        outputState = 3;
        doOutput = 1;
      } else if (outputState == 3) {
        doOutput = 0;
      } else if (outputState == 4) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output POINT_DATA again after intervening CELL_DATA");
      }
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
      } else if (outputState == 2) {
        doOutput = 0;
      } else if (outputState == 3) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output CELL_DATA again after intervening POINT_DATA");
      } else if (outputState == 4) {
        doOutput = 0;
      }
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
        ierr = PetscViewerASCIIPrintf(viewer,"%G",xv[i*bs+b]);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_VTK_COORDS) {
    PetscInt bs, b;

    ierr = VecGetBlockSize(xin, &bs);CHKERRQ(ierr);
    if ((bs < 1) || (bs > 3)) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %d", bs);
    }
    for (i=0; i<n/bs; i++) {
      for (b=0; b<bs; b++) {
        if (b > 0) {
          ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
        }
#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%G",xv[i*bs+b]);CHKERRQ(ierr);
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
    if ((bs < 1) || (bs > 3)) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "PCICE can only handle up to 3D objects, but vector dimension is %d", bs);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n", xin->map->N/bs);CHKERRQ(ierr);
    for (i=0; i<n/bs; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"%7D   ", i+1);CHKERRQ(ierr);
      for (b=0; b<bs; b++) {
        if (b > 0) {
          ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
        }
#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"% 12.5E",xv[i*bs+b]);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    PetscFunctionReturn(0);
  } else {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)xin,viewer,"Vector Object");CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if (format == PETSC_VIEWER_ASCII_INDEX) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D: ",i);CHKERRQ(ierr);
      }
#if defined(PETSC_USE_COMPLEX)
      if (PetscImaginaryPart(xv[i]) > 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%G + %G i\n",PetscRealPart(xv[i]),PetscImaginaryPart(xv[i]));CHKERRQ(ierr);
      } else if (PetscImaginaryPart(xv[i]) < 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%G - %G i\n",PetscRealPart(xv[i]),-PetscImaginaryPart(xv[i]));CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"%G\n",PetscRealPart(xv[i]));CHKERRQ(ierr);
      }
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%G\n",xv[i]);CHKERRQ(ierr);
#endif
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xin,&xv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq_Draw_LG"
PetscErrorCode VecView_Seq_Draw_LG(Vec xin,PetscViewer v)
{
  PetscErrorCode    ierr;
  PetscInt          i,n = xin->map->n;
  PetscDraw         win;
  PetscReal         *xx;
  PetscDrawLG       lg;
  const PetscScalar *xv;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDrawLG(v,0,&lg);CHKERRQ(ierr);
  ierr = PetscDrawLGGetDraw(lg,&win);CHKERRQ(ierr);
  ierr = PetscDrawCheckResizedWindow(win);CHKERRQ(ierr);
  ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
  ierr = PetscMalloc((n+1)*sizeof(PetscReal),&xx);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    xx[i] = (PetscReal) i;
  }
  ierr = VecGetArrayRead(xin,&xv);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscDrawLGAddPoints(lg,n,&xx,(PetscReal**)&xv);CHKERRQ(ierr);
#else 
  {
    PetscReal *yy;
    ierr = PetscMalloc((n+1)*sizeof(PetscReal),&yy);CHKERRQ(ierr);    
    for (i=0; i<n; i++) {
      yy[i] = PetscRealPart(xv[i]);
    }
    ierr = PetscDrawLGAddPoints(lg,n,&xx,&yy);CHKERRQ(ierr);
    ierr = PetscFree(yy);CHKERRQ(ierr);
  }
#endif
  ierr = VecRestoreArrayRead(xin,&xv);CHKERRQ(ierr);
  ierr = PetscFree(xx);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  ierr = PetscDrawSynchronizedFlush(win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq_Draw"
PetscErrorCode VecView_Seq_Draw(Vec xin,PetscViewer v)
{
  PetscErrorCode    ierr;
  PetscDraw         draw;
  PetscBool         isnull;
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

  PetscFunctionBegin;

  /* Write vector header */
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipHeader);CHKERRQ(ierr);
  if (!skipHeader) {
    ierr = PetscViewerBinaryWrite(viewer,&classid,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&n,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  }

  /* Write vector contents */
#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscViewerBinaryGetMPIIO(viewer,&isMPIIO);CHKERRQ(ierr);
  if (!isMPIIO) {
#endif
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fdes);CHKERRQ(ierr);
    ierr = VecGetArrayRead(xin,&xv);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fdes,(void*)xv,n,PETSC_SCALAR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xin,&xv);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  } else {
    MPI_Offset   off;
    MPI_File     mfdes;
    PetscMPIInt  gsizes[1],lsizes[1],lstarts[1];
    MPI_Datatype view;

    gsizes[0]  = PetscMPIIntCast(n);
    lsizes[0]  = PetscMPIIntCast(n);
    lstarts[0] = 0;
    ierr = MPI_Type_create_subarray(1,gsizes,lsizes,lstarts,MPI_ORDER_FORTRAN,MPIU_SCALAR,&view);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&view);CHKERRQ(ierr);

    ierr = PetscViewerBinaryGetMPIIODescriptor(viewer,&mfdes);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetMPIIOOffset(viewer,&off);CHKERRQ(ierr);
    ierr = VecGetArrayRead(xin,&xv);CHKERRQ(ierr);
    ierr = MPIU_File_write_all(mfdes,(void*)xv,lsizes[0],MPIU_SCALAR,MPI_STATUS_IGNORE);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xin,&xv);CHKERRQ(ierr);
    ierr = PetscViewerBinaryAddMPIIOOffset(viewer,n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = MPI_Type_free(&view);CHKERRQ(ierr);    
  }
#endif

  ierr = PetscViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
  if (file) {
    if (((PetscObject)xin)->prefix) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,file,"-%s_vecload_block_size %D\n",((PetscObject)xin)->prefix,xin->map->bs);CHKERRQ(ierr);
    } else {
      ierr = PetscFPrintf(PETSC_COMM_SELF,file,"-vecload_block_size %D\n",xin->map->bs);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <mat.h>   /* MATLAB include file */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq_Matlab"
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
EXTERN_C_END
#endif

#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq"
PetscErrorCode VecView_Seq(Vec xin,PetscViewer viewer)
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

  if (isdraw){ 
    ierr = VecView_Seq_Draw(xin,viewer);CHKERRQ(ierr);
  } else if (iascii){
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
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  } else if (ismatlab) {
    ierr = VecView_Seq_Matlab(xin,viewer);CHKERRQ(ierr);
#endif
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by this vector object",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetValues_Seq"
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

#undef __FUNCT__  
#define __FUNCT__ "VecSetValues_Seq"
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

#undef __FUNCT__  
#define __FUNCT__ "VecSetValuesBlocked_Seq"
PetscErrorCode VecSetValuesBlocked_Seq(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar yin[],InsertMode m)
{
  PetscScalar    *xx,*y = (PetscScalar*)yin;
  PetscInt       i,bs = xin->map->bs,start,j;
  PetscErrorCode ierr;

  /*
       For optimization could treat bs = 2, 3, 4, 5 as special cases with loop unrolling
  */
  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  if (m == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      start = bs*ix[i];
      if (start < 0) continue;
#if defined(PETSC_USE_DEBUG)
      if (start >= xin->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",start,xin->map->n);
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
#if defined(PETSC_USE_DEBUG)
      if (start >= xin->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",start,xin->map->n);
#endif
      for (j=0; j<bs; j++) {
        xx[start+j] += y[j];
      }
      y += bs;
    }  
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDestroy_Seq"
PetscErrorCode VecDestroy_Seq(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectDepublish(v);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr);
  ierr = PetscFree(v->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetOption_Seq"
PetscErrorCode VecSetOption_Seq(Vec v,VecOption op,PetscBool  flag)
{
  PetscFunctionBegin;
  if (op == VEC_IGNORE_NEGATIVE_INDICES) {
    v->stash.ignorenegidx = flag;
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_Seq"
PetscErrorCode VecDuplicate_Seq(Vec win,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(((PetscObject)win)->comm,V);CHKERRQ(ierr);
  ierr = PetscObjectSetPrecision((PetscObject)*V,((PetscObject)win)->precision);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,win->map->n,win->map->n);CHKERRQ(ierr);
  ierr = VecSetType(*V,((PetscObject)win)->type_name);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*V)->map);CHKERRQ(ierr);
  ierr = PetscOListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);

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
   	    VecStrideScatter_Default
          };


/*
      This is called by VecCreate_Seq() (i.e. VecCreateSeq()) and VecCreateSeqWithArray()
*/
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_Seq_Private"
PetscErrorCode VecCreate_Seq_Private(Vec v,const PetscScalar array[])
{
  Vec_Seq        *s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(v,Vec_Seq,&s);CHKERRQ(ierr);
  ierr = PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  v->data            = (void*)s;
  v->petscnative     = PETSC_TRUE;
  s->array           = (PetscScalar *)array;
  s->array_allocated = 0;

  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECSEQ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscMatlabEnginePut_C","VecMatlabEnginePut_Default",VecMatlabEnginePut_Default);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscMatlabEngineGet_C","VecMatlabEngineGet_Default",VecMatlabEngineGet_Default);CHKERRQ(ierr);
#endif
#if defined(PETSC_USE_MIXED_PRECISION)
  ((PetscObject)v)->precision = (PetscPrecision)sizeof(PetscScalar);
#endif
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






