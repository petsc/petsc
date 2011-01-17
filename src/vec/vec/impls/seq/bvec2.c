#define PETSCVEC_DLL
/*
   Implements the sequential vectors.
*/

#include "private/vecimpl.h"          /*I "petscvec.h" I*/
#include "../src/vec/vec/impls/dvecimpl.h" 
#include "petscblaslapack.h"
#if defined(PETSC_HAVE_PNETCDF)
EXTERN_C_BEGIN
#include "pnetcdf.h"
EXTERN_C_END
#endif

#if defined(PETSC_HAVE_HDF5)
extern PetscErrorCode VecView_MPI_HDF5(Vec,PetscViewer);
#endif

#include "../src/vec/vec/impls/seq/ftn-kernels/fnorm.h"
#undef __FUNCT__  
#define __FUNCT__ "VecNorm_Seq"
PetscErrorCode VecNorm_Seq(Vec xin,NormType type,PetscReal* z)
{
  PetscScalar    *xx;
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n;
  PetscBLASInt   one = 1, bn = PetscBLASIntCast(n);

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
    /*
      This is because the Fortran BLAS 1 Norm is very slow! 
    */
#if defined(PETSC_HAVE_SLOW_BLAS_NORM2)
#if defined(PETSC_USE_FORTRAN_KERNEL_NORM)
    fortrannormsqr_(xx,&n,z);
    *z = sqrt(*z);
#elif defined(PETSC_USE_UNROLLED_NORM)
    {
    PetscReal work = 0.0;
    switch (n & 0x3) {
      case 3: work += PetscRealPart(xx[0]*PetscConj(xx[0])); xx++;
      case 2: work += PetscRealPart(xx[0]*PetscConj(xx[0])); xx++;
      case 1: work += PetscRealPart(xx[0]*PetscConj(xx[0])); xx++; n -= 4;
    }
    while (n>0) {
      work += PetscRealPart(xx[0]*PetscConj(xx[0])+xx[1]*PetscConj(xx[1])+
                        xx[2]*PetscConj(xx[2])+xx[3]*PetscConj(xx[3]));
      xx += 4; n -= 4;
    } 
    *z = sqrt(work);}
#else
    {
      PetscInt         i;
      PetscScalar sum=0.0;
      for (i=0; i<n; i++) {
        sum += (xx[i])*(PetscConj(xx[i]));
      }
      *z = sqrt(PetscRealPart(sum));
    }
#endif
#else
    *z = BLASnrm2_(&bn,xx,&one);
#endif
    ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
    ierr = PetscLogFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    PetscInt          i;
    PetscReal    max = 0.0,tmp;

    ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if ((tmp = PetscAbsScalar(*xx)) > max) max = tmp;
      /* check special case of tmp == NaN */
      if (tmp != tmp) {max = tmp; break;}
      xx++;
    }
    ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
    *z   = max;
  } else if (type == NORM_1) {
    ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
    *z = BLASasum_(&bn,xx,&one);
    ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
    ierr = PetscLogFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    ierr = VecNorm_Seq(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_Seq(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq_File"
PetscErrorCode VecView_Seq_File(Vec xin,PetscViewer viewer)
{
  Vec_Seq           *x = (Vec_Seq *)xin->data;
  PetscErrorCode    ierr;
  PetscInt          i,n = xin->map->n;
  const char        *name;
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
    PetscTruth hasState;
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
      SETERRQ1(PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %d", bs);
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
        SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Tried to output POINT_DATA again after intervening CELL_DATA");
      }
      if (doOutput) {
        ierr = PetscViewerASCIIPrintf(viewer, "POINT_DATA %d\n", n);CHKERRQ(ierr);
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
        SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Tried to output CELL_DATA again after intervening POINT_DATA");
      } else if (outputState == 4) {
        doOutput = 0;
      }
      if (doOutput) {
        ierr = PetscViewerASCIIPrintf(viewer, "CELL_DATA %d\n", n);CHKERRQ(ierr);
      }
    }
    ierr = PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, outputState);CHKERRQ(ierr);
    if (name) {
      ierr = PetscViewerASCIIPrintf(viewer, "SCALARS %s double %d\n", name, bs);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "SCALARS scalars double %d\n", bs);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n");CHKERRQ(ierr);
    for (i=0; i<n/bs; i++) {
      for (b=0; b<bs; b++) {
        if (b > 0) {
          ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
        }
#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%G",x->array[i*bs+b]);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_VTK_COORDS) {
    PetscInt bs, b;

    ierr = VecGetBlockSize(xin, &bs);CHKERRQ(ierr);
    if ((bs < 1) || (bs > 3)) {
      SETERRQ1(PETSC_ERR_ARG_WRONGSTATE, "VTK can only handle 3D objects, but vector dimension is %d", bs);
    }
    for (i=0; i<n/bs; i++) {
      for (b=0; b<bs; b++) {
        if (b > 0) {
          ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
        }
#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%G",x->array[i*bs+b]);CHKERRQ(ierr);
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
      SETERRQ1(PETSC_ERR_ARG_WRONGSTATE, "PCICE can only handle up to 3D objects, but vector dimension is %d", bs);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"%D\n", xin->map->N/bs);CHKERRQ(ierr);
    for (i=0; i<n/bs; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"%7D   ", i+1);CHKERRQ(ierr);
      for (b=0; b<bs; b++) {
        if (b > 0) {
          ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
        }
#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"% 12.5E",x->array[i*bs+b]);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_PYLITH) {
    PetscInt bs, b;

    ierr = VecGetBlockSize(xin, &bs);CHKERRQ(ierr);
    if (bs != 3) {
      SETERRQ1(PETSC_ERR_ARG_WRONGSTATE, "PyLith can only handle 3D objects, but vector dimension is %d", bs);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#  Node      X-coord           Y-coord           Z-coord\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    for (i=0; i<n/bs; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"%7D ", i+1);CHKERRQ(ierr);
      for (b=0; b<bs; b++) {
        if (b > 0) {
          ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
        }
#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"% 16.8E",x->array[i*bs+b]);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
  } else {
    for (i=0; i<n; i++) {
      if (format == PETSC_VIEWER_ASCII_INDEX) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D: ",i);CHKERRQ(ierr);
      }
#if defined(PETSC_USE_COMPLEX)
      if (PetscImaginaryPart(x->array[i]) > 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%G + %G i\n",PetscRealPart(x->array[i]),PetscImaginaryPart(x->array[i]));CHKERRQ(ierr);
      } else if (PetscImaginaryPart(x->array[i]) < 0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%G - %G i\n",PetscRealPart(x->array[i]),-PetscImaginaryPart(x->array[i]));CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"%G\n",PetscRealPart(x->array[i]));CHKERRQ(ierr);
      }
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%G\n",x->array[i]);CHKERRQ(ierr);
#endif
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq_Draw_LG"
static PetscErrorCode VecView_Seq_Draw_LG(Vec xin,PetscViewer v)
{
  Vec_Seq        *x = (Vec_Seq *)xin->data;
  PetscErrorCode ierr;
  PetscInt       i,n = xin->map->n;
  PetscDraw      win;
  PetscReal      *xx;
  PetscDrawLG    lg;

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
static PetscErrorCode VecView_Seq_Draw(Vec xin,PetscViewer v)
{
  PetscErrorCode    ierr;
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
static PetscErrorCode VecView_Seq_Binary(Vec xin,PetscViewer viewer)
{
  Vec_Seq        *x = (Vec_Seq *)xin->data;
  PetscErrorCode ierr;
  int            fdes;
  PetscInt       n = xin->map->n,cookie=VEC_FILE_COOKIE;
  FILE           *file;
#if defined(PETSC_HAVE_MPIIO)
  PetscTruth     isMPIIO;
#endif

  PetscFunctionBegin;

  /* Write vector header */
  ierr = PetscViewerBinaryWrite(viewer,&cookie,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&n,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);

  /* Write vector contents */
#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscViewerBinaryGetMPIIO(viewer,&isMPIIO);CHKERRQ(ierr);
  if (!isMPIIO) {
#endif
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fdes);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fdes,x->array,n,PETSC_SCALAR,PETSC_FALSE);CHKERRQ(ierr);
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
    ierr = MPIU_File_write_all(mfdes,x->array,lsizes[0],MPIU_SCALAR,MPI_STATUS_IGNORE);CHKERRQ(ierr);
    ierr = PetscViewerBinaryAddMPIIOOffset(viewer,n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = MPI_Type_free(&view);CHKERRQ(ierr);    
  }
#endif

  ierr = PetscViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
  if (file && xin->map->bs > 1) {
    if (((PetscObject)xin)->prefix) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,file,"-%s_vecload_block_size %D\n",((PetscObject)xin)->prefix,xin->map->bs);CHKERRQ(ierr);
    } else {
      ierr = PetscFPrintf(PETSC_COMM_SELF,file,"-vecload_block_size %D\n",xin->map->bs);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_PNETCDF)
#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq_Netcdf"
PetscErrorCode VecView_Seq_Netcdf(Vec xin,PetscViewer v)
{
  PetscErrorCode ierr;
  int            n = xin->map->n,ncid,xdim,xdim_num=1,xin_id,xstart=0;
  PetscScalar    *xarray;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  ierr = VecGetArray(xin,&xarray);CHKERRQ(ierr);
  ierr = PetscViewerNetcdfGetID(v,&ncid);CHKERRQ(ierr);
  if (ncid < 0) SETERRQ(PETSC_ERR_ORDER,"First call PetscViewerNetcdfOpen to create NetCDF dataset");
  /* define dimensions */
  ierr = ncmpi_def_dim(ncid,"PETSc_Vector_Global_Size",n,&xdim);CHKERRQ(ierr);
  /* define variables */
  ierr = ncmpi_def_var(ncid,"PETSc_Vector_Seq",NC_DOUBLE,xdim_num,&xdim,&xin_id);CHKERRQ(ierr);
  /* leave define mode */
  ierr = ncmpi_enddef(ncid);CHKERRQ(ierr);
  /* store the vector */
  ierr = VecGetOwnershipRange(xin,&xstart,PETSC_NULL);CHKERRQ(ierr);
  ierr = ncmpi_put_vara_double_all(ncid,xin_id,(const MPI_Offset*)&xstart,(const MPI_Offset*)&n,xarray);CHKERRQ(ierr);
#else 
    PetscPrintf(PETSC_COMM_WORLD,"NetCDF viewer not supported for complex numbers\n");
#endif
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include "mat.h"   /* Matlab include file */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq_Matlab"
PetscErrorCode VecView_Seq_Matlab(Vec vec,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       n;
  PetscScalar    *array;
  
  PetscFunctionBegin;
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  ierr = PetscObjectName((PetscObject)vec);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  ierr = PetscViewerMatlabPutArray(viewer,n,1,array,((PetscObject)vec)->name);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif

#undef __FUNCT__  
#define __FUNCT__ "VecView_Seq"
PetscErrorCode VecView_Seq(Vec xin,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth     isdraw,iascii,issocket,isbinary;
#if defined(PETSC_HAVE_MATHEMATICA)
  PetscTruth     ismathematica;
#endif
#if defined(PETSC_HAVE_PNETCDF)
  PetscTruth     isnetcdf;
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscTruth     ismatlab;
#endif
#if defined(PETSC_HAVE_HDF5)
  PetscTruth     ishdf5;
#endif

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATHEMATICA)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_MATHEMATICA,&ismathematica);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_HDF5,&ishdf5);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PNETCDF)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_NETCDF,&isnetcdf);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_MATLAB,&ismatlab);CHKERRQ(ierr);
#endif

  if (isdraw){ 
    ierr = VecView_Seq_Draw(xin,viewer);CHKERRQ(ierr);
  } else if (iascii){
    ierr = VecView_Seq_File(xin,viewer);CHKERRQ(ierr);
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
#if defined(PETSC_HAVE_PNETCDF)
  } else if (isnetcdf) {
    ierr = VecView_Seq_Netcdf(xin,viewer);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  } else if (ismatlab) {
    ierr = VecView_Seq_Matlab(xin,viewer);CHKERRQ(ierr);
#endif
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by this vector object",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetValues_Seq"
PetscErrorCode VecGetValues_Seq(Vec xin,PetscInt ni,const PetscInt ix[],PetscScalar y[])
{
  Vec_Seq     *x = (Vec_Seq *)xin->data;
  PetscScalar *xx = x->array;
  PetscInt    i;

  PetscFunctionBegin;
  for (i=0; i<ni; i++) {
    if (xin->stash.ignorenegidx && ix[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
    if (ix[i] < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D cannot be negative",ix[i]);
    if (ix[i] >= xin->map->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D to large maximum allowed %D",ix[i],xin->map->n);
#endif
    y[i] = xx[ix[i]];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetValues_Seq"
PetscErrorCode VecSetValues_Seq(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode m)
{
  Vec_Seq     *x = (Vec_Seq *)xin->data;
  PetscScalar *xx = x->array;
  PetscInt    i;

  PetscFunctionBegin;
  if (m == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      if (xin->stash.ignorenegidx && ix[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
      if (ix[i] < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D cannot be negative",ix[i]);
      if (ix[i] >= xin->map->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",ix[i],xin->map->n);
#endif
      xx[ix[i]] = y[i];
    }
  } else {
    for (i=0; i<ni; i++) {
      if (xin->stash.ignorenegidx && ix[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
      if (ix[i] < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D cannot be negative",ix[i]);
      if (ix[i] >= xin->map->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",ix[i],xin->map->n);
#endif
      xx[ix[i]] += y[i];
    }  
  }  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetValuesBlocked_Seq"
PetscErrorCode VecSetValuesBlocked_Seq(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar yin[],InsertMode m)
{
  Vec_Seq     *x = (Vec_Seq *)xin->data;
  PetscScalar *xx = x->array,*y = (PetscScalar*)yin;
  PetscInt    i,bs = xin->map->bs,start,j;

  /*
       For optimization could treat bs = 2, 3, 4, 5 as special cases with loop unrolling
  */
  PetscFunctionBegin;
  if (m == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      start = bs*ix[i];
      if (start < 0) continue;
#if defined(PETSC_USE_DEBUG)
      if (start >= xin->map->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",start,xin->map->n);
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
      if (start >= xin->map->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",start,xin->map->n);
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
PetscErrorCode VecDestroy_Seq(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(v);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr);
  ierr = PetscFree(vs);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__  
#define __FUNCT__ "VecPublish_Seq"
static PetscErrorCode VecPublish_Seq(PetscObject obj)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

EXTERN PetscErrorCode VecLoad_Binary(PetscViewer, const VecType, Vec*);

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
            VecGetArray_Seq,
            VecGetSize_Seq,
            VecGetSize_Seq,
            VecRestoreArray_Seq,
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
            VecLoadIntoVector_Default,
            0, /* VecLoadIntoVectorNative */
            VecReciprocal_Default,
            0, /* VecViewNative */
            VecConjugate_Seq,
	    0,
	    0,
            VecResetArray_Seq,
            0,
            VecMaxPointwiseDivide_Seq,
            VecLoad_Binary, /* 50 */
            VecPointwiseMax_Seq,
            VecPointwiseMaxAbs_Seq,
            VecPointwiseMin_Seq,
            VecGetValues_Seq
          };


/*
      This is called by VecCreate_Seq() (i.e. VecCreateSeq()) and VecCreateSeqWithArray()
*/
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_Seq_Private"
static PetscErrorCode VecCreate_Seq_Private(Vec v,const PetscScalar array[])
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

  if (v->map->bs == -1) v->map->bs = 1;
  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECSEQ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
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
PetscErrorCode PETSCVEC_DLLEXPORT VecCreateSeqWithArray(MPI_Comm comm,PetscInt n,const PetscScalar array[],Vec *V)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = VecCreate(comm,V);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,n,n);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot create VECSEQ on more than one process");
  }
  ierr = VecCreate_Seq_Private(*V,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   VECSEQ - VECSEQ = "seq" - The basic sequential vector

   Options Database Keys:
. -vec_type seq - sets the vector type to VECSEQ during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateSeqWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateSeq()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_Seq"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Seq(Vec V)
{
  Vec_Seq        *s;
  PetscScalar    *array;
  PetscErrorCode ierr;
  PetscInt       n = PetscMax(V->map->n,V->map->N);
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)V)->comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot create VECSEQ on more than one process");
  }
  ierr = PetscMalloc(n*sizeof(PetscScalar),&array);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(V, n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(array,n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecCreate_Seq_Private(V,array);CHKERRQ(ierr);
  s    = (Vec_Seq*)V->data;
  s->array_allocated = array;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_Seq"
PetscErrorCode VecDuplicate_Seq(Vec win,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateSeq(((PetscObject)win)->comm,win->map->n,V);CHKERRQ(ierr);
  if (win->mapping) {
    ierr = PetscObjectReference((PetscObject)win->mapping);CHKERRQ(ierr);
    (*V)->mapping = win->mapping;
  }
  if (win->bmapping) {
    ierr = PetscObjectReference((PetscObject)win->bmapping);CHKERRQ(ierr);
    (*V)->bmapping = win->bmapping;
  }
  (*V)->map->bs = win->map->bs;
  ierr = PetscOListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);

  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetOption_Seq"
PetscErrorCode VecSetOption_Seq(Vec v,VecOption op,PetscTruth flag)
{
  PetscFunctionBegin;
  if (op == VEC_IGNORE_NEGATIVE_INDICES) {
    v->stash.ignorenegidx = flag;
  } 
  PetscFunctionReturn(0);
}
