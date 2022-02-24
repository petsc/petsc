
/*
   This file contains simple binary input routines for vectors.  The
   analogous output routines are within each vector implementation's
   VecView (with viewer types PETSCVIEWERBINARY)
 */

#include <petscsys.h>
#include <petscvec.h>         /*I  "petscvec.h"  I*/
#include <petsc/private/vecimpl.h>
#include <petsc/private/viewerimpl.h>
#include <petsclayouthdf5.h>

PetscErrorCode VecView_Binary(Vec vec,PetscViewer viewer)
{
  PetscBool         skipHeader;
  PetscLayout       map;
  PetscInt          tr[2],n,s,N;
  const PetscScalar *xarray;

  PetscFunctionBegin;
  PetscCheckSameComm(vec,1,viewer,2);
  CHKERRQ(PetscViewerSetUp(viewer));
  CHKERRQ(PetscViewerBinaryGetSkipHeader(viewer,&skipHeader));

  CHKERRQ(VecGetLayout(vec,&map));
  CHKERRQ(PetscLayoutGetLocalSize(map,&n));
  CHKERRQ(PetscLayoutGetRange(map,&s,NULL));
  CHKERRQ(PetscLayoutGetSize(map,&N));

  tr[0] = VEC_FILE_CLASSID; tr[1] = N;
  if (!skipHeader) CHKERRQ(PetscViewerBinaryWrite(viewer,tr,2,PETSC_INT));

  CHKERRQ(VecGetArrayRead(vec,&xarray));
  CHKERRQ(PetscViewerBinaryWriteAll(viewer,xarray,n,s,N,PETSC_SCALAR));
  CHKERRQ(VecRestoreArrayRead(vec,&xarray));

  { /* write to the viewer's .info file */
    FILE             *info;
    PetscMPIInt       rank;
    PetscViewerFormat format;
    const char        *name,*pre;

    CHKERRQ(PetscViewerBinaryGetInfoPointer(viewer,&info));
    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)vec),&rank));

    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_BINARY_MATLAB) {
      CHKERRQ(PetscObjectGetName((PetscObject)vec,&name));
      if (rank == 0 && info) {
        CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,info,"#--- begin code written by PetscViewerBinary for MATLAB format ---#\n"));
        CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,info,"#$$ Set.%s = PetscBinaryRead(fd);\n",name));
        CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,info,"#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n"));
      }
    }

    CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)vec,&pre));
    if (rank == 0 && info) CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,info,"-%svecload_block_size %" PetscInt_FMT "\n",pre?pre:"",PetscAbs(vec->map->bs)));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Binary(Vec vec, PetscViewer viewer)
{
  PetscBool      skipHeader,flg;
  PetscInt       tr[2],rows,N,n,s,bs;
  PetscScalar    *array;
  PetscLayout    map;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerSetUp(viewer));
  CHKERRQ(PetscViewerBinaryGetSkipHeader(viewer,&skipHeader));

  CHKERRQ(VecGetLayout(vec,&map));
  CHKERRQ(PetscLayoutGetSize(map,&N));

  /* read vector header */
  if (!skipHeader) {
    CHKERRQ(PetscViewerBinaryRead(viewer,tr,2,NULL,PETSC_INT));
    PetscCheckFalse(tr[0] != VEC_FILE_CLASSID,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Not a vector next in file");
    PetscCheckFalse(tr[1] < 0,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Vector size (%" PetscInt_FMT ") in file is negative",tr[1]);
    PetscCheckFalse(N >= 0 && N != tr[1],PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Vector in file different size (%" PetscInt_FMT ") than input vector (%" PetscInt_FMT ")",tr[1],N);
    rows = tr[1];
  } else {
    PetscCheckFalse(N < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"Vector binary file header was skipped, thus the user must specify the global size of input vector");
    rows = N;
  }

  /* set vector size, blocksize, and type if not already set; block size first so that local sizes will be compatible. */
  CHKERRQ(PetscLayoutGetBlockSize(map,&bs));
  CHKERRQ(PetscOptionsGetInt(((PetscObject)viewer)->options,((PetscObject)vec)->prefix,"-vecload_block_size",&bs,&flg));
  if (flg) CHKERRQ(VecSetBlockSize(vec,bs));
  CHKERRQ(PetscLayoutGetLocalSize(map,&n));
  if (N < 0) CHKERRQ(VecSetSizes(vec,n,rows));
  CHKERRQ(VecSetUp(vec));

  /* get vector sizes and check global size */
  CHKERRQ(VecGetSize(vec,&N));
  CHKERRQ(VecGetLocalSize(vec,&n));
  CHKERRQ(VecGetOwnershipRange(vec,&s,NULL));
  PetscCheckFalse(N != rows,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Vector in file different size (%" PetscInt_FMT ") than input vector (%" PetscInt_FMT ")",rows,N);

  /* read vector values */
  CHKERRQ(VecGetArray(vec,&array));
  CHKERRQ(PetscViewerBinaryReadAll(viewer,array,n,s,N,PETSC_SCALAR));
  CHKERRQ(VecRestoreArray(vec,&array));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode VecLoad_HDF5(Vec xin, PetscViewer viewer)
{
  hid_t          scalartype; /* scalar type (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  PetscScalar    *x,*arr;
  const char     *vecname;

  PetscFunctionBegin;
  PetscCheckFalse(!((PetscObject)xin)->name,PetscObjectComm((PetscObject)xin), PETSC_ERR_SUP, "Vec name must be set with PetscObjectSetName() before VecLoad()");
#if defined(PETSC_USE_REAL_SINGLE)
  scalartype = H5T_NATIVE_FLOAT;
#elif defined(PETSC_USE_REAL___FLOAT128)
#error "HDF5 output with 128 bit floats not supported."
#elif defined(PETSC_USE_REAL___FP16)
#error "HDF5 output with 16 bit floats not supported."
#else
  scalartype = H5T_NATIVE_DOUBLE;
#endif
  CHKERRQ(PetscObjectGetName((PetscObject)xin, &vecname));
  CHKERRQ(PetscViewerHDF5Load(viewer,vecname,xin->map,scalartype,(void**)&x));
  CHKERRQ(VecSetUp(xin)); /* VecSetSizes might have not been called so ensure ops->create has been called */
  if (!xin->ops->replacearray) {
    CHKERRQ(VecGetArray(xin,&arr));
    CHKERRQ(PetscArraycpy(arr,x,xin->map->n));
    CHKERRQ(PetscFree(x));
    CHKERRQ(VecRestoreArray(xin,&arr));
  } else {
    CHKERRQ(VecReplaceArray(xin,x));
  }
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_ADIOS)
#include <adios.h>
#include <adios_read.h>
#include <petsc/private/vieweradiosimpl.h>
#include <petsc/private/viewerimpl.h>

PetscErrorCode VecLoad_ADIOS(Vec xin, PetscViewer viewer)
{
  PetscViewer_ADIOS *adios = (PetscViewer_ADIOS*)viewer->data;
  PetscScalar       *x;
  PetscInt          Nfile,N,rstart,n;
  uint64_t          N_t,rstart_t;
  const char        *vecname;
  ADIOS_VARINFO     *v;
  ADIOS_SELECTION   *sel;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject) xin, &vecname));

  v    = adios_inq_var(adios->adios_fp, vecname);
  PetscCheckFalse(v->ndim != 1,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Vector in file is not of dimension 1 (%" PetscInt_FMT ")", v->ndim);
  Nfile = (PetscInt) v->dims[0];

  /* Set Vec sizes,blocksize,and type if not already set */
  if ((xin)->map->n < 0 && (xin)->map->N < 0) CHKERRQ(VecSetSizes(xin, PETSC_DECIDE, Nfile));
  /* If sizes and type already set,check if the vector global size is correct */
  CHKERRQ(VecGetSize(xin, &N));
  CHKERRQ(VecGetLocalSize(xin, &n));
  PetscCheckFalse(N != Nfile,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Vector in file different length (%" PetscInt_FMT ") then input vector (%" PetscInt_FMT ")", Nfile, N);

  CHKERRQ(VecGetOwnershipRange(xin,&rstart,NULL));
  rstart_t = rstart;
  N_t  = n;
  sel  = adios_selection_boundingbox (v->ndim, &rstart_t, &N_t);
  CHKERRQ(VecGetArray(xin,&x));
  adios_schedule_read(adios->adios_fp, sel, vecname, 0, 1, x);
  adios_perform_reads (adios->adios_fp, 1);
  CHKERRQ(VecRestoreArray(xin,&x));
  adios_selection_delete(sel);

  PetscFunctionReturn(0);
}
#endif

PetscErrorCode  VecLoad_Default(Vec newvec, PetscViewer viewer)
{
  PetscBool      isbinary;
#if defined(PETSC_HAVE_HDF5)
  PetscBool      ishdf5;
#endif
#if defined(PETSC_HAVE_ADIOS)
  PetscBool      isadios;
#endif

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
#if defined(PETSC_HAVE_HDF5)
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
#endif
#if defined(PETSC_HAVE_ADIOS)
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS,&isadios));
#endif

#if defined(PETSC_HAVE_HDF5)
  if (ishdf5) {
    if (!((PetscObject)newvec)->name) {
      CHKERRQ(PetscLogEventEnd(VEC_Load,viewer,0,0,0));
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Since HDF5 format gives ASCII name for each object in file; must use VecLoad() after setting name of Vec with PetscObjectSetName()");
    }
    CHKERRQ(VecLoad_HDF5(newvec, viewer));
  } else
#endif
#if defined(PETSC_HAVE_ADIOS)
  if (isadios) {
    if (!((PetscObject)newvec)->name) {
      CHKERRQ(PetscLogEventEnd(VEC_Load,viewer,0,0,0));
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Since ADIOS format gives ASCII name for each object in file; must use VecLoad() after setting name of Vec with PetscObjectSetName()");
    }
    CHKERRQ(VecLoad_ADIOS(newvec, viewer));
  } else
#endif
  {
    CHKERRQ(VecLoad_Binary(newvec, viewer));
  }
  PetscFunctionReturn(0);
}

/*@
  VecChop - Set all values in the vector with an absolute value less than the tolerance to zero

  Input Parameters:
+ v   - The vector
- tol - The zero tolerance

  Output Parameters:
. v - The chopped vector

  Level: intermediate

.seealso: VecCreate(), VecSet()
@*/
PetscErrorCode VecChop(Vec v, PetscReal tol)
{
  PetscScalar    *a;
  PetscInt       n, i;

  PetscFunctionBegin;
  CHKERRQ(VecGetLocalSize(v, &n));
  CHKERRQ(VecGetArray(v, &a));
  for (i = 0; i < n; ++i) {
    if (PetscAbsScalar(a[i]) < tol) a[i] = 0.0;
  }
  CHKERRQ(VecRestoreArray(v, &a));
  PetscFunctionReturn(0);
}
