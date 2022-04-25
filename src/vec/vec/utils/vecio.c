
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
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscViewerBinaryGetSkipHeader(viewer,&skipHeader));

  PetscCall(VecGetLayout(vec,&map));
  PetscCall(PetscLayoutGetLocalSize(map,&n));
  PetscCall(PetscLayoutGetRange(map,&s,NULL));
  PetscCall(PetscLayoutGetSize(map,&N));

  tr[0] = VEC_FILE_CLASSID; tr[1] = N;
  if (!skipHeader) PetscCall(PetscViewerBinaryWrite(viewer,tr,2,PETSC_INT));

  PetscCall(VecGetArrayRead(vec,&xarray));
  PetscCall(PetscViewerBinaryWriteAll(viewer,xarray,n,s,N,PETSC_SCALAR));
  PetscCall(VecRestoreArrayRead(vec,&xarray));

  { /* write to the viewer's .info file */
    FILE             *info;
    PetscMPIInt       rank;
    PetscViewerFormat format;
    const char        *name,*pre;

    PetscCall(PetscViewerBinaryGetInfoPointer(viewer,&info));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)vec),&rank));

    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_BINARY_MATLAB) {
      PetscCall(PetscObjectGetName((PetscObject)vec,&name));
      if (rank == 0 && info) {
        PetscCall(PetscFPrintf(PETSC_COMM_SELF,info,"#--- begin code written by PetscViewerBinary for MATLAB format ---#\n"));
        PetscCall(PetscFPrintf(PETSC_COMM_SELF,info,"#$$ Set.%s = PetscBinaryRead(fd);\n",name));
        PetscCall(PetscFPrintf(PETSC_COMM_SELF,info,"#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n"));
      }
    }

    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)vec,&pre));
    if (rank == 0 && info) PetscCall(PetscFPrintf(PETSC_COMM_SELF,info,"-%svecload_block_size %" PetscInt_FMT "\n",pre?pre:"",PetscAbs(vec->map->bs)));
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
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscViewerBinaryGetSkipHeader(viewer,&skipHeader));

  PetscCall(VecGetLayout(vec,&map));
  PetscCall(PetscLayoutGetSize(map,&N));

  /* read vector header */
  if (!skipHeader) {
    PetscCall(PetscViewerBinaryRead(viewer,tr,2,NULL,PETSC_INT));
    PetscCheck(tr[0] == VEC_FILE_CLASSID,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Not a vector next in file");
    PetscCheck(tr[1] >= 0,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Vector size (%" PetscInt_FMT ") in file is negative",tr[1]);
    PetscCheck(N < 0 || N == tr[1],PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Vector in file different size (%" PetscInt_FMT ") than input vector (%" PetscInt_FMT ")",tr[1],N);
    rows = tr[1];
  } else {
    PetscCheck(N >= 0,PETSC_COMM_SELF,PETSC_ERR_USER,"Vector binary file header was skipped, thus the user must specify the global size of input vector");
    rows = N;
  }

  /* set vector size, blocksize, and type if not already set; block size first so that local sizes will be compatible. */
  PetscCall(PetscLayoutGetBlockSize(map,&bs));
  PetscCall(PetscOptionsGetInt(((PetscObject)viewer)->options,((PetscObject)vec)->prefix,"-vecload_block_size",&bs,&flg));
  if (flg) PetscCall(VecSetBlockSize(vec,bs));
  PetscCall(PetscLayoutGetLocalSize(map,&n));
  if (N < 0) PetscCall(VecSetSizes(vec,n,rows));
  PetscCall(VecSetUp(vec));

  /* get vector sizes and check global size */
  PetscCall(VecGetSize(vec,&N));
  PetscCall(VecGetLocalSize(vec,&n));
  PetscCall(VecGetOwnershipRange(vec,&s,NULL));
  PetscCheck(N == rows,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Vector in file different size (%" PetscInt_FMT ") than input vector (%" PetscInt_FMT ")",rows,N);

  /* read vector values */
  PetscCall(VecGetArray(vec,&array));
  PetscCall(PetscViewerBinaryReadAll(viewer,array,n,s,N,PETSC_SCALAR));
  PetscCall(VecRestoreArray(vec,&array));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode VecLoad_HDF5(Vec xin, PetscViewer viewer)
{
  hid_t          scalartype; /* scalar type (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  PetscScalar    *x,*arr;
  const char     *vecname;

  PetscFunctionBegin;
  PetscCheck(((PetscObject)xin)->name,PetscObjectComm((PetscObject)xin), PETSC_ERR_SUP, "Vec name must be set with PetscObjectSetName() before VecLoad()");
#if defined(PETSC_USE_REAL_SINGLE)
  scalartype = H5T_NATIVE_FLOAT;
#elif defined(PETSC_USE_REAL___FLOAT128)
#error "HDF5 output with 128 bit floats not supported."
#elif defined(PETSC_USE_REAL___FP16)
#error "HDF5 output with 16 bit floats not supported."
#else
  scalartype = H5T_NATIVE_DOUBLE;
#endif
  PetscCall(PetscObjectGetName((PetscObject)xin, &vecname));
  PetscCall(PetscViewerHDF5Load(viewer,vecname,xin->map,scalartype,(void**)&x));
  PetscCall(VecSetUp(xin)); /* VecSetSizes might have not been called so ensure ops->create has been called */
  if (!xin->ops->replacearray) {
    PetscCall(VecGetArray(xin,&arr));
    PetscCall(PetscArraycpy(arr,x,xin->map->n));
    PetscCall(PetscFree(x));
    PetscCall(VecRestoreArray(xin,&arr));
  } else {
    PetscCall(VecReplaceArray(xin,x));
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
  PetscCall(PetscObjectGetName((PetscObject) xin, &vecname));

  v    = adios_inq_var(adios->adios_fp, vecname);
  PetscCheck(v->ndim == 1,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Vector in file is not of dimension 1 (%" PetscInt_FMT ")", v->ndim);
  Nfile = (PetscInt) v->dims[0];

  /* Set Vec sizes,blocksize,and type if not already set */
  if ((xin)->map->n < 0 && (xin)->map->N < 0) PetscCall(VecSetSizes(xin, PETSC_DECIDE, Nfile));
  /* If sizes and type already set,check if the vector global size is correct */
  PetscCall(VecGetSize(xin, &N));
  PetscCall(VecGetLocalSize(xin, &n));
  PetscCheck(N == Nfile,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Vector in file different length (%" PetscInt_FMT ") then input vector (%" PetscInt_FMT ")", Nfile, N);

  PetscCall(VecGetOwnershipRange(xin,&rstart,NULL));
  rstart_t = rstart;
  N_t  = n;
  sel  = adios_selection_boundingbox (v->ndim, &rstart_t, &N_t);
  PetscCall(VecGetArray(xin,&x));
  adios_schedule_read(adios->adios_fp, sel, vecname, 0, 1, x);
  adios_perform_reads (adios->adios_fp, 1);
  PetscCall(VecRestoreArray(xin,&x));
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
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
#endif
#if defined(PETSC_HAVE_ADIOS)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS,&isadios));
#endif

#if defined(PETSC_HAVE_HDF5)
  if (ishdf5) {
    if (!((PetscObject)newvec)->name) {
      PetscCall(PetscLogEventEnd(VEC_Load,viewer,0,0,0));
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Since HDF5 format gives ASCII name for each object in file; must use VecLoad() after setting name of Vec with PetscObjectSetName()");
    }
    PetscCall(VecLoad_HDF5(newvec, viewer));
  } else
#endif
#if defined(PETSC_HAVE_ADIOS)
  if (isadios) {
    if (!((PetscObject)newvec)->name) {
      PetscCall(PetscLogEventEnd(VEC_Load,viewer,0,0,0));
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Since ADIOS format gives ASCII name for each object in file; must use VecLoad() after setting name of Vec with PetscObjectSetName()");
    }
    PetscCall(VecLoad_ADIOS(newvec, viewer));
  } else
#endif
  {
    PetscCall(VecLoad_Binary(newvec, viewer));
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

.seealso: `VecCreate()`, `VecSet()`
@*/
PetscErrorCode VecChop(Vec v, PetscReal tol)
{
  PetscScalar    *a;
  PetscInt       n, i;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetArray(v, &a));
  for (i = 0; i < n; ++i) {
    if (PetscAbsScalar(a[i]) < tol) a[i] = 0.0;
  }
  PetscCall(VecRestoreArray(v, &a));
  PetscFunctionReturn(0);
}
