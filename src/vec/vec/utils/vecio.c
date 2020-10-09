
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
  PetscErrorCode    ierr;
  PetscBool         skipHeader;
  PetscLayout       map;
  PetscInt          tr[2],n,s,N;
  const PetscScalar *xarray;

  PetscFunctionBegin;
  PetscCheckSameComm(vec,1,viewer,2);
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipHeader);CHKERRQ(ierr);

  ierr = VecGetLayout(vec,&map);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(map,&n);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(map,&s,NULL);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(map,&N);CHKERRQ(ierr);

  tr[0] = VEC_FILE_CLASSID; tr[1] = N;
  if (!skipHeader) {ierr = PetscViewerBinaryWrite(viewer,tr,2,PETSC_INT);CHKERRQ(ierr);}

  ierr = VecGetArrayRead(vec,&xarray);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWriteAll(viewer,xarray,n,s,N,PETSC_SCALAR);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vec,&xarray);CHKERRQ(ierr);

  { /* write to the viewer's .info file */
    FILE             *info;
    PetscMPIInt       rank;
    PetscViewerFormat format;
    const char        *name,*pre;

    ierr = PetscViewerBinaryGetInfoPointer(viewer,&info);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)vec),&rank);CHKERRQ(ierr);

    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_BINARY_MATLAB) {
      ierr = PetscObjectGetName((PetscObject)vec,&name);CHKERRQ(ierr);
      if (!rank && info) {
        ierr = PetscFPrintf(PETSC_COMM_SELF,info,"#--- begin code written by PetscViewerBinary for MATLAB format ---#\n");CHKERRQ(ierr);
        ierr = PetscFPrintf(PETSC_COMM_SELF,info,"#$$ Set.%s = PetscBinaryRead(fd);\n",name);CHKERRQ(ierr);
        ierr = PetscFPrintf(PETSC_COMM_SELF,info,"#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n");CHKERRQ(ierr);
      }
    }

    ierr = PetscObjectGetOptionsPrefix((PetscObject)vec,&pre);CHKERRQ(ierr);
    if (!rank && info) {ierr = PetscFPrintf(PETSC_COMM_SELF,info,"-%svecload_block_size %D\n",pre?pre:"",PetscAbs(vec->map->bs));CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Binary(Vec vec, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      skipHeader,flg;
  PetscInt       tr[2],rows,N,n,s,bs;
  PetscScalar    *array;
  PetscLayout    map;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipHeader);CHKERRQ(ierr);

  ierr = VecGetLayout(vec,&map);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(map,&N);CHKERRQ(ierr);

  /* read vector header */
  if (!skipHeader) {
    ierr = PetscViewerBinaryRead(viewer,tr,2,NULL,PETSC_INT);CHKERRQ(ierr);
    if (tr[0] != VEC_FILE_CLASSID) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Not a vector next in file");
    if (tr[1] < 0) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Vector size (%D) in file is negative",tr[1]);
    if (N >= 0 && N != tr[1]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Vector in file different size (%D) than input vector (%D)",tr[1],N);
    rows = tr[1];
  } else {
    if (N < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Vector binary file header was skipped, thus the user must specify the global size of input vector");
    rows = N;
  }

  /* set vector size, blocksize, and type if not already set; block size first so that local sizes will be compatible. */
  ierr = PetscLayoutGetBlockSize(map,&bs);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(((PetscObject)viewer)->options,((PetscObject)vec)->prefix,"-vecload_block_size",&bs,&flg);CHKERRQ(ierr);
  if (flg) {ierr = VecSetBlockSize(vec,bs);CHKERRQ(ierr);}
  ierr = PetscLayoutGetLocalSize(map,&n);CHKERRQ(ierr);
  if (N < 0) {ierr = VecSetSizes(vec,n,rows);CHKERRQ(ierr);}
  ierr = VecSetUp(vec);CHKERRQ(ierr);

  /* get vector sizes and check global size */
  ierr = VecGetSize(vec,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(vec,&s,NULL);CHKERRQ(ierr);
  if (N != rows) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Vector in file different size (%D) than input vector (%D)",rows,N);

  /* read vector values */
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  ierr = PetscViewerBinaryReadAll(viewer,array,n,s,N,PETSC_SCALAR);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode VecLoad_HDF5(Vec xin, PetscViewer viewer)
{
  hid_t          scalartype; /* scalar type (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  PetscScalar    *x,*arr;
  const char     *vecname;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!((PetscObject)xin)->name) SETERRQ(PetscObjectComm((PetscObject)xin), PETSC_ERR_SUP, "Vec name must be set with PetscObjectSetName() before VecLoad()");
#if defined(PETSC_USE_REAL_SINGLE)
  scalartype = H5T_NATIVE_FLOAT;
#elif defined(PETSC_USE_REAL___FLOAT128)
#error "HDF5 output with 128 bit floats not supported."
#elif defined(PETSC_USE_REAL___FP16)
#error "HDF5 output with 16 bit floats not supported."
#else
  scalartype = H5T_NATIVE_DOUBLE;
#endif
  ierr = PetscObjectGetName((PetscObject)xin, &vecname);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Load(viewer,vecname,xin->map,scalartype,(void**)&x);CHKERRQ(ierr);
  ierr = VecSetUp(xin);CHKERRQ(ierr); /* VecSetSizes might have not been called so ensure ops->create has been called */
  if (!xin->ops->replacearray) {
    ierr = VecGetArray(xin,&arr);CHKERRQ(ierr);
    ierr = PetscArraycpy(arr,x,xin->map->n);CHKERRQ(ierr);
    ierr = PetscFree(x);CHKERRQ(ierr);
    ierr = VecRestoreArray(xin,&arr);CHKERRQ(ierr);
  } else {
    ierr = VecReplaceArray(xin,x);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscScalar       *x;
  PetscInt          Nfile,N,rstart,n;
  uint64_t          N_t,rstart_t;
  const char        *vecname;
  ADIOS_VARINFO     *v;
  ADIOS_SELECTION   *sel;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject) xin, &vecname);CHKERRQ(ierr);

  v    = adios_inq_var(adios->adios_fp, vecname);
  if (v->ndim != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Vector in file is not of dimension 1 (%D)", v->ndim);
  Nfile = (PetscInt) v->dims[0];

  /* Set Vec sizes,blocksize,and type if not already set */
  if ((xin)->map->n < 0 && (xin)->map->N < 0) {
    ierr = VecSetSizes(xin, PETSC_DECIDE, Nfile);CHKERRQ(ierr);
  }
  /* If sizes and type already set,check if the vector global size is correct */
  ierr = VecGetSize(xin, &N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(xin, &n);CHKERRQ(ierr);
  if (N != Nfile) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Vector in file different length (%D) then input vector (%D)", Nfile, N);

  ierr = VecGetOwnershipRange(xin,&rstart,NULL);CHKERRQ(ierr);
  rstart_t = rstart;
  N_t  = n;
  sel  = adios_selection_boundingbox (v->ndim, &rstart_t, &N_t);
  ierr = VecGetArray(xin,&x);CHKERRQ(ierr);
  adios_schedule_read(adios->adios_fp, sel, vecname, 0, 1, x);
  adios_perform_reads (adios->adios_fp, 1);
  ierr = VecRestoreArray(xin,&x);CHKERRQ(ierr);
  adios_selection_delete(sel);

  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_ADIOS2)
#include <adios2_c.h>
#include <petsc/private/vieweradios2impl.h>
#include <petsc/private/viewerimpl.h>

PetscErrorCode VecLoad_ADIOS2(Vec xin, PetscViewer viewer)
{
  PetscViewer_ADIOS2 *adios2 = (PetscViewer_ADIOS2*)viewer->data;
  PetscErrorCode     ierr;
  PetscScalar        *x;
  PetscInt           Nfile,N,rstart,n;
  const char         *vecname;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject) xin, &vecname);CHKERRQ(ierr);

  /* Set Vec sizes,blocksize,and type if not already set */
  if ((xin)->map->n < 0 && (xin)->map->N < 0) {
    ierr = VecSetSizes(xin, PETSC_DECIDE, Nfile);CHKERRQ(ierr);
  }
  /* If sizes and type already set,check if the vector global size is correct */
  ierr = VecGetSize(xin, &N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(xin, &n);CHKERRQ(ierr);
  if (N != Nfile) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Vector in file different length (%D) then input vector (%D)", Nfile, N);

  ierr = VecGetOwnershipRange(xin,&rstart,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(xin,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(xin,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode  VecLoad_Default(Vec newvec, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;
#if defined(PETSC_HAVE_HDF5)
  PetscBool      ishdf5;
#endif
#if defined(PETSC_HAVE_ADIOS)
  PetscBool      isadios;
#endif
#if defined(PETSC_HAVE_ADIOS2)
  PetscBool      isadios2;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ADIOS)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS,&isadios);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ADIOS2)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS2,&isadios2);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_HDF5)
  if (ishdf5) {
    if (!((PetscObject)newvec)->name) {
      ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Since HDF5 format gives ASCII name for each object in file; must use VecLoad() after setting name of Vec with PetscObjectSetName()");
    }
    ierr = VecLoad_HDF5(newvec, viewer);CHKERRQ(ierr);
  } else
#endif
#if defined(PETSC_HAVE_ADIOS)
  if (isadios) {
    if (!((PetscObject)newvec)->name) {
      ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Since ADIOS format gives ASCII name for each object in file; must use VecLoad() after setting name of Vec with PetscObjectSetName()");
    }
    ierr = VecLoad_ADIOS(newvec, viewer);CHKERRQ(ierr);
  } else
#endif
#if defined(PETSC_HAVE_ADIOS2)
  if (isadios2) {
    if (!((PetscObject)newvec)->name) {
      ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Since ADIOS2 format gives ASCII name for each object in file; must use VecLoad() after setting name of Vec with PetscObjectSetName()");
    }
    ierr = VecLoad_ADIOS2(newvec, viewer);CHKERRQ(ierr);
  } else
#endif
  {
    ierr = VecLoad_Binary(newvec, viewer);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  for (i = 0; i < n; ++i) {
    if (PetscAbsScalar(a[i]) < tol) a[i] = 0.0;
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
