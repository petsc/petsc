
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

static PetscErrorCode PetscViewerBinaryReadVecHeader_Private(PetscViewer viewer,PetscInt *rows)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt       tr[2],type;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  /* Read vector header */
  ierr = PetscViewerBinaryRead(viewer,tr,2,NULL,PETSC_INT);CHKERRQ(ierr);
  type = tr[0];
  if (type != VEC_FILE_CLASSID) {
    ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a vector next in file");
  }
  *rows = tr[1];
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode VecLoad_Binary_MPIIO(Vec vec, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    lsize;
  PetscScalar    *avec;
  MPI_File       mfdes;
  MPI_Offset     off;

  PetscFunctionBegin;
  ierr = VecGetArray(vec,&avec);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(vec->map->n,&lsize);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetMPIIODescriptor(viewer,&mfdes);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetMPIIOOffset(viewer,&off);CHKERRQ(ierr);
  off += vec->map->rstart*sizeof(PetscScalar);
  ierr = MPI_File_set_view(mfdes,off,MPIU_SCALAR,MPIU_SCALAR,(char*)"native",MPI_INFO_NULL);CHKERRQ(ierr);
  ierr = MPIU_File_read_all(mfdes,avec,lsize,MPIU_SCALAR,MPI_STATUS_IGNORE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryAddMPIIOOffset(viewer,vec->map->N*sizeof(PetscScalar));CHKERRQ(ierr);

  ierr = VecRestoreArray(vec,&avec);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode VecLoad_Binary(Vec vec, PetscViewer viewer)
{
  PetscMPIInt    size,rank,tag;
  int            fd;
  PetscInt       i,rows = 0,n,*range,N,bs;
  PetscErrorCode ierr;
  PetscBool      flag,skipheader;
  PetscScalar    *avec,*avecwork;
  MPI_Comm       comm;
  MPI_Request    request;
  MPI_Status     status;
#if defined(PETSC_HAVE_MPIIO)
  PetscBool      useMPIIO;
#endif

  PetscFunctionBegin;
  /* force binary viewer to load .info file if it has not yet done so */
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);
  if (!skipheader) {
    ierr = PetscViewerBinaryReadVecHeader_Private(viewer,&rows);CHKERRQ(ierr);
  } else {
    VecType vtype;
    ierr = VecGetType(vec,&vtype);CHKERRQ(ierr);
    if (!vtype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER, "Vector binary file header was skipped, thus the user must specify the type and length of input vector");
    ierr = VecGetSize(vec,&N);CHKERRQ(ierr);
    if (!N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER, "Vector binary file header was skipped, thus the user must specify the length of input vector");
    rows = N;
  }
  /* Set Vec sizes,blocksize,and type if not already set. Block size first so that local sizes will be compatible. */
  ierr = PetscOptionsGetInt(NULL,((PetscObject)vec)->prefix, "-vecload_block_size", &bs, &flag);CHKERRQ(ierr);
  if (flag) {
    ierr = VecSetBlockSize(vec, bs);CHKERRQ(ierr);
  }
  if (vec->map->n < 0 && vec->map->N < 0) {
    ierr = VecSetSizes(vec,PETSC_DECIDE,rows);CHKERRQ(ierr);
  }

  /* If sizes and type already set,check if the vector global size is correct */
  ierr = VecGetSize(vec, &N);CHKERRQ(ierr);
  if (N != rows) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Vector in file different length (%D) then input vector (%D)", rows, N);

#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscViewerBinaryGetUseMPIIO(viewer,&useMPIIO);CHKERRQ(ierr);
  if (useMPIIO) {
    ierr = VecLoad_Binary_MPIIO(vec, viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)viewer,&tag);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&avec);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscBinaryRead(fd,avec,n,NULL,PETSC_SCALAR);CHKERRQ(ierr);

    if (size > 1) {
      /* read in other chuncks and send to other processors */
      /* determine maximum chunck owned by other */
      range = vec->map->range;
      n = 1;
      for (i=1; i<size; i++) n = PetscMax(n,range[i+1] - range[i]);

      ierr = PetscMalloc1(n,&avecwork);CHKERRQ(ierr);
      for (i=1; i<size; i++) {
        n    = range[i+1] - range[i];
        ierr = PetscBinaryRead(fd,avecwork,n,NULL,PETSC_SCALAR);CHKERRQ(ierr);
        ierr = MPI_Isend(avecwork,n,MPIU_SCALAR,i,tag,comm,&request);CHKERRQ(ierr);
        ierr = MPI_Wait(&request,&status);CHKERRQ(ierr);
      }
      ierr = PetscFree(avecwork);CHKERRQ(ierr);
    }
  } else {
    ierr = MPI_Recv(avec,n,MPIU_SCALAR,0,tag,comm,&status);CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(vec,&avec);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode VecLoad_HDF5(Vec xin, PetscViewer viewer)
{
  hid_t          scalartype; /* scalar type (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  PetscScalar    *x;
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
  ierr = PetscViewerHDF5Load(viewer, vecname, xin->map, scalartype, (void**)&x);CHKERRQ(ierr);
  ierr = VecSetUp(xin);CHKERRQ(ierr);
  ierr = VecReplaceArray(xin, x);CHKERRQ(ierr);
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
