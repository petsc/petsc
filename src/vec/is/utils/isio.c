#include <petscis.h>         /*I  "petscis.h"  I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/viewerimpl.h>
#include <petsclayouthdf5.h>

#if defined(PETSC_HAVE_HDF5)
/*
     This should handle properly the cases where PetscInt is 32 or 64 and hsize_t is 32 or 64. These means properly casting with
   checks back and forth between the two types of variables.
*/
PetscErrorCode ISLoad_HDF5(IS is, PetscViewer viewer)
{
  hid_t           inttype;    /* int type (H5T_NATIVE_INT or H5T_NATIVE_LLONG) */
  PetscInt       *ind;
  const char     *isname;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!((PetscObject)is)->name) SETERRQ(PetscObjectComm((PetscObject)is), PETSC_ERR_SUP, "Since HDF5 format gives ASCII name for each object in file; must use ISLoad() after setting name of Vec with PetscObjectSetName()");
#if defined(PETSC_USE_64BIT_INDICES)
  inttype = H5T_NATIVE_LLONG;
#else
  inttype = H5T_NATIVE_INT;
#endif
  ierr = PetscObjectGetName((PetscObject)is, &isname);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Load(viewer, isname, is->map, inttype, (void**)&ind);CHKERRQ(ierr);
  ierr = ISGeneralSetIndices(is, is->map->n, ind, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode ISLoad_Binary(IS is, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isgeneral,skipHeader,useMPIIO;
  int            fd;
  PetscInt       tr[2],N,ln,*idx;
  MPI_Request    request;
  MPI_Status     status;
  MPI_Comm       comm;
  PetscMPIInt    rank,size,tag;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)is,&comm);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)is,ISGENERAL,&isgeneral);CHKERRQ(ierr);
  if (!isgeneral) SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"IS must be of type ISGENERAL to load into it");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipHeader);CHKERRQ(ierr);
  if (skipHeader) SETERRQ(comm,PETSC_ERR_USER, "Currently no support for binary files without headers");
  /* force binary viewer to load .info file if it has not yet done so */
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryRead(viewer,tr,2,NULL,PETSC_INT);CHKERRQ(ierr);
  if (tr[0] != IS_FILE_CLASSID) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Not an IS next in file");

  /* Has IS already had its layout defined */
  /* ierr = ISGetLayout(is,&map);CHKERRQ(ierr); */
  ierr = PetscLayoutGetSize(is->map,&N);CHKERRQ(ierr);
  if (N > -1 && N != tr[1]) SETERRQ2(comm,PETSC_ERR_ARG_SIZ,"Size of IS in file %D does not match size of IS provided",tr[1],N);
  if (N == -1) {
    N = tr[1];
    ierr = PetscLayoutSetSize(is->map,N);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(is->map);CHKERRQ(ierr);
  }
  ierr = PetscLayoutGetLocalSize(is->map,&ln);CHKERRQ(ierr);
  ierr = PetscMalloc1(ln,&idx);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetUseMPIIO(viewer,&useMPIIO);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  if (useMPIIO) {
    MPI_File    mfdes;
    MPI_Offset  off;
    PetscMPIInt lsize;
    PetscInt    rstart;

    ierr = PetscMPIIntCast(ln,&lsize);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetMPIIODescriptor(viewer,&mfdes);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetMPIIOOffset(viewer,&off);CHKERRQ(ierr);
    ierr = PetscLayoutGetRange(is->map,&rstart,NULL);CHKERRQ(ierr);
    off += rstart*(MPI_Offset)sizeof(PetscInt);
    ierr = MPI_File_set_view(mfdes,off,MPIU_INT,MPIU_INT,(char*)"native",MPI_INFO_NULL);CHKERRQ(ierr);
    ierr = MPIU_File_read_all(mfdes,idx,lsize,MPIU_INT,MPI_STATUS_IGNORE);CHKERRQ(ierr);
    ierr = PetscViewerBinaryAddMPIIOOffset(viewer,N*(MPI_Offset)sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISGeneralSetIndices(is,ln,idx,PETSC_OWN_POINTER);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)viewer,&tag);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);

  if (!rank) {
    ierr = PetscBinaryRead(fd,idx,ln,NULL,PETSC_INT);CHKERRQ(ierr);

    if (size > 1) {
      PetscInt *range,n,i,*idxwork;

      /* read in other chuncks and send to other processors */
      /* determine maximum chunck owned by other */
      range = is->map->range;
      n = 1;
      for (i=1; i<size; i++) n = PetscMax(n,range[i+1] - range[i]);

      ierr = PetscMalloc1(n,&idxwork);CHKERRQ(ierr);
      for (i=1; i<size; i++) {
        n    = range[i+1] - range[i];
        ierr = PetscBinaryRead(fd,idxwork,n,NULL,PETSC_INT);CHKERRQ(ierr);
        ierr = MPI_Isend(idxwork,n,MPIU_INT,i,tag,comm,&request);CHKERRQ(ierr);
        ierr = MPI_Wait(&request,&status);CHKERRQ(ierr);
      }
      ierr = PetscFree(idxwork);CHKERRQ(ierr);
    }
  } else {
    ierr = MPI_Recv(idx,ln,MPIU_INT,0,tag,comm,&status);CHKERRQ(ierr);
  }
  ierr = ISGeneralSetIndices(is,ln,idx,PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ISLoad_Default(IS is, PetscViewer viewer)
{
  PetscBool      isbinary;
#if defined(PETSC_HAVE_HDF5)
  PetscBool      ishdf5;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5);CHKERRQ(ierr);
#endif
  if (isbinary) {
    ierr = ISLoad_Binary(is, viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    ierr = ISLoad_HDF5(is, viewer);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}
