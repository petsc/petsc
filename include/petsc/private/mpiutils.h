#if !defined(MPIUTILS_H)
#define MPIUTILS_H

#include <petscsys.h>

#if !defined(PETSC_HAVE_MPI_LARGE_COUNT) /* No matter PetscInt is 32-bit or 64-bit, without MPI large count we always do casting before MPI calls */
  /* Cast PetscInt <a> to PetscMPIInt <b>, where <a> is likely used for the 'count' argument in MPI routines.
    It is similar to PetscMPIIntCast() execept that here it returns an MPI error code.
  */
  PETSC_STATIC_INLINE PetscMPIInt PetscMPIIntCast_Internal(PetscInt a,PetscMPIInt *b)
  {
   #if defined(PETSC_USE_64BIT_INDICES)
    if (a > PETSC_MPI_INT_MAX) return MPI_ERR_COUNT;
   #endif
    *b = (PetscMPIInt)(a);
    return MPI_SUCCESS;
  }

  PETSC_STATIC_INLINE PetscMPIInt MPIU_Send(const void *buf,PetscInt count,MPI_Datatype datatype,PetscMPIInt dest,PetscMPIInt tag,MPI_Comm comm)
  {
    PetscMPIInt  ierr,count2;
    ierr = PetscMPIIntCast_Internal(count,&count2); if (ierr) return ierr;
    ierr = MPI_Send(buf,count2,datatype,dest,tag,comm); return ierr;
  }

  PETSC_STATIC_INLINE PetscMPIInt MPIU_Send_init(const void *buf,PetscInt count,MPI_Datatype datatype,PetscMPIInt dest,PetscMPIInt tag,MPI_Comm comm,MPI_Request *request)
  {
    PetscMPIInt  ierr,count2;
    ierr = PetscMPIIntCast_Internal(count,&count2); if (ierr) return ierr;
    ierr = MPI_Send_init(buf,count2,datatype,dest,tag,comm,request); return ierr;
  }

  PETSC_STATIC_INLINE PetscMPIInt MPIU_Isend(const void *buf,PetscInt count,MPI_Datatype datatype,PetscMPIInt dest,PetscMPIInt tag,MPI_Comm comm,MPI_Request *request)
  {
    PetscMPIInt  ierr,count2;
    ierr = PetscMPIIntCast_Internal(count,&count2); if (ierr) return ierr;
    ierr = MPI_Isend(buf,count2,datatype,dest,tag,comm,request); return ierr;
  }

  PETSC_STATIC_INLINE PetscMPIInt MPIU_Recv(void *buf,PetscInt count,MPI_Datatype datatype,PetscMPIInt source,PetscMPIInt tag,MPI_Comm comm,MPI_Status *status)
  {
    PetscMPIInt  ierr,count2;
    ierr = PetscMPIIntCast_Internal(count,&count2); if (ierr) return ierr;
    ierr = MPI_Recv(buf,count2,datatype,source,tag,comm,status); return ierr;
  }

  PETSC_STATIC_INLINE PetscMPIInt MPIU_Recv_init(void *buf,PetscInt count,MPI_Datatype datatype,PetscMPIInt source,PetscMPIInt tag,MPI_Comm comm,MPI_Request* request)
  {
    PetscMPIInt  ierr,count2;
    ierr = PetscMPIIntCast_Internal(count,&count2); if (ierr) return ierr;
    ierr = MPI_Recv_init(buf,count2,datatype,source,tag,comm,request); return ierr;
  }

  PETSC_STATIC_INLINE PetscMPIInt MPIU_Irecv(void *buf,PetscInt count,MPI_Datatype datatype,PetscMPIInt source,PetscMPIInt tag,MPI_Comm comm,MPI_Request* request)
  {
    PetscMPIInt  ierr,count2;
    ierr = PetscMPIIntCast_Internal(count,&count2); if (ierr) return ierr;
    ierr = MPI_Irecv(buf,count2,datatype,source,tag,comm,request); return ierr;
  }
 #if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
  PETSC_STATIC_INLINE PetscMPIInt MPIU_Reduce_local(const void *inbuf,void *inoutbuf,PetscInt count,MPI_Datatype datatype,MPI_Op op)
  {
    PetscMPIInt  ierr,count2;
    ierr = PetscMPIIntCast_Internal(count,&count2); if (ierr) return ierr;
    ierr = MPI_Reduce_local(inbuf,inoutbuf,count,datatype,op); return ierr;
  }
 #endif

#elif defined(PETSC_USE_64BIT_INDICES)
  #define MPIU_Send(buf,count,datatype,dest,tag,comm)                         MPI_Send_c(buf,count,datatype,dest,tag,comm)
  #define MPIU_Send_init(buf,count,datatype,dest,tag,comm,request)            MPI_Send_init_c(buf,count,datatype,dest,tag,comm,request)
  #define MPIU_Isend(buf,count,datatype,dest,tag,comm,request)                MPI_Isend_c(buf,count,datatype,dest,tag,comm,request)
  #define MPIU_Recv(buf,count,datatype,source,tag,comm,status)                MPI_Recv_c(buf,count,datatype,source,tag,comm,status)
  #define MPIU_Recv_init(buf,count,datatype,source,tag,comm,request)          MPI_Recv_init_c(buf,count,datatype,source,tag,comm,request)
  #define MPIU_Irecv(buf,count,datatype,source,tag,comm,request)              MPI_Irecv_c(buf,count,datatype,source,tag,comm,request)
 #if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
  #define MPIU_Reduce_local(inbuf,inoutbuf,count,datatype,op)                 MPI_Reduce_local_c(inbuf,inoutbuf,count,datatype,op)
 #endif
#else
  #define MPIU_Send(buf,count,datatype,dest,tag,comm)                         MPI_Send(buf,count,datatype,dest,tag,comm)
  #define MPIU_Send_init(buf,count,datatype,dest,tag,comm,request)            MPI_Send_init(buf,count,datatype,dest,tag,comm,request)
  #define MPIU_Isend(buf,count,datatype,dest,tag,comm,request)                MPI_Isend(buf,count,datatype,dest,tag,comm,request)
  #define MPIU_Recv(buf,count,datatype,source,tag,comm,status)                MPI_Recv(buf,count,datatype,source,tag,comm,status)
  #define MPIU_Recv_init(buf,count,datatype,source,tag,comm,request)          MPI_Recv_init(buf,count,datatype,source,tag,comm,request)
  #define MPIU_Irecv(buf,count,datatype,source,tag,comm,request)              MPI_Irecv(buf,count,datatype,source,tag,comm,request)
 #if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
  #define MPIU_Reduce_local(inbuf,inoutbuf,count,datatype,op)                 MPI_Reduce_local(inbuf,inoutbuf,count,datatype,op)
 #endif
#endif

/* These APIs use arrays of MPI_Count/MPI_Aint */
#if defined(PETSC_HAVE_MPI_LARGE_COUNT) && defined(PETSC_USE_64BIT_INDICES)
  #define MPIU_Neighbor_alltoallv(a,b,c,d,e,f,g,h,i)                          MPI_Neighbor_alltoallv_c(a,b,c,d,e,f,g,h,i)
  #define MPIU_Ineighbor_alltoallv(a,b,c,d,e,f,g,h,i,j)                       MPI_Ineighbor_alltoallv_c(a,b,c,d,e,f,g,h,i,j)
#else
  #define MPIU_Neighbor_alltoallv(a,b,c,d,e,f,g,h,i)                          MPI_Neighbor_alltoallv(a,b,c,d,e,f,g,h,i)
  #define MPIU_Ineighbor_alltoallv(a,b,c,d,e,f,g,h,i,j)                       MPI_Ineighbor_alltoallv(a,b,c,d,e,f,g,h,i,j)
#endif

#endif
