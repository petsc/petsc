/*
      This provides a few of the MPI-uni functions that cannot be implemented
    with C macros
*/
#include <mpiuni/mpi.h>
#if !defined(__MPIUNI_H)
#error "Wrong mpi.h included! require mpi.h from MPIUNI"
#endif
#if !defined(PETSC_STDCALL)
#define PETSC_STDCALL
#endif
#include <stdio.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#define MPI_SUCCESS 0
#define MPI_FAILURE 1
void    *MPIUNI_TMP        = 0;
int     MPIUNI_DATASIZE[10] = {sizeof(int),sizeof(float),sizeof(double),2*sizeof(double),sizeof(char),2*sizeof(int),4*sizeof(double),4,8,2*sizeof(double)};
/*
       With MPI Uni there is only one communicator, which is called 1.
*/
#define MAX_ATTR 128

typedef struct {
  void                *extra_state;
  void                *attribute_val;
  int                 active;
  MPI_Delete_function *del;
} MPI_Attr;

static MPI_Attr attr[MAX_ATTR];
static int      num_attr = 1,mpi_tag_ub = 100000000;

#if defined(__cplusplus)
extern "C" {
#endif

/* 
   To avoid problems with prototypes to the system memcpy() it is duplicated here
*/
int MPIUNI_Memcpy(void *a,const void* b,int n) {
  int  i;
  char *aa= (char*)a;
  char *bb= (char*)b;

  if (b == MPI_IN_PLACE) return 0;
  for (i=0; i<n; i++) aa[i] = bb[i];
  return 0;
}

/*
   Used to set the built-in MPI_TAG_UB attribute
*/
static int Keyval_setup(void)
{
  attr[0].active        = 1;
  attr[0].attribute_val = &mpi_tag_ub;
  return 0;
}

int MPI_Keyval_create(MPI_Copy_function *copy_fn,MPI_Delete_function *delete_fn,int *keyval,void *extra_state)
{
  if (num_attr >= MAX_ATTR) MPI_Abort(MPI_COMM_WORLD,1);

  attr[num_attr].extra_state = extra_state;
  attr[num_attr].del         = delete_fn;
  *keyval                    = num_attr++;
  return 0;
}

int MPI_Keyval_free(int *keyval)
{
  attr[*keyval].active = 0;
  return MPI_SUCCESS;
}

int MPI_Attr_put(MPI_Comm comm,int keyval,void *attribute_val)
{
  attr[keyval].active        = 1;
  attr[keyval].attribute_val = attribute_val;
  return MPI_SUCCESS;
}
  
int MPI_Attr_delete(MPI_Comm comm,int keyval)
{
  if (attr[keyval].active && attr[keyval].del) {
    void* save_attribute_val   = attr[keyval].attribute_val;
    attr[keyval].active        = 0;
    attr[keyval].attribute_val = 0;
    (*(attr[keyval].del))(comm,keyval,save_attribute_val,attr[keyval].extra_state);
  }
  return MPI_SUCCESS;
}

int MPI_Attr_get(MPI_Comm comm,int keyval,void *attribute_val,int *flag)
{
  if (!keyval) Keyval_setup();
  *flag                   = attr[keyval].active;
  *(void **)attribute_val = attr[keyval].attribute_val;
  return MPI_SUCCESS;
}

static int dups = 0;
int MPI_Comm_create(MPI_Comm comm,MPI_Group group,MPI_Comm *newcomm)
{
  dups++;
  *newcomm =  comm;
  return MPI_SUCCESS;
}

int MPI_Comm_dup(MPI_Comm comm,MPI_Comm *out)
{
  *out = comm;
  dups++;
  return 0;
}

int MPI_Comm_free(MPI_Comm *comm)
{
  int i;

  if (--dups) return MPI_SUCCESS;
  for (i=0; i<num_attr; i++) {
    if (attr[i].active && attr[i].del) {
      (*attr[i].del)(*comm,i,attr[i].attribute_val,attr[i].extra_state);
    }
    attr[i].active = 0;
  }
  return MPI_SUCCESS;
}

int MPI_Comm_size(MPI_Comm comm, int*size)
{
  *size=1;
  return MPI_SUCCESS;
}

int MPI_Comm_rank(MPI_Comm comm, int*rank)
{
  *rank=0;
  return MPI_SUCCESS;
}

int MPI_Abort(MPI_Comm comm,int errorcode)
{
  abort();
  return MPI_SUCCESS;
}

/* --------------------------------------------------------------------------*/
  
static int MPI_was_initialized = 0;
static int MPI_was_finalized   = 0;

int MPI_Init(int *argc, char ***argv)
{
  if (MPI_was_initialized) return 1;
  if (MPI_was_finalized) return 1;
  MPI_was_initialized = 1;
  return 0;
}

int MPI_Finalize(void)
{
  if (MPI_was_finalized) return 1;
  if (!MPI_was_initialized) return 1;
  MPI_was_finalized = 1;
  return 0;
}

int MPI_Initialized(int *flag)
{
  *flag = MPI_was_initialized;
  return 0;
}

int MPI_Finalized(int *flag)
{
  *flag = MPI_was_finalized;
  return 0;
}

/* -------------------     Fortran versions of several routines ------------------ */

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define mpi_init_             MPI_INIT
#define mpi_finalize_         MPI_FINALIZE
#define mpi_comm_size_        MPI_COMM_SIZE
#define mpi_comm_rank_        MPI_COMM_RANK
#define mpi_abort_            MPI_ABORT
#define mpi_reduce_           MPI_REDUCE
#define mpi_allreduce_        MPI_ALLREDUCE
#define mpi_barrier_          MPI_BARRIER
#define mpi_bcast_            MPI_BCAST
#define mpi_gather_           MPI_GATHER
#define mpi_allgather_        MPI_ALLGATHER
#define mpi_comm_split_       MPI_COMM_SPLIT
#define mpi_scan_             MPI_SCAN
#define mpi_send_             MPI_SEND
#define mpi_recv_             MPI_RECV
#define mpi_reduce_scatter_   MPI_REDUCE_SCATTER
#define mpi_irecv_            MPI_IRECV
#define mpi_isend_            MPI_ISEND
#define mpi_sendrecv_         MPI_SENDRECV
#define mpi_test_             MPI_TEST
#define mpi_waitall_          MPI_WAITALL
#define mpi_waitany_          MPI_WAITANY
#define mpi_allgatherv_       MPI_ALLGATHERV
#define mpi_alltoallv_        MPI_ALLTOALLV
#define mpi_comm_create_      MPI_COMM_CREATE
#define mpi_address_          MPI_ADDRESS
#define mpi_pack_             MPI_PACK
#define mpi_unpack_           MPI_UNPACK
#define mpi_pack_size_        MPI_PACK_SIZE
#define mpi_type_struct_      MPI_TYPE_STRUCT
#define mpi_type_commit_      MPI_TYPE_COMMIT
#define mpi_wtime_            MPI_WTIME
#define mpi_cancel_           MPI_CANCEL
#define mpi_comm_dup_         MPI_COMM_DUP
#define mpi_comm_free_        MPI_COMM_FREE
#define mpi_get_count_        MPI_GET_COUNT
#define mpi_get_processor_name_ MPI_GET_PROCESSOR_NAME
#define mpi_initialized_      MPI_INITIALIZED
#define mpi_iprobe_           MPI_IPROBE
#define mpi_probe_            MPI_PROBE
#define mpi_request_free_     MPI_REQUEST_FREE
#define mpi_ssend_            MPI_SSEND
#define mpi_wait_             MPI_WAIT
#define mpi_comm_group_       MPI_COMM_GROUP
#define mpi_exscan_           MPI_EXSCAN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define mpi_init_             mpi_init
#define mpi_finalize_         mpi_finalize
#define mpi_comm_size_        mpi_comm_size
#define mpi_comm_rank_        mpi_comm_rank
#define mpi_abort_            mpi_abort
#define mpi_reduce_           mpi_reduce
#define mpi_allreduce_        mpi_allreduce
#define mpi_barrier_          mpi_barrier
#define mpi_bcast_            mpi_bcast
#define mpi_gather_           mpi_gather
#define mpi_allgather_        mpi_allgather
#define mpi_comm_split_       mpi_comm_split
#define mpi_scan_             mpi_scan
#define mpi_send_             mpi_send
#define mpi_recv_             mpi_recv
#define mpi_reduce_scatter_   mpi_reduce_scatter
#define mpi_irecv_            mpi_irecv
#define mpi_isend_            mpi_isend
#define mpi_sendrecv_         mpi_sendrecv
#define mpi_test_             mpi_test
#define mpi_waitall_          mpi_waitall
#define mpi_waitany_          mpi_waitany
#define mpi_allgatherv_       mpi_allgatherv
#define mpi_alltoallv_        mpi_alltoallv
#define mpi_comm_create_      mpi_comm_create
#define mpi_address_          mpi_address
#define mpi_pack_             mpi_pack
#define mpi_unpack_           mpi_unpack
#define mpi_pack_size_        mpi_pack_size
#define mpi_type_struct_      mpi_type_struct
#define mpi_type_commit_      mpi_type_commit
#define mpi_wtime_            mpi_wtime
#define mpi_cancel_           mpi_cancel
#define mpi_comm_dup_         mpi_comm_dup
#define mpi_comm_free_        mpi_comm_free
#define mpi_get_count_        mpi_get_count
#define mpi_get_processor_name_ mpi_get_processor_name
#define mpi_initialized_      mpi_initialized
#define mpi_iprobe_           mpi_iprobe
#define mpi_probe_            mpi_probe
#define mpi_request_free_     mpi_request_free
#define mpi_ssend_            mpi_ssend
#define mpi_wait_             mpi_wait
#define mpi_comm_group_       mpi_comm_group
#define mpi_exscan_           mpi_exscan
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
#define mpi_init_             mpi_init__
#define mpi_finalize_         mpi_finalize__
#define mpi_comm_size_        mpi_comm_size__
#define mpi_comm_rank_        mpi_comm_rank__
#define mpi_abort_            mpi_abort__
#define mpi_reduce_           mpi_reduce__
#define mpi_allreduce_        mpi_allreduce__
#define mpi_barrier_          mpi_barrier__
#define mpi_bcast_            mpi_bcast__
#define mpi_gather_           mpi_gather__
#define mpi_allgather_        mpi_allgather__
#define mpi_comm_split_       mpi_comm_split__
#define mpi_scan_             mpi_scan__
#define mpi_send_             mpi_send__
#define mpi_recv_             mpi_recv__
#define mpi_reduce_scatter_   mpi_reduce_scatter__
#define mpi_irecv_            mpi_irecv__
#define mpi_isend_            mpi_isend__
#define mpi_sendrecv_         mpi_sendrecv__
#define mpi_test_             mpi_test__
#define mpi_waitall_          mpi_waitall__
#define mpi_waitany_          mpi_waitany__
#define mpi_allgatherv_       mpi_allgatherv__
#define mpi_alltoallv_        mpi_alltoallv__
#define mpi_comm_create_      mpi_comm_create__
#define mpi_address_          mpi_address__
#define mpi_pack_             mpi_pack__
#define mpi_unpack_           mpi_unpack__
#define mpi_pack_size_        mpi_pack_size__
#define mpi_type_struct_      mpi_type_struct__
#define mpi_type_commit_      mpi_type_commit__
#define mpi_wtime_            mpi_wtime__
#define mpi_cancel_           mpi_cancel__
#define mpi_comm_dup_         mpi_comm_dup__
#define mpi_comm_free_        mpi_comm_free__
#define mpi_get_count_        mpi_get_count__
#define mpi_get_processor_name_ mpi_get_processor_name__
#define mpi_initialized_      mpi_initialized__
#define mpi_iprobe_           mpi_iprobe__
#define mpi_probe_            mpi_probe__
#define mpi_request_free_     mpi_request_free__
#define mpi_ssend_            mpi_ssend__
#define mpi_wait_             mpi_wait__
#define mpi_comm_group_       mpi_comm_group__
#define mpi_exscan_           mpi_exscan__
#endif


/* Do not build fortran interface if MPI namespace colision is to be avoided */
#if !defined(MPIUNI_AVOID_MPI_NAMESPACE)

void PETSC_STDCALL  mpi_init_(int *ierr)
{
  *ierr = MPI_Init((int*)0, (char***)0);
}

void PETSC_STDCALL  mpi_finalize_(int *ierr)
{
  *ierr = MPI_Finalize();
}

void PETSC_STDCALL mpi_comm_size_(MPI_Comm *comm,int *size,int *ierr) 
{
  *size = 1;
  *ierr = 0;
}

void PETSC_STDCALL mpi_comm_rank_(MPI_Comm *comm,int *rank,int *ierr)
{
  *rank=0;
  *ierr=MPI_SUCCESS;
}

void PETSC_STDCALL mpi_comm_split_(MPI_Comm *comm,int *color,int *key, MPI_Comm *newcomm, int *ierr)
{
  *newcomm = *comm;
  *ierr=MPI_SUCCESS;
}

void PETSC_STDCALL mpi_abort_(MPI_Comm *comm,int *errorcode,int *ierr) 
{
  abort();
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_reduce_(void *sendbuf,void *recvbuf,int *count,int *datatype,int *op,int *root,int *comm,int *ierr)
{
  MPIUNI_Memcpy(recvbuf,sendbuf,(*count)*MPIUNI_DATASIZE[*datatype]);
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_allreduce_(void *sendbuf,void *recvbuf,int *count,int *datatype,int *op,int *comm,int *ierr) 
{
  MPIUNI_Memcpy(recvbuf,sendbuf,(*count)*MPIUNI_DATASIZE[*datatype]);
  *ierr = MPI_SUCCESS;
} 

void PETSC_STDCALL mpi_barrier_(MPI_Comm *comm,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_bcast_(void *buf,int *count,int *datatype,int *root,int *comm,int *ierr)
{
  *ierr = MPI_SUCCESS;
}


void PETSC_STDCALL mpi_gather_(void *sendbuf,int *scount,int *sdatatype, void* recvbuf, int* rcount, int* rdatatype, int *root,int *comm,int *ierr)
{
  MPIUNI_Memcpy(recvbuf,sendbuf,(*scount)*MPIUNI_DATASIZE[*sdatatype]);
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_allgather_(void *sendbuf,int *scount,int *sdatatype, void* recvbuf, int* rcount, int* rdatatype,int *comm,int *ierr)
{
  MPIUNI_Memcpy(recvbuf,sendbuf,(*scount)*MPIUNI_DATASIZE[*sdatatype]);
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_scan_(void *sendbuf,void *recvbuf,int *count,int *datatype,int *op,int *comm,int *ierr)
{
  MPIUNI_Memcpy(recvbuf,sendbuf,(*count)*MPIUNI_DATASIZE[*datatype]);
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_send_(void*buf,int *count,int *datatype,int *dest,int *tag,int *comm,int *ierr )
{
  *ierr = MPI_Abort(MPI_COMM_WORLD,0);
}

void PETSC_STDCALL mpi_recv_(void*buf,int *count,int *datatype,int *source,int *tag,int *comm,int status,int *ierr )
{
  *ierr = MPI_Abort(MPI_COMM_WORLD,0);
}

void PETSC_STDCALL mpi_reduce_scatter_(void*sendbuf,void*recvbuf,int *recvcounts,int *datatype,int *op,int *comm,int *ierr)
{
  *ierr = MPI_Abort(MPI_COMM_WORLD,0);
}

void PETSC_STDCALL mpi_irecv_(void*buf,int *count, int *datatype, int *source, int *tag, int *comm, int *request, int *ierr)
{
  *ierr = MPI_Abort(MPI_COMM_WORLD,0);
}

void PETSC_STDCALL mpi_isend_(void*buf,int *count,int *datatype,int *dest,int *tag,int *comm,int *request, int *ierr)
{
  *ierr = MPI_Abort(MPI_COMM_WORLD,0);
}

void PETSC_STDCALL mpi_sendrecv_(void*sendbuf,int *sendcount,int *sendtype,int *dest,int *sendtag,void*recvbuf,int *recvcount,int *recvtype,int *source,int *recvtag,int *comm,int *status,int *ierr)
{
  MPIUNI_Memcpy(recvbuf,sendbuf,(*sendcount)*MPIUNI_DATASIZE[*sendtype]);
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_test_(int *request,int *flag,int *status,int *ierr)
{
  *ierr = MPI_Abort(MPI_COMM_WORLD,0);
}

void PETSC_STDCALL mpi_waitall_(int *count,int *array_of_requests,int *array_of_statuses,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

  void PETSC_STDCALL mpi_waitany_(int *count,int *array_of_requests,int * index, int *status,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_allgatherv_(void*sendbuf,int *sendcount,int *sendtype,void*recvbuf,int *recvcounts,int *displs,int *recvtype,int *comm,int *ierr)
{
  MPIUNI_Memcpy(recvbuf,sendbuf,(*sendcount)*MPIUNI_DATASIZE[*sendtype]);
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_alltoallv_(void*sendbuf,int *sendcounts,int *sdispls,int *sendtype,void*recvbuf,int *recvcounts,int *rdispls,int *recvtype,int *comm,int *ierr)
{
  MPIUNI_Memcpy(recvbuf,sendbuf,(*sendcounts)*MPIUNI_DATASIZE[*sendtype]);
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_comm_create_(int *comm,int *group,int *newcomm,int *ierr)
{
  *newcomm =  *comm;
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_address_(void*location,MPIUNI_INTPTR *address,int *ierr)
{
  *address =  (MPIUNI_INTPTR) location;
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_pack_(void*inbuf,int *incount,int *datatype,void*outbuf,int *outsize,int *position,int *comm,int *ierr)
{
  *ierr = MPI_Abort(MPI_COMM_WORLD,0);
}

void PETSC_STDCALL mpi_unpack_(void*inbuf,int *insize,int *position,void*outbuf,int *outcount,int *datatype,int *comm,int *ierr)
{
  *ierr = MPI_Abort(MPI_COMM_WORLD,0);
}

void PETSC_STDCALL mpi_pack_size_(int *incount,int *datatype,int *comm,int *size,int *ierr)
{
  *ierr = MPI_Abort(MPI_COMM_WORLD,0);
}

void PETSC_STDCALL mpi_type_struct_(int *count,int *array_of_blocklengths,int * array_of_displaments,int *array_of_types,int *newtype,int *ierr)
{
  *ierr = MPI_Abort(MPI_COMM_WORLD,0);
}

void PETSC_STDCALL mpi_type_commit_(int *datatype,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

double PETSC_STDCALL mpi_wtime_(void)
{
  return 0.0;
}

void PETSC_STDCALL mpi_cancel_(int *request,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_comm_dup_(int *comm,int *out,int *ierr)
{
  *out = *comm;
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_comm_free_(int *comm,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_get_count_(int *status,int *datatype,int *count,int *ierr)
{
  *ierr = MPI_Abort(MPI_COMM_WORLD,0);
}

/* duplicate from fortranimpl.h */
#if defined(PETSC_HAVE_FORTRAN_MIXED_STR_ARG)
#define PETSC_MIXED_LEN(len) ,int len
#define PETSC_END_LEN(len)
#else
#define PETSC_MIXED_LEN(len)
#define PETSC_END_LEN(len)   ,int len
#endif

void PETSC_STDCALL mpi_get_processor_name_(char *name PETSC_MIXED_LEN(len),int *result_len,int *ierr PETSC_END_LEN(len))
{
  MPIUNI_Memcpy(name,"localhost",9*sizeof(char));
  *result_len = 9;
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_initialized_(int *flag,int *ierr)
{
  *flag = MPI_was_initialized;
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_iprobe_(int *source,int *tag,int *comm,int *glag,int *status,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_probe_(int *source,int *tag,int *comm,int *flag,int *status,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_request_free_(int *request,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_ssend_(void*buf,int *count,int *datatype,int *dest,int *tag,int *comm,int *ierr)
{
  *ierr = MPI_Abort(MPI_COMM_WORLD,0);
}

void PETSC_STDCALL mpi_wait_(int *request,int *status,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_comm_group_(int*comm,int*group,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL mpi_exscan_(void*sendbuf,void*recvbuf,int*count,int*datatype,int*op,int*comm,int*ierr)
{
  *ierr = MPI_SUCCESS;
}

#endif /* MPIUNI_AVOID_MPI_NAMESPACE */

#if defined(__cplusplus)
}
#endif
