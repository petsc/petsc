/*
      This provides a few of the MPI-uni functions that cannot be implemented
    with C macros
*/
#include <petscsys.h>
#if !defined(MPIUNI_H)
#error "Wrong mpi.h included! require mpi.h from MPIUNI"
#endif

#include <petsc/private/petscimpl.h> /* for PetscCUPMInitialized */

#if defined(PETSC_HAVE_CUDA)
  #include <cuda_runtime.h>
#endif

#if defined(PETSC_HAVE_HIP)
  #include <hip/hip_runtime.h>
#endif

#define MPI_SUCCESS 0
#define MPI_FAILURE 1

void *MPIUNI_TMP = NULL;

/*
       With MPI Uni there are exactly four distinct communicators:
    MPI_COMM_SELF, MPI_COMM_WORLD, and a MPI_Comm_dup() of each of these (duplicates of duplicates return the same communictor)

    MPI_COMM_SELF and MPI_COMM_WORLD are MPI_Comm_free() in MPI_Finalize() but in general with PETSc,
     the other communicators are freed once the last PETSc object is freed (before MPI_Finalize()).

*/
#define MAX_ATTR 256
#define MAX_COMM 128

typedef struct {
  void *attribute_val;
  int  active;
} MPI_Attr;

typedef struct {
  void                *extra_state;
  MPI_Delete_function *del;
  int                 active;  /* Is this keyval in use by some comm? */
} MPI_Attr_keyval;

static MPI_Attr_keyval attr_keyval[MAX_ATTR];
static MPI_Attr        attr[MAX_COMM][MAX_ATTR];
static int             comm_active[MAX_COMM];  /* Boolean array indicating which comms are in use */
static int             mpi_tag_ub = 100000000;
static int             num_attr = 1; /* Maximal number of keyvals/attributes ever created, including the predefined MPI_TAG_UB attribute. */
static int             MaxComm  = 2; /* Maximal number of communicators ever created, including comm_self(1), comm_world(2), but not comm_null(0) */
static void*           MPIUNIF_mpi_in_place = 0;

#define CommIdx(comm)  ((comm)-1)  /* the communicator's internal index used in attr[idx][] and comm_active[idx]. comm_null does not occupy slots in attr[][] */

#if defined(__cplusplus)
extern "C" {
#endif

/*
   To avoid problems with prototypes to the system memcpy() it is duplicated here
*/
int MPIUNI_Memcpy(void *dst,const void *src,int n)
{
  if (dst == MPI_IN_PLACE || dst == MPIUNIF_mpi_in_place) return MPI_SUCCESS;
  if (src == MPI_IN_PLACE || src == MPIUNIF_mpi_in_place) return MPI_SUCCESS;
  if (!n) return MPI_SUCCESS;

  /* GPU-aware MPIUNI. Use synchronous copy per MPI semantics */
#if defined(PETSC_HAVE_CUDA)
  if (PetscCUDAInitialized) {cudaError_t cerr = cudaMemcpy(dst,src,n,cudaMemcpyDefault);if (cerr != cudaSuccess) return MPI_FAILURE;} else
#elif defined(PETSC_HAVE_HIP)
  if (PetscHIPInitialized)  {hipError_t  cerr = hipMemcpy(dst,src,n,hipMemcpyDefault);  if (cerr != hipSuccess)  return MPI_FAILURE;} else
#endif
  {memcpy(dst,src,n);}
  return MPI_SUCCESS;
}

static int classcnt = 0;
static int codecnt = 0;

int MPI_Add_error_class(int *cl)
{
  *cl = classcnt++;
  return MPI_SUCCESS;
}

int MPI_Add_error_code(int cl,int *co)
{
  if (cl >= classcnt) return MPI_FAILURE;
  *co = codecnt++;
  return MPI_SUCCESS;
}

int MPI_Type_get_envelope(MPI_Datatype datatype,int *num_integers,int *num_addresses,int *num_datatypes,int *combiner)
{
  int comb = datatype >> 28;
  switch (comb) {
  case MPI_COMBINER_NAMED:
    *num_integers = 0;
    *num_addresses = 0;
    *num_datatypes = 0;
    *combiner = comb;
    break;
  case MPI_COMBINER_DUP:
    *num_integers = 0;
    *num_addresses = 0;
    *num_datatypes = 1;
    *combiner = comb;
    break;
  case MPI_COMBINER_CONTIGUOUS:
    *num_integers = 1;
    *num_addresses = 0;
    *num_datatypes = 1;
    *combiner = comb;
    break;
  default:
    return MPIUni_Abort(MPI_COMM_SELF,1);
  }
  return MPI_SUCCESS;
}

int MPI_Type_get_contents(MPI_Datatype datatype,int max_integers,int max_addresses,int max_datatypes,int *array_of_integers,MPI_Aint *array_of_addresses,MPI_Datatype *array_of_datatypes)
{
  int comb = datatype >> 28;
  switch (comb) {
  case MPI_COMBINER_NAMED:
    return MPIUni_Abort(MPI_COMM_SELF,1);
    break;
  case MPI_COMBINER_DUP:
    if (max_datatypes < 1) return MPIUni_Abort(MPI_COMM_SELF,1);
    array_of_datatypes[0] = datatype & 0x0fffffff;
    break;
  case MPI_COMBINER_CONTIGUOUS:
    if (max_integers < 1 || max_datatypes < 1) return MPIUni_Abort(MPI_COMM_SELF,1);
    array_of_integers[0] = (datatype >> 8) & 0xfff; /* count */
    array_of_datatypes[0] = (datatype & 0x0ff000ff) | 0x100;  /* basic named type (count=1) from which the contiguous type is derived */
    break;
  default:
    return MPIUni_Abort(MPI_COMM_SELF,1);
  }
  return MPI_SUCCESS;
}

/*
   Used to set the built-in MPI_TAG_UB attribute
*/
static int Keyval_setup(void)
{
  attr[CommIdx(MPI_COMM_WORLD)][0].active        = 1;
  attr[CommIdx(MPI_COMM_WORLD)][0].attribute_val = &mpi_tag_ub;
  attr[CommIdx(MPI_COMM_SELF )][0].active        = 1;
  attr[CommIdx(MPI_COMM_SELF )][0].attribute_val = &mpi_tag_ub;
  attr_keyval[0].active                          = 1;
  return MPI_SUCCESS;
}

int MPI_Comm_create_keyval(MPI_Copy_function *copy_fn,MPI_Delete_function *delete_fn,int *keyval,void *extra_state)
{
  int i,keyid;
  for (i=1; i<num_attr; i++) { /* the first attribute is always in use */
    if (!attr_keyval[i].active) {
      keyid = i;
      goto found;
    }
  }
  if (num_attr >= MAX_ATTR) return MPIUni_Abort(MPI_COMM_WORLD,1);
  keyid = num_attr++;

found:
  attr_keyval[keyid].extra_state = extra_state;
  attr_keyval[keyid].del         = delete_fn;
  attr_keyval[keyid].active      = 1;
  *keyval                        = keyid;
  return MPI_SUCCESS;
}

int MPI_Comm_free_keyval(int *keyval)
{
  attr_keyval[*keyval].extra_state = 0;
  attr_keyval[*keyval].del         = 0;
  attr_keyval[*keyval].active      = 0;
  *keyval = 0;
  return MPI_SUCCESS;
}

int MPI_Comm_set_attr(MPI_Comm comm,int keyval,void *attribute_val)
{
  int idx = CommIdx(comm);
  if (comm < 1 || comm > MaxComm) return MPI_FAILURE;
  attr[idx][keyval].active        = 1;
  attr[idx][keyval].attribute_val = attribute_val;
  return MPI_SUCCESS;
}

int MPI_Comm_delete_attr(MPI_Comm comm,int keyval)
{
  int idx = CommIdx(comm);
  if (comm < 1 || comm > MaxComm) return MPI_FAILURE;
  if (attr[idx][keyval].active && attr_keyval[keyval].del) {
    void *save_attribute_val        = attr[idx][keyval].attribute_val;
    attr[idx][keyval].active        = 0;
    attr[idx][keyval].attribute_val = 0;
    (*(attr_keyval[keyval].del))(comm,keyval,save_attribute_val,attr_keyval[keyval].extra_state);
  }
  return MPI_SUCCESS;
}

int MPI_Comm_get_attr(MPI_Comm comm,int keyval,void *attribute_val,int *flag)
{
  int idx = CommIdx(comm);
  if (comm < 1 || comm > MaxComm) return MPI_FAILURE;
  if (!keyval) Keyval_setup();
  *flag                  = attr[idx][keyval].active;
  *(void**)attribute_val = attr[idx][keyval].attribute_val;
  return MPI_SUCCESS;
}

int MPI_Comm_create(MPI_Comm comm,MPI_Group group,MPI_Comm *newcomm)
{
  int j;
  if (comm < 1 || comm > MaxComm) return MPI_FAILURE;
  for (j=3; j<=MaxComm; j++) {
    if (!comm_active[CommIdx(j)]) {
      comm_active[CommIdx(j)] = 1;
      *newcomm = j;
      return MPI_SUCCESS;
    }
  }
  if (MaxComm >= MAX_COMM) return MPI_FAILURE;
  *newcomm = ++MaxComm;
  comm_active[CommIdx(*newcomm)] = 1;
  return MPI_SUCCESS;
}

int MPI_Comm_dup(MPI_Comm comm,MPI_Comm *out)
{
  int j;
  if (comm < 1 || comm > MaxComm) return MPI_FAILURE;
  for (j=3; j<=MaxComm; j++) {
    if (!comm_active[CommIdx(j)]) {
      comm_active[CommIdx(j)] = 1;
      *out = j;
      return MPI_SUCCESS;
    }
  }
  if (MaxComm >= MAX_COMM) return MPI_FAILURE;
  *out = ++MaxComm;
  comm_active[CommIdx(*out)] = 1;
  return MPI_SUCCESS;
}

int MPI_Comm_free(MPI_Comm *comm)
{
  int i;
  int idx = CommIdx(*comm);

  if (*comm < 1 || *comm > MaxComm) return MPI_FAILURE;
  for (i=0; i<num_attr; i++) {
    if (attr[idx][i].active && attr_keyval[i].del) (*attr_keyval[i].del)(*comm,i,attr[idx][i].attribute_val,attr_keyval[i].extra_state);
    attr[idx][i].active        = 0;
    attr[idx][i].attribute_val = 0;
  }
  if (*comm >= 3) comm_active[idx] = 0;
  *comm = 0;
  return MPI_SUCCESS;
}

int MPI_Comm_size(MPI_Comm comm, int *size)
{
  if (comm < 1 || comm > MaxComm) return MPI_FAILURE;
  *size=1;
  return MPI_SUCCESS;
}

int MPI_Comm_rank(MPI_Comm comm, int *rank)
{
  if (comm < 1 || comm > MaxComm) return MPI_FAILURE;
  *rank=0;
  return MPI_SUCCESS;
}

int MPIUni_Abort(MPI_Comm comm,int errorcode)
{
  printf("MPI operation not supported by PETSc's sequential MPI wrappers\n");
  return MPI_FAILURE;
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
  if (MPI_was_initialized) return MPI_FAILURE;
  if (MPI_was_finalized) return MPI_FAILURE; /* MPI standard: once MPI_FINALIZE returns, no MPI routine (not even MPI_INIT) may be called, except ... */
  MPI_was_initialized = 1;
  return MPI_SUCCESS;
}

int MPI_Finalize(void)
{
  MPI_Comm comm;
  if (MPI_was_finalized) return MPI_FAILURE;
  if (!MPI_was_initialized) return MPI_FAILURE;
  comm = MPI_COMM_WORLD;
  MPI_Comm_free(&comm);
  comm = MPI_COMM_SELF;
  MPI_Comm_free(&comm);
#if defined(PETSC_USE_DEBUG)
  {
    int i;
    for (i=3; i<=MaxComm; i++) {
      if (comm_active[CommIdx(i)]) printf("MPIUni warning: MPI communicator %d is not freed before MPI_Finalize()\n", i);
    }
  }
#endif
  /* reset counters */
  MaxComm  = 2;
  num_attr = 1;
  MPI_was_finalized = 1;
  return MPI_SUCCESS;
}

int MPI_Initialized(int *flag)
{
  *flag = MPI_was_initialized;
  return MPI_SUCCESS;
}

int MPI_Finalized(int *flag)
{
  *flag = MPI_was_finalized;
  return MPI_SUCCESS;
}

/* -------------------     Fortran versions of several routines ------------------ */

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define mpiunisetmoduleblock_          MPIUNISETMODULEBLOCK
#define mpiunisetfortranbasepointers_  MPIUNISETFORTRANBASEPOINTERS
#define petsc_mpi_init_                PETSC_MPI_INIT
#define petsc_mpi_finalize_            PETSC_MPI_FINALIZE
#define petsc_mpi_comm_size_           PETSC_MPI_COMM_SIZE
#define petsc_mpi_comm_rank_           PETSC_MPI_COMM_RANK
#define petsc_mpi_abort_               PETSC_MPI_ABORT
#define petsc_mpi_reduce_              PETSC_MPI_REDUCE
#define petsc_mpi_allreduce_           PETSC_MPI_ALLREDUCE
#define petsc_mpi_barrier_             PETSC_MPI_BARRIER
#define petsc_mpi_bcast_               PETSC_MPI_BCAST
#define petsc_mpi_gather_              PETSC_MPI_GATHER
#define petsc_mpi_allgather_           PETSC_MPI_ALLGATHER
#define petsc_mpi_comm_split_          PETSC_MPI_COMM_SPLIT
#define petsc_mpi_scan_                PETSC_MPI_SCAN
#define petsc_mpi_send_                PETSC_MPI_SEND
#define petsc_mpi_recv_                PETSC_MPI_RECV
#define petsc_mpi_reduce_scatter_      PETSC_MPI_REDUCE_SCATTER
#define petsc_mpi_irecv_               PETSC_MPI_IRECV
#define petsc_mpi_isend_               PETSC_MPI_ISEND
#define petsc_mpi_sendrecv_            PETSC_MPI_SENDRECV
#define petsc_mpi_test_                PETSC_MPI_TEST
#define petsc_mpi_waitall_             PETSC_MPI_WAITALL
#define petsc_mpi_waitany_             PETSC_MPI_WAITANY
#define petsc_mpi_allgatherv_          PETSC_MPI_ALLGATHERV
#define petsc_mpi_alltoallv_           PETSC_MPI_ALLTOALLV
#define petsc_mpi_comm_create_         PETSC_MPI_COMM_CREATE
#define petsc_mpi_address_             PETSC_MPI_ADDRESS
#define petsc_mpi_pack_                PETSC_MPI_PACK
#define petsc_mpi_unpack_              PETSC_MPI_UNPACK
#define petsc_mpi_pack_size_           PETSC_MPI_PACK_SIZE
#define petsc_mpi_type_struct_         PETSC_MPI_TYPE_STRUCT
#define petsc_mpi_type_commit_         PETSC_MPI_TYPE_COMMIT
#define petsc_mpi_wtime_               PETSC_MPI_WTIME
#define petsc_mpi_cancel_              PETSC_MPI_CANCEL
#define petsc_mpi_comm_dup_            PETSC_MPI_COMM_DUP
#define petsc_mpi_comm_free_           PETSC_MPI_COMM_FREE
#define petsc_mpi_get_count_           PETSC_MPI_GET_COUNT
#define petsc_mpi_get_processor_name_  PETSC_MPI_GET_PROCESSOR_NAME
#define petsc_mpi_initialized_         PETSC_MPI_INITIALIZED
#define petsc_mpi_iprobe_              PETSC_MPI_IPROBE
#define petsc_mpi_probe_               PETSC_MPI_PROBE
#define petsc_mpi_request_free_        PETSC_MPI_REQUEST_FREE
#define petsc_mpi_ssend_               PETSC_MPI_SSEND
#define petsc_mpi_wait_                PETSC_MPI_WAIT
#define petsc_mpi_comm_group_          PETSC_MPI_COMM_GROUP
#define petsc_mpi_exscan_              PETSC_MPI_EXSCAN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define mpiunisetmoduleblock_          mpiunisetmoduleblock
#define mpiunisetfortranbasepointers_  mpiunisetfortranbasepointers
#define petsc_mpi_init_                petsc_mpi_init
#define petsc_mpi_finalize_            petsc_mpi_finalize
#define petsc_mpi_comm_size_           petsc_mpi_comm_size
#define petsc_mpi_comm_rank_           petsc_mpi_comm_rank
#define petsc_mpi_abort_               petsc_mpi_abort
#define petsc_mpi_reduce_              petsc_mpi_reduce
#define petsc_mpi_allreduce_           petsc_mpi_allreduce
#define petsc_mpi_barrier_             petsc_mpi_barrier
#define petsc_mpi_bcast_               petsc_mpi_bcast
#define petsc_mpi_gather_              petsc_mpi_gather
#define petsc_mpi_allgather_           petsc_mpi_allgather
#define petsc_mpi_comm_split_          petsc_mpi_comm_split
#define petsc_mpi_scan_                petsc_mpi_scan
#define petsc_mpi_send_                petsc_mpi_send
#define petsc_mpi_recv_                petsc_mpi_recv
#define petsc_mpi_reduce_scatter_      petsc_mpi_reduce_scatter
#define petsc_mpi_irecv_               petsc_mpi_irecv
#define petsc_mpi_isend_               petsc_mpi_isend
#define petsc_mpi_sendrecv_            petsc_mpi_sendrecv
#define petsc_mpi_test_                petsc_mpi_test
#define petsc_mpi_waitall_             petsc_mpi_waitall
#define petsc_mpi_waitany_             petsc_mpi_waitany
#define petsc_mpi_allgatherv_          petsc_mpi_allgatherv
#define petsc_mpi_alltoallv_           petsc_mpi_alltoallv
#define petsc_mpi_comm_create_         petsc_mpi_comm_create
#define petsc_mpi_address_             petsc_mpi_address
#define petsc_mpi_pack_                petsc_mpi_pack
#define petsc_mpi_unpack_              petsc_mpi_unpack
#define petsc_mpi_pack_size_           petsc_mpi_pack_size
#define petsc_mpi_type_struct_         petsc_mpi_type_struct
#define petsc_mpi_type_commit_         petsc_mpi_type_commit
#define petsc_mpi_wtime_               petsc_mpi_wtime
#define petsc_mpi_cancel_              petsc_mpi_cancel
#define petsc_mpi_comm_dup_            petsc_mpi_comm_dup
#define petsc_mpi_comm_free_           petsc_mpi_comm_free
#define petsc_mpi_get_count_           petsc_mpi_get_count
#define petsc_mpi_get_processor_name_  petsc_mpi_get_processor_name
#define petsc_mpi_initialized_         petsc_mpi_initialized
#define petsc_mpi_iprobe_              petsc_mpi_iprobe
#define petsc_mpi_probe_               petsc_mpi_probe
#define petsc_mpi_request_free_        petsc_mpi_request_free
#define petsc_mpi_ssend_               petsc_mpi_ssend
#define petsc_mpi_wait_                petsc_mpi_wait
#define petsc_mpi_comm_group_          petsc_mpi_comm_group
#define petsc_mpi_exscan_              petsc_mpi_exscan
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
#define petsc_mpi_init_                petsc_mpi_init__
#define petsc_mpi_finalize_            petsc_mpi_finalize__
#define petsc_mpi_comm_size_           petsc_mpi_comm_size__
#define petsc_mpi_comm_rank_           petsc_mpi_comm_rank__
#define petsc_mpi_abort_               petsc_mpi_abort__
#define petsc_mpi_reduce_              petsc_mpi_reduce__
#define petsc_mpi_allreduce_           petsc_mpi_allreduce__
#define petsc_mpi_barrier_             petsc_mpi_barrier__
#define petsc_mpi_bcast_               petsc_mpi_bcast__
#define petsc_mpi_gather_              petsc_mpi_gather__
#define petsc_mpi_allgather_           petsc_mpi_allgather__
#define petsc_mpi_comm_split_          petsc_mpi_comm_split__
#define petsc_mpi_scan_                petsc_mpi_scan__
#define petsc_mpi_send_                petsc_mpi_send__
#define petsc_mpi_recv_                petsc_mpi_recv__
#define petsc_mpi_reduce_scatter_      petsc_mpi_reduce_scatter__
#define petsc_mpi_irecv_               petsc_mpi_irecv__
#define petsc_mpi_isend_               petsc_mpi_isend__
#define petsc_mpi_sendrecv_            petsc_mpi_sendrecv__
#define petsc_mpi_test_                petsc_mpi_test__
#define petsc_mpi_waitall_             petsc_mpi_waitall__
#define petsc_mpi_waitany_             petsc_mpi_waitany__
#define petsc_mpi_allgatherv_          petsc_mpi_allgatherv__
#define petsc_mpi_alltoallv_           petsc_mpi_alltoallv__
#define petsc_mpi_comm_create_         petsc_mpi_comm_create__
#define petsc_mpi_address_             petsc_mpi_address__
#define petsc_mpi_pack_                petsc_mpi_pack__
#define petsc_mpi_unpack_              petsc_mpi_unpack__
#define petsc_mpi_pack_size_           petsc_mpi_pack_size__
#define petsc_mpi_type_struct_         petsc_mpi_type_struct__
#define petsc_mpi_type_commit_         petsc_mpi_type_commit__
#define petsc_mpi_wtime_               petsc_mpi_wtime__
#define petsc_mpi_cancel_              petsc_mpi_cancel__
#define petsc_mpi_comm_dup_            petsc_mpi_comm_dup__
#define petsc_mpi_comm_free_           petsc_mpi_comm_free__
#define petsc_mpi_get_count_           petsc_mpi_get_count__
#define petsc_mpi_get_processor_name_  petsc_mpi_get_processor_name__
#define petsc_mpi_initialized_         petsc_mpi_initialized__
#define petsc_mpi_iprobe_              petsc_mpi_iprobe__
#define petsc_mpi_probe_               petsc_mpi_probe__
#define petsc_mpi_request_free_        petsc_mpi_request_free__
#define petsc_mpi_ssend_               petsc_mpi_ssend__
#define petsc_mpi_wait_                petsc_mpi_wait__
#define petsc_mpi_comm_group_          petsc_mpi_comm_group__
#define petsc_mpi_exscan_              petsc_mpi_exscan__
#endif

/* Do not build fortran interface if MPI namespace colision is to be avoided */
#if defined(PETSC_HAVE_FORTRAN)

PETSC_EXTERN void mpiunisetmoduleblock_(void);

PETSC_EXTERN void mpiunisetfortranbasepointers_(void *f_mpi_in_place)
{
  MPIUNIF_mpi_in_place   = f_mpi_in_place;
}

PETSC_EXTERN void petsc_mpi_init_(int *ierr)
{
  mpiunisetmoduleblock_();
  *ierr = MPI_Init((int*)0, (char***)0);
}

PETSC_EXTERN void petsc_mpi_finalize_(int *ierr)
{
  *ierr = MPI_Finalize();
}

PETSC_EXTERN void petsc_mpi_comm_size_(MPI_Comm *comm,int *size,int *ierr)
{
  *size = 1;
  *ierr = 0;
}

PETSC_EXTERN void petsc_mpi_comm_rank_(MPI_Comm *comm,int *rank,int *ierr)
{
  *rank = 0;
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_comm_split_(MPI_Comm *comm,int *color,int *key, MPI_Comm *newcomm, int *ierr)
{
  *newcomm = *comm;
  *ierr    = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_abort_(MPI_Comm *comm,int *errorcode,int *ierr)
{
  abort();
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_reduce_(void *sendbuf,void *recvbuf,int *count,int *datatype,int *op,int *root,int *comm,int *ierr)
{
  *ierr = MPI_Reduce(sendbuf,recvbuf,*count,*datatype,*op,*root,*comm);
}

PETSC_EXTERN void petsc_mpi_allreduce_(void *sendbuf,void *recvbuf,int *count,int *datatype,int *op,int *comm,int *ierr)
{
  *ierr = MPI_Allreduce(sendbuf,recvbuf,*count,*datatype,*op,*comm);
}

PETSC_EXTERN void petsc_mpi_barrier_(MPI_Comm *comm,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_bcast_(void *buf,int *count,int *datatype,int *root,int *comm,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_gather_(void *sendbuf,int *scount,int *sdatatype, void *recvbuf, int *rcount, int *rdatatype, int *root,int *comm,int *ierr)
{
  *ierr = MPI_Gather(sendbuf,*scount,*sdatatype,recvbuf,rcount,rdatatype,*root,*comm);
}

PETSC_EXTERN void petsc_mpi_allgather_(void *sendbuf,int *scount,int *sdatatype, void *recvbuf, int *rcount, int *rdatatype,int *comm,int *ierr)
{
  *ierr = MPI_Allgather(sendbuf,*scount,*sdatatype,recvbuf,rcount,rdatatype,*comm);
}

PETSC_EXTERN void petsc_mpi_scan_(void *sendbuf,void *recvbuf,int *count,int *datatype,int *op,int *comm,int *ierr)
{
  *ierr = MPIUNI_Memcpy(recvbuf,sendbuf,(*count)*MPI_sizeof(*datatype));
}

PETSC_EXTERN void petsc_mpi_send_(void *buf,int *count,int *datatype,int *dest,int *tag,int *comm,int *ierr)
{
  *ierr = MPIUni_Abort(MPI_COMM_WORLD,0);
}

PETSC_EXTERN void petsc_mpi_recv_(void *buf,int *count,int *datatype,int *source,int *tag,int *comm,int status,int *ierr)
{
  *ierr = MPIUni_Abort(MPI_COMM_WORLD,0);
}

PETSC_EXTERN void petsc_mpi_reduce_scatter_(void *sendbuf,void *recvbuf,int *recvcounts,int *datatype,int *op,int *comm,int *ierr)
{
  *ierr = MPIUni_Abort(MPI_COMM_WORLD,0);
}

PETSC_EXTERN void petsc_mpi_irecv_(void *buf,int *count, int *datatype, int *source, int *tag, int *comm, int *request, int *ierr)
{
  *ierr = MPIUni_Abort(MPI_COMM_WORLD,0);
}

PETSC_EXTERN void petsc_mpi_isend_(void *buf,int *count,int *datatype,int *dest,int *tag,int *comm,int *request, int *ierr)
{
  *ierr = MPIUni_Abort(MPI_COMM_WORLD,0);
}

PETSC_EXTERN void petsc_mpi_sendrecv_(void *sendbuf,int *sendcount,int *sendtype,int *dest,int *sendtag,void *recvbuf,int *recvcount,int *recvtype,int *source,int *recvtag,int *comm,int *status,int *ierr)
{
  *ierr = MPIUNI_Memcpy(recvbuf,sendbuf,(*sendcount)*MPI_sizeof(*sendtype));
}

PETSC_EXTERN void petsc_mpi_test_(int *request,int *flag,int *status,int *ierr)
{
  *ierr = MPIUni_Abort(MPI_COMM_WORLD,0);
}

PETSC_EXTERN void petsc_mpi_waitall_(int *count,int *array_of_requests,int *array_of_statuses,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_waitany_(int *count,int *array_of_requests,int * index, int *status,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_allgatherv_(void *sendbuf,int *sendcount,int *sendtype,void *recvbuf,int *recvcounts,int *displs,int *recvtype,int *comm,int *ierr)
{
  *ierr = MPI_Allgatherv(sendbuf,*sendcount,*sendtype,recvbuf,recvcounts,displs,*recvtype,*comm);
}

PETSC_EXTERN void petsc_mpi_alltoallv_(void *sendbuf,int *sendcounts,int *sdispls,int *sendtype,void *recvbuf,int *recvcounts,int *rdispls,int *recvtype,int *comm,int *ierr)
{
  *ierr = MPI_Alltoallv(sendbuf,sendcounts,sdispls,*sendtype,recvbuf,recvcounts,rdispls,*recvtype,*comm);
}

PETSC_EXTERN void petsc_mpi_comm_create_(int *comm,int *group,int *newcomm,int *ierr)
{
  *newcomm =  *comm;
  *ierr    = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_address_(void *location,MPI_Aint *address,int *ierr)
{
  *address =  (MPI_Aint) ((char *)location);
  *ierr    = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_pack_(void *inbuf,int *incount,int *datatype,void *outbuf,int *outsize,int *position,int *comm,int *ierr)
{
  *ierr = MPIUni_Abort(MPI_COMM_WORLD,0);
}

PETSC_EXTERN void petsc_mpi_unpack_(void *inbuf,int *insize,int *position,void *outbuf,int *outcount,int *datatype,int *comm,int *ierr)
{
  *ierr = MPIUni_Abort(MPI_COMM_WORLD,0);
}

PETSC_EXTERN void petsc_mpi_pack_size_(int *incount,int *datatype,int *comm,int *size,int *ierr)
{
  *ierr = MPIUni_Abort(MPI_COMM_WORLD,0);
}

PETSC_EXTERN void petsc_mpi_type_struct_(int *count,int *array_of_blocklengths,int * array_of_displaments,int *array_of_types,int *newtype,int *ierr)
{
  *ierr = MPIUni_Abort(MPI_COMM_WORLD,0);
}

PETSC_EXTERN void petsc_mpi_type_commit_(int *datatype,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

double petsc_mpi_wtime_(void)
{
  return 0.0;
}

PETSC_EXTERN void petsc_mpi_cancel_(int *request,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_comm_dup_(int *comm,int *out,int *ierr)
{
  *out  = *comm;
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_comm_free_(int *comm,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_get_count_(int *status,int *datatype,int *count,int *ierr)
{
  *ierr = MPIUni_Abort(MPI_COMM_WORLD,0);
}

PETSC_EXTERN void petsc_mpi_get_processor_name_(char *name,int *result_len,int *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  MPIUNI_Memcpy(name,"localhost",9*sizeof(char));
  *result_len = 9;
  *ierr       = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_initialized_(int *flag,int *ierr)
{
  *flag = MPI_was_initialized;
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_iprobe_(int *source,int *tag,int *comm,int *glag,int *status,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_probe_(int *source,int *tag,int *comm,int *flag,int *status,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_request_free_(int *request,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_ssend_(void *buf,int *count,int *datatype,int *dest,int *tag,int *comm,int *ierr)
{
  *ierr = MPIUni_Abort(MPI_COMM_WORLD,0);
}

PETSC_EXTERN void petsc_mpi_wait_(int *request,int *status,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_comm_group_(int *comm,int *group,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

PETSC_EXTERN void petsc_mpi_exscan_(void *sendbuf,void *recvbuf,int *count,int *datatype,int *op,int *comm,int *ierr)
{
  *ierr = MPI_SUCCESS;
}

#endif /* PETSC_HAVE_FORTRAN */

#if defined(__cplusplus)
}
#endif
