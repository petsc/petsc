/*
      This provides a few of the MPI-uni functions that cannot be implemented
    with C macros
*/
#include "include/mpiuni/mpi.h"

#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#define MPI_SUCCESS 0
#define MPI_FAILURE 1
void    *MPIUNI_TMP        = 0;
int     MPIUNI_DATASIZE[5] = { sizeof(int),sizeof(float),sizeof(double),2*sizeof(double),sizeof(char)};
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

/*
         These functions are mapped to the Petsc_ name by ./mpi.h
*/
int Petsc_MPI_Keyval_create(MPI_Copy_function *copy_fn,MPI_Delete_function *delete_fn,int *keyval,void *extra_state)
{
  if (num_attr >= MAX_ATTR) MPI_Abort(MPI_COMM_WORLD,1);

  attr[num_attr].extra_state = extra_state;
  attr[num_attr].del         = delete_fn;
  *keyval                    = num_attr++;
  return 0;
}

int Petsc_MPI_Keyval_free(int *keyval)
{
  attr[*keyval].active = 0;
  return MPI_SUCCESS;
}

int Petsc_MPI_Attr_put(MPI_Comm comm,int keyval,void *attribute_val)
{
  attr[keyval].active        = 1;
  attr[keyval].attribute_val = attribute_val;
  return MPI_SUCCESS;
}
  
int Petsc_MPI_Attr_delete(MPI_Comm comm,int keyval)
{
  if (attr[keyval].active && attr[keyval].del) {
    (*(attr[keyval].del))(comm,keyval,attr[keyval].attribute_val,attr[keyval].extra_state);
  }
  attr[keyval].active        = 0;
  attr[keyval].attribute_val = 0;
  return MPI_SUCCESS;
}

int Petsc_MPI_Attr_get(MPI_Comm comm,int keyval,void *attribute_val,int *flag)
{
  if (!keyval) Keyval_setup();
  *flag                   = attr[keyval].active;
  *(void **)attribute_val = attr[keyval].attribute_val;
  return MPI_SUCCESS;
}

static int dups = 0;
int Petsc_MPI_Comm_dup(MPI_Comm comm,MPI_Comm *out)
{
  *out = comm;
  dups++;
  return 0;
}

int Petsc_MPI_Comm_free(MPI_Comm *comm)
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

/* --------------------------------------------------------------------------*/

int Petsc_MPI_Abort(MPI_Comm comm,int errorcode) 
{
  abort();
  return MPI_SUCCESS;
}

static int MPI_was_initialized = 0;

int Petsc_MPI_Initialized(int *flag)
{
  *flag = MPI_was_initialized;
  return 0;
}

int Petsc_MPI_Finalize(void)
{
  MPI_was_initialized = 0;
  return 0;
}

/* -------------------     Fortran versions of several routines ------------------ */

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define mpi_init_             MPI_INIT
#define mpi_finalize_         MPI_FINALIZE
#define mpi_comm_size_        MPI_COMM_SIZE
#define mpi_comm_rank_        MPI_COMM_RANK
#define mpi_abort_            MPI_ABORT
#define mpi_allreduce_        MPI_ALLREDUCE
#define mpi_barrier_          MPI_BARRIER
#define mpi_bcast_            MPI_BCAST
#define mpi_gather_           MPI_GATHER
#define mpi_allgather_        MPI_ALLGATHER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define mpi_init_             mpi_init
#define mpi_finalize_         mpi_finalize
#define mpi_comm_size_        mpi_comm_size
#define mpi_comm_rank_        mpi_comm_rank
#define mpi_abort_            mpi_abort
#define mpi_allreduce_        mpi_allreduce
#define mpi_barrier_          mpi_barrier
#define mpi_bcast_            mpi_bcast
#define mpi_gather_           mpi_gather
#define mpi_allgather_        mpi_allgather
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
#define mpi_init_             mpi_init__
#define mpi_finalize_         mpi_finalize__
#define mpi_comm_size_        mpi_comm_size__
#define mpi_comm_rank_        mpi_comm_rank__
#define mpi_abort_            mpi_abort__
#define mpi_allreduce_        mpi_allreduce__
#define mpi_barrier_          mpi_barrier__
#define mpi_bcast_            mpi_bcast__
#define mpi_gather_           mpi_gather__
#define mpi_allgather_        mpi_allgather__
#endif

void PETSC_STDCALL  mpi_init_(int *ierr)
{
  MPI_was_initialized = 1;
  *ierr = MPI_SUCCESS;
}

void PETSC_STDCALL  mpi_finalize_(int *ierr)
{
  *ierr = MPI_SUCCESS;
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

void PETSC_STDCALL mpi_comm_split(MPI_Comm *comm,int *color,int *key, MPI_Comm *newcomm, int *ierr)
{
  *newcomm = *comm;
  *ierr=MPI_SUCCESS;
}

void PETSC_STDCALL mpi_abort_(MPI_Comm *comm,int *errorcode,int *ierr) 
{
  abort();
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

#if defined(__cplusplus)
}
#endif


