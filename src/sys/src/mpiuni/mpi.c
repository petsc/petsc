#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mpi.c,v 1.45 1998/04/01 00:21:03 balay Exp balay $";
#endif

#include "petsc.h"               /*I   "petsc.h"   I*/
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h" 
#define MPI_SUCCESS 0
void    *MPIUNI_TMP   = 0;
int     MPIUNI_DATASIZE[5] = { sizeof(int),sizeof(float),sizeof(double),
                               2*sizeof(double),sizeof(char)};

PLogDouble MPI_Wtime(void)
{
  PLogDouble d;
  PetscGetTime(&d);
  return d;
}

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

/*
   Used to set the built-in MPI_TAG_UB attribute
*/
static int Keyval_setup(void)
{
  attr[0].active        = 1;
  attr[0].attribute_val = &mpi_tag_ub;
  return 0;
}

int MPI_Keyval_create(MPI_Copy_function *copy_fn,MPI_Delete_function *delete_fn,int *keyval,
                      void *extra_state)
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

int MPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val)
{
  attr[keyval].active        = 1;
  attr[keyval].attribute_val = attribute_val;
  return MPI_SUCCESS;
}
  
int MPI_Attr_delete(MPI_Comm comm, int keyval)
{
  if (attr[keyval].active && attr[keyval].del) {
    (*(attr[keyval].del))(comm,keyval,attr[keyval].attribute_val,attr[keyval].extra_state);
  }
  attr[keyval].active        = 0;
  attr[keyval].attribute_val = 0;
  return MPI_SUCCESS;
}

int MPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag)
{
  if (keyval == 0) Keyval_setup();
  *flag                  = attr[keyval].active;
  *(int **)attribute_val = (int *)attr[keyval].attribute_val;
  return MPI_SUCCESS;
}

static int dups = 0;
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
  for ( i=0; i<num_attr; i++ ) {
    if (attr[i].active && attr[i].del) {
      (*attr[i].del)(*comm,i,attr[i].attribute_val,attr[i].extra_state);
    }
    attr[i].active = 0;
  }
  return MPI_SUCCESS;
}

/* --------------------------------------------------------------------------*/

int MPI_Abort(MPI_Comm comm,int errorcode) 
{
  PetscError(__LINE__,"mpi_abort",__FILE__,__SDIR__,errorcode,0,"[0] Aborting program!");
  exit(errorcode); 
  return MPI_SUCCESS;
}

static int MPI_was_initialized = 0;

int MPI_Initialized(int *flag)
{
  *flag = MPI_was_initialized;
  return 0;
}

int MPI_Finalize(void)
{
  MPI_was_initialized = 0;
  return 0;
}

/* -------------------     Fortran versions of several routines ------------------ */

#if defined(__cplusplus)
extern "C" {
#endif

/******mpi_init*******/
void  mpi_init(int *ierr)
{
  MPI_was_initialized = 1;
  *ierr = MPI_SUCCESS;
}

void  mpi_init_(int *ierr)
{
  MPI_was_initialized = 1;
  *ierr = MPI_SUCCESS;
}

void  mpi_init__(int *ierr)
{
  MPI_was_initialized = 1;
  *ierr = MPI_SUCCESS;
}

void  MPI_INIT(int *ierr)
{
  MPI_was_initialized = 1;
  *ierr = MPI_SUCCESS;
}

/******mpi_comm_size*******/
void mpi_comm_size(MPI_Comm *comm,int *size,int *ierr) 
{
  *size = 1;
  *ierr = 0;
}

void mpi_comm_size_(MPI_Comm *comm,int *size,int *ierr) 
{
  *size = 1;
  *ierr = 0;
}

void mpi_comm_size__(MPI_Comm *comm,int *size,int *ierr) 
{
  *size = 1;
  *ierr = 0;
}

void MPI_COMM_SIZE(MPI_Comm *comm,int *size,int *ierr) 
{
  *size = 1;
  *ierr = 0;
}

/******mpi_comm_rank*******/
void mpi_comm_rank(MPI_Comm *comm,int *rank,int *ierr)
{
  *rank=0;
  *ierr=MPI_SUCCESS;
}

void mpi_comm_rank_(MPI_Comm *comm,int *rank,int *ierr)
{
  *rank=0;
  *ierr=MPI_SUCCESS;
}

void mpi_comm_rank__(MPI_Comm *comm,int *rank,int *ierr)
{
  *rank=0;
  *ierr=MPI_SUCCESS;
}

void MPI_COMM_RANK(MPI_Comm *comm,int *rank,int *ierr)
{
  *rank=0;
  *ierr=MPI_SUCCESS;
}

/******mpi_wtick*******/
PLogDouble mpi_wtick(void) 
{
  (*PetscErrorPrintf)("MPI_Wtime: use PetscGetTime instead.\n");
  return 0.0;
}

PLogDouble mpi_wtick_(void) 
{
  (*PetscErrorPrintf)("MPI_Wtime: use PetscGetTime instead.\n");
  return 0.0;
}

PLogDouble mpi_wtick__(void) 
{
  (*PetscErrorPrintf)("MPI_Wtime: use PetscGetTime instead.\n");
  return 0.0;
}

PLogDouble MPI_WTICK(void) 
{
  (*PetscErrorPrintf)("MPI_Wtime: use PetscGetTime instead.\n");
  return 0.0;
}

/*******mpi_wtime******/
PLogDouble mpi_wtime(void)
{
  PLogDouble d;
  PetscGetTime(&d);
  return d;
}

PLogDouble mpi_wtime_(void)
{
  PLogDouble d;
  PetscGetTime(&d);
  return d;
}

PLogDouble mpi_wtime__(void)
{
  PLogDouble d;
  PetscGetTime(&d);
  return d;
}

PLogDouble MPI_WTIME(void)
{
  PLogDouble d;
  PetscGetTime(&d);
  return d;
}

/*******mpi_abort******/
void mpi_abort(MPI_Comm *comm,int *errorcode,int *ierr) 
{
  PetscError(__LINE__,"mpi_abort",__FILE__,__SDIR__,*errorcode,0,"[0] Aborting program!");
  exit(*errorcode); 
  *ierr = MPI_SUCCESS;
}

void mpi_abort_(MPI_Comm *comm,int *errorcode,int *ierr) 
{
  PetscError(__LINE__,"mpi_abort",__FILE__,__SDIR__,*errorcode,0,"[0] Aborting program!");
  exit(*errorcode);
  *ierr = MPI_SUCCESS;
}

void mpi_abort__(MPI_Comm *comm,int *errorcode,int *ierr) 
{
  PetscError(__LINE__,"mpi_abort",__FILE__,__SDIR__,*errorcode,0,"[0] Aborting program!");
  exit(*errorcode);
  *ierr = MPI_SUCCESS;
}

void MPI_ABORT(MPI_Comm *comm,int *errorcode,int *ierr) 
{
  PetscError(__LINE__,"mpi_abort",__FILE__,__SDIR__,*errorcode,0,"[0] Aborting program!");
  exit(*errorcode);
  *ierr = MPI_SUCCESS;
}
/*******mpi_allreduce******/
void mpi_allreduce(void *sendbuf,void *recvbuf,int *count,int *datatype,
                   int *op,int *comm,int *ierr) 
{
  PetscMemcpy( recvbuf, sendbuf, (*count)*MPIUNI_DATASIZE[*datatype]);
  *ierr = MPI_SUCCESS;
} 
void mpi_allreduce_(void *sendbuf,void *recvbuf,int *count,int *datatype,
                   int *op,int *comm,int *ierr) 
{
  PetscMemcpy( recvbuf, sendbuf, (*count)*MPIUNI_DATASIZE[*datatype]);
  *ierr = MPI_SUCCESS;
} 
void mpi_allreduce__(void *sendbuf,void *recvbuf,int *count,int *datatype,
                   int *op,int *comm,int *ierr) 
{
  PetscMemcpy( recvbuf, sendbuf, (*count)*MPIUNI_DATASIZE[*datatype]);
  *ierr = MPI_SUCCESS;
} 
void MPI_ALLREDUCE(void *sendbuf,void *recvbuf,int *count,int *datatype,
                   int *op,int *comm,int *ierr) 
{
  PetscMemcpy( recvbuf, sendbuf, (*count)*MPIUNI_DATASIZE[*datatype]);
  *ierr = MPI_SUCCESS;
} 


#if defined(__cplusplus)
}
#endif


