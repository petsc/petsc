
#if !defined(__PVECIMPL)
#define __PVECIMPL

#include <../src/vec/vec/impls/dvecimpl.h>

typedef struct {
  PetscInt insertmode;
  PetscInt count;
  PetscInt bcount;
} VecAssemblyHeader;

typedef struct {
  PetscInt *ints;
  PetscInt *intb;
  PetscScalar *scalars;
  PetscScalar *scalarb;
  char        pendings;
  char        pendingb;
} VecAssemblyFrame;

typedef struct {
  VECHEADER
  PetscInt    nghost;                   /* number of ghost points on this process */
  Vec         localrep;                 /* local representation of vector */
  VecScatter  localupdate;              /* scatter to update ghost values */

  PetscBool   assembly_subset;          /* Subsequent assemblies will set a subset (perhaps equal) of off-process entries set on first assembly */
  PetscBool   first_assembly_done;      /* Is the first time assembly done? */
  PetscBool   use_status;               /* Use MPI_Status to determine number of items in each message */
  PetscMPIInt nsendranks;
  PetscMPIInt nrecvranks;
  PetscMPIInt *sendranks;
  PetscMPIInt *recvranks;
  VecAssemblyHeader *sendhdr,*recvhdr;
  VecAssemblyFrame *sendptrs;   /* pointers to the main messages */
  MPI_Request    *sendreqs;
  MPI_Request    *recvreqs;
  PetscSegBuffer segrecvint;
  PetscSegBuffer segrecvscalar;
  PetscSegBuffer segrecvframe;
 #if defined(PETSC_HAVE_NVSHMEM)
  PetscBool      use_nvshmem; /* Try to use NVSHMEM in communication of, for example, VecNorm */
 #endif

  /* COO fields, assuming m is the vector's local size */
  PetscCount  coo_n;
  PetscCount  tot1; /* Total local entries in COO arrays */
  PetscCount  *jmap1; /* [m+1]: i-th entry of the vector has jmap1[i+1]-jmap1[i] repeats in COO arrays */
  PetscCount  *perm1; /* [tot1]: permutation array for local entries */

  PetscCount  nnz2;   /* Unique entries in recvbuf */
  PetscCount  *imap2; /* [nnz2]: i-th unique entry in recvbuf is imap2[i]-th entry in the vector */
  PetscCount  *jmap2; /* [nnz2+1] */
  PetscCount  *perm2; /* [recvlen] */

  PetscSF     coo_sf;
  PetscCount  sendlen,recvlen; /* Lengths (in unit of PetscScalar) of send/recvbuf */
  PetscCount  *Cperm; /* [sendlen]: permutation array to fill sendbuf[]. 'C' for communication */
  PetscScalar *sendbuf,*recvbuf;  /* Buffers for remote values in VecSetValuesCOO() */
} Vec_MPI;

PETSC_INTERN PetscErrorCode VecDot_MPI(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecMDot_MPI(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_INTERN PetscErrorCode VecTDot_MPI(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecMTDot_MPI(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_INTERN PetscErrorCode VecNorm_MPI(Vec,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode VecMax_MPI(Vec,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode VecMin_MPI(Vec,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode VecDestroy_MPI(Vec);
PETSC_INTERN PetscErrorCode VecView_MPI_Binary(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_Draw_LG(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_Socket(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_HDF5(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_ADIOS(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_MPI(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecGetSize_MPI(Vec,PetscInt*);
PETSC_INTERN PetscErrorCode VecGetValues_MPI(Vec,PetscInt,const PetscInt [], PetscScalar []);
PETSC_INTERN PetscErrorCode VecSetValues_MPI(Vec,PetscInt,const PetscInt [],const PetscScalar[],InsertMode);
PETSC_INTERN PetscErrorCode VecSetValuesBlocked_MPI(Vec,PetscInt,const PetscInt [],const PetscScalar[],InsertMode);
PETSC_INTERN PetscErrorCode VecAssemblyBegin_MPI(Vec);
PETSC_INTERN PetscErrorCode VecAssemblyEnd_MPI(Vec);
PETSC_INTERN PetscErrorCode VecAssemblyReset_MPI(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPI_Private(Vec,PetscBool,PetscInt,const PetscScalar[]);
PETSC_EXTERN PetscErrorCode VecCreate_MPI(Vec);
PETSC_INTERN PetscErrorCode VecDuplicate_MPI(Vec,Vec*);
PETSC_INTERN PetscErrorCode VecSetPreallocationCOO_MPI(Vec,PetscCount,const PetscInt[]);
PETSC_INTERN PetscErrorCode VecSetValuesCOO_MPI(Vec,const PetscScalar[],InsertMode);
#endif

