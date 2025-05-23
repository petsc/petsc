#pragma once

#include <../src/vec/vec/impls/dvecimpl.h>

typedef struct {
  PetscInt insertmode;
  PetscInt count;
  PetscInt bcount;
} VecAssemblyHeader;

typedef struct {
  PetscInt    *ints;
  PetscInt    *intb;
  PetscScalar *scalars;
  PetscScalar *scalarb;
  char         pendings;
  char         pendingb;
} VecAssemblyFrame;

typedef struct {
  VECHEADER
  PetscInt   nghost;      /* number of ghost points on this process */
  IS         ghost;       /* global indices of ghost values */
  Vec        localrep;    /* local representation of vector */
  VecScatter localupdate; /* scatter to update ghost values */

  PetscBool          assembly_subset;     /* Subsequent assemblies will set a subset (perhaps equal) of off-process entries set on first assembly */
  PetscBool          first_assembly_done; /* Is the first time assembly done? */
  PetscBool          use_status;          /* Use MPI_Status to determine number of items in each message */
  PetscMPIInt        nsendranks;
  PetscMPIInt        nrecvranks;
  PetscMPIInt       *sendranks;
  PetscMPIInt       *recvranks;
  VecAssemblyHeader *sendhdr, *recvhdr;
  VecAssemblyFrame  *sendptrs; /* pointers to the main messages */
  MPI_Request       *sendreqs;
  MPI_Request       *recvreqs;
  PetscSegBuffer     segrecvint;
  PetscSegBuffer     segrecvscalar;
  PetscSegBuffer     segrecvframe;
#if defined(PETSC_HAVE_NVSHMEM)
  PetscBool use_nvshmem; /* Try to use NVSHMEM in communication of, for example, VecNorm */
#endif

  /* COO fields, assuming m is the vector's local size */
  PetscCount  coo_n;
  PetscCount  tot1;  /* Total local entries in COO arrays */
  PetscCount *jmap1; /* [m+1]: i-th entry of the vector has jmap1[i+1]-jmap1[i] repeats in COO arrays */
  PetscCount *perm1; /* [tot1]: permutation array for local entries */

  PetscCount  nnz2;  /* Unique entries in recvbuf */
  PetscCount *imap2; /* [nnz2]: i-th unique entry in recvbuf is imap2[i]-th entry in the vector */
  PetscCount *jmap2; /* [nnz2+1] */
  PetscCount *perm2; /* [recvlen] */

  PetscSF      coo_sf;
  PetscCount   sendlen, recvlen;  /* Lengths (in unit of PetscScalar) of send/recvbuf */
  PetscCount  *Cperm;             /* [sendlen]: permutation array to fill sendbuf[]. 'C' for communication */
  PetscScalar *sendbuf, *recvbuf; /* Buffers for remote values in VecSetValuesCOO() */
} Vec_MPI;

PETSC_INTERN PetscErrorCode VecMTDot_MPI(Vec, PetscInt, const Vec[], PetscScalar *);
PETSC_INTERN PetscErrorCode VecView_MPI_Binary(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_Draw_LG(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_Socket(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_HDF5(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_ADIOS(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecGetSize_MPI(Vec, PetscInt *);
PETSC_INTERN PetscErrorCode VecGetValues_MPI(Vec, PetscInt, const PetscInt[], PetscScalar[]);
PETSC_INTERN PetscErrorCode VecSetValues_MPI(Vec, PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
PETSC_INTERN PetscErrorCode VecSetValuesBlocked_MPI(Vec, PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
PETSC_INTERN PetscErrorCode VecAssemblyBegin_MPI(Vec);
PETSC_INTERN PetscErrorCode VecAssemblyEnd_MPI(Vec);
PETSC_INTERN PetscErrorCode VecAssemblyReset_MPI(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_MPI(Vec);
PETSC_INTERN PetscErrorCode VecMDot_MPI_GEMV(Vec, PetscInt, const Vec[], PetscScalar *);
PETSC_INTERN PetscErrorCode VecMTDot_MPI_GEMV(Vec, PetscInt, const Vec[], PetscScalar *);

PETSC_INTERN PetscErrorCode VecDuplicate_MPI(Vec, Vec *);
PETSC_INTERN PetscErrorCode VecDuplicateWithArray_MPI(Vec, const PetscScalar *, Vec *);
PETSC_INTERN PetscErrorCode VecSetPreallocationCOO_MPI(Vec, PetscCount, const PetscInt[]);
PETSC_INTERN PetscErrorCode VecSetValuesCOO_MPI(Vec, const PetscScalar[], InsertMode);

PETSC_INTERN PetscErrorCode VecDot_MPI(Vec, Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecMDot_MPI(Vec, PetscInt, const Vec[], PetscScalar *);
PETSC_INTERN PetscErrorCode VecTDot_MPI(Vec, Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecNorm_MPI(Vec, NormType, PetscReal *);
PETSC_INTERN PetscErrorCode VecMax_MPI(Vec, PetscInt *, PetscReal *);
PETSC_INTERN PetscErrorCode VecMin_MPI(Vec, PetscInt *, PetscReal *);
PETSC_INTERN PetscErrorCode VecMaxPointwiseDivide_MPI(Vec, Vec, PetscReal *);
PETSC_INTERN PetscErrorCode VecPlaceArray_MPI(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecResetArray_MPI(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPI_Private(Vec, PetscBool, PetscInt, const PetscScalar[]);

PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode VecView_MPI(Vec, PetscViewer);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode VecDestroy_MPI(Vec);

static inline PetscErrorCode VecMXDot_MPI_Default(Vec xin, PetscInt nv, const Vec y[], PetscScalar *z, PetscErrorCode (*VecMXDot_SeqFn)(Vec, PetscInt, const Vec[], PetscScalar *))
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_SeqFn(xin, nv, y, z));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, z, nv, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)xin)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode VecXDot_MPI_Default(Vec xin, Vec yin, PetscScalar *z, PetscErrorCode (*VecXDot_SeqFn)(Vec, Vec, PetscScalar *))
{
  PetscFunctionBegin;
  PetscCall(VecXDot_SeqFn(xin, yin, z));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, z, 1, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)xin)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode VecMinMax_MPI_Default(Vec xin, PetscInt *idx, PetscReal *z, PetscErrorCode (*VecMinMax_SeqFn)(Vec, PetscInt *, PetscReal *), const MPI_Op ops[2])
{
  PetscFunctionBegin;
  /* Find the local max */
  PetscCall(VecMinMax_SeqFn(xin, idx, z));
  if (PetscDefined(HAVE_MPIUNI)) PetscFunctionReturn(PETSC_SUCCESS);
  /* Find the global max */
  if (idx) {
    struct {
      PetscReal v;
      PetscInt  i;
    } in = {*z, *idx + xin->map->rstart};

    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &in, 1, MPIU_REAL_INT, ops[0], PetscObjectComm((PetscObject)xin)));
    *z   = in.v;
    *idx = in.i;
  } else {
    /* User does not need idx */
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, z, 1, MPIU_REAL, ops[1], PetscObjectComm((PetscObject)xin)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode VecDotNorm2_MPI_Default(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm, PetscErrorCode (*VecDotNorm2_SeqFn)(Vec, Vec, PetscScalar *, PetscScalar *))
{
  PetscFunctionBegin;
  PetscCall(VecDotNorm2_SeqFn(s, t, dp, nm));
  {
    PetscScalar sum[] = {*dp, *nm};

    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &sum, 2, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)s)));
    *dp = sum[0];
    *nm = sum[1];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode VecNorm_MPI_Default(Vec xin, NormType type, PetscReal *z, PetscErrorCode (*VecNorm_SeqFn)(Vec, NormType, PetscReal *))
{
  PetscMPIInt zn = 1;
  MPI_Op      op = MPIU_SUM;

  PetscFunctionBegin;
  PetscCall(VecNorm_SeqFn(xin, type, z));
  switch (type) {
  case NORM_1_AND_2:
    // the 2 norm needs to be squared below before being summed. NORM_2 stores the norm in the
    // first slot but while NORM_1_AND_2 stores it in the second
    z[1] *= z[1];
    zn = 2;
    break;
  case NORM_2:
  case NORM_FROBENIUS:
    z[0] *= z[0];
  case NORM_1:
    break;
  case NORM_INFINITY:
    op = MPIU_MAX;
    break;
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, z, zn, MPIU_REAL, op, PetscObjectComm((PetscObject)xin)));
  if (type == NORM_2 || type == NORM_FROBENIUS || type == NORM_1_AND_2) z[type == NORM_1_AND_2] = PetscSqrtReal(z[type == NORM_1_AND_2]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode VecErrorWeightedNorms_MPI_Default(Vec U, Vec Y, Vec E, NormType wnormtype, PetscReal atol, Vec vatol, PetscReal rtol, Vec vrtol, PetscReal ignore_max, PetscReal *norm, PetscInt *norm_loc, PetscReal *norma, PetscInt *norma_loc, PetscReal *normr, PetscInt *normr_loc, PetscErrorCode (*SeqFn)(Vec, Vec, Vec, NormType, PetscReal, Vec, PetscReal, Vec, PetscReal, PetscReal *, PetscInt *, PetscReal *, PetscInt *, PetscReal *, PetscInt *))
{
  PetscReal loc[6];

  PetscFunctionBegin;
  PetscCall(SeqFn(U, Y, E, wnormtype, atol, vatol, rtol, vrtol, ignore_max, norm, norm_loc, norma, norma_loc, normr, normr_loc));
  if (wnormtype == NORM_2) {
    loc[0] = PetscSqr(*norm);
    loc[1] = PetscSqr(*norma);
    loc[2] = PetscSqr(*normr);
  } else {
    loc[0] = *norm;
    loc[1] = *norma;
    loc[2] = *normr;
  }
  loc[3] = (PetscReal)*norm_loc;
  loc[4] = (PetscReal)*norma_loc;
  loc[5] = (PetscReal)*normr_loc;
  if (wnormtype == NORM_2) {
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, loc, 6, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)U)));
  } else {
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, loc, 3, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)U)));
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, loc + 3, 3, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)U)));
  }
  if (wnormtype == NORM_2) {
    *norm  = PetscSqrtReal(loc[0]);
    *norma = PetscSqrtReal(loc[1]);
    *normr = PetscSqrtReal(loc[2]);
  } else {
    *norm  = loc[0];
    *norma = loc[1];
    *normr = loc[2];
  }
  *norm_loc  = loc[3];
  *norma_loc = loc[4];
  *normr_loc = loc[5];
  PetscFunctionReturn(PETSC_SUCCESS);
}
