#define PETSCKSP_DLL

#include <../src/ksp/ksp/impls/gmres/agmres/agmresimpl.h>
/*
 *  This file implements the RODDEC algorithm : its purpose is to orthogonalize a set of vectors distributed across several processes. These processes are organized in a virtual ring.
 * References : [1] Sidje, Roger B. Alternatives for parallel Krylov subspace basis computation. Numer. Linear Algebra Appl. 4 (1997), no. 4, 305-331
 *
 *
 * initial author R. B. SIDJE,
 * modified : G.-A Atenekeng-Kahou, 2008
 * modified : D. NUENTSA WAKAM, 2011
 *
 */


/*
 * Take the processes that own the vectors and Organize them on a virtual ring.
 */
static PetscErrorCode  KSPAGMRESRoddecGivens(PetscReal*, PetscReal*, PetscReal*, PetscInt);

PetscErrorCode KSPAGMRESRoddecInitNeighboor(KSP ksp)
{
  MPI_Comm       comm;
  KSP_AGMRES     *agmres = (KSP_AGMRES*)(ksp->data);
  PetscErrorCode ierr;
  PetscMPIInt    First, Last, rank, size;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)agmres->vecs[0], &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&rank, &First, 1, MPI_INT, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&rank, &Last, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);

  if ((rank != Last) && (!rank)) {
    agmres->Ileft  = rank - 1;
    agmres->Iright = rank + 1;
  } else {
    if (rank == Last) {
      agmres->Ileft  = rank - 1;
      agmres->Iright = First;
    } else {
      {
        agmres->Ileft  = Last;
        agmres->Iright = rank + 1;
      }
    }
  }
  agmres->rank  = rank;
  agmres->size  = size;
  agmres->First = First;
  agmres->Last  = Last;
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPAGMRESRoddecGivens(PetscReal * c, PetscReal * s, PetscReal * r, PetscInt make_r)
{
  PetscReal a, b, t;

  PetscFunctionBegin;
  if (make_r == 1) {
    a = *c;
    b = *s;
    if (b == 0.e0) {
      *c = 1.e0;
      *s = 0.e0;
    } else {
      if (PetscAbsReal(b) > PetscAbsReal(a)) {
        t  = -a / b;
        *s = 1.e0 / PetscSqrtReal(1.e0 + t * t);
        *c = (*s) * t;
      } else {
        t  = -b / a;
        *c = 1.e0 / PetscSqrtReal(1.e0 + t * t);
        *s = (*c) * t;
      }
    }
    if (*c == 0.e0) {
      *r = 1.e0;
    } else {
      if (PetscAbsReal(*s) < PetscAbsReal(*c)) {
        *r = PetscSign(*c) * (*s) / 2.e0;
      } else {
        *r = PetscSign(*s) * 2.e0 / (*c);
      }
    }
  }

  if (*r == 1.e0) {
    *c = 0.e0;
    *s = 1.e0;
  } else {
    if (PetscAbsReal(*r) < 1.e0) {
      *s = 2.e0 * (*r);
      *c = PetscSqrtReal(1.e0 - (*s) * (*s));
    } else {
      *c = 2.e0 / (*r);
      *s = PetscSqrtReal(1.e0 - (*c) * (*c));
    }
  }
  PetscFunctionReturn(0);

}

/*
 * Compute the QR factorization of the Krylov basis vectors
 * Input :
 *  - the vectors are availabe in agmres->vecs (alias VEC_V)
 *  - nvec :  the number of vectors
 * Output :
 *  - agmres->Qloc : product of elementary reflectors for the QR of the (local part) of the vectors.
 *  - agmres->sgn :  Sign of the rotations
 *  - agmres->tloc : scalar factors of the elementary reflectors.

 */
PetscErrorCode KSPAGMRESRoddec(KSP ksp, PetscInt nvec)
{
  KSP_AGMRES     *agmres = (KSP_AGMRES*) ksp->data;
  MPI_Comm       comm;
  PetscScalar    *Qloc   = agmres->Qloc;
  PetscScalar    *sgn    = agmres->sgn;
  PetscScalar    *tloc   = agmres->tloc;
  PetscErrorCode ierr;
  PetscReal      *wbufptr = agmres->wbufptr;
  PetscMPIInt    rank     = agmres->rank;
  PetscMPIInt    First    = agmres->First;
  PetscMPIInt    Last     = agmres->Last;
  PetscBLASInt   pas,len,bnloc,bpos;
  PetscInt       nloc,d, i, j, k;
  PetscInt       pos;
  PetscReal      c, s, rho, Ajj, val, tt, old;
  PetscScalar    *col;
  MPI_Status     status;
  PetscBLASInt   N = MAXKSPSIZE + 1;


  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(KSP_AGMRESRoddec,ksp,0,0,0);CHKERRQ(ierr);
  ierr = PetscArrayzero(agmres->Rloc, N*N);CHKERRQ(ierr);
  /* check input arguments */
  if (nvec < 1) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE, "The number of input vectors shoud be positive");
  ierr = VecGetLocalSize(VEC_V(0), &nloc);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nloc,&bnloc);CHKERRQ(ierr);
  if (nvec > nloc) SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_WRONG, "In QR factorization, the number of local rows should be greater or equal to the number of columns");
  pas = 1;
  /* Copy the vectors of the basis */
  for (j = 0; j < nvec; j++) {
    ierr = VecGetArray(VEC_V(j), &col);CHKERRQ(ierr);
    PetscStackCallBLAS("BLAScopy",BLAScopy_(&bnloc, col, &pas, &Qloc[j*nloc], &pas));
    ierr = VecRestoreArray(VEC_V(j), &col);CHKERRQ(ierr);
  }
  /* Each process performs a local QR on its own block */
  for (j = 0; j < nvec; j++) {
    len = nloc - j;
    Ajj = Qloc[j*nloc+j];
    PetscStackCallBLAS("BLASnrm2",rho = -PetscSign(Ajj) * BLASnrm2_(&len, &(Qloc[j*nloc+j]), &pas));
    if (rho == 0.0) tloc[j] = 0.0;
    else {
      tloc[j] = (Ajj - rho) / rho;
      len     = len - 1;
      val     = 1.0 / (Ajj - rho);
      PetscStackCallBLAS("BLASscal",BLASscal_(&len, &val, &(Qloc[j*nloc+j+1]), &pas));
      Qloc[j*nloc+j] = 1.0;
      len            = len + 1;
      for (k = j + 1; k < nvec; k++) {
        PetscStackCallBLAS("BLASdot",tt = tloc[j] * BLASdot_(&len, &(Qloc[j*nloc+j]), &pas, &(Qloc[k*nloc+j]), &pas));
        PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&len, &tt, &(Qloc[j*nloc+j]), &pas, &(Qloc[k*nloc+j]), &pas));
      }
      Qloc[j*nloc+j] = rho;
    }
  }
  /* annihilate undesirable Rloc, diagonal by diagonal*/
  for (d = 0; d < nvec; d++) {
    len = nvec - d;
    if (rank == First) {
      PetscStackCallBLAS("BLAScopy",BLAScopy_(&len, &(Qloc[d*nloc+d]), &bnloc, &(wbufptr[d]), &pas));
      ierr = MPI_Send(&(wbufptr[d]), len, MPIU_SCALAR, rank + 1, agmres->tag, comm);CHKERRQ(ierr);
    } else {
      ierr = MPI_Recv(&(wbufptr[d]), len, MPIU_SCALAR, rank - 1, agmres->tag, comm, &status);CHKERRQ(ierr);
      /* Elimination of Rloc(1,d)*/
      c    = wbufptr[d];
      s    = Qloc[d*nloc];
      ierr = KSPAGMRESRoddecGivens(&c, &s, &rho, 1);CHKERRQ(ierr);
      /* Apply Givens Rotation*/
      for (k = d; k < nvec; k++) {
        old          = wbufptr[k];
        wbufptr[k]   =  c * old - s * Qloc[k*nloc];
        Qloc[k*nloc] =  s * old + c * Qloc[k*nloc];
      }
      Qloc[d*nloc] = rho;
      if (rank != Last) {
        ierr = MPI_Send(& (wbufptr[d]), len, MPIU_SCALAR, rank + 1, agmres->tag, comm);CHKERRQ(ierr);
      }
      /* zero-out the d-th diagonal of Rloc ...*/
      for (j = d + 1; j < nvec; j++) {
        /* elimination of Rloc[i][j]*/
        i    = j - d;
        c    = Qloc[j*nloc+i-1];
        s    = Qloc[j*nloc+i];
        ierr = KSPAGMRESRoddecGivens(&c, &s, &rho, 1);CHKERRQ(ierr);
        for (k = j; k < nvec; k++) {
          old              = Qloc[k*nloc+i-1];
          Qloc[k*nloc+i-1] = c * old - s * Qloc[k*nloc+i];
          Qloc[k*nloc+i]   =   s * old + c * Qloc[k*nloc+i];
        }
        Qloc[j*nloc+i] = rho;
      }
      if (rank == Last) {
        PetscStackCallBLAS("BLAScopy",BLAScopy_(&len, &(wbufptr[d]), &pas, RLOC(d,d), &N));
        for (k = d + 1; k < nvec; k++) *RLOC(k,d) = 0.0;
      }
    }
  }

  if (rank == Last) {
    for (d = 0; d < nvec; d++) {
      pos    = nvec - d;
      ierr = PetscBLASIntCast(pos,&bpos);CHKERRQ(ierr);
      sgn[d] = PetscSign(*RLOC(d,d));
      PetscStackCallBLAS("BLASscal",BLASscal_(&bpos, &(sgn[d]), RLOC(d,d), &N));
    }
  }
  /* BroadCast Rloc to all other processes
   * NWD : should not be needed
   */
  ierr = MPI_Bcast(agmres->Rloc,N*N,MPIU_SCALAR,Last,comm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(KSP_AGMRESRoddec,ksp,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
 *  Computes Out <-- Q * In where Q is the orthogonal matrix from AGMRESRoddec
 *  Input
 *   - Qloc, sgn, tloc, nvec (see AGMRESRoddec above)
 *   - In : input vector (size nvec)
 *  Output :
 *   - Out : Petsc vector (distributed as the basis vectors)
 */
PetscErrorCode KSPAGMRESRodvec(KSP ksp, PetscInt nvec, PetscScalar *In, Vec Out)
{
  KSP_AGMRES     *agmres  = (KSP_AGMRES*) ksp->data;
  MPI_Comm       comm;
  PetscScalar    *Qloc    = agmres->Qloc;
  PetscScalar    *sgn     = agmres->sgn;
  PetscScalar    *tloc    = agmres->tloc;
  PetscMPIInt    rank     = agmres->rank;
  PetscMPIInt    First    = agmres->First, Last = agmres->Last;
  PetscMPIInt    Iright   = agmres->Iright, Ileft = agmres->Ileft;
  PetscScalar    *y, *zloc;
  PetscErrorCode ierr;
  PetscInt       nloc,d, len, i, j;
  PetscBLASInt   bnvec,pas,blen;
  PetscInt       dpt;
  PetscReal      c, s, rho, zp, zq, yd=0.0, tt;
  MPI_Status     status;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(nvec,&bnvec);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
  pas  = 1;
  ierr = VecGetLocalSize(VEC_V(0), &nloc);CHKERRQ(ierr);
  ierr = PetscMalloc1(nvec, &y);CHKERRQ(ierr);
  ierr = PetscArraycpy(y, In, nvec);CHKERRQ(ierr);
  ierr = VecGetArray(Out, &zloc);CHKERRQ(ierr);

  if (rank == Last) {
    for (i = 0; i < nvec; i++) y[i] = sgn[i] * y[i];
  }
  for (i = 0; i < nloc; i++) zloc[i] = 0.0;
  if (agmres->size == 1) PetscStackCallBLAS("BLAScopy",BLAScopy_(&bnvec, y, &pas, &(zloc[0]), &pas));
  else {
    for (d = nvec - 1; d >= 0; d--) {
      if (rank == First) {
        ierr = MPI_Recv(&(zloc[d]), 1, MPIU_SCALAR, Iright, agmres->tag, comm, &status);CHKERRQ(ierr);
      } else {
        for (j = nvec - 1; j >= d + 1; j--) {
          i         = j - d;
          ierr      = KSPAGMRESRoddecGivens(&c, &s, &(Qloc[j * nloc + i]), 0);CHKERRQ(ierr);
          zp        = zloc[i-1];
          zq        = zloc[i];
          zloc[i-1] =     c * zp + s * zq;
          zloc[i]   =     -s * zp + c * zq;
        }
        ierr = KSPAGMRESRoddecGivens(&c, &s, &(Qloc[d * nloc]), 0);CHKERRQ(ierr);
        if (rank == Last) {
          zp      = y[d];
          zq      = zloc[0];
          y[d]    =      c * zp + s * zq;
          zloc[0] =   -s * zp + c * zq;
          ierr    = MPI_Send(&(y[d]), 1, MPIU_SCALAR, Ileft, agmres->tag, comm);CHKERRQ(ierr);
        } else {
          ierr    = MPI_Recv(&yd, 1, MPIU_SCALAR, Iright, agmres->tag, comm, &status);CHKERRQ(ierr);
          zp      = yd;
          zq      = zloc[0];
          yd      =      c * zp + s * zq;
          zloc[0] =   -s * zp + c * zq;
          ierr    = MPI_Send(&yd, 1, MPIU_SCALAR, Ileft, agmres->tag, comm);CHKERRQ(ierr);
        }
      }
    }
  }
  for (j = nvec - 1; j >= 0; j--) {
    dpt = j * nloc + j;
    if (tloc[j] != 0.0) {
      len       = nloc - j;
      ierr      = PetscBLASIntCast(len,&blen);CHKERRQ(ierr);
      rho       = Qloc[dpt];
      Qloc[dpt] = 1.0;
      tt        = tloc[j] * (BLASdot_(&blen, &(Qloc[dpt]), &pas, &(zloc[j]), &pas));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&blen, &tt, &(Qloc[dpt]), &pas, &(zloc[j]), &pas));
      Qloc[dpt] = rho;
    }
  }
  ierr = VecRestoreArray(Out, &zloc);CHKERRQ(ierr);
  ierr = PetscFree(y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
