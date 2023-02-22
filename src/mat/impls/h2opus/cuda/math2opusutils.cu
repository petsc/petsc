#include <petsc/private/matimpl.h>
#include <petsc/private/vecimpl.h>
#include <petscsf.h>
#if defined(PETSC_HAVE_CUDA)
  #include <thrust/for_each.h>
  #include <thrust/device_vector.h>
  #include <thrust/execution_policy.h>
#endif

PETSC_INTERN PetscErrorCode PetscSFGetVectorSF(PetscSF sf, PetscInt nv, PetscInt ldr, PetscInt ldl, PetscSF *vsf)
{
  PetscSF            rankssf;
  const PetscSFNode *iremote;
  PetscSFNode       *viremote, *rremotes;
  const PetscInt    *ilocal;
  PetscInt          *vilocal = NULL, *ldrs;
  const PetscMPIInt *ranks;
  PetscMPIInt       *sranks;
  PetscInt           nranks, nr, nl, vnr, vnl, i, v, j, maxl;
  MPI_Comm           comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscValidLogicalCollectiveInt(sf, nv, 2);
  PetscValidPointer(vsf, 5);
  if (nv == 1) {
    PetscCall(PetscObjectReference((PetscObject)sf));
    *vsf = sf;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCall(PetscSFGetGraph(sf, &nr, &nl, &ilocal, &iremote));
  PetscCall(PetscSFGetLeafRange(sf, NULL, &maxl));
  maxl += 1;
  if (ldl == PETSC_DECIDE) ldl = maxl;
  if (ldr == PETSC_DECIDE) ldr = nr;
  PetscCheck(ldr >= nr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid leading dimension %" PetscInt_FMT " < %" PetscInt_FMT, ldr, nr);
  PetscCheck(ldl >= maxl, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid leading dimension %" PetscInt_FMT " < %" PetscInt_FMT, ldl, maxl);
  vnr = nr * nv;
  vnl = nl * nv;
  PetscCall(PetscMalloc1(vnl, &viremote));
  if (ilocal) PetscCall(PetscMalloc1(vnl, &vilocal));

  /* TODO: Should this special SF be available, e.g.
     PetscSFGetRanksSF or similar? */
  PetscCall(PetscSFGetRootRanks(sf, &nranks, &ranks, NULL, NULL, NULL));
  PetscCall(PetscMalloc1(nranks, &sranks));
  PetscCall(PetscArraycpy(sranks, ranks, nranks));
  PetscCall(PetscSortMPIInt(nranks, sranks));
  PetscCall(PetscMalloc1(nranks, &rremotes));
  for (i = 0; i < nranks; i++) {
    rremotes[i].rank  = sranks[i];
    rremotes[i].index = 0;
  }
  PetscCall(PetscSFDuplicate(sf, PETSCSF_DUPLICATE_CONFONLY, &rankssf));
  PetscCall(PetscSFSetGraph(rankssf, 1, nranks, NULL, PETSC_OWN_POINTER, rremotes, PETSC_OWN_POINTER));
  PetscCall(PetscMalloc1(nranks, &ldrs));
  PetscCall(PetscSFBcastBegin(rankssf, MPIU_INT, &ldr, ldrs, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(rankssf, MPIU_INT, &ldr, ldrs, MPI_REPLACE));
  PetscCall(PetscSFDestroy(&rankssf));

  j = -1;
  for (i = 0; i < nl; i++) {
    const PetscInt r  = iremote[i].rank;
    const PetscInt ii = iremote[i].index;

    if (j < 0 || sranks[j] != r) PetscCall(PetscFindMPIInt(r, nranks, sranks, &j));
    PetscCheck(j >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to locate neighbor rank %" PetscInt_FMT, r);
    for (v = 0; v < nv; v++) {
      viremote[v * nl + i].rank  = r;
      viremote[v * nl + i].index = v * ldrs[j] + ii;
      if (ilocal) vilocal[v * nl + i] = v * ldl + ilocal[i];
    }
  }
  PetscCall(PetscFree(sranks));
  PetscCall(PetscFree(ldrs));
  PetscCall(PetscSFCreate(comm, vsf));
  PetscCall(PetscSFSetGraph(*vsf, vnr, vnl, vilocal, PETSC_OWN_POINTER, viremote, PETSC_OWN_POINTER));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatDenseGetH2OpusVectorSF(Mat A, PetscSF h2sf, PetscSF *osf)
{
  PetscSF asf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(h2sf, PETSCSF_CLASSID, 2);
  PetscValidPointer(osf, 3);
  PetscCall(PetscObjectQuery((PetscObject)A, "_math2opus_vectorsf", (PetscObject *)&asf));
  if (!asf) {
    PetscInt lda;

    PetscCall(MatDenseGetLDA(A, &lda));
    PetscCall(PetscSFGetVectorSF(h2sf, A->cmap->N, lda, PETSC_DECIDE, &asf));
    PetscCall(PetscObjectCompose((PetscObject)A, "_math2opus_vectorsf", (PetscObject)asf));
    PetscCall(PetscObjectDereference((PetscObject)asf));
  }
  *osf = asf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_CUDA)
struct SignVector_Functor {
  const PetscScalar *v;
  PetscScalar       *s;
  SignVector_Functor(const PetscScalar *_v, PetscScalar *_s) : v(_v), s(_s) { }

  __host__ __device__ void operator()(PetscInt i) { s[i] = (v[i] < 0 ? -1 : 1); }
};
#endif

PETSC_INTERN PetscErrorCode VecSign(Vec v, Vec s)
{
  const PetscScalar *av;
  PetscScalar       *as;
  PetscInt           i, n;
#if defined(PETSC_HAVE_CUDA)
  PetscBool viscuda, siscuda;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(s, VEC_CLASSID, 2);
  PetscCall(VecGetLocalSize(s, &n));
  PetscCall(VecGetLocalSize(v, &i));
  PetscCheck(i == n, PETSC_COMM_SELF, PETSC_ERR_SUP, "Invalid local sizes %" PetscInt_FMT " != %" PetscInt_FMT, i, n);
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscObjectTypeCompareAny((PetscObject)v, &viscuda, VECSEQCUDA, VECMPICUDA, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)s, &siscuda, VECSEQCUDA, VECMPICUDA, ""));
  viscuda = (PetscBool)(viscuda && !v->boundtocpu);
  siscuda = (PetscBool)(siscuda && !s->boundtocpu);
  if (viscuda && siscuda) {
    PetscCall(VecCUDAGetArrayRead(v, &av));
    PetscCall(VecCUDAGetArrayWrite(s, &as));
    SignVector_Functor sign_vector(av, as);
    thrust::for_each(thrust::device, thrust::counting_iterator<PetscInt>(0), thrust::counting_iterator<PetscInt>(n), sign_vector);
    PetscCall(VecCUDARestoreArrayWrite(s, &as));
    PetscCall(VecCUDARestoreArrayRead(v, &av));
  } else
#endif
  {
    PetscCall(VecGetArrayRead(v, &av));
    PetscCall(VecGetArrayWrite(s, &as));
    for (i = 0; i < n; i++) as[i] = PetscAbsScalar(av[i]) < 0 ? -1. : 1.;
    PetscCall(VecRestoreArrayWrite(s, &as));
    PetscCall(VecRestoreArrayRead(v, &av));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_CUDA)
struct StandardBasis_Functor {
  PetscScalar *v;
  PetscInt     j;

  StandardBasis_Functor(PetscScalar *_v, PetscInt _j) : v(_v), j(_j) { }
  __host__ __device__ void operator()(PetscInt i) { v[i] = (i == j ? 1 : 0); }
};
#endif

PETSC_INTERN PetscErrorCode VecSetDelta(Vec x, PetscInt i)
{
#if defined(PETSC_HAVE_CUDA)
  PetscBool iscuda;
#endif
  PetscInt st, en;

  PetscFunctionBegin;
  PetscCall(VecGetOwnershipRange(x, &st, &en));
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscObjectTypeCompareAny((PetscObject)x, &iscuda, VECSEQCUDA, VECMPICUDA, ""));
  iscuda = (PetscBool)(iscuda && !x->boundtocpu);
  if (iscuda) {
    PetscScalar *ax;
    PetscCall(VecCUDAGetArrayWrite(x, &ax));
    StandardBasis_Functor delta(ax, i - st);
    thrust::for_each(thrust::device, thrust::counting_iterator<PetscInt>(0), thrust::counting_iterator<PetscInt>(en - st), delta);
    PetscCall(VecCUDARestoreArrayWrite(x, &ax));
  } else
#endif
  {
    PetscCall(VecSet(x, 0.));
    if (st <= i && i < en) PetscCall(VecSetValue(x, i, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* these are approximate norms */
/* NORM_2: Estimating the matrix p-norm Nicholas J. Higham
   NORM_1/NORM_INFINITY: A block algorithm for matrix 1-norm estimation, with an application to 1-norm pseudospectra Higham, Nicholas J. and Tisseur, Francoise */
PETSC_INTERN PetscErrorCode MatApproximateNorm_Private(Mat A, NormType normtype, PetscInt normsamples, PetscReal *n)
{
  Vec         x, y, w, z;
  PetscReal   normz, adot;
  PetscScalar dot;
  PetscInt    i, j, N, jold = -1;
  PetscBool   boundtocpu = PETSC_TRUE;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_DEVICE)
  boundtocpu = A->boundtocpu;
#endif
  switch (normtype) {
  case NORM_INFINITY:
  case NORM_1:
    if (normsamples < 0) normsamples = 10; /* pure guess */
    if (normtype == NORM_INFINITY) {
      Mat B;
      PetscCall(MatCreateTranspose(A, &B));
      A = B;
    } else {
      PetscCall(PetscObjectReference((PetscObject)A));
    }
    PetscCall(MatCreateVecs(A, &x, &y));
    PetscCall(MatCreateVecs(A, &z, &w));
    PetscCall(VecBindToCPU(x, boundtocpu));
    PetscCall(VecBindToCPU(y, boundtocpu));
    PetscCall(VecBindToCPU(z, boundtocpu));
    PetscCall(VecBindToCPU(w, boundtocpu));
    PetscCall(VecGetSize(x, &N));
    PetscCall(VecSet(x, 1. / N));
    *n = 0.0;
    for (i = 0; i < normsamples; i++) {
      PetscCall(MatMult(A, x, y));
      PetscCall(VecSign(y, w));
      PetscCall(MatMultTranspose(A, w, z));
      PetscCall(VecNorm(z, NORM_INFINITY, &normz));
      PetscCall(VecDot(x, z, &dot));
      adot = PetscAbsScalar(dot);
      PetscCall(PetscInfo(A, "%s norm it %" PetscInt_FMT " -> (%g %g)\n", NormTypes[normtype], i, (double)normz, (double)adot));
      if (normz <= adot && i > 0) {
        PetscCall(VecNorm(y, NORM_1, n));
        break;
      }
      PetscCall(VecMax(z, &j, &normz));
      if (j == jold) {
        PetscCall(VecNorm(y, NORM_1, n));
        PetscCall(PetscInfo(A, "%s norm it %" PetscInt_FMT " -> breakdown (j==jold)\n", NormTypes[normtype], i));
        break;
      }
      jold = j;
      PetscCall(VecSetDelta(x, j));
    }
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&w));
    PetscCall(VecDestroy(&y));
    PetscCall(VecDestroy(&z));
    break;
  case NORM_2:
    if (normsamples < 0) normsamples = 20; /* pure guess */
    PetscCall(MatCreateVecs(A, &x, &y));
    PetscCall(MatCreateVecs(A, &z, NULL));
    PetscCall(VecBindToCPU(x, boundtocpu));
    PetscCall(VecBindToCPU(y, boundtocpu));
    PetscCall(VecBindToCPU(z, boundtocpu));
    PetscCall(VecSetRandom(x, NULL));
    PetscCall(VecNormalize(x, NULL));
    *n = 0.0;
    for (i = 0; i < normsamples; i++) {
      PetscCall(MatMult(A, x, y));
      PetscCall(VecNormalize(y, n));
      PetscCall(MatMultTranspose(A, y, z));
      PetscCall(VecNorm(z, NORM_2, &normz));
      PetscCall(VecDot(x, z, &dot));
      adot = PetscAbsScalar(dot);
      PetscCall(PetscInfo(A, "%s norm it %" PetscInt_FMT " -> %g (%g %g)\n", NormTypes[normtype], i, (double)*n, (double)normz, (double)adot));
      if (normz <= adot) break;
      if (i < normsamples - 1) {
        Vec t;

        PetscCall(VecNormalize(z, NULL));
        t = x;
        x = z;
        z = t;
      }
    }
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(VecDestroy(&z));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "%s norm not supported", NormTypes[normtype]);
  }
  PetscCall(PetscInfo(A, "%s norm %g computed in %" PetscInt_FMT " iterations\n", NormTypes[normtype], (double)*n, i));
  PetscFunctionReturn(PETSC_SUCCESS);
}
