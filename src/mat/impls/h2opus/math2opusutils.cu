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
  PetscSF           rankssf;
  const PetscSFNode *iremote;
  PetscSFNode       *viremote,*rremotes;
  const PetscInt    *ilocal;
  PetscInt          *vilocal = NULL,*ldrs;
  const PetscMPIInt *ranks;
  PetscMPIInt       *sranks;
  PetscInt          nranks,nr,nl,vnr,vnl,i,v,j,maxl;
  MPI_Comm          comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidLogicalCollectiveInt(sf,nv,2);
  PetscValidPointer(vsf,5);
  if (nv == 1) {
    CHKERRQ(PetscObjectReference((PetscObject)sf));
    *vsf = sf;
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscObjectGetComm((PetscObject)sf,&comm));
  CHKERRQ(PetscSFGetGraph(sf,&nr,&nl,&ilocal,&iremote));
  CHKERRQ(PetscSFGetLeafRange(sf,NULL,&maxl));
  maxl += 1;
  if (ldl == PETSC_DECIDE) ldl = maxl;
  if (ldr == PETSC_DECIDE) ldr = nr;
  PetscCheck(ldr >= nr,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid leading dimension %" PetscInt_FMT " < %" PetscInt_FMT,ldr,nr);
  PetscCheck(ldl >= maxl,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid leading dimension %" PetscInt_FMT " < %" PetscInt_FMT,ldl,maxl);
  vnr  = nr*nv;
  vnl  = nl*nv;
  CHKERRQ(PetscMalloc1(vnl,&viremote));
  if (ilocal) CHKERRQ(PetscMalloc1(vnl,&vilocal));

  /* TODO: Should this special SF be available, e.g.
     PetscSFGetRanksSF or similar? */
  CHKERRQ(PetscSFGetRootRanks(sf,&nranks,&ranks,NULL,NULL,NULL));
  CHKERRQ(PetscMalloc1(nranks,&sranks));
  CHKERRQ(PetscArraycpy(sranks,ranks,nranks));
  CHKERRQ(PetscSortMPIInt(nranks,sranks));
  CHKERRQ(PetscMalloc1(nranks,&rremotes));
  for (i=0;i<nranks;i++) {
    rremotes[i].rank  = sranks[i];
    rremotes[i].index = 0;
  }
  CHKERRQ(PetscSFDuplicate(sf,PETSCSF_DUPLICATE_CONFONLY,&rankssf));
  CHKERRQ(PetscSFSetGraph(rankssf,1,nranks,NULL,PETSC_OWN_POINTER,rremotes,PETSC_OWN_POINTER));
  CHKERRQ(PetscMalloc1(nranks,&ldrs));
  CHKERRQ(PetscSFBcastBegin(rankssf,MPIU_INT,&ldr,ldrs,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(rankssf,MPIU_INT,&ldr,ldrs,MPI_REPLACE));
  CHKERRQ(PetscSFDestroy(&rankssf));

  j = -1;
  for (i=0;i<nl;i++) {
    const PetscInt r  = iremote[i].rank;
    const PetscInt ii = iremote[i].index;

    if (j < 0 || sranks[j] != r) {
      CHKERRQ(PetscFindMPIInt(r,nranks,sranks,&j));
    }
    PetscCheck(j >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to locate neighbor rank %" PetscInt_FMT,r);
    for (v=0;v<nv;v++) {
      viremote[v*nl + i].rank  = r;
      viremote[v*nl + i].index = v*ldrs[j] + ii;
      if (ilocal) vilocal[v*nl + i] = v*ldl + ilocal[i];
    }
  }
  CHKERRQ(PetscFree(sranks));
  CHKERRQ(PetscFree(ldrs));
  CHKERRQ(PetscSFCreate(comm,vsf));
  CHKERRQ(PetscSFSetGraph(*vsf,vnr,vnl,vilocal,PETSC_OWN_POINTER,viremote,PETSC_OWN_POINTER));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatDenseGetH2OpusVectorSF(Mat A, PetscSF h2sf, PetscSF *osf)
{
  PetscSF asf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(h2sf,PETSCSF_CLASSID,2);
  PetscValidPointer(osf,3);
  CHKERRQ(PetscObjectQuery((PetscObject)A,"_math2opus_vectorsf",(PetscObject*)&asf));
  if (!asf) {
    PetscInt lda;

    CHKERRQ(MatDenseGetLDA(A,&lda));
    CHKERRQ(PetscSFGetVectorSF(h2sf,A->cmap->N,lda,PETSC_DECIDE,&asf));
    CHKERRQ(PetscObjectCompose((PetscObject)A,"_math2opus_vectorsf",(PetscObject)asf));
    CHKERRQ(PetscObjectDereference((PetscObject)asf));
  }
  *osf = asf;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
struct SignVector_Functor
{
    const PetscScalar *v;
    PetscScalar *s;
    SignVector_Functor(const PetscScalar *_v, PetscScalar *_s) : v(_v), s(_s) {}

    __host__ __device__ void operator()(PetscInt i)
    {
        s[i] = (v[i] < 0 ? -1 : 1);
    }
};
#endif

PETSC_INTERN PetscErrorCode VecSign(Vec v, Vec s)
{
  const PetscScalar *av;
  PetscScalar       *as;
  PetscInt          i,n;
#if defined(PETSC_HAVE_CUDA)
  PetscBool         viscuda,siscuda;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(s,VEC_CLASSID,2);
  CHKERRQ(VecGetLocalSize(s,&n));
  CHKERRQ(VecGetLocalSize(v,&i));
  PetscCheck(i == n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Invalid local sizes %" PetscInt_FMT " != %" PetscInt_FMT,i,n);
#if defined(PETSC_HAVE_CUDA)
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)v,&viscuda,VECSEQCUDA,VECMPICUDA,""));
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)s,&siscuda,VECSEQCUDA,VECMPICUDA,""));
  viscuda = (PetscBool)(viscuda && !v->boundtocpu);
  siscuda = (PetscBool)(siscuda && !s->boundtocpu);
  if (viscuda && siscuda) {
    CHKERRQ(VecCUDAGetArrayRead(v,&av));
    CHKERRQ(VecCUDAGetArrayWrite(s,&as));
    SignVector_Functor sign_vector(av, as);
    thrust::for_each(thrust::device,thrust::counting_iterator<PetscInt>(0),
                     thrust::counting_iterator<PetscInt>(n), sign_vector);
    CHKERRQ(VecCUDARestoreArrayWrite(s,&as));
    CHKERRQ(VecCUDARestoreArrayRead(v,&av));
  } else
#endif
  {
    CHKERRQ(VecGetArrayRead(v,&av));
    CHKERRQ(VecGetArrayWrite(s,&as));
    for (i=0;i<n;i++) as[i] = PetscAbsScalar(av[i]) < 0 ? -1. : 1.;
    CHKERRQ(VecRestoreArrayWrite(s,&as));
    CHKERRQ(VecRestoreArrayRead(v,&av));
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
struct StandardBasis_Functor
{
    PetscScalar *v;
    PetscInt j;

    StandardBasis_Functor(PetscScalar *_v, PetscInt _j) : v(_v), j(_j) {}
    __host__ __device__ void operator()(PetscInt i)
    {
        v[i] = (i == j ? 1 : 0);
    }
};
#endif

PETSC_INTERN PetscErrorCode VecSetDelta(Vec x, PetscInt i)
{
#if defined(PETSC_HAVE_CUDA)
  PetscBool iscuda;
#endif
  PetscInt  st,en;

  PetscFunctionBegin;
  CHKERRQ(VecGetOwnershipRange(x,&st,&en));
#if defined(PETSC_HAVE_CUDA)
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)x,&iscuda,VECSEQCUDA,VECMPICUDA,""));
  iscuda = (PetscBool)(iscuda && !x->boundtocpu);
  if (iscuda) {
    PetscScalar *ax;
    CHKERRQ(VecCUDAGetArrayWrite(x,&ax));
    StandardBasis_Functor delta(ax, i-st);
    thrust::for_each(thrust::device,thrust::counting_iterator<PetscInt>(0),
                     thrust::counting_iterator<PetscInt>(en-st), delta);
    CHKERRQ(VecCUDARestoreArrayWrite(x,&ax));
  } else
#endif
  {
    CHKERRQ(VecSet(x,0.));
    if (st <= i && i < en) {
      CHKERRQ(VecSetValue(x,i,1.0,INSERT_VALUES));
    }
    CHKERRQ(VecAssemblyBegin(x));
    CHKERRQ(VecAssemblyEnd(x));
  }
  PetscFunctionReturn(0);
}

/* these are approximate norms */
/* NORM_2: Estimating the matrix p-norm Nicholas J. Higham
   NORM_1/NORM_INFINITY: A block algorithm for matrix 1-norm estimation, with an application to 1-norm pseudospectra Higham, Nicholas J. and Tisseur, Francoise */
PETSC_INTERN PetscErrorCode MatApproximateNorm_Private(Mat A, NormType normtype, PetscInt normsamples, PetscReal* n)
{
  Vec         x,y,w,z;
  PetscReal   normz,adot;
  PetscScalar dot;
  PetscInt    i,j,N,jold = -1;
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
      CHKERRQ(MatCreateTranspose(A,&B));
      A = B;
    } else {
      CHKERRQ(PetscObjectReference((PetscObject)A));
    }
    CHKERRQ(MatCreateVecs(A,&x,&y));
    CHKERRQ(MatCreateVecs(A,&z,&w));
    CHKERRQ(VecBindToCPU(x,boundtocpu));
    CHKERRQ(VecBindToCPU(y,boundtocpu));
    CHKERRQ(VecBindToCPU(z,boundtocpu));
    CHKERRQ(VecBindToCPU(w,boundtocpu));
    CHKERRQ(VecGetSize(x,&N));
    CHKERRQ(VecSet(x,1./N));
    *n   = 0.0;
    for (i = 0; i < normsamples; i++) {
      CHKERRQ(MatMult(A,x,y));
      CHKERRQ(VecSign(y,w));
      CHKERRQ(MatMultTranspose(A,w,z));
      CHKERRQ(VecNorm(z,NORM_INFINITY,&normz));
      CHKERRQ(VecDot(x,z,&dot));
      adot = PetscAbsScalar(dot);
      CHKERRQ(PetscInfo(A,"%s norm it %" PetscInt_FMT " -> (%g %g)\n",NormTypes[normtype],i,(double)normz,(double)adot));
      if (normz <= adot && i > 0) {
        CHKERRQ(VecNorm(y,NORM_1,n));
        break;
      }
      CHKERRQ(VecMax(z,&j,&normz));
      if (j == jold) {
        CHKERRQ(VecNorm(y,NORM_1,n));
        CHKERRQ(PetscInfo(A,"%s norm it %" PetscInt_FMT " -> breakdown (j==jold)\n",NormTypes[normtype],i));
        break;
      }
      jold = j;
      CHKERRQ(VecSetDelta(x,j));
    }
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&w));
    CHKERRQ(VecDestroy(&y));
    CHKERRQ(VecDestroy(&z));
    break;
  case NORM_2:
    if (normsamples < 0) normsamples = 20; /* pure guess */
    CHKERRQ(MatCreateVecs(A,&x,&y));
    CHKERRQ(MatCreateVecs(A,&z,NULL));
    CHKERRQ(VecBindToCPU(x,boundtocpu));
    CHKERRQ(VecBindToCPU(y,boundtocpu));
    CHKERRQ(VecBindToCPU(z,boundtocpu));
    CHKERRQ(VecSetRandom(x,NULL));
    CHKERRQ(VecNormalize(x,NULL));
    *n   = 0.0;
    for (i = 0; i < normsamples; i++) {
      CHKERRQ(MatMult(A,x,y));
      CHKERRQ(VecNormalize(y,n));
      CHKERRQ(MatMultTranspose(A,y,z));
      CHKERRQ(VecNorm(z,NORM_2,&normz));
      CHKERRQ(VecDot(x,z,&dot));
      adot = PetscAbsScalar(dot);
      CHKERRQ(PetscInfo(A,"%s norm it %" PetscInt_FMT " -> %g (%g %g)\n",NormTypes[normtype],i,(double)*n,(double)normz,(double)adot));
      if (normz <= adot) break;
      if (i < normsamples - 1) {
        Vec t;

        CHKERRQ(VecNormalize(z,NULL));
        t = x;
        x = z;
        z = t;
      }
    }
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y));
    CHKERRQ(VecDestroy(&z));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"%s norm not supported",NormTypes[normtype]);
  }
  CHKERRQ(PetscInfo(A,"%s norm %g computed in %" PetscInt_FMT " iterations\n",NormTypes[normtype],(double)*n,i));
  PetscFunctionReturn(0);
}
