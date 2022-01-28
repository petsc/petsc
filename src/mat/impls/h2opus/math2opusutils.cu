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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidLogicalCollectiveInt(sf,nv,2);
  PetscValidPointer(vsf,5);
  if (nv == 1) {
    ierr = PetscObjectReference((PetscObject)sf);CHKERRQ(ierr);
    *vsf = sf;
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf,&nr,&nl,&ilocal,&iremote);CHKERRQ(ierr);
  ierr = PetscSFGetLeafRange(sf,NULL,&maxl);CHKERRQ(ierr);
  maxl += 1;
  if (ldl == PETSC_DECIDE) ldl = maxl;
  if (ldr == PETSC_DECIDE) ldr = nr;
  if (ldr < nr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid leading dimension %" PetscInt_FMT " < %" PetscInt_FMT,ldr,nr);
  if (ldl < maxl) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid leading dimension %" PetscInt_FMT " < %" PetscInt_FMT,ldl,maxl);
  vnr  = nr*nv;
  vnl  = nl*nv;
  ierr = PetscMalloc1(vnl,&viremote);CHKERRQ(ierr);
  if (ilocal) {
    ierr = PetscMalloc1(vnl,&vilocal);CHKERRQ(ierr);
  }

  /* TODO: Should this special SF be available, e.g.
     PetscSFGetRanksSF or similar? */
  ierr = PetscSFGetRootRanks(sf,&nranks,&ranks,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(nranks,&sranks);CHKERRQ(ierr);
  ierr = PetscArraycpy(sranks,ranks,nranks);CHKERRQ(ierr);
  ierr = PetscSortMPIInt(nranks,sranks);CHKERRQ(ierr);
  ierr = PetscMalloc1(nranks,&rremotes);CHKERRQ(ierr);
  for (i=0;i<nranks;i++) {
    rremotes[i].rank  = sranks[i];
    rremotes[i].index = 0;
  }
  ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_CONFONLY,&rankssf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(rankssf,1,nranks,NULL,PETSC_OWN_POINTER,rremotes,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscMalloc1(nranks,&ldrs);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(rankssf,MPIU_INT,&ldr,ldrs,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(rankssf,MPIU_INT,&ldr,ldrs,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&rankssf);CHKERRQ(ierr);

  j = -1;
  for (i=0;i<nl;i++) {
    const PetscInt r  = iremote[i].rank;
    const PetscInt ii = iremote[i].index;

    if (j < 0 || sranks[j] != r) {
      ierr = PetscFindMPIInt(r,nranks,sranks,&j);CHKERRQ(ierr);
    }
    if (j < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to locate neighbor rank %" PetscInt_FMT,r);
    for (v=0;v<nv;v++) {
      viremote[v*nl + i].rank  = r;
      viremote[v*nl + i].index = v*ldrs[j] + ii;
      if (ilocal) vilocal[v*nl + i] = v*ldl + ilocal[i];
    }
  }
  ierr = PetscFree(sranks);CHKERRQ(ierr);
  ierr = PetscFree(ldrs);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm,vsf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*vsf,vnr,vnl,vilocal,PETSC_OWN_POINTER,viremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
#if defined(PETSC_HAVE_CUDA)
  PetscBool         viscuda,siscuda;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(s,VEC_CLASSID,2);
  ierr = VecGetLocalSize(s,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&i);CHKERRQ(ierr);
  if (i != n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Invalid local sizes %" PetscInt_FMT " != %" PetscInt_FMT,i,n);
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)v,&viscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)s,&siscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  viscuda = (PetscBool)(viscuda && !v->boundtocpu);
  siscuda = (PetscBool)(siscuda && !s->boundtocpu);
  if (viscuda && siscuda) {
    ierr = VecCUDAGetArrayRead(v,&av);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayWrite(s,&as);CHKERRQ(ierr);
    SignVector_Functor sign_vector(av, as);
    thrust::for_each(thrust::device,thrust::counting_iterator<PetscInt>(0),
                     thrust::counting_iterator<PetscInt>(n), sign_vector);
    ierr = VecCUDARestoreArrayWrite(s,&as);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(v,&av);CHKERRQ(ierr);
  } else
#endif
  {
    ierr = VecGetArrayRead(v,&av);CHKERRQ(ierr);
    ierr = VecGetArrayWrite(s,&as);CHKERRQ(ierr);
    for (i=0;i<n;i++) as[i] = PetscAbsScalar(av[i]) < 0 ? -1. : 1.;
    ierr = VecRestoreArrayWrite(s,&as);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v,&av);CHKERRQ(ierr);
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
  PetscBool      iscuda;
#endif
  PetscInt       st,en;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetOwnershipRange(x,&st,&en);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&iscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  iscuda = (PetscBool)(iscuda && !x->boundtocpu);
  if (iscuda) {
    PetscScalar *ax;
    ierr = VecCUDAGetArrayWrite(x,&ax);CHKERRQ(ierr);
    StandardBasis_Functor delta(ax, i-st);
    thrust::for_each(thrust::device,thrust::counting_iterator<PetscInt>(0),
                     thrust::counting_iterator<PetscInt>(en-st), delta);
    ierr = VecCUDARestoreArrayWrite(x,&ax);CHKERRQ(ierr);
  } else
#endif
  {
    ierr = VecSet(x,0.);CHKERRQ(ierr);
    if (st <= i && i < en) {
      ierr = VecSetValue(x,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* these are approximate norms */
/* NORM_2: Estimating the matrix p-norm Nicholas J. Higham
   NORM_1/NORM_INFINITY: A block algorithm for matrix 1-norm estimation, with an application to 1-norm pseudospectra Higham, Nicholas J. and Tisseur, Francoise */
PETSC_INTERN PetscErrorCode MatApproximateNorm_Private(Mat A, NormType normtype, PetscInt normsamples, PetscReal* n)
{
  Vec            x,y,w,z;
  PetscReal      normz,adot;
  PetscScalar    dot;
  PetscInt       i,j,N,jold = -1;
  PetscErrorCode ierr;
  PetscBool      boundtocpu = PETSC_TRUE;

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
      ierr = MatCreateTranspose(A,&B);CHKERRQ(ierr);
      A = B;
    } else {
      ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,&z,&w);CHKERRQ(ierr);
    ierr = VecBindToCPU(x,boundtocpu);CHKERRQ(ierr);
    ierr = VecBindToCPU(y,boundtocpu);CHKERRQ(ierr);
    ierr = VecBindToCPU(z,boundtocpu);CHKERRQ(ierr);
    ierr = VecBindToCPU(w,boundtocpu);CHKERRQ(ierr);
    ierr = VecGetSize(x,&N);CHKERRQ(ierr);
    ierr = VecSet(x,1./N);CHKERRQ(ierr);
    *n   = 0.0;
    for (i = 0; i < normsamples; i++) {
      ierr = MatMult(A,x,y);CHKERRQ(ierr);
      ierr = VecSign(y,w);CHKERRQ(ierr);
      ierr = MatMultTranspose(A,w,z);CHKERRQ(ierr);
      ierr = VecNorm(z,NORM_INFINITY,&normz);CHKERRQ(ierr);
      ierr = VecDot(x,z,&dot);CHKERRQ(ierr);
      adot = PetscAbsScalar(dot);
      ierr = PetscInfo4(A,"%s norm it %" PetscInt_FMT " -> (%g %g)\n",NormTypes[normtype],i,(double)normz,(double)adot);CHKERRQ(ierr);
      if (normz <= adot && i > 0) {
        ierr = VecNorm(y,NORM_1,n);CHKERRQ(ierr);
        break;
      }
      ierr = VecMax(z,&j,&normz);CHKERRQ(ierr);
      if (j == jold) {
        ierr = VecNorm(y,NORM_1,n);CHKERRQ(ierr);
        ierr = PetscInfo2(A,"%s norm it %" PetscInt_FMT " -> breakdown (j==jold)\n",NormTypes[normtype],i);CHKERRQ(ierr);
        break;
      }
      jold = j;
      ierr = VecSetDelta(x,j);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&w);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = VecDestroy(&z);CHKERRQ(ierr);
    break;
  case NORM_2:
    if (normsamples < 0) normsamples = 20; /* pure guess */
    ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,&z,NULL);CHKERRQ(ierr);
    ierr = VecBindToCPU(x,boundtocpu);CHKERRQ(ierr);
    ierr = VecBindToCPU(y,boundtocpu);CHKERRQ(ierr);
    ierr = VecBindToCPU(z,boundtocpu);CHKERRQ(ierr);
    ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
    ierr = VecNormalize(x,NULL);CHKERRQ(ierr);
    *n   = 0.0;
    for (i = 0; i < normsamples; i++) {
      ierr = MatMult(A,x,y);CHKERRQ(ierr);
      ierr = VecNormalize(y,n);CHKERRQ(ierr);
      ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);
      ierr = VecNorm(z,NORM_2,&normz);CHKERRQ(ierr);
      ierr = VecDot(x,z,&dot);CHKERRQ(ierr);
      adot = PetscAbsScalar(dot);
      ierr = PetscInfo5(A,"%s norm it %" PetscInt_FMT " -> %g (%g %g)\n",NormTypes[normtype],i,(double)*n,(double)normz,(double)adot);CHKERRQ(ierr);
      if (normz <= adot) break;
      if (i < normsamples - 1) {
        Vec t;

        ierr = VecNormalize(z,NULL);CHKERRQ(ierr);
        t = x;
        x = z;
        z = t;
      }
    }
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = VecDestroy(&z);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"%s norm not supported",NormTypes[normtype]);
  }
  ierr = PetscInfo3(A,"%s norm %g computed in %" PetscInt_FMT " iterations\n",NormTypes[normtype],(double)*n,i);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

