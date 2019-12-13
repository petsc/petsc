/*
  Common tools for constructing discretizations
*/
#if !defined(PETSCDT_H)
#define PETSCDT_H

#include <petscsys.h>

PETSC_EXTERN PetscClassId PETSCQUADRATURE_CLASSID;

/*S
  PetscQuadrature - Quadrature rule for integration.

  Level: beginner

.seealso:  PetscQuadratureCreate(), PetscQuadratureDestroy()
S*/
typedef struct _p_PetscQuadrature *PetscQuadrature;

/*E
  PetscGaussLobattoLegendreCreateType - algorithm used to compute the Gauss-Lobatto-Legendre nodes and weights

  Level: intermediate

$  PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA - compute the nodes via linear algebra
$  PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON - compute the nodes by solving a nonlinear equation with Newton's method

E*/
typedef enum {PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON} PetscGaussLobattoLegendreCreateType;

PETSC_EXTERN PetscErrorCode PetscQuadratureCreate(MPI_Comm, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscQuadratureDuplicate(PetscQuadrature, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetOrder(PetscQuadrature, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetOrder(PetscQuadrature, PetscInt);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetNumComponents(PetscQuadrature, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetNumComponents(PetscQuadrature, PetscInt);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetData(PetscQuadrature, PetscInt*, PetscInt*, PetscInt*, const PetscReal *[], const PetscReal *[]);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetData(PetscQuadrature, PetscInt, PetscInt, PetscInt, const PetscReal [], const PetscReal []);
PETSC_EXTERN PetscErrorCode PetscQuadratureView(PetscQuadrature, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscQuadratureDestroy(PetscQuadrature *);

PETSC_EXTERN PetscErrorCode PetscQuadratureExpandComposite(PetscQuadrature, PetscInt, const PetscReal[], const PetscReal[], PetscQuadrature *);

PETSC_EXTERN PetscErrorCode PetscQuadraturePushForward(PetscQuadrature, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, PetscQuadrature *);

PETSC_EXTERN PetscErrorCode PetscDTLegendreEval(PetscInt,const PetscReal*,PetscInt,const PetscInt*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussQuadrature(PetscInt,PetscReal,PetscReal,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussLobattoLegendreQuadrature(PetscInt,PetscGaussLobattoLegendreCreateType,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTReconstructPoly(PetscInt,PetscInt,const PetscReal*,PetscInt,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussTensorQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*);
PETSC_EXTERN PetscErrorCode PetscDTGaussJacobiQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*);

PETSC_EXTERN PetscErrorCode PetscDTTanhSinhTensorQuadrature(PetscInt, PetscInt, PetscReal, PetscReal, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscDTTanhSinhIntegrate(void (*)(PetscReal, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTTanhSinhIntegrateMPFR(void (*)(PetscReal, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreIntegrate(PetscInt, PetscReal *, PetscReal *, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementLaplacianCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementLaplacianDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementGradientCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementGradientDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementAdvectionCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementAdvectionDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementMassCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementMassDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***);

PETSC_EXTERN PetscErrorCode PetscDTAltVApply(PetscInt, PetscInt, const PetscReal *, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVWedge(PetscInt, PetscInt, PetscInt, const PetscReal *, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVWedgeMatrix(PetscInt, PetscInt, PetscInt, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVPullback(PetscInt, PetscInt, const PetscReal *, PetscInt, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVPullbackMatrix(PetscInt, PetscInt, const PetscReal *, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVInterior(PetscInt, PetscInt, const PetscReal *, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVInteriorMatrix(PetscInt, PetscInt, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVInteriorPattern(PetscInt, PetscInt, PetscInt (*)[3]);
PETSC_EXTERN PetscErrorCode PetscDTAltVStar(PetscInt, PetscInt, PetscInt, const PetscReal *, PetscReal *);

#if defined(PETSC_USE_64BIT_INDICES)
#define PETSC_FACTORIAL_MAX 20
#define PETSC_BINOMIAL_MAX  61
#else
#define PETSC_FACTORIAL_MAX 12
#define PETSC_BINOMIAL_MAX  29
#endif

/*MC
   PetscDTFactorial - Approximate n! as a real number

   Input Arguments:
.  n - a non-negative integer

   Output Arguments:
.  factorial - n!

   Level: beginner
M*/
PETSC_STATIC_INLINE PetscErrorCode PetscDTFactorial(PetscInt n, PetscReal *factorial)
{
  PetscReal f = 1.0;
  PetscInt  i;

  PetscFunctionBegin;
  *factorial = -1.;
  if (n < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Factorial called with negative number %D\n", n);
  for (i = 1; i < n+1; ++i) f *= i;
  *factorial = f;
  PetscFunctionReturn(0);
}

/*MC
   PetscDTFactorialInt - Compute n! as an integer

   Input Arguments:
.  n - a non-negative integer

   Output Arguments:
.  factorial - n!

   Level: beginner

   Note: this is limited to n such that n! can be represented by PetscInt, which is 12 if PetscInt is a signed 32-bit integer and 20 if PetscInt is a signed 64-bit integer.
M*/
PETSC_STATIC_INLINE PetscErrorCode PetscDTFactorialInt(PetscInt n, PetscInt *factorial)
{
  PetscInt facLookup[13] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600};

  PetscFunctionBegin;
  *factorial = -1;
  if (n < 0 || n > PETSC_FACTORIAL_MAX) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of elements %D is not in supported range [0,%D]\n",n,PETSC_FACTORIAL_MAX);
  if (n <= 12) {
    *factorial = facLookup[n];
  } else {
    PetscInt f = facLookup[12];
    PetscInt i;

    for (i = 13; i < n+1; ++i) f *= i;
    *factorial = f;
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscDTBinomial - Approximate the binomial coefficient "n choose k"

   Input Arguments:
+  n - a non-negative integer
-  k - an integer between 0 and n, inclusive

   Output Arguments:
.  binomial - approximation of the binomial coefficient n choose k

   Level: beginner
M*/
PETSC_STATIC_INLINE PetscErrorCode PetscDTBinomial(PetscInt n, PetscInt k, PetscReal *binomial)
{
  PetscFunctionBeginHot;
  if (n < 0 || k < 0 || k > n) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Binomial arguments (%D %D) must be non-negative, k <= n\n", n, k);
  if (n <= 3) {
    PetscInt binomLookup[4][4] = {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 2, 1, 0}, {1, 3, 3, 1}};

    *binomial = binomLookup[n][k];
  } else {
    PetscReal binom = 1.;
    PetscInt  i;

    k = PetscMin(k, n - k);
    for (i = 0; i < k; i++) binom = (binom * (n - i)) / (i + 1);
    *binomial = binom;
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscDTBinomialInt - Compute the binomial coefficient "n choose k"

   Input Arguments:
+  n - a non-negative integer
-  k - an integer between 0 and n, inclusive

   Output Arguments:
.  binomial - the binomial coefficient n choose k

   Note: this is limited by integers that can be represented by PetscInt

   Level: beginner
M*/
PETSC_STATIC_INLINE PetscErrorCode PetscDTBinomialInt(PetscInt n, PetscInt k, PetscInt *binomial)
{
  PetscInt bin;

  PetscFunctionBegin;
  *binomial = -1;
  if (n < 0 || k < 0 || k > n) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Binomial arguments (%D %D) must be non-negative, k <= n\n", n, k);
  if (n > PETSC_BINOMIAL_MAX) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Binomial elements %D is larger than max for PetscInt, %D\n", n, PETSC_BINOMIAL_MAX);
  if (n <= 3) {
    PetscInt binomLookup[4][4] = {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 2, 1, 0}, {1, 3, 3, 1}};

    bin = binomLookup[n][k];
  } else {
    PetscInt  binom = 1;
    PetscInt  i;

    k = PetscMin(k, n - k);
    for (i = 0; i < k; i++) binom = (binom * (n - i)) / (i + 1);
    bin = binom;
  }
  *binomial = bin;
  PetscFunctionReturn(0);
}

/*MC
   PetscDTEnumPerm - Get a permutation of n integers from its encoding into the integers [0, n!) as a sequence of swaps.

   A permutation can be described by the operations that convert the lists [0, 1, ..., n-1] into the permutation,
   by a sequence of swaps, where the ith step swaps whatever number is in ith position with a number that is in
   some position j >= i.  This swap is encoded as the difference (j - i).  The difference d_i at step i is less than
   (n - i).  This sequence of n-1 differences [d_0, ..., d_{n-2}] is encoded as the number
   (n-1)! * d_0 + (n-2)! * d_1 + ... + 1! * d_{n-2}.

   Input Arguments:
+  n - a non-negative integer (see note about limits below)
-  k - an integer in [0, n!)

   Output Arguments:
+  perm - the permuted list of the integers [0, ..., n-1]
-  isOdd - if not NULL, returns wether the permutation used an even or odd number of swaps.

   Note: this is limited to n such that n! can be represented by PetscInt, which is 12 if PetscInt is a signed 32-bit integer and 20 if PetscInt is a signed 64-bit integer.

   Level: beginner
M*/
PETSC_STATIC_INLINE PetscErrorCode PetscDTEnumPerm(PetscInt n, PetscInt k, PetscInt *perm, PetscBool *isOdd)
{
  PetscInt  odd = 0;
  PetscInt  i;
  PetscInt  work[PETSC_FACTORIAL_MAX];
  PetscInt *w;

  PetscFunctionBegin;
  if (isOdd) *isOdd = PETSC_FALSE;
  if (n < 0 || n > PETSC_FACTORIAL_MAX) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of elements %D is not in supported range [0,%D]\n",n,PETSC_FACTORIAL_MAX);
  w = &work[n - 2];
  for (i = 2; i <= n; i++) {
    *(w--) = k % i;
    k /= i;
  }
  for (i = 0; i < n; i++) perm[i] = i;
  for (i = 0; i < n - 1; i++) {
    PetscInt s = work[i];
    PetscInt swap = perm[i];

    perm[i] = perm[i + s];
    perm[i + s] = swap;
    odd ^= (!!s);
  }
  if (isOdd) *isOdd = odd ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*MC
   PetscDTPermIndex - Encode a permutation of n into an integer in [0, n!).  This inverts PetscDTEnumPerm.

   Input Arguments:
+  n - a non-negative integer (see note about limits below)
-  perm - the permuted list of the integers [0, ..., n-1]

   Output Arguments:
+  k - an integer in [0, n!)
.  isOdd - if not NULL, returns wether the permutation used an even or odd number of swaps.

   Note: this is limited to n such that n! can be represented by PetscInt, which is 12 if PetscInt is a signed 32-bit integer and 20 if PetscInt is a signed 64-bit integer.

   Level: beginner
M*/
PETSC_STATIC_INLINE PetscErrorCode PetscDTPermIndex(PetscInt n, const PetscInt *perm, PetscInt *k, PetscBool *isOdd)
{
  PetscInt  odd = 0;
  PetscInt  i, idx;
  PetscInt  work[PETSC_FACTORIAL_MAX];
  PetscInt  iwork[PETSC_FACTORIAL_MAX];

  PetscFunctionBeginHot;
  *k = -1;
  if (isOdd) *isOdd = PETSC_FALSE;
  if (n < 0 || n > PETSC_FACTORIAL_MAX) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of elements %D is not in supported range [0,%D]\n",n,PETSC_FACTORIAL_MAX);
  for (i = 0; i < n; i++) work[i] = i;  /* partial permutation */
  for (i = 0; i < n; i++) iwork[i] = i; /* partial permutation inverse */
  for (idx = 0, i = 0; i < n - 1; i++) {
    PetscInt j = perm[i];
    PetscInt icur = work[i];
    PetscInt jloc = iwork[j];
    PetscInt diff = jloc - i;

    idx = idx * (n - i) + diff;
    /* swap (i, jloc) */
    work[i] = j;
    work[jloc] = icur;
    iwork[j] = i;
    iwork[icur] = jloc;
    odd ^= (!!diff);
  }
  *k = idx;
  if (isOdd) *isOdd = odd ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*MC
   PetscDTEnumSubset - Get an ordered subset of the integers [0, ..., n - 1] from its encoding as an integers in [0, n choose k).
   The encoding is in lexicographic order.

   Input Arguments:
+  n - a non-negative integer (see note about limits below)
.  k - an integer in [0, n]
-  j - an index in [0, n choose k)

   Output Arguments:
.  subset - the jth subset of size k of the integers [0, ..., n - 1]

   Note: this is limited by arguments such that n choose k can be represented by PetscInt

   Level: beginner

.seealso: PetscDTSubsetIndex()
M*/
PETSC_STATIC_INLINE PetscErrorCode PetscDTEnumSubset(PetscInt n, PetscInt k, PetscInt j, PetscInt *subset)
{
  PetscInt       Nk, i, l;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscDTBinomialInt(n, k, &Nk);CHKERRQ(ierr);
  for (i = 0, l = 0; i < n && l < k; i++) {
    PetscInt Nminuskminus = (Nk * (k - l)) / (n - i);
    PetscInt Nminusk = Nk - Nminuskminus;

    if (j < Nminuskminus) {
      subset[l++] = i;
      Nk = Nminuskminus;
    } else {
      j -= Nminuskminus;
      Nk = Nminusk;
    }
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscDTSubsetIndex - Convert an ordered subset of k integers from the set [0, ..., n - 1] to its encoding as an integers in [0, n choose k) in lexicographic order.  This is the inverse of PetscDTEnumSubset.

   Input Arguments:
+  n - a non-negative integer (see note about limits below)
.  k - an integer in [0, n]
-  subset - an ordered subset of the integers [0, ..., n - 1]

   Output Arguments:
.  index - the rank of the subset in lexicographic order

   Note: this is limited by arguments such that n choose k can be represented by PetscInt

   Level: beginner

.seealso: PetscDTEnumSubset()
M*/
PETSC_STATIC_INLINE PetscErrorCode PetscDTSubsetIndex(PetscInt n, PetscInt k, const PetscInt *subset, PetscInt *index)
{
  PetscInt       i, j = 0, l, Nk;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *index = -1;
  ierr = PetscDTBinomialInt(n, k, &Nk);CHKERRQ(ierr);
  for (i = 0, l = 0; i < n && l < k; i++) {
    PetscInt Nminuskminus = (Nk * (k - l)) / (n - i);
    PetscInt Nminusk = Nk - Nminuskminus;

    if (subset[l] == i) {
      l++;
      Nk = Nminuskminus;
    } else {
      j += Nminuskminus;
      Nk = Nminusk;
    }
  }
  *index = j;
  PetscFunctionReturn(0);
}

/*MC
   PetscDTEnumSubset - Split the integers [0, ..., n - 1] into two complementary ordered subsets, the first subset of size k and being the jth subset of that size in lexicographic order.

   Input Arguments:
+  n - a non-negative integer (see note about limits below)
.  k - an integer in [0, n]
-  j - an index in [0, n choose k)

   Output Arguments:
+  perm - the jth subset of size k of the integers [0, ..., n - 1], followed by its complementary set.
-  isOdd - if not NULL, return whether perm is an even or odd permutation.

   Note: this is limited by arguments such that n choose k can be represented by PetscInt

   Level: beginner

.seealso: PetscDTEnumSubset(), PetscDTSubsetIndex()
M*/
PETSC_STATIC_INLINE PetscErrorCode PetscDTEnumSplit(PetscInt n, PetscInt k, PetscInt j, PetscInt *perm, PetscBool *isOdd)
{
  PetscInt       i, l, m, *subcomp, Nk;
  PetscInt       odd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (isOdd) *isOdd = PETSC_FALSE;
  ierr = PetscDTBinomialInt(n, k, &Nk);CHKERRQ(ierr);
  odd = 0;
  subcomp = &perm[k];
  for (i = 0, l = 0, m = 0; i < n && l < k; i++) {
    PetscInt Nminuskminus = (Nk * (k - l)) / (n - i);
    PetscInt Nminusk = Nk - Nminuskminus;

    if (j < Nminuskminus) {
      perm[l++] = i;
      Nk = Nminuskminus;
    } else {
      subcomp[m++] = i;
      j -= Nminuskminus;
      odd ^= ((k - l) & 1);
      Nk = Nminusk;
    }
  }
  for (; i < n; i++) {
    subcomp[m++] = i;
  }
  if (isOdd) *isOdd = odd ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#endif
