const char help[] = "Tests PetscDTPTrimmedEvalJet()";

#include <petscdt.h>
#include <petscblaslapack.h>
#include <petscmat.h>

static PetscErrorCode constructTabulationAndMass(PetscInt dim, PetscInt deg, PetscInt form, PetscInt jetDegree, PetscInt npoints,
                                                 const PetscReal *points, const PetscReal *weights,
                                                 PetscInt *_Nb, PetscInt *_Nf, PetscInt *_Nk,
                                                 PetscReal **B, PetscScalar **M)
{
  PetscInt       Nf; // Number of form components
  PetscInt       Nbpt; // number of trimmed polynomials
  PetscInt       Nk; // jet size
  PetscReal     *p_trimmed;

  PetscFunctionBegin;
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(form), &Nf));
  PetscCall(PetscDTPTrimmedSize(dim, deg, form, &Nbpt));
  PetscCall(PetscDTBinomialInt(dim + jetDegree, dim, &Nk));
  PetscCall(PetscMalloc1(Nbpt * Nf * Nk * npoints, &p_trimmed));
  PetscCall(PetscDTPTrimmedEvalJet(dim, npoints, points, deg, form, jetDegree, p_trimmed));

  // compute the direct mass matrix
  PetscScalar *M_trimmed;
  PetscCall(PetscCalloc1(Nbpt * Nbpt, &M_trimmed));
  for (PetscInt i = 0; i < Nbpt; i++) {
    for (PetscInt j = 0; j < Nbpt; j++) {
      PetscReal v = 0.;

      for (PetscInt f = 0; f < Nf; f++) {
        const PetscReal *p_i = &p_trimmed[(i * Nf + f) * Nk * npoints];
        const PetscReal *p_j = &p_trimmed[(j * Nf + f) * Nk * npoints];

        for (PetscInt pt = 0; pt < npoints; pt++) {
          v += p_i[pt] * p_j[pt] * weights[pt];
        }
      }
      M_trimmed[i * Nbpt + j] += v;
    }
  }
  *_Nb = Nbpt;
  *_Nf = Nf;
  *_Nk = Nk;
  *B = p_trimmed;
  *M = M_trimmed;
  PetscFunctionReturn(0);
}

static PetscErrorCode test(PetscInt dim, PetscInt deg, PetscInt form, PetscInt jetDegree, PetscBool cond)
{
  PetscQuadrature  q;
  PetscInt         npoints;
  const PetscReal *points;
  const PetscReal *weights;
  PetscInt         Nf; // Number of form components
  PetscInt         Nk; // jet size
  PetscInt         Nbpt; // number of trimmed polynomials
  PetscReal       *p_trimmed;
  PetscScalar     *M_trimmed;
  PetscReal       *p_scalar;
  PetscInt         Nbp; // number of scalar polynomials
  PetscScalar     *Mcopy;
  PetscScalar     *M_moments;
  PetscReal        frob_err = 0.;
  Mat              mat_trimmed;
  Mat              mat_moments_T;
  Mat              AinvB;
  PetscInt         Nbm1;
  Mat              Mm1;
  PetscReal       *p_trimmed_copy;
  PetscReal       *M_moment_real;

  PetscFunctionBegin;
  // Construct an appropriate quadrature
  PetscCall(PetscDTStroudConicalQuadrature(dim, 1, deg + 2, -1., 1., &q));
  PetscCall(PetscQuadratureGetData(q, NULL, NULL, &npoints, &points, &weights));

  PetscCall(constructTabulationAndMass(dim, deg, form, jetDegree, npoints, points, weights, &Nbpt, &Nf, &Nk, &p_trimmed, &M_trimmed));

  PetscCall(PetscDTBinomialInt(dim + deg, dim, &Nbp));
  PetscCall(PetscMalloc1(Nbp * Nk * npoints, &p_scalar));
  PetscCall(PetscDTPKDEvalJet(dim, npoints, points, deg, jetDegree, p_scalar));

  PetscCall(PetscMalloc1(Nbpt * Nbpt, &Mcopy));
  // Print the condition numbers (useful for testing out different bases internally in PetscDTPTrimmedEvalJet())
#if !defined(PETSC_USE_COMPLEX)
  if (cond) {
    PetscReal *S;
    PetscScalar *work;
    PetscBLASInt n = Nbpt;
    PetscBLASInt lwork = 5 * Nbpt;
    PetscBLASInt lierr;

    PetscCall(PetscMalloc1(Nbpt, &S));
    PetscCall(PetscMalloc1(5*Nbpt, &work));
    PetscCall(PetscArraycpy(Mcopy, M_trimmed, Nbpt * Nbpt));

    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","N",&n,&n,Mcopy,&n,S,NULL,&n,NULL,&n,work,&lwork,&lierr));
    PetscReal cond = S[0] / S[Nbpt - 1];
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dimension %" PetscInt_FMT ", degree %" PetscInt_FMT ", form %" PetscInt_FMT ": condition number %g\n", dim, deg, form, (double) cond));
    PetscCall(PetscFree(work));
    PetscCall(PetscFree(S));
  }
#endif

  // compute the moments with the orthonormal polynomials
  PetscCall(PetscCalloc1(Nbpt * Nbp * Nf, &M_moments));
  for (PetscInt i = 0; i < Nbp; i++) {
    for (PetscInt j = 0; j < Nbpt; j++) {
      for (PetscInt f = 0; f < Nf; f++) {
        PetscReal        v = 0.;
        const PetscReal *p_i = &p_scalar[i * Nk * npoints];
        const PetscReal *p_j = &p_trimmed[(j * Nf + f) * Nk * npoints];

        for (PetscInt pt = 0; pt < npoints; pt++) {
          v += p_i[pt] * p_j[pt] * weights[pt];
        }
        M_moments[(i * Nf + f) * Nbpt + j] += v;
      }
    }
  }

  // subtract M_moments^T * M_moments from M_trimmed: because the trimmed polynomials should be contained in
  // the full polynomials, the result should be zero
  PetscCall(PetscArraycpy(Mcopy, M_trimmed, Nbpt * Nbpt));
  {
    PetscBLASInt m = Nbpt;
    PetscBLASInt n = Nbpt;
    PetscBLASInt k = Nbp * Nf;
    PetscScalar mone = -1.;
    PetscScalar one = 1.;

    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&m,&n,&k,&mone,M_moments,&m,M_moments,&m,&one,Mcopy,&m));
  }

  frob_err = 0.;
  for (PetscInt i = 0; i < Nbpt * Nbpt; i++) frob_err += PetscRealPart(Mcopy[i]) * PetscRealPart(Mcopy[i]);
  frob_err = PetscSqrtReal(frob_err);

  if (frob_err > PETSC_SMALL) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dimension %" PetscInt_FMT ", degree %" PetscInt_FMT ", form %" PetscInt_FMT ": trimmed projection error %g", dim, deg, form, (double) frob_err);
  }

  // P trimmed is also supposed to contain the polynomials of one degree less: construction M_moment[0:sub,:] * M_trimmed^{-1} * M_moments[0:sub,:]^T should be the identity matrix
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Nbpt, Nbpt, M_trimmed, &mat_trimmed));
  PetscCall(PetscDTBinomialInt(dim + deg - 1, dim, &Nbm1));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Nbpt, Nbm1 * Nf, M_moments, &mat_moments_T));
  PetscCall(MatDuplicate(mat_moments_T, MAT_DO_NOT_COPY_VALUES, &AinvB));
  PetscCall(MatLUFactor(mat_trimmed, NULL, NULL, NULL));
  PetscCall(MatMatSolve(mat_trimmed, mat_moments_T, AinvB));
  PetscCall(MatTransposeMatMult(mat_moments_T, AinvB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mm1));
  PetscCall(MatShift(Mm1, -1.));
  PetscCall(MatNorm(Mm1, NORM_FROBENIUS, &frob_err));
  if (frob_err > PETSC_SMALL) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dimension %" PetscInt_FMT ", degree %" PetscInt_FMT ", form %" PetscInt_FMT ": trimmed reverse projection error %g", dim, deg, form, (double) frob_err);
  }
  PetscCall(MatDestroy(&Mm1));
  PetscCall(MatDestroy(&AinvB));
  PetscCall(MatDestroy(&mat_moments_T));

  // The Koszul differential applied to P trimmed (Lambda k+1) should be contained in P trimmed (Lambda k)
  if (PetscAbsInt(form) < dim) {
    PetscInt     Nf1, Nbpt1, Nk1;
    PetscReal   *p_trimmed1;
    PetscScalar *M_trimmed1;
    PetscInt   (*pattern)[3];
    PetscReal   *p_koszul;
    PetscScalar *M_koszul;
    PetscScalar *M_k_moment;
    Mat          mat_koszul;
    Mat          mat_k_moment_T;
    Mat          AinvB;
    Mat          prod;

    PetscCall(constructTabulationAndMass(dim, deg, form < 0 ? form - 1 : form + 1, 0, npoints, points, weights, &Nbpt1, &Nf1, &Nk1,
                                       &p_trimmed1, &M_trimmed1));

    PetscCall(PetscMalloc1(Nf1 * (PetscAbsInt(form) + 1), &pattern));
    PetscCall(PetscDTAltVInteriorPattern(dim, PetscAbsInt(form) + 1, pattern));

    // apply the Koszul operator
    PetscCall(PetscCalloc1(Nbpt1 * Nf * npoints, &p_koszul));
    for (PetscInt b = 0; b < Nbpt1; b++) {
      for (PetscInt a = 0; a < Nf1 * (PetscAbsInt(form) + 1); a++) {
        PetscInt         i,j,k;
        PetscReal        sign;
        PetscReal       *p_i;
        const PetscReal *p_j;

        i = pattern[a][0];
        if (form < 0) {
          i = Nf-1-i;
        }
        j = pattern[a][1];
        if (form < 0) {
          j = Nf1-1-j;
        }
        k = pattern[a][2] < 0 ? -(pattern[a][2] + 1) : pattern[a][2];
        sign = pattern[a][2] < 0 ? -1 : 1;
        if (form < 0 && (i & 1) ^ (j & 1)) {
          sign = -sign;
        }

        p_i = &p_koszul[(b * Nf + i) * npoints];
        p_j = &p_trimmed1[(b * Nf1 + j) * npoints];
        for (PetscInt pt = 0; pt < npoints; pt++) {
          p_i[pt] += p_j[pt] * points[pt * dim + k] * sign;
        }
      }
    }

    // mass matrix of the result
    PetscCall(PetscMalloc1(Nbpt1 * Nbpt1, &M_koszul));
    for (PetscInt i = 0; i < Nbpt1; i++) {
      for (PetscInt j = 0; j < Nbpt1; j++) {
        PetscReal val = 0.;

        for (PetscInt v = 0; v < Nf; v++) {
          const PetscReal *p_i = &p_koszul[(i * Nf + v) * npoints];
          const PetscReal *p_j = &p_koszul[(j * Nf + v) * npoints];

          for (PetscInt pt = 0; pt < npoints; pt++) {
            val += p_i[pt] * p_j[pt] * weights[pt];
          }
        }
        M_koszul[i * Nbpt1 + j] = val;
      }
    }

    // moment matrix between the result and P trimmed
    PetscCall(PetscMalloc1(Nbpt * Nbpt1, &M_k_moment));
    for (PetscInt i = 0; i < Nbpt1; i++) {
      for (PetscInt j = 0; j < Nbpt; j++) {
        PetscReal val = 0.;

        for (PetscInt v = 0; v < Nf; v++) {
          const PetscReal *p_i = &p_koszul[(i * Nf + v) * npoints];
          const PetscReal *p_j = &p_trimmed[(j * Nf + v) * Nk * npoints];

          for (PetscInt pt = 0; pt < npoints; pt++) {
            val += p_i[pt] * p_j[pt] * weights[pt];
          }
        }
        M_k_moment[i * Nbpt + j] = val;
      }
    }

    // M_k_moment M_trimmed^{-1} M_k_moment^T == M_koszul
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Nbpt1, Nbpt1, M_koszul, &mat_koszul));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Nbpt, Nbpt1, M_k_moment, &mat_k_moment_T));
    PetscCall(MatDuplicate(mat_k_moment_T, MAT_DO_NOT_COPY_VALUES, &AinvB));
    PetscCall(MatMatSolve(mat_trimmed, mat_k_moment_T, AinvB));
    PetscCall(MatTransposeMatMult(mat_k_moment_T, AinvB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &prod));
    PetscCall(MatAXPY(prod, -1., mat_koszul, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(prod, NORM_FROBENIUS, &frob_err));
    if (frob_err > PETSC_SMALL) {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dimension %" PetscInt_FMT ", degree %" PetscInt_FMT ", forms (%" PetscInt_FMT ", %" PetscInt_FMT "): koszul projection error %g", dim, deg, form, form < 0 ? (form-1):(form+1), (double) frob_err);
    }

    PetscCall(MatDestroy(&prod));
    PetscCall(MatDestroy(&AinvB));
    PetscCall(MatDestroy(&mat_k_moment_T));
    PetscCall(MatDestroy(&mat_koszul));
    PetscCall(PetscFree(M_k_moment));
    PetscCall(PetscFree(M_koszul));
    PetscCall(PetscFree(p_koszul));
    PetscCall(PetscFree(pattern));
    PetscCall(PetscFree(p_trimmed1));
    PetscCall(PetscFree(M_trimmed1));
  }

  // M_moments has shape [Nbp][Nf][Nbpt]
  // p_scalar has shape [Nbp][Nk][npoints]
  // contracting on [Nbp] should be the same shape as
  // p_trimmed, which is [Nbpt][Nf][Nk][npoints]
  PetscCall(PetscCalloc1(Nbpt * Nf * Nk * npoints, &p_trimmed_copy));
  PetscCall(PetscMalloc1(Nbp * Nf * Nbpt, &M_moment_real));
  for (PetscInt i = 0; i < Nbp * Nf * Nbpt; i++) {
    M_moment_real[i] = PetscRealPart(M_moments[i]);
  }
  for (PetscInt f = 0; f < Nf; f++) {
    PetscBLASInt m = Nk * npoints;
    PetscBLASInt n = Nbpt;
    PetscBLASInt k = Nbp;
    PetscBLASInt lda = Nk * npoints;
    PetscBLASInt ldb = Nf * Nbpt;
    PetscBLASInt ldc = Nf * Nk * npoints;
    PetscReal    alpha = 1.0;
    PetscReal    beta = 1.0;

    PetscStackCallBLAS("BLASREALgemm",BLASREALgemm_("N","T",&m,&n,&k,&alpha,p_scalar,&lda,&M_moment_real[f * Nbpt],&ldb,&beta,&p_trimmed_copy[f * Nk * npoints],&ldc));
  }
  frob_err = 0.;
  for (PetscInt i = 0; i < Nbpt * Nf * Nk * npoints; i++) {
    frob_err += (p_trimmed_copy[i] - p_trimmed[i]) * (p_trimmed_copy[i] - p_trimmed[i]);
  }
  frob_err = PetscSqrtReal(frob_err);

  if (frob_err > PETSC_SMALL) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dimension %" PetscInt_FMT ", degree %" PetscInt_FMT ", form %" PetscInt_FMT ": jet error %g", dim, deg, form, (double) frob_err);
  }

  PetscCall(PetscFree(M_moment_real));
  PetscCall(PetscFree(p_trimmed_copy));
  PetscCall(MatDestroy(&mat_trimmed));
  PetscCall(PetscFree(Mcopy));
  PetscCall(PetscFree(M_moments));
  PetscCall(PetscFree(M_trimmed));
  PetscCall(PetscFree(p_trimmed));
  PetscCall(PetscFree(p_scalar));
  PetscCall(PetscQuadratureDestroy(&q));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt       max_dim = 3;
  PetscInt       max_deg = 4;
  PetscInt       k       = 3;
  PetscBool      cond    = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD,"","Options for PetscDTPTrimmedEvalJet() tests","none");
  PetscCall(PetscOptionsInt("-max_dim", "Maximum dimension of the simplex",__FILE__,max_dim,&max_dim,NULL));
  PetscCall(PetscOptionsInt("-max_degree", "Maximum degree of the trimmed polynomial space",__FILE__,max_deg,&max_deg,NULL));
  PetscCall(PetscOptionsInt("-max_jet", "The number of derivatives to test",__FILE__,k,&k,NULL));
  PetscCall(PetscOptionsBool("-cond", "Compute the condition numbers of the mass matrices of the bases",__FILE__,cond,&cond,NULL));
  PetscOptionsEnd();
  for (PetscInt dim = 2; dim <= max_dim; dim++) {
    for (PetscInt deg = 1; deg <= max_deg; deg++) {
      for (PetscInt form = -dim+1; form <= dim; form++) {
        PetscCall(test(dim, deg, form, PetscMax(1, k), cond));
      }
    }
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: !single
    args:

TEST*/
