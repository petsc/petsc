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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(form), &Nf);CHKERRQ(ierr);
  ierr = PetscDTPTrimmedSize(dim, deg, form, &Nbpt);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim + jetDegree, dim, &Nk);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nbpt * Nf * Nk * npoints, &p_trimmed);CHKERRQ(ierr);
  ierr = PetscDTPTrimmedEvalJet(dim, npoints, points, deg, form, jetDegree, p_trimmed);CHKERRQ(ierr);

  // compute the direct mass matrix
  PetscScalar *M_trimmed;
  ierr = PetscCalloc1(Nbpt * Nbpt, &M_trimmed);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  // Construct an appropriate quadrature
  ierr = PetscDTStroudConicalQuadrature(dim, 1, deg + 2, -1., 1., &q);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q, NULL, NULL, &npoints, &points, &weights);CHKERRQ(ierr);

  ierr = constructTabulationAndMass(dim, deg, form, jetDegree, npoints, points, weights, &Nbpt, &Nf, &Nk, &p_trimmed, &M_trimmed);CHKERRQ(ierr);

  ierr = PetscDTBinomialInt(dim + deg, dim, &Nbp);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nbp * Nk * npoints, &p_scalar);CHKERRQ(ierr);
  ierr = PetscDTPKDEvalJet(dim, npoints, points, deg, jetDegree, p_scalar);CHKERRQ(ierr);

  ierr = PetscMalloc1(Nbpt * Nbpt, &Mcopy);CHKERRQ(ierr);
  // Print the condition numbers (useful for testing out different bases internally in PetscDTPTrimmedEvalJet())
#if !defined(PETSC_USE_COMPLEX)
  if (cond) {
    PetscReal *S;
    PetscScalar *work;
    PetscBLASInt n = Nbpt;
    PetscBLASInt lwork = 5 * Nbpt;
    PetscBLASInt lierr;

    ierr = PetscMalloc1(Nbpt, &S);CHKERRQ(ierr);
    ierr = PetscMalloc1(5*Nbpt, &work);CHKERRQ(ierr);
    ierr = PetscArraycpy(Mcopy, M_trimmed, Nbpt * Nbpt);CHKERRQ(ierr);

    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","N",&n,&n,Mcopy,&n,S,NULL,&n,NULL,&n,work,&lwork,&lierr));
    PetscReal cond = S[0] / S[Nbpt - 1];
    ierr = PetscPrintf(PETSC_COMM_WORLD, "dimension %D, degree %D, form %D: condition number %g\n", dim, deg, form, (double) cond);
    ierr = PetscFree(work);CHKERRQ(ierr);
    ierr = PetscFree(S);CHKERRQ(ierr);
  }
#endif

  // compute the moments with the orthonormal polynomials
  ierr = PetscCalloc1(Nbpt * Nbp * Nf, &M_moments);CHKERRQ(ierr);
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
  ierr = PetscArraycpy(Mcopy, M_trimmed, Nbpt * Nbpt);CHKERRQ(ierr);
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
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dimension %D, degree %D, form %D: trimmed projection error %g", dim, deg, form, (double) frob_err);
  }

  // P trimmed is also supposed to contain the polynomials of one degree less: construction M_moment[0:sub,:] * M_trimmed^{-1} * M_moments[0:sub,:]^T should be the identity matrix
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, Nbpt, Nbpt, M_trimmed, &mat_trimmed);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim + deg - 1, dim, &Nbm1);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, Nbpt, Nbm1 * Nf, M_moments, &mat_moments_T);CHKERRQ(ierr);
  ierr = MatDuplicate(mat_moments_T, MAT_DO_NOT_COPY_VALUES, &AinvB);CHKERRQ(ierr);
  ierr = MatLUFactor(mat_trimmed, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = MatMatSolve(mat_trimmed, mat_moments_T, AinvB);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(mat_moments_T, AinvB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mm1);CHKERRQ(ierr);
  ierr = MatShift(Mm1, -1.);CHKERRQ(ierr);
  ierr = MatNorm(Mm1, NORM_FROBENIUS, &frob_err);CHKERRQ(ierr);
  if (frob_err > PETSC_SMALL) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dimension %D, degree %D, form %D: trimmed reverse projection error %g", dim, deg, form, (double) frob_err);
  }
  ierr = MatDestroy(&Mm1);CHKERRQ(ierr);
  ierr = MatDestroy(&AinvB);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_moments_T);CHKERRQ(ierr);

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

    ierr = constructTabulationAndMass(dim, deg, form < 0 ? form - 1 : form + 1, 0, npoints, points, weights, &Nbpt1, &Nf1, &Nk1,
                                      &p_trimmed1, &M_trimmed1);CHKERRQ(ierr);

    ierr = PetscMalloc1(Nf1 * (PetscAbsInt(form) + 1), &pattern);CHKERRQ(ierr);
    ierr = PetscDTAltVInteriorPattern(dim, PetscAbsInt(form) + 1, pattern);CHKERRQ(ierr);

    // apply the Koszul operator
    ierr = PetscCalloc1(Nbpt1 * Nf * npoints, &p_koszul);CHKERRQ(ierr);
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
    ierr = PetscMalloc1(Nbpt1 * Nbpt1, &M_koszul);CHKERRQ(ierr);
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
    ierr = PetscMalloc1(Nbpt * Nbpt1, &M_k_moment);CHKERRQ(ierr);
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
    ierr = MatCreateSeqDense(PETSC_COMM_SELF, Nbpt1, Nbpt1, M_koszul, &mat_koszul);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF, Nbpt, Nbpt1, M_k_moment, &mat_k_moment_T);CHKERRQ(ierr);
    ierr = MatDuplicate(mat_k_moment_T, MAT_DO_NOT_COPY_VALUES, &AinvB);CHKERRQ(ierr);
    ierr = MatMatSolve(mat_trimmed, mat_k_moment_T, AinvB);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(mat_k_moment_T, AinvB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &prod);CHKERRQ(ierr);
    ierr = MatAXPY(prod, -1., mat_koszul, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(prod, NORM_FROBENIUS, &frob_err);CHKERRQ(ierr);
    if (frob_err > PETSC_SMALL) {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dimension %D, degree %D, forms (%D, %D): koszul projection error %g", dim, deg, form, form < 0 ? (form-1):(form+1), (double) frob_err);
    }

    ierr = MatDestroy(&prod);CHKERRQ(ierr);
    ierr = MatDestroy(&AinvB);CHKERRQ(ierr);
    ierr = MatDestroy(&mat_k_moment_T);CHKERRQ(ierr);
    ierr = MatDestroy(&mat_koszul);CHKERRQ(ierr);
    ierr = PetscFree(M_k_moment);CHKERRQ(ierr);
    ierr = PetscFree(M_koszul);CHKERRQ(ierr);
    ierr = PetscFree(p_koszul);CHKERRQ(ierr);
    ierr = PetscFree(pattern);CHKERRQ(ierr);
    ierr = PetscFree(p_trimmed1);CHKERRQ(ierr);
    ierr = PetscFree(M_trimmed1);CHKERRQ(ierr);
  }

  // M_moments has shape [Nbp][Nf][Nbpt]
  // p_scalar has shape [Nbp][Nk][npoints]
  // contracting on [Nbp] should be the same shape as
  // p_trimmed, which is [Nbpt][Nf][Nk][npoints]
  ierr = PetscCalloc1(Nbpt * Nf * Nk * npoints, &p_trimmed_copy);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nbp * Nf * Nbpt, &M_moment_real);CHKERRQ(ierr);
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
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dimension %D, degree %D, form %D: jet error %g", dim, deg, form, (double) frob_err);
  }

  ierr = PetscFree(M_moment_real);CHKERRQ(ierr);
  ierr = PetscFree(p_trimmed_copy);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_trimmed);CHKERRQ(ierr);
  ierr = PetscFree(Mcopy);CHKERRQ(ierr);
  ierr = PetscFree(M_moments);CHKERRQ(ierr);
  ierr = PetscFree(M_trimmed);CHKERRQ(ierr);
  ierr = PetscFree(p_trimmed);CHKERRQ(ierr);
  ierr = PetscFree(p_scalar);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt  max_dim = 3;
  PetscInt  max_deg = 4;
  PetscInt  k = 3;
  PetscBool cond = PETSC_FALSE;

  PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Options for PetscDTPTrimmedEvalJet() tests","none");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-max_dim", "Maximum dimension of the simplex",__FILE__,max_dim,&max_dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-max_degree", "Maximum degree of the trimmed polynomial space",__FILE__,max_deg,&max_deg,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-max_jet", "The number of derivatives to test",__FILE__,k,&k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cond", "Compute the condition numbers of the mass matrices of the bases",__FILE__,cond,&cond,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  for (PetscInt dim = 2; dim <= max_dim; dim++) {
    for (PetscInt deg = 1; deg <= max_deg; deg++) {
      for (PetscInt form = -dim+1; form <= dim; form++) {
        ierr = test(dim, deg, form, PetscMax(1, k), cond);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    requires: !single
    args:

TEST*/
