const char help[] = "Test basic creation and evaluation of PETSCSPACEPTRIMMED";

#include <petscfe.h>

static PetscErrorCode test(PetscInt dim, PetscInt formDegree, PetscInt degree, PetscInt nCopies)
{
  MPI_Comm         comm = PETSC_COMM_SELF;
  PetscSpace       sp;
  PetscInt         Nf, Nb;
  PetscInt         maxDexp, maxD, d;
  PetscInt         Nbexp, Bsize, Dsize, Hsize;
  PetscReal       *B, *D, *H;
  PetscQuadrature  quad;
  PetscInt         npoints;
  const PetscReal *points;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceCreate(comm, &sp);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sp, "ptrimmed");CHKERRQ(ierr);
  ierr = PetscSpaceSetType(sp, PETSCSPACEPTRIMMED);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(sp, dim);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &Nf);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(sp, Nf * nCopies);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(sp, degree, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscSpacePTrimmedSetFormDegree(sp, formDegree);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
  ierr = PetscSpaceView(sp, NULL);CHKERRQ(ierr);

  ierr = PetscDTPTrimmedSize(dim, formDegree == 0 ? degree : degree + 1, PetscAbsInt(formDegree), &Nbexp);CHKERRQ(ierr);
  Nbexp *= nCopies;
  ierr = PetscSpaceGetDimension(sp, &Nb);CHKERRQ(ierr);
  if (Nb != Nbexp) SETERRQ(comm, PETSC_ERR_PLIB, "Space dimension mismatch, %D != %D", Nbexp, Nb);

  maxDexp = (PetscAbsInt(formDegree) == dim || formDegree == 0) ? degree : degree + 1;
  ierr = PetscSpaceGetDegree(sp, &d, &maxD);CHKERRQ(ierr);
  if (degree != d) SETERRQ(comm, PETSC_ERR_PLIB, "Space degree mismatch, %D != %D", degree, d);
  if (maxDexp != maxD) SETERRQ(comm, PETSC_ERR_PLIB, "Space max degree mismatch, %D != %D", maxDexp, maxD);

  ierr = PetscDTStroudConicalQuadrature(dim, 1, maxD + 1, -1., 1., &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &npoints, &points, NULL);CHKERRQ(ierr);

  Bsize = npoints * Nb * Nf * nCopies;
  Dsize = dim * Bsize;
  Hsize = dim * Dsize;
  ierr = PetscMalloc3(Bsize, &B, Dsize, &D, Hsize, &H);CHKERRQ(ierr);
  ierr = PetscSpaceEvaluate(sp, npoints, points, B, D, H);CHKERRQ(ierr);
  for (PetscInt i = 0; i < Bsize; i++) {
    if (PetscIsInfOrNanReal(B[i])) SETERRQ(comm, PETSC_ERR_PLIB, "Bad value B[%D]", i);
  }
  for (PetscInt i = 0; i < Dsize; i++) {
    if (PetscIsInfOrNanReal(D[i])) SETERRQ(comm, PETSC_ERR_PLIB, "Bad value D[%D]", i);
  }
  for (PetscInt i = 0; i < Hsize; i++) {
    if (PetscIsInfOrNanReal(H[i])) SETERRQ(comm, PETSC_ERR_PLIB, "Bad value H[%H]", i);
  }
  ierr = PetscFree3(B, D, H);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  for (PetscInt dim = 0; dim <= 3; dim++) {
    for (PetscInt formDegree = -dim; formDegree <= dim; formDegree++) {
      for (PetscInt degree = 0; degree <= 4; degree++) {
        if (formDegree == 0 && degree == 0) continue;
        for (PetscInt nCopies = 1; nCopies <= PetscMax(2,dim); nCopies++) {
          ierr = test(dim, formDegree, degree, nCopies);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:

TEST*/
