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

  PetscFunctionBegin;
  PetscCall(PetscSpaceCreate(comm, &sp));
  PetscCall(PetscObjectSetName((PetscObject)sp, "ptrimmed"));
  PetscCall(PetscSpaceSetType(sp, PETSCSPACEPTRIMMED));
  PetscCall(PetscSpaceSetNumVariables(sp, dim));
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &Nf));
  PetscCall(PetscSpaceSetNumComponents(sp, Nf * nCopies));
  PetscCall(PetscSpaceSetDegree(sp, degree, PETSC_DETERMINE));
  PetscCall(PetscSpacePTrimmedSetFormDegree(sp, formDegree));
  PetscCall(PetscSpaceSetUp(sp));
  PetscCall(PetscSpaceView(sp, NULL));

  PetscCall(PetscDTPTrimmedSize(dim, formDegree == 0 ? degree : degree + 1, PetscAbsInt(formDegree), &Nbexp));
  Nbexp *= nCopies;
  PetscCall(PetscSpaceGetDimension(sp, &Nb));
  PetscCheckFalse(Nb != Nbexp,comm, PETSC_ERR_PLIB, "Space dimension mismatch, %D != %D", Nbexp, Nb);

  maxDexp = (PetscAbsInt(formDegree) == dim || formDegree == 0) ? degree : degree + 1;
  PetscCall(PetscSpaceGetDegree(sp, &d, &maxD));
  PetscCheckFalse(degree != d,comm, PETSC_ERR_PLIB, "Space degree mismatch, %D != %D", degree, d);
  PetscCheckFalse(maxDexp != maxD,comm, PETSC_ERR_PLIB, "Space max degree mismatch, %D != %D", maxDexp, maxD);

  PetscCall(PetscDTStroudConicalQuadrature(dim, 1, maxD + 1, -1., 1., &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &npoints, &points, NULL));

  Bsize = npoints * Nb * Nf * nCopies;
  Dsize = dim * Bsize;
  Hsize = dim * Dsize;
  PetscCall(PetscMalloc3(Bsize, &B, Dsize, &D, Hsize, &H));
  PetscCall(PetscSpaceEvaluate(sp, npoints, points, B, D, H));
  for (PetscInt i = 0; i < Bsize; i++) {
    PetscCheckFalse(PetscIsInfOrNanReal(B[i]),comm, PETSC_ERR_PLIB, "Bad value B[%D]", i);
  }
  for (PetscInt i = 0; i < Dsize; i++) {
    PetscCheckFalse(PetscIsInfOrNanReal(D[i]),comm, PETSC_ERR_PLIB, "Bad value D[%D]", i);
  }
  for (PetscInt i = 0; i < Hsize; i++) {
    PetscCheckFalse(PetscIsInfOrNanReal(H[i]),comm, PETSC_ERR_PLIB, "Bad value H[%H]", i);
  }
  PetscCall(PetscFree3(B, D, H));
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscCall(PetscSpaceDestroy(&sp));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  for (PetscInt dim = 0; dim <= 3; dim++) {
    for (PetscInt formDegree = -dim; formDegree <= dim; formDegree++) {
      for (PetscInt degree = 0; degree <= 4; degree++) {
        if (formDegree == 0 && degree == 0) continue;
        for (PetscInt nCopies = 1; nCopies <= PetscMax(2,dim); nCopies++) {
          PetscCall(test(dim, formDegree, degree, nCopies));
        }
      }
    }
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:

TEST*/
