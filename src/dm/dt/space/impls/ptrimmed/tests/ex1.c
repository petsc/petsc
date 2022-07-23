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
  PetscCheck(Nb == Nbexp,comm, PETSC_ERR_PLIB, "Space dimension mismatch, %" PetscInt_FMT " != %" PetscInt_FMT, Nbexp, Nb);

  maxDexp = (PetscAbsInt(formDegree) == dim || formDegree == 0) ? degree : degree + 1;
  PetscCall(PetscSpaceGetDegree(sp, &d, &maxD));
  PetscCheck(degree == d,comm, PETSC_ERR_PLIB, "Space degree mismatch, %" PetscInt_FMT " != %" PetscInt_FMT, degree, d);
  PetscCheck(maxDexp == maxD,comm, PETSC_ERR_PLIB, "Space max degree mismatch, %" PetscInt_FMT " != %" PetscInt_FMT, maxDexp, maxD);

  PetscCall(PetscDTStroudConicalQuadrature(dim, 1, maxD + 1, -1., 1., &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &npoints, &points, NULL));

  Bsize = npoints * Nb * Nf * nCopies;
  Dsize = dim * Bsize;
  Hsize = dim * Dsize;
  PetscCall(PetscMalloc3(Bsize, &B, Dsize, &D, Hsize, &H));
  PetscCall(PetscSpaceEvaluate(sp, npoints, points, B, D, H));
  for (PetscInt i = 0; i < Bsize; i++) {
    PetscCheck(!PetscIsInfOrNanReal(B[i]),comm, PETSC_ERR_PLIB, "Bad value B[%" PetscInt_FMT "]", i);
  }
  for (PetscInt i = 0; i < Dsize; i++) {
    PetscCheck(!PetscIsInfOrNanReal(D[i]),comm, PETSC_ERR_PLIB, "Bad value D[%" PetscInt_FMT "]", i);
  }
  for (PetscInt i = 0; i < Hsize; i++) {
    PetscCheck(!PetscIsInfOrNanReal(H[i]),comm, PETSC_ERR_PLIB, "Bad value H[%" PetscInt_FMT "]", i);
  }
  PetscCall(PetscFree3(B, D, H));
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscCall(PetscSpaceDestroy(&sp));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{

  PetscFunctionBeginUser;
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
