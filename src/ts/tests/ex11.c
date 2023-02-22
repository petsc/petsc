static char help[] = "Demonstrates previous memory leak for XXXRegister()\n\n";

#include <petscts.h>
#include <petsccharacteristic.h>
#include <petscdraw.h>
#include <petscdm.h>
#include <petscpf.h>
#include <petscsf.h>
#include <petscao.h>

static PetscErrorCode TSGLLEAdaptCreate_Dummy(TSGLLEAdapt ts)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSGLLECreate_Dummy(TS ts)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAdaptCreate_Dummy(TSAdapt ts)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSCreate_Dummy(TS ts)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if !defined(PETSC_USE_COMPLEX)
static PetscErrorCode CharacteristicCreate_Dummy(Characteristic chr)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode SNESLineSearchCreate_Dummy(SNESLineSearch sneslinesearch)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESCreate_Dummy(SNES snes)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPCreate_Dummy(KSP ksp)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPGuessCreate_Dummy(KSPGuess ksp)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCCreate_Dummy(PC pc)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreate_Dummy(DM dm)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatOrderingCreate_Dummy(Mat mat, MatOrderingType mtype, IS *isr, IS *isc)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatPartitioningCreate_Dummy(MatPartitioning mat)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreate_Dummy(Mat mat)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PFCreate_Dummy(PF pf, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecCreate_Dummy(Vec vec)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecScatterCreate_Dummy(VecScatter vec)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFCreate_Dummy(PetscSF sf)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ISLocalToGlobalMappingCreate_Dummy(ISLocalToGlobalMapping is)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ISCreate_Dummy(IS is)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AOCreate_Dummy(AO ao)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawCreate_Dummy(PetscDraw draw)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerCreate_Dummy(PetscViewer viewer)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRandomCreate_Dummy(PetscRandom arand)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscReal A[1], Gamma[1] = {1.0}, b[1], c[1], d[1];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  /* TaoLineSearchRegister() also has the same memory leak */
  /* TaoRegister() also has the same memory leak */
  PetscCall(TSGLLEAdaptRegister("dummy", TSGLLEAdaptCreate_Dummy));
  PetscCall(TSGLLERegister("dummy", TSGLLECreate_Dummy));
  PetscCall(TSRKRegister("dummy", 0, 0, A, 0, 0, 0, 0, 0));
  PetscCall(TSGLEERegister("dummy", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(TSARKIMEXRegister("dummy", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(TSRosWRegister("dummy", 0, 1, A, Gamma, b, 0, 0, 0));
  PetscCall(TSBasicSymplecticRegister("dummy", 0, 0, c, d));
  PetscCall(TSAdaptRegister("dummy", TSAdaptCreate_Dummy));
  PetscCall(TSRegister("dummy", TSCreate_Dummy));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(CharacteristicRegister("dummy", CharacteristicCreate_Dummy));
#endif
  PetscCall(SNESLineSearchRegister("dummy", SNESLineSearchCreate_Dummy));
  PetscCall(SNESRegister("dummy", SNESCreate_Dummy));
  PetscCall(KSPGuessRegister("dummy", KSPGuessCreate_Dummy));
  PetscCall(KSPRegister("dummy", KSPCreate_Dummy));
  PetscCall(PCRegister("dummy", PCCreate_Dummy));
  PetscCall(DMRegister("dummy", DMCreate_Dummy));
  PetscCall(MatOrderingRegister("dummy", MatOrderingCreate_Dummy));
  PetscCall(MatPartitioningRegister("dummy", MatPartitioningCreate_Dummy));
  PetscCall(MatRegister("dummy", MatCreate_Dummy));
  PetscCall(PFRegister("dummy", PFCreate_Dummy));
  PetscCall(VecScatterRegister("dummy", VecScatterCreate_Dummy));
  PetscCall(VecRegister("dummy", VecCreate_Dummy));
  PetscCall(PetscSFRegister("dummy", PetscSFCreate_Dummy));
  PetscCall(ISLocalToGlobalMappingRegister("dummy", ISLocalToGlobalMappingCreate_Dummy));
  PetscCall(ISRegister("dummy", ISCreate_Dummy));
  PetscCall(AORegister("dummy", AOCreate_Dummy));
  PetscCall(PetscDrawRegister("dummy", PetscDrawCreate_Dummy));
  PetscCall(PetscViewerRegister("dummy", PetscViewerCreate_Dummy));
  PetscCall(PetscRandomRegister("dummy", PetscRandomCreate_Dummy));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
