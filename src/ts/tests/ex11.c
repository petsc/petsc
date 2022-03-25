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
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLECreate_Dummy(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptCreate_Dummy(TSAdapt ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSCreate_Dummy(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#if !defined(PETSC_USE_COMPLEX)
static PetscErrorCode CharacteristicCreate_Dummy(Characteristic chr)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode SNESLineSearchCreate_Dummy(SNESLineSearch sneslinesearch)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESCreate_Dummy(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPCreate_Dummy(KSP ksp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessCreate_Dummy(KSPGuess ksp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCCreate_Dummy(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreate_Dummy(DM dm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatOrderingCreate_Dummy(Mat mat,MatOrderingType mtype,IS *isr,IS *isc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPartitioningCreate_Dummy(MatPartitioning mat)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreate_Dummy(Mat mat)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PFCreate_Dummy(PF pf,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecCreate_Dummy(Vec vec)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterCreate_Dummy(VecScatter vec)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFCreate_Dummy(PetscSF sf)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISLocalToGlobalMappingCreate_Dummy(ISLocalToGlobalMapping is)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISCreate_Dummy(IS is)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode AOCreate_Dummy(AO ao)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawCreate_Dummy(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerCreate_Dummy(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscRandomCreate_Dummy(PetscRandom arand)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscReal      A[1],Gamma[1] = {1.0},b[1],c[1],d[1];

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  /* TaoLineSearchRegister() also has the same memory leak */
  /* TaoRegister() also has the same memory leak */
  PetscCall(TSGLLEAdaptRegister("dummy",TSGLLEAdaptCreate_Dummy));
  PetscCall(TSGLLERegister("dummy",TSGLLECreate_Dummy));
  PetscCall(TSRKRegister("dummy",0,0,A,0,0,0,0,0));
  PetscCall(TSGLEERegister("dummy",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0));
  PetscCall(TSARKIMEXRegister("dummy",0,0,0,0,0,0,0,0,0,0,0,0,0));
  PetscCall(TSRosWRegister("dummy",0,1,A,Gamma,b,0,0,0));
  PetscCall(TSBasicSymplecticRegister("dummy",0,0,c,d));
  PetscCall(TSAdaptRegister("dummy",TSAdaptCreate_Dummy));
  PetscCall(TSRegister("dummy",TSCreate_Dummy));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(CharacteristicRegister("dummy",CharacteristicCreate_Dummy));
#endif
  PetscCall(SNESLineSearchRegister("dummy",SNESLineSearchCreate_Dummy));
  PetscCall(SNESRegister("dummy",SNESCreate_Dummy));
  PetscCall(KSPGuessRegister("dummy",KSPGuessCreate_Dummy));
  PetscCall(KSPRegister("dummy",KSPCreate_Dummy));
  PetscCall(PCRegister("dummy",PCCreate_Dummy));
  PetscCall(DMRegister("dummy",DMCreate_Dummy));
  PetscCall(MatOrderingRegister("dummy",MatOrderingCreate_Dummy));
  PetscCall(MatPartitioningRegister("dummy",MatPartitioningCreate_Dummy));
  PetscCall(MatRegister("dummy",MatCreate_Dummy));
  PetscCall(PFRegister("dummy",PFCreate_Dummy));
  PetscCall(VecScatterRegister("dummy",VecScatterCreate_Dummy));
  PetscCall(VecRegister("dummy",VecCreate_Dummy));
  PetscCall(PetscSFRegister("dummy",PetscSFCreate_Dummy));
  PetscCall(ISLocalToGlobalMappingRegister("dummy",ISLocalToGlobalMappingCreate_Dummy));
  PetscCall(ISRegister("dummy",ISCreate_Dummy));
  PetscCall(AORegister("dummy",AOCreate_Dummy));
  PetscCall(PetscDrawRegister("dummy",PetscDrawCreate_Dummy));
  PetscCall(PetscViewerRegister("dummy",PetscViewerCreate_Dummy));
  PetscCall(PetscRandomRegister("dummy",PetscRandomCreate_Dummy));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
