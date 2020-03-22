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
  PetscErrorCode ierr;
  PetscReal      A[1],Gamma[1] = {1.0},b[1],c[1],d[1];
  
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* TaoLineSearchRegister() also has the same memory leak */
  /* TaoRegister() also has the same memory leak */
  ierr = TSGLLEAdaptRegister("dummy",TSGLLEAdaptCreate_Dummy);CHKERRQ(ierr);
  ierr = TSGLLERegister("dummy",TSGLLECreate_Dummy);CHKERRQ(ierr);
  ierr = TSRKRegister("dummy",0,0,A,0,0,0,0,0);CHKERRQ(ierr);
  ierr = TSGLEERegister("dummy",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = TSARKIMEXRegister("dummy",0,0,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = TSRosWRegister("dummy",0,1,A,Gamma,b,0,0,0);CHKERRQ(ierr);
  ierr = TSBasicSymplecticRegister("dummy",0,0,c,d);CHKERRQ(ierr);
  ierr = TSAdaptRegister("dummy",TSAdaptCreate_Dummy);CHKERRQ(ierr);
  ierr = TSRegister("dummy",TSCreate_Dummy);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = CharacteristicRegister("dummy",CharacteristicCreate_Dummy);CHKERRQ(ierr);
#endif
  ierr = SNESLineSearchRegister("dummy",SNESLineSearchCreate_Dummy);CHKERRQ(ierr);
  ierr = SNESRegister("dummy",SNESCreate_Dummy);CHKERRQ(ierr);
  ierr = KSPGuessRegister("dummy",KSPGuessCreate_Dummy);CHKERRQ(ierr);
  ierr = KSPRegister("dummy",KSPCreate_Dummy);CHKERRQ(ierr);
  ierr = PCRegister("dummy",PCCreate_Dummy);CHKERRQ(ierr);
  ierr = DMRegister("dummy",DMCreate_Dummy);CHKERRQ(ierr);
  ierr = MatOrderingRegister("dummy",MatOrderingCreate_Dummy);CHKERRQ(ierr);
  ierr = MatPartitioningRegister("dummy",MatPartitioningCreate_Dummy);CHKERRQ(ierr);
  ierr = MatRegister("dummy",MatCreate_Dummy);CHKERRQ(ierr);
  ierr = PFRegister("dummy",PFCreate_Dummy);CHKERRQ(ierr);
  ierr = VecScatterRegister("dummy",VecScatterCreate_Dummy);CHKERRQ(ierr);
  ierr = VecRegister("dummy",VecCreate_Dummy);CHKERRQ(ierr);
  ierr = PetscSFRegister("dummy",PetscSFCreate_Dummy);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRegister("dummy",ISLocalToGlobalMappingCreate_Dummy);CHKERRQ(ierr);
  ierr = ISRegister("dummy",ISCreate_Dummy);CHKERRQ(ierr);
  ierr = AORegister("dummy",AOCreate_Dummy);CHKERRQ(ierr);
  ierr = PetscDrawRegister("dummy",PetscDrawCreate_Dummy);CHKERRQ(ierr);
  ierr = PetscViewerRegister("dummy",PetscViewerCreate_Dummy);CHKERRQ(ierr);
  ierr = PetscRandomRegister("dummy",PetscRandomCreate_Dummy);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/

