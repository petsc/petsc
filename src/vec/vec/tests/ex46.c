
static char help[] = "Tests PetscViewerBinary VecView()/VecLoad() function correctly when binary header is skipped.\n\n";

/*T
 Concepts: viewers^skipheader^mpiio
T*/

#include <petscviewer.h>
#include <petscvec.h>

#define VEC_LEN 10
const PetscReal test_values[] = { 0.311256, 88.068, 11.077444, 9953.62, 7.345, 64.8943, 3.1458, 6699.95, 0.00084, 0.0647 };

PetscErrorCode MyVecDump(const char fname[],PetscBool skippheader,PetscBool usempiio,Vec x)
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscBool      ismpiio,isskip;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject)x,&comm));

  CHKERRQ(PetscViewerCreate(comm,&viewer));
  CHKERRQ(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  if (skippheader) CHKERRQ(PetscViewerBinarySetSkipHeader(viewer,PETSC_TRUE));
  CHKERRQ(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  if (usempiio) CHKERRQ(PetscViewerBinarySetUseMPIIO(viewer,PETSC_TRUE));
  CHKERRQ(PetscViewerFileSetName(viewer,fname));

  CHKERRQ(VecView(x,viewer));

  CHKERRQ(PetscViewerBinaryGetUseMPIIO(viewer,&ismpiio));
  if (ismpiio) CHKERRQ(PetscPrintf(comm,"*** PetscViewer[write] using MPI-IO ***\n"));
  CHKERRQ(PetscViewerBinaryGetSkipHeader(viewer,&isskip));
  if (isskip) CHKERRQ(PetscPrintf(comm,"*** PetscViewer[write] skipping header ***\n"));

  CHKERRQ(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode MyVecLoad(const char fname[],PetscBool skippheader,PetscBool usempiio,Vec x)
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscBool      ismpiio,isskip;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject)x,&comm));

  CHKERRQ(PetscViewerCreate(comm,&viewer));
  CHKERRQ(PetscViewerSetType(viewer,PETSCVIEWERBINARY));
  if (skippheader) CHKERRQ(PetscViewerBinarySetSkipHeader(viewer,PETSC_TRUE));
  CHKERRQ(PetscViewerFileSetMode(viewer,FILE_MODE_READ));
  if (usempiio) CHKERRQ(PetscViewerBinarySetUseMPIIO(viewer,PETSC_TRUE));
  CHKERRQ(PetscViewerFileSetName(viewer,fname));

  CHKERRQ(VecLoad(x,viewer));

  CHKERRQ(PetscViewerBinaryGetSkipHeader(viewer,&isskip));
  if (isskip) CHKERRQ(PetscPrintf(comm,"*** PetscViewer[load] skipping header ***\n"));
  CHKERRQ(PetscViewerBinaryGetUseMPIIO(viewer,&ismpiio));
  if (ismpiio) CHKERRQ(PetscPrintf(comm,"*** PetscViewer[load] using MPI-IO ***\n"));

  CHKERRQ(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode VecFill(Vec x)
{
  PetscInt       i,s,e;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetOwnershipRange(x,&s,&e));
  for (i=s; i<e; i++) {
    CHKERRQ(VecSetValue(x,i,(PetscScalar)test_values[i],INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));
  PetscFunctionReturn(0);
}

PetscErrorCode VecCompare(Vec a,Vec b)
{
  PetscInt       locmin[2],locmax[2];
  PetscReal      min[2],max[2];
  Vec            ref;

  PetscFunctionBeginUser;
  CHKERRQ(VecMin(a,&locmin[0],&min[0]));
  CHKERRQ(VecMax(a,&locmax[0],&max[0]));

  CHKERRQ(VecMin(b,&locmin[1],&min[1]));
  CHKERRQ(VecMax(b,&locmax[1],&max[1]));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"VecCompare\n"));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  min(a)   = %+1.2e [loc %" PetscInt_FMT "]\n",(double)min[0],locmin[0]));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  max(a)   = %+1.2e [loc %" PetscInt_FMT "]\n",(double)max[0],locmax[0]));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  min(b)   = %+1.2e [loc %" PetscInt_FMT "]\n",(double)min[1],locmin[1]));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  max(b)   = %+1.2e [loc %" PetscInt_FMT "]\n",(double)max[1],locmax[1]));

  CHKERRQ(VecDuplicate(a,&ref));
  CHKERRQ(VecCopy(a,ref));
  CHKERRQ(VecAXPY(ref,-1.0,b));
  CHKERRQ(VecMin(ref,&locmin[0],&min[0]));
  if (PetscAbsReal(min[0]) > 1.0e-10) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  ERROR: min(a-b) > 1.0e-10\n"));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  min(a-b) = %+1.10e\n",(double)PetscAbsReal(min[0])));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  min(a-b) < 1.0e-10\n"));
  }
  CHKERRQ(VecDestroy(&ref));
  PetscFunctionReturn(0);
}

PetscErrorCode HeaderlessBinaryRead(const char name[])
{
  int            fdes;
  PetscScalar    buffer[VEC_LEN];
  PetscInt       i;
  PetscMPIInt    rank;
  PetscBool      dataverified = PETSC_TRUE;

  PetscFunctionBeginUser;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank == 0) {
    CHKERRQ(PetscBinaryOpen(name,FILE_MODE_READ,&fdes));
    CHKERRQ(PetscBinaryRead(fdes,buffer,VEC_LEN,NULL,PETSC_SCALAR));
    CHKERRQ(PetscBinaryClose(fdes));

    for (i=0; i<VEC_LEN; i++) {
      PetscScalar v;
      v = PetscAbsScalar(test_values[i]-buffer[i]);
#if defined(PETSC_USE_COMPLEX)
      if ((PetscRealPart(v) > 1.0e-10) || (PetscImaginaryPart(v) > 1.0e-10)) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"ERROR: Difference > 1.0e-10 occurred (delta = (%+1.12e,%+1.12e) [loc %" PetscInt_FMT "])\n",(double)PetscRealPart(buffer[i]),(double)PetscImaginaryPart(buffer[i]),i));
        dataverified = PETSC_FALSE;
      }
#else
      if (PetscRealPart(v) > 1.0e-10) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"ERROR: Difference > 1.0e-10 occurred (delta = %+1.12e [loc %" PetscInt_FMT "])\n",(double)PetscRealPart(buffer[i]),i));
        dataverified = PETSC_FALSE;
      }
#endif
    }
    if (dataverified) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Headerless read of data verified\n"));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TestBinary(void)
{
  Vec            x,y;
  PetscBool      skipheader = PETSC_TRUE;
  PetscBool      usempiio = PETSC_FALSE;

  PetscFunctionBeginUser;
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,VEC_LEN));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecFill(x));
  CHKERRQ(MyVecDump("xH.pbvec",skipheader,usempiio,x));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&y));
  CHKERRQ(VecSetSizes(y,PETSC_DECIDE,VEC_LEN));
  CHKERRQ(VecSetFromOptions(y));

  CHKERRQ(MyVecLoad("xH.pbvec",skipheader,usempiio,y));
  CHKERRQ(VecCompare(x,y));

  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&x));

  CHKERRQ(HeaderlessBinaryRead("xH.pbvec"));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
PetscErrorCode TestBinaryMPIIO(void)
{
  Vec            x,y;
  PetscBool      skipheader = PETSC_TRUE;
  PetscBool      usempiio = PETSC_TRUE;

  PetscFunctionBeginUser;
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,VEC_LEN));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecFill(x));
  CHKERRQ(MyVecDump("xHmpi.pbvec",skipheader,usempiio,x));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&y));
  CHKERRQ(VecSetSizes(y,PETSC_DECIDE,VEC_LEN));
  CHKERRQ(VecSetFromOptions(y));

  CHKERRQ(MyVecLoad("xHmpi.pbvec",skipheader,usempiio,y));
  CHKERRQ(VecCompare(x,y));

  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&x));

  CHKERRQ(HeaderlessBinaryRead("xHmpi.pbvec"));
  PetscFunctionReturn(0);
}
#endif

int main(int argc,char **args)
{
  PetscBool      usempiio = PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-usempiio",&usempiio,NULL));
  if (!usempiio) {
    CHKERRQ(TestBinary());
  } else {
#if defined(PETSC_HAVE_MPIIO)
    CHKERRQ(TestBinaryMPIIO());
#else
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: Executing TestBinaryMPIIO() requires a working MPI-2 implementation\n"));
#endif
  }
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      output_file: output/ex46_1_p1.out

   test:
      suffix: 2
      nsize: 6
      output_file: output/ex46_1_p6.out

   test:
      suffix: 3
      nsize: 12
      output_file: output/ex46_1_p12.out

   testset:
      requires: mpiio
      args: -usempiio
      test:
         suffix: mpiio_1
         output_file: output/ex46_2_p1.out
      test:
         suffix: mpiio_2
         nsize: 6
         output_file: output/ex46_2_p6.out
      test:
         suffix: mpiio_3
         nsize: 12
         output_file: output/ex46_2_p12.out

TEST*/
