
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  if (skippheader) { ierr = PetscViewerBinarySetSkipHeader(viewer,PETSC_TRUE);CHKERRQ(ierr); }
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  if (usempiio) { ierr = PetscViewerBinarySetUseMPIIO(viewer,PETSC_TRUE);CHKERRQ(ierr); }
  ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);

  ierr = VecView(x,viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetUseMPIIO(viewer,&ismpiio);CHKERRQ(ierr);
  if (ismpiio) { ierr = PetscPrintf(comm,"*** PetscViewer[write] using MPI-IO ***\n");CHKERRQ(ierr); }
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&isskip);CHKERRQ(ierr);
  if (isskip) { ierr = PetscPrintf(comm,"*** PetscViewer[write] skipping header ***\n");CHKERRQ(ierr); }

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MyVecLoad(const char fname[],PetscBool skippheader,PetscBool usempiio,Vec x)
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscBool      ismpiio,isskip;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  if (skippheader) { ierr = PetscViewerBinarySetSkipHeader(viewer,PETSC_TRUE);CHKERRQ(ierr); }
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_READ);CHKERRQ(ierr);
  if (usempiio) { ierr = PetscViewerBinarySetUseMPIIO(viewer,PETSC_TRUE);CHKERRQ(ierr); }
  ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);

  ierr = VecLoad(x,viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetSkipHeader(viewer,&isskip);CHKERRQ(ierr);
  if (isskip) { ierr = PetscPrintf(comm,"*** PetscViewer[load] skipping header ***\n");CHKERRQ(ierr); }
  ierr = PetscViewerBinaryGetUseMPIIO(viewer,&ismpiio);CHKERRQ(ierr);
  if (ismpiio) { ierr = PetscPrintf(comm,"*** PetscViewer[load] using MPI-IO ***\n");CHKERRQ(ierr); }

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecFill(Vec x)
{
  PetscErrorCode ierr;
  PetscInt       i,s,e;

  PetscFunctionBeginUser;
  ierr = VecGetOwnershipRange(x,&s,&e);CHKERRQ(ierr);
  for (i=s; i<e; i++) {
    ierr = VecSetValue(x,i,(PetscScalar)test_values[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecCompare(Vec a,Vec b)
{
  PetscInt       locmin[2],locmax[2];
  PetscReal      min[2],max[2];
  Vec            ref;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecMin(a,&locmin[0],&min[0]);CHKERRQ(ierr);
  ierr = VecMax(a,&locmax[0],&max[0]);CHKERRQ(ierr);

  ierr = VecMin(b,&locmin[1],&min[1]);CHKERRQ(ierr);
  ierr = VecMax(b,&locmax[1],&max[1]);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecCompare\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  min(a)   = %+1.2e [loc %D]\n",(double)min[0],locmin[0]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  max(a)   = %+1.2e [loc %D]\n",(double)max[0],locmax[0]);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"  min(b)   = %+1.2e [loc %D]\n",(double)min[1],locmin[1]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  max(b)   = %+1.2e [loc %D]\n",(double)max[1],locmax[1]);CHKERRQ(ierr);

  ierr = VecDuplicate(a,&ref);CHKERRQ(ierr);
  ierr = VecCopy(a,ref);CHKERRQ(ierr);
  ierr = VecAXPY(ref,-1.0,b);CHKERRQ(ierr);
  ierr = VecMin(ref,&locmin[0],&min[0]);CHKERRQ(ierr);
  if (PetscAbsReal(min[0]) > 1.0e-10) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  ERROR: min(a-b) > 1.0e-10\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  min(a-b) = %+1.10e\n",(double)PetscAbsReal(min[0]));CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  min(a-b) < 1.0e-10\n");CHKERRQ(ierr);
  }
  ierr = VecDestroy(&ref);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode HeaderlessBinaryRead(const char name[])
{
  PetscErrorCode ierr;
  int            fdes;
  PetscScalar    buffer[VEC_LEN];
  PetscInt       i;
  PetscMPIInt    rank;
  PetscBool      dataverified = PETSC_TRUE;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  if (!rank) {
    ierr = PetscBinaryOpen(name,FILE_MODE_READ,&fdes);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fdes,buffer,VEC_LEN,NULL,PETSC_SCALAR);CHKERRQ(ierr);
    ierr = PetscBinaryClose(fdes);CHKERRQ(ierr);

    for (i=0; i<VEC_LEN; i++) {
      PetscScalar v;
      v = PetscAbsScalar(test_values[i]-buffer[i]);
#if defined(PETSC_USE_COMPLEX)
      if ((PetscRealPart(v) > 1.0e-10) || (PetscImaginaryPart(v) > 1.0e-10)) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"ERROR: Difference > 1.0e-10 occurred (delta = (%+1.12e,%+1.12e) [loc %D])\n",(double)PetscRealPart(buffer[i]),(double)PetscImaginaryPart(buffer[i]),i);CHKERRQ(ierr);
        dataverified = PETSC_FALSE;
      }
#else
      if (PetscRealPart(v) > 1.0e-10) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"ERROR: Difference > 1.0e-10 occurred (delta = %+1.12e [loc %D])\n",(double)PetscRealPart(buffer[i]),i);CHKERRQ(ierr);
        dataverified = PETSC_FALSE;
      }
#endif
    }
    if (dataverified) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Headerless read of data verified\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TestBinary(void)
{
  PetscErrorCode ierr;
  Vec            x,y;
  PetscBool      skipheader = PETSC_TRUE;
  PetscBool      usempiio = PETSC_FALSE;

  PetscFunctionBeginUser;
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,VEC_LEN);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecFill(x);CHKERRQ(ierr);
  ierr = MyVecDump("xH.pbvec",skipheader,usempiio,x);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,VEC_LEN);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);

  ierr = MyVecLoad("xH.pbvec",skipheader,usempiio,y);CHKERRQ(ierr);
  ierr = VecCompare(x,y);CHKERRQ(ierr);

  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = HeaderlessBinaryRead("xH.pbvec");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
PetscErrorCode TestBinaryMPIIO(void)
{
  PetscErrorCode ierr;
  Vec            x,y;
  PetscBool      skipheader = PETSC_TRUE;
  PetscBool      usempiio = PETSC_TRUE;

  PetscFunctionBeginUser;
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,VEC_LEN);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecFill(x);CHKERRQ(ierr);
  ierr = MyVecDump("xHmpi.pbvec",skipheader,usempiio,x);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,VEC_LEN);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);

  ierr = MyVecLoad("xHmpi.pbvec",skipheader,usempiio,y);CHKERRQ(ierr);
  ierr = VecCompare(x,y);CHKERRQ(ierr);

  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = HeaderlessBinaryRead("xHmpi.pbvec");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscBool      usempiio = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL,NULL,"-usempiio",&usempiio,NULL);CHKERRQ(ierr);
  if (!usempiio) {
    ierr = TestBinary();CHKERRQ(ierr);
  } else {
#if defined(PETSC_HAVE_MPIIO)
    ierr = TestBinaryMPIIO();CHKERRQ(ierr);
#else
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Executing TestBinaryMPIIO() requires a working MPI-2 implementation\n");CHKERRQ(ierr);
#endif
  }
  ierr = PetscFinalize();
  return ierr;
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
