static char help[] = "Tests binary viewers.\n\n";

#include <petscsys.h>
#include <petscviewer.h>

static PetscErrorCode TestOpen(PetscFileMode mode,PetscViewer *viewer)
{
  const char     *name;
  PetscBool      skipinfo,skipheader,skipoptions;

  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"binary.dat",mode,viewer));
  PetscCall(PetscViewerBinarySkipInfo(*viewer));
  PetscCall(PetscViewerBinarySetSkipInfo(*viewer,PETSC_FALSE));
  PetscCall(PetscViewerBinarySetSkipHeader(*viewer,PETSC_FALSE));
  PetscCall(PetscViewerBinarySetSkipOptions(*viewer,PETSC_FALSE));
  PetscCall(PetscViewerSetUp(*viewer));
  PetscCall(PetscViewerFileGetName(*viewer,&name));
  PetscCall(PetscViewerFileGetMode(*viewer,&mode));
  PetscCall(PetscViewerBinaryGetSkipInfo(*viewer,&skipinfo));
  PetscCall(PetscViewerBinaryGetSkipHeader(*viewer,&skipheader));
  PetscCall(PetscViewerBinaryGetSkipOptions(*viewer,&skipoptions));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestWrite(PetscViewer viewer)
{
  PetscInt       idata = 42;
  PetscReal      rdata = 42;
  PetscInt       s = PETSC_DETERMINE, t = PETSC_DETERMINE;
  PetscViewer    subviewer;

  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryWrite(viewer,&idata,1,PETSC_INT));
  PetscCall(PetscViewerBinaryWrite(viewer,&rdata,1,PETSC_REAL));

  PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&subviewer));
  if (subviewer) {
    PetscCall(PetscViewerBinaryWrite(subviewer,&idata,1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(subviewer,&rdata,1,PETSC_REAL));
  }
  PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&subviewer));

  PetscCall(PetscViewerBinaryWriteAll(viewer,&idata,1,s,t,PETSC_INT));
  PetscCall(PetscViewerBinaryWriteAll(viewer,&rdata,1,s,t,PETSC_REAL));

  PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&subviewer));
  if (subviewer) {
    PetscCall(PetscViewerBinaryWrite(subviewer,&idata,1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(subviewer,&rdata,1,PETSC_REAL));
  }
  PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&subviewer));

  PetscCall(PetscViewerBinaryWrite(viewer,&idata,1,PETSC_INT));
  PetscCall(PetscViewerBinaryWrite(viewer,&rdata,1,PETSC_REAL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestRead(PetscViewer viewer)
{
  PetscInt       idata = 0;
  PetscReal      rdata = 0;
  PetscInt       s = PETSC_DETERMINE, t = PETSC_DETERMINE;
  PetscViewer    subviewer;
  MPI_Comm       comm = PetscObjectComm((PetscObject)viewer);

  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryRead(viewer,&idata,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer,&rdata,1,NULL,PETSC_REAL));
  PetscCheck(idata == 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
  PetscCheck(rdata == 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);

  PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&subviewer));
  if (subviewer) {
    MPI_Comm subcomm = PetscObjectComm((PetscObject)subviewer);
    PetscCall(PetscViewerBinaryRead(subviewer,&idata,1,NULL,PETSC_INT));
    PetscCall(PetscViewerBinaryRead(subviewer,&rdata,1,NULL,PETSC_REAL));
    PetscCheck(idata == 42,subcomm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
    PetscCheck(rdata == 42,subcomm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);
  }
  PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&subviewer));

  PetscCall(PetscViewerBinaryReadAll(viewer,&idata,1,s,t,PETSC_INT));
  PetscCall(PetscViewerBinaryReadAll(viewer,&rdata,1,s,t,PETSC_REAL));
  PetscCheck(idata == 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
  PetscCheck(rdata == 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);

  PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&subviewer));
  if (subviewer) {
    MPI_Comm subcomm = PetscObjectComm((PetscObject)subviewer);
    PetscCall(PetscViewerBinaryRead(subviewer,&idata,1,NULL,PETSC_INT));
    PetscCall(PetscViewerBinaryRead(subviewer,&rdata,1,NULL,PETSC_REAL));
    PetscCheck(idata == 42,subcomm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
    PetscCheck(rdata == 42,subcomm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);
  }
  PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&subviewer));

  PetscCall(PetscViewerBinaryRead(viewer,&idata,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer,&rdata,1,NULL,PETSC_REAL));
  PetscCheck(idata == 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
  PetscCheck(rdata == 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestEOF(PetscViewer viewer)
{
  char           data;
  PetscInt       count = PETSC_MAX_INT;
  MPI_Comm       comm = PetscObjectComm((PetscObject)viewer);

  PetscFunctionBegin;
  PetscCall(PetscViewerRead(viewer,&data,1,&count,PETSC_CHAR));
  PetscCheck(!count,comm,PETSC_ERR_FILE_UNEXPECTED,"Expected EOF");
  PetscFunctionReturn(0);
}

static PetscErrorCode TestClose(PetscViewer *viewer)
{
  PetscFileMode  mode;

  PetscFunctionBegin;
  PetscCall(PetscViewerFileGetMode(*viewer,&mode));
  if (mode == FILE_MODE_READ) PetscCall(TestEOF(*viewer));
  PetscCall(PetscViewerDestroy(viewer));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscViewer    viewer;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,NULL,help));

  PetscCall(TestOpen(FILE_MODE_WRITE,&viewer));
  PetscCall(TestWrite(viewer));
  PetscCall(TestClose(&viewer));

  PetscCall(TestOpen(FILE_MODE_READ,&viewer));
  PetscCall(TestRead(viewer));
  PetscCall(TestClose(&viewer));

  PetscCall(TestOpen(FILE_MODE_APPEND,&viewer));
  PetscCall(TestWrite(viewer));
  PetscCall(TestClose(&viewer));

  PetscCall(TestOpen(FILE_MODE_READ,&viewer));
  PetscCall(TestRead(viewer));
  PetscCall(TestRead(viewer));
  PetscCall(TestClose(&viewer));

  PetscCall(TestOpen(FILE_MODE_APPEND,&viewer));
  PetscCall(TestWrite(viewer));
  PetscCall(TestClose(&viewer));

  PetscCall(TestOpen(FILE_MODE_READ,&viewer));
  PetscCall(TestRead(viewer));
  PetscCall(TestRead(viewer));
  PetscCall(TestRead(viewer));
  PetscCall(TestClose(&viewer));

  PetscCall(TestOpen(FILE_MODE_WRITE,&viewer));
  PetscCall(TestWrite(viewer));
  PetscCall(TestClose(&viewer));

  PetscCall(TestOpen(FILE_MODE_READ,&viewer));
  PetscCall(TestRead(viewer));
  PetscCall(TestClose(&viewer));

  PetscCall(TestOpen(FILE_MODE_WRITE,&viewer));
  PetscCall(TestClose(&viewer));
  PetscCall(TestOpen(FILE_MODE_READ,&viewer));
  PetscCall(TestClose(&viewer));
  PetscCall(TestOpen(FILE_MODE_APPEND,&viewer));
  PetscCall(TestClose(&viewer));
  PetscCall(TestOpen(FILE_MODE_READ,&viewer));
  PetscCall(TestClose(&viewer));

  {
    FILE        *info;
    PetscMPIInt rank;

    PetscCall(TestOpen(FILE_MODE_WRITE,&viewer));
    PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_BINARY_MATLAB));
    PetscCall(PetscViewerBinaryGetInfoPointer(viewer,&info));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank));
    PetscCheck(rank != 0 || info,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing info pointer");
    PetscCall(TestClose(&viewer));
  }

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
     nsize: {{1 2 3}separate_output}
     args: -viewer_view
     test:
       suffix: stdio
       args: -viewer_binary_mpiio 0
     test:
       requires: mpiio
       suffix: mpiio
       args: -viewer_binary_mpiio 1

TEST*/
