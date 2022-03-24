static char help[] = "Tests binary viewers.\n\n";

#include <petscsys.h>
#include <petscviewer.h>

static PetscErrorCode TestOpen(PetscFileMode mode,PetscViewer *viewer)
{
  const char     *name;
  PetscBool      skipinfo,skipheader,skipoptions;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"binary.dat",mode,viewer));
  CHKERRQ(PetscViewerBinarySkipInfo(*viewer));
  CHKERRQ(PetscViewerBinarySetSkipInfo(*viewer,PETSC_FALSE));
  CHKERRQ(PetscViewerBinarySetSkipHeader(*viewer,PETSC_FALSE));
  CHKERRQ(PetscViewerBinarySetSkipOptions(*viewer,PETSC_FALSE));
  CHKERRQ(PetscViewerSetUp(*viewer));
  CHKERRQ(PetscViewerFileGetName(*viewer,&name));
  CHKERRQ(PetscViewerFileGetMode(*viewer,&mode));
  CHKERRQ(PetscViewerBinaryGetSkipInfo(*viewer,&skipinfo));
  CHKERRQ(PetscViewerBinaryGetSkipHeader(*viewer,&skipheader));
  CHKERRQ(PetscViewerBinaryGetSkipOptions(*viewer,&skipoptions));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestWrite(PetscViewer viewer)
{
  PetscInt       idata = 42;
  PetscReal      rdata = 42;
  PetscInt       s = PETSC_DETERMINE, t = PETSC_DETERMINE;
  PetscViewer    subviewer;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerBinaryWrite(viewer,&idata,1,PETSC_INT));
  CHKERRQ(PetscViewerBinaryWrite(viewer,&rdata,1,PETSC_REAL));

  CHKERRQ(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&subviewer));
  if (subviewer) {
    CHKERRQ(PetscViewerBinaryWrite(subviewer,&idata,1,PETSC_INT));
    CHKERRQ(PetscViewerBinaryWrite(subviewer,&rdata,1,PETSC_REAL));
  }
  CHKERRQ(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&subviewer));

  CHKERRQ(PetscViewerBinaryWriteAll(viewer,&idata,1,s,t,PETSC_INT));
  CHKERRQ(PetscViewerBinaryWriteAll(viewer,&rdata,1,s,t,PETSC_REAL));

  CHKERRQ(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&subviewer));
  if (subviewer) {
    CHKERRQ(PetscViewerBinaryWrite(subviewer,&idata,1,PETSC_INT));
    CHKERRQ(PetscViewerBinaryWrite(subviewer,&rdata,1,PETSC_REAL));
  }
  CHKERRQ(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&subviewer));

  CHKERRQ(PetscViewerBinaryWrite(viewer,&idata,1,PETSC_INT));
  CHKERRQ(PetscViewerBinaryWrite(viewer,&rdata,1,PETSC_REAL));
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
  CHKERRQ(PetscViewerBinaryRead(viewer,&idata,1,NULL,PETSC_INT));
  CHKERRQ(PetscViewerBinaryRead(viewer,&rdata,1,NULL,PETSC_REAL));
  PetscCheckFalse(idata != 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
  PetscCheckFalse(rdata != 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);

  CHKERRQ(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&subviewer));
  if (subviewer) {
    MPI_Comm subcomm = PetscObjectComm((PetscObject)subviewer);
    CHKERRQ(PetscViewerBinaryRead(subviewer,&idata,1,NULL,PETSC_INT));
    CHKERRQ(PetscViewerBinaryRead(subviewer,&rdata,1,NULL,PETSC_REAL));
    PetscCheckFalse(idata != 42,subcomm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
    PetscCheckFalse(rdata != 42,subcomm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);
  }
  CHKERRQ(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&subviewer));

  CHKERRQ(PetscViewerBinaryReadAll(viewer,&idata,1,s,t,PETSC_INT));
  CHKERRQ(PetscViewerBinaryReadAll(viewer,&rdata,1,s,t,PETSC_REAL));
  PetscCheckFalse(idata != 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
  PetscCheckFalse(rdata != 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);

  CHKERRQ(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&subviewer));
  if (subviewer) {
    MPI_Comm subcomm = PetscObjectComm((PetscObject)subviewer);
    CHKERRQ(PetscViewerBinaryRead(subviewer,&idata,1,NULL,PETSC_INT));
    CHKERRQ(PetscViewerBinaryRead(subviewer,&rdata,1,NULL,PETSC_REAL));
    PetscCheckFalse(idata != 42,subcomm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
    PetscCheckFalse(rdata != 42,subcomm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);
  }
  CHKERRQ(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&subviewer));

  CHKERRQ(PetscViewerBinaryRead(viewer,&idata,1,NULL,PETSC_INT));
  CHKERRQ(PetscViewerBinaryRead(viewer,&rdata,1,NULL,PETSC_REAL));
  PetscCheckFalse(idata != 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
  PetscCheckFalse(rdata != 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestEOF(PetscViewer viewer)
{
  char           data;
  PetscInt       count = PETSC_MAX_INT;
  MPI_Comm       comm = PetscObjectComm((PetscObject)viewer);

  PetscFunctionBegin;
  CHKERRQ(PetscViewerRead(viewer,&data,1,&count,PETSC_CHAR));
  PetscCheck(!count,comm,PETSC_ERR_FILE_UNEXPECTED,"Expected EOF");
  PetscFunctionReturn(0);
}

static PetscErrorCode TestClose(PetscViewer *viewer)
{
  PetscFileMode  mode;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerFileGetMode(*viewer,&mode));
  if (mode == FILE_MODE_READ) CHKERRQ(TestEOF(*viewer));
  CHKERRQ(PetscViewerDestroy(viewer));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscViewer    viewer;

  CHKERRQ(PetscInitialize(&argc,&args,NULL,help));

  CHKERRQ(TestOpen(FILE_MODE_WRITE,&viewer));
  CHKERRQ(TestWrite(viewer));
  CHKERRQ(TestClose(&viewer));

  CHKERRQ(TestOpen(FILE_MODE_READ,&viewer));
  CHKERRQ(TestRead(viewer));
  CHKERRQ(TestClose(&viewer));

  CHKERRQ(TestOpen(FILE_MODE_APPEND,&viewer));
  CHKERRQ(TestWrite(viewer));
  CHKERRQ(TestClose(&viewer));

  CHKERRQ(TestOpen(FILE_MODE_READ,&viewer));
  CHKERRQ(TestRead(viewer));
  CHKERRQ(TestRead(viewer));
  CHKERRQ(TestClose(&viewer));

  CHKERRQ(TestOpen(FILE_MODE_APPEND,&viewer));
  CHKERRQ(TestWrite(viewer));
  CHKERRQ(TestClose(&viewer));

  CHKERRQ(TestOpen(FILE_MODE_READ,&viewer));
  CHKERRQ(TestRead(viewer));
  CHKERRQ(TestRead(viewer));
  CHKERRQ(TestRead(viewer));
  CHKERRQ(TestClose(&viewer));

  CHKERRQ(TestOpen(FILE_MODE_WRITE,&viewer));
  CHKERRQ(TestWrite(viewer));
  CHKERRQ(TestClose(&viewer));

  CHKERRQ(TestOpen(FILE_MODE_READ,&viewer));
  CHKERRQ(TestRead(viewer));
  CHKERRQ(TestClose(&viewer));

  CHKERRQ(TestOpen(FILE_MODE_WRITE,&viewer));
  CHKERRQ(TestClose(&viewer));
  CHKERRQ(TestOpen(FILE_MODE_READ,&viewer));
  CHKERRQ(TestClose(&viewer));
  CHKERRQ(TestOpen(FILE_MODE_APPEND,&viewer));
  CHKERRQ(TestClose(&viewer));
  CHKERRQ(TestOpen(FILE_MODE_READ,&viewer));
  CHKERRQ(TestClose(&viewer));

  {
    FILE        *info;
    PetscMPIInt rank;

    CHKERRQ(TestOpen(FILE_MODE_WRITE,&viewer));
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_BINARY_MATLAB));
    CHKERRQ(PetscViewerBinaryGetInfoPointer(viewer,&info));
    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank));
    PetscCheckFalse(rank == 0 && !info,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing info pointer");
    CHKERRQ(TestClose(&viewer));
  }

  CHKERRQ(PetscFinalize());
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
