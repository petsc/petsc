static char help[] = "Tests binary viewers.\n\n";

#include <petscsys.h>
#include <petscviewer.h>

static PetscErrorCode TestOpen(PetscFileMode mode,PetscViewer *viewer)
{
  const char     *name;
  PetscBool      skipinfo,skipheader,skipoptions;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"binary.dat",mode,viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinarySkipInfo(*viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipInfo(*viewer,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipHeader(*viewer,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipOptions(*viewer,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerSetUp(*viewer);CHKERRQ(ierr);
  ierr = PetscViewerFileGetName(*viewer,&name);CHKERRQ(ierr);
  ierr = PetscViewerFileGetMode(*viewer,&mode);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetSkipInfo(*viewer,&skipinfo);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetSkipHeader(*viewer,&skipheader);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetSkipOptions(*viewer,&skipoptions);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestWrite(PetscViewer viewer)
{
  PetscInt       idata = 42;
  PetscReal      rdata = 42;
  PetscInt       s = PETSC_DETERMINE, t = PETSC_DETERMINE;
  PetscViewer    subviewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryWrite(viewer,&idata,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&rdata,1,PETSC_REAL);CHKERRQ(ierr);

  ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&subviewer);CHKERRQ(ierr);
  if (subviewer) {
    ierr = PetscViewerBinaryWrite(subviewer,&idata,1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(subviewer,&rdata,1,PETSC_REAL);CHKERRQ(ierr);
  }
  ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&subviewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryWriteAll(viewer,&idata,1,s,t,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWriteAll(viewer,&rdata,1,s,t,PETSC_REAL);CHKERRQ(ierr);

  ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&subviewer);CHKERRQ(ierr);
  if (subviewer) {
    ierr = PetscViewerBinaryWrite(subviewer,&idata,1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(subviewer,&rdata,1,PETSC_REAL);CHKERRQ(ierr);
  }
  ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&subviewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryWrite(viewer,&idata,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&rdata,1,PETSC_REAL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestRead(PetscViewer viewer)
{
  PetscInt       idata = 0;
  PetscReal      rdata = 0;
  PetscInt       s = PETSC_DETERMINE, t = PETSC_DETERMINE;
  PetscViewer    subviewer;
  MPI_Comm       comm = PetscObjectComm((PetscObject)viewer);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryRead(viewer,&idata,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&rdata,1,NULL,PETSC_REAL);CHKERRQ(ierr);
  PetscAssertFalse(idata != 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
  PetscAssertFalse(rdata != 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);

  ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&subviewer);CHKERRQ(ierr);
  if (subviewer) {
    MPI_Comm subcomm = PetscObjectComm((PetscObject)subviewer);
    ierr = PetscViewerBinaryRead(subviewer,&idata,1,NULL,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(subviewer,&rdata,1,NULL,PETSC_REAL);CHKERRQ(ierr);
    PetscAssertFalse(idata != 42,subcomm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
    PetscAssertFalse(rdata != 42,subcomm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);
  }
  ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&subviewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryReadAll(viewer,&idata,1,s,t,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryReadAll(viewer,&rdata,1,s,t,PETSC_REAL);CHKERRQ(ierr);
  PetscAssertFalse(idata != 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
  PetscAssertFalse(rdata != 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);

  ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&subviewer);CHKERRQ(ierr);
  if (subviewer) {
    MPI_Comm subcomm = PetscObjectComm((PetscObject)subviewer);
    ierr = PetscViewerBinaryRead(subviewer,&idata,1,NULL,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(subviewer,&rdata,1,NULL,PETSC_REAL);CHKERRQ(ierr);
    PetscAssertFalse(idata != 42,subcomm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
    PetscAssertFalse(rdata != 42,subcomm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);
  }
  ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&subviewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryRead(viewer,&idata,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&rdata,1,NULL,PETSC_REAL);CHKERRQ(ierr);
  PetscAssertFalse(idata != 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected idata=%" PetscInt_FMT,idata);
  PetscAssertFalse(rdata != 42,comm,PETSC_ERR_FILE_UNEXPECTED,"Unexpected rdata=%g",(double)rdata);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestEOF(PetscViewer viewer)
{
  char           data;
  PetscInt       count = PETSC_MAX_INT;
  MPI_Comm       comm = PetscObjectComm((PetscObject)viewer);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerRead(viewer,&data,1,&count,PETSC_CHAR);CHKERRQ(ierr);
  PetscAssertFalse(count,comm,PETSC_ERR_FILE_UNEXPECTED,"Expected EOF");
  PetscFunctionReturn(0);
}

static PetscErrorCode TestClose(PetscViewer *viewer)
{
  PetscFileMode  mode;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerFileGetMode(*viewer,&mode);CHKERRQ(ierr);
  if (mode == FILE_MODE_READ) {ierr = TestEOF(*viewer);CHKERRQ(ierr);}
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;

  ierr = TestOpen(FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = TestWrite(viewer);CHKERRQ(ierr);
  ierr = TestClose(&viewer);CHKERRQ(ierr);

  ierr = TestOpen(FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = TestRead(viewer);CHKERRQ(ierr);
  ierr = TestClose(&viewer);CHKERRQ(ierr);

  ierr = TestOpen(FILE_MODE_APPEND,&viewer);CHKERRQ(ierr);
  ierr = TestWrite(viewer);CHKERRQ(ierr);
  ierr = TestClose(&viewer);CHKERRQ(ierr);

  ierr = TestOpen(FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = TestRead(viewer);CHKERRQ(ierr);
  ierr = TestRead(viewer);CHKERRQ(ierr);
  ierr = TestClose(&viewer);CHKERRQ(ierr);

  ierr = TestOpen(FILE_MODE_APPEND,&viewer);CHKERRQ(ierr);
  ierr = TestWrite(viewer);CHKERRQ(ierr);
  ierr = TestClose(&viewer);CHKERRQ(ierr);

  ierr = TestOpen(FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = TestRead(viewer);CHKERRQ(ierr);
  ierr = TestRead(viewer);CHKERRQ(ierr);
  ierr = TestRead(viewer);CHKERRQ(ierr);
  ierr = TestClose(&viewer);CHKERRQ(ierr);

  ierr = TestOpen(FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = TestWrite(viewer);CHKERRQ(ierr);
  ierr = TestClose(&viewer);CHKERRQ(ierr);

  ierr = TestOpen(FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = TestRead(viewer);CHKERRQ(ierr);
  ierr = TestClose(&viewer);CHKERRQ(ierr);

  ierr = TestOpen(FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = TestClose(&viewer);CHKERRQ(ierr);
  ierr = TestOpen(FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = TestClose(&viewer);CHKERRQ(ierr);
  ierr = TestOpen(FILE_MODE_APPEND,&viewer);CHKERRQ(ierr);
  ierr = TestClose(&viewer);CHKERRQ(ierr);
  ierr = TestOpen(FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = TestClose(&viewer);CHKERRQ(ierr);

  {
    FILE        *info;
    PetscMPIInt rank;

    ierr = TestOpen(FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetInfoPointer(viewer,&info);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank);CHKERRMPI(ierr);
    PetscAssertFalse(rank == 0 && !info,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing info pointer");
    ierr = TestClose(&viewer);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();
  return ierr;
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
