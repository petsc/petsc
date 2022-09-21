static const char help[] = "Tests PetscDeviceContextView().\n\n";

#include "petscdevicetestcommon.h"
#include <petscviewer.h>

static PetscErrorCode TestView(PetscDeviceContext dctx)
{
  PetscViewer viewer;

  PetscFunctionBegin;
  /* test stdout world */
  PetscCall(PetscDeviceContextView(dctx, NULL));

  /* test creating our own viewer */
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  PetscCall(PetscDeviceContextView(dctx, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  MPI_Comm           comm;
  PetscDeviceContext dctx, dup;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(TestView(dctx));

  PetscCall(PetscDeviceContextDuplicate(dctx, &dup));
  PetscCall(TestView(dup));
  PetscCall(PetscDeviceContextDestroy(&dup));

  PetscCall(PetscPrintf(comm, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

 build:
   requires: defined(PETSC_HAVE_CXX)

 testset:
   args: -root_device_context_stream_type \
     {{global_blocking default_blocking global_nonblocking}separate output}
   filter: grep -ve "ex6 on a" -ve "\[0\] "
   test:
     requires: !device
     suffix: host_no_device
   test:
     requires: device
     args: -root_device_context_device_type host
     suffix: host_with_device
   test:
     requires: cuda
     args: -root_device_context_device_type cuda
     suffix: cuda
   test:
     requires: hip
     args: -root_device_context_device_type hip
     suffix: hip
   test:
     requires: sycl
     args: -root_device_context_device_type sycl
     suffix: sycl

TEST*/
