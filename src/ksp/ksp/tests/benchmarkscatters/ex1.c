
static char help[] = "Used to benchmark changes to the PETSc VecScatter routines\n\n";
#include <petscksp.h>
extern PetscErrorCode PetscLogView_VecScatter(PetscViewer);

int main(int argc, char **args)
{
  KSP         ksp;
  Mat         A;
  Vec         x, b;
  PetscViewer fd;
  char        file[PETSC_MAX_PATH_LEN];
  PetscBool   flg, preload = PETSC_TRUE;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));

  PetscPreLoadBegin(preload, "Load system");

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSetFromOptions(b));
  PetscCall(VecSet(b, 1.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSetUpOnBlocks(ksp));

  PetscPreLoadStage("KSPSolve");
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));
  PetscPreLoadEnd();
  PetscCall(PetscLogView_VecScatter(PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscFinalize());
  return 0;
}

#include <petsctime.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/kspimpl.h>
#include <petscmachineinfo.h>
#include <petscconfiginfo.h>
/*
   This is a special log viewer that prints out detailed information only for the VecScatter routines
*/
typedef enum {
  COUNT,
  TIME,
  NUMMESS,
  MESSLEN,
  REDUCT,
  FLOPS
} Stats;
PetscErrorCode PetscLogView_VecScatter(PetscViewer viewer)
{
  MPI_Comm            comm      = PetscObjectComm((PetscObject)viewer);
  PetscEventPerfInfo *eventInfo = NULL;
  PetscLogDouble      locTotalTime, stats[6], maxstats[6], minstats[6], sumstats[6], avetime, ksptime;
  PetscStageLog       stageLog;
  const int           stage = 2;
  int                 event, events[] = {VEC_ScatterBegin, VEC_ScatterEnd};
  PetscMPIInt         rank, size;
  PetscInt            i;
  char                arch[128], hostname[128], username[128], pname[PETSC_MAX_PATH_LEN], date[128], version[256];

  PetscFunctionBegin;
  PetscCall(PetscTime(&locTotalTime));
  locTotalTime -= petsc_BaseTime;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscViewerASCIIPrintf(viewer, "numProcs   = %d\n", size));

  PetscCall(PetscGetArchType(arch, sizeof(arch)));
  PetscCall(PetscGetHostName(hostname, sizeof(hostname)));
  PetscCall(PetscGetUserName(username, sizeof(username)));
  PetscCall(PetscGetProgramName(pname, sizeof(pname)));
  PetscCall(PetscGetDate(date, sizeof(date)));
  PetscCall(PetscGetVersion(version, sizeof(version)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s on a %s named %s with %d processors, by %s %s\n", pname, arch, hostname, size, username, date));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Using %s\n", version));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Configure options: %s", petscconfigureoptions));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s", petscmachineinfo));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s", petsccompilerinfo));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s", petsccompilerflagsinfo));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s", petsclinkerinfo));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s\n", PETSC_MPICC_SHOW));
  PetscCall(PetscOptionsView(NULL, viewer));
#if defined(PETSC_HAVE_HWLOC)
  PetscCall(PetscProcessPlacementView(viewer));
#endif
  PetscCall(PetscViewerASCIIPrintf(viewer, "----------------------------------------------------\n"));

  PetscCall(PetscViewerASCIIPrintf(viewer, "                Time     Min to Max Range   Proportion of KSP\n"));

  eventInfo = stageLog->stageInfo[stage].eventLog->eventInfo;
  PetscCall(MPIU_Allreduce(&eventInfo[KSP_Solve].time, &ksptime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_WORLD));
  ksptime = ksptime / size;

  for (i = 0; i < (int)(sizeof(events) / sizeof(int)); i++) {
    event          = events[i];
    stats[COUNT]   = eventInfo[event].count;
    stats[TIME]    = eventInfo[event].time;
    stats[NUMMESS] = eventInfo[event].numMessages;
    stats[MESSLEN] = eventInfo[event].messageLength;
    stats[REDUCT]  = eventInfo[event].numReductions;
    stats[FLOPS]   = eventInfo[event].flops;
    PetscCall(MPIU_Allreduce(stats, maxstats, 6, MPIU_PETSCLOGDOUBLE, MPI_MAX, PETSC_COMM_WORLD));
    PetscCall(MPIU_Allreduce(stats, minstats, 6, MPIU_PETSCLOGDOUBLE, MPI_MIN, PETSC_COMM_WORLD));
    PetscCall(MPIU_Allreduce(stats, sumstats, 6, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_WORLD));

    avetime = sumstats[1] / size;
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s %4.2e   -%5.1f %% %5.1f %%   %4.2e %%\n", stageLog->eventLog->eventInfo[event].name, avetime, 100. * (avetime - minstats[1]) / avetime, 100. * (maxstats[1] - avetime) / avetime, 100. * avetime / ksptime));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
     requires: defined(PETSC_USE_LOG)

   test:
     TODO: need to implement

TEST*/
