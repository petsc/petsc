
static char help[] = "Used to benchmark changes to the PETSc VecScatter routines\n\n";
#include <petscksp.h>
extern PetscErrorCode  PetscLogView_VecScatter(PetscViewer);

int main(int argc,char **args)
{
  KSP            ksp;
  Mat            A;
  Vec            x,b;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscBool      flg,preload = PETSC_TRUE;

  PetscInitialize(&argc,&args,(char*)0,help);
  CHKERRQ(PetscLogDefaultBegin());
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));

  PetscPreLoadBegin(preload,"Load system");

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  CHKERRQ(MatCreateVecs(A,&x,&b));
  CHKERRQ(VecSetFromOptions(b));
  CHKERRQ(VecSet(b,1.0));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(KSPSetUpOnBlocks(ksp));

  PetscPreLoadStage("KSPSolve");
  CHKERRQ(KSPSolve(ksp,b,x));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(KSPDestroy(&ksp));
  PetscPreLoadEnd();
  CHKERRQ(PetscLogView_VecScatter(PETSC_VIEWER_STDOUT_WORLD));

  ierr = PetscFinalize();
  return ierr;
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
typedef enum { COUNT,TIME,NUMMESS,MESSLEN,REDUCT,FLOPS} Stats;
PetscErrorCode  PetscLogView_VecScatter(PetscViewer viewer)
{
  MPI_Comm           comm       = PetscObjectComm((PetscObject) viewer);
  PetscEventPerfInfo *eventInfo = NULL;
  PetscLogDouble     locTotalTime,stats[6],maxstats[6],minstats[6],sumstats[6],avetime,ksptime;
  PetscStageLog      stageLog;
  const int          stage = 2;
  int                event,events[] = {VEC_ScatterBegin,VEC_ScatterEnd};
  PetscMPIInt        rank,size;
  PetscErrorCode     ierr;
  PetscInt           i;
  char               arch[128],hostname[128],username[128],pname[PETSC_MAX_PATH_LEN],date[128],version[256];

  PetscFunctionBegin;
  PetscTime(&locTotalTime);  locTotalTime -= petsc_BaseTime;
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscLogGetStageLog(&stageLog));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"numProcs   = %d\n",size));

  CHKERRQ(PetscGetArchType(arch,sizeof(arch)));
  CHKERRQ(PetscGetHostName(hostname,sizeof(hostname)));
  CHKERRQ(PetscGetUserName(username,sizeof(username)));
  CHKERRQ(PetscGetProgramName(pname,sizeof(pname)));
  CHKERRQ(PetscGetDate(date,sizeof(date)));
  CHKERRQ(PetscGetVersion(version,sizeof(version)));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s on a %s named %s with %d processors, by %s %s\n", pname, arch, hostname, size, username, date));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Using %s\n", version));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Configure options: %s",petscconfigureoptions));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "%s", petscmachineinfo));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "%s", petsccompilerinfo));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "%s", petsccompilerflagsinfo));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "%s", petsclinkerinfo));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "%s\n", PETSC_MPICC_SHOW));
  CHKERRQ(PetscOptionsView(NULL,viewer));
#if defined(PETSC_HAVE_HWLOC)
  CHKERRQ(PetscProcessPlacementView(viewer));
#endif
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "----------------------------------------------------\n"));

  CHKERRQ(PetscViewerASCIIPrintf(viewer,"                Time     Min to Max Range   Proportion of KSP\n"));

  eventInfo = stageLog->stageInfo[stage].eventLog->eventInfo;
  CHKERRMPI(MPI_Allreduce(&eventInfo[KSP_Solve].time,&ksptime,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  ksptime = ksptime/size;

  for (i=0; i<(int)(sizeof(events)/sizeof(int)); i++) {
    event = events[i];
    stats[COUNT]   = eventInfo[event].count;
    stats[TIME]    = eventInfo[event].time;
    stats[NUMMESS] = eventInfo[event].numMessages;
    stats[MESSLEN] = eventInfo[event].messageLength;
    stats[REDUCT]  = eventInfo[event].numReductions;
    stats[FLOPS]   = eventInfo[event].flops;
    CHKERRMPI(MPI_Allreduce(stats,maxstats,6,MPIU_PETSCLOGDOUBLE,MPI_MAX,PETSC_COMM_WORLD));
    CHKERRMPI(MPI_Allreduce(stats,minstats,6,MPIU_PETSCLOGDOUBLE,MPI_MIN,PETSC_COMM_WORLD));
    CHKERRMPI(MPI_Allreduce(stats,sumstats,6,MPIU_PETSCLOGDOUBLE,MPI_SUM,PETSC_COMM_WORLD));

    avetime  = sumstats[1]/size;
    ierr = PetscViewerASCIIPrintf(viewer,"%s %4.2e   -%5.1f %% %5.1f %%   %4.2e %%\n",stageLog->eventLog->eventInfo[event].name,
                                  avetime,100.*(avetime-minstats[1])/avetime,100.*(maxstats[1]-avetime)/avetime,100.*avetime/ksptime);CHKERRQ(ierr);
  }
  CHKERRQ(PetscViewerFlush(viewer));
  PetscFunctionReturn(0);
}

/*TEST

   build:
     requires: defined(PETSC_USE_LOG)

   test:
     TODO: need to implement

TEST*/
