
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
  ierr = PetscLogDefaultBegin();CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);

  PetscPreLoadBegin(preload,"Load system");

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecSet(b,1.0);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);

  PetscPreLoadStage("KSPSolve");
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  PetscPreLoadEnd();
  ierr = PetscLogView_VecScatter(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

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
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"numProcs   = %d\n",size);CHKERRQ(ierr);

  ierr = PetscGetArchType(arch,sizeof(arch));CHKERRQ(ierr);
  ierr = PetscGetHostName(hostname,sizeof(hostname));CHKERRQ(ierr);
  ierr = PetscGetUserName(username,sizeof(username));CHKERRQ(ierr);
  ierr = PetscGetProgramName(pname,sizeof(pname));CHKERRQ(ierr);
  ierr = PetscGetDate(date,sizeof(date));CHKERRQ(ierr);
  ierr = PetscGetVersion(version,sizeof(version));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%s on a %s named %s with %d processors, by %s %s\n", pname, arch, hostname, size, username, date);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Using %s\n", version);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Configure options: %s",petscconfigureoptions);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "%s", petscmachineinfo);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "%s", petsccompilerinfo);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "%s", petsccompilerflagsinfo);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "%s", petsclinkerinfo);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "%s\n", PETSC_MPICC_SHOW);CHKERRQ(ierr);
  ierr = PetscOptionsView(NULL,viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HWLOC)
  ierr = PetscProcessPlacementView(viewer);CHKERRQ(ierr);
#endif
  ierr = PetscViewerASCIIPrintf(viewer, "----------------------------------------------------\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"                Time     Min to Max Range   Proportion of KSP\n");CHKERRQ(ierr);

  eventInfo = stageLog->stageInfo[stage].eventLog->eventInfo;
  ierr = MPI_Allreduce(&eventInfo[KSP_Solve].time,&ksptime,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRMPI(ierr);
  ksptime = ksptime/size;

  for (i=0; i<(int)(sizeof(events)/sizeof(int)); i++) {
    event = events[i];
    stats[COUNT]   = eventInfo[event].count;
    stats[TIME]    = eventInfo[event].time;
    stats[NUMMESS] = eventInfo[event].numMessages;
    stats[MESSLEN] = eventInfo[event].messageLength;
    stats[REDUCT]  = eventInfo[event].numReductions;
    stats[FLOPS]   = eventInfo[event].flops;
    ierr = MPI_Allreduce(stats,maxstats,6,MPIU_PETSCLOGDOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRMPI(ierr);
    ierr = MPI_Allreduce(stats,minstats,6,MPIU_PETSCLOGDOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRMPI(ierr);
    ierr = MPI_Allreduce(stats,sumstats,6,MPIU_PETSCLOGDOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRMPI(ierr);

    avetime  = sumstats[1]/size;
    ierr = PetscViewerASCIIPrintf(viewer,"%s %4.2e   -%5.1f %% %5.1f %%   %4.2e %%\n",stageLog->eventLog->eventInfo[event].name,
                                  avetime,100.*(avetime-minstats[1])/avetime,100.*(maxstats[1]-avetime)/avetime,100.*avetime/ksptime);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   build:
     requires: defined(PETSC_USE_LOG)

   test:
     TODO: need to implement

TEST*/
