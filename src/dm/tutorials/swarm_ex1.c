
static char help[] = "Tests DMSwarm\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>

PETSC_EXTERN PetscErrorCode DMSwarmCollect_General(DM,PetscErrorCode (*)(DM,void*,PetscInt*,PetscInt**),size_t,void*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMSwarmCollect_DMDABoundingBox(DM,PetscInt*);

PetscErrorCode ex1_1(void)
{
  DM             dms;
  Vec            x;
  PetscMPIInt    rank,size;
  PetscInt       p;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCheck(!(size > 1) || !(size != 4),PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must be run wuth 4 MPI ranks");

  PetscCall(DMCreate(PETSC_COMM_WORLD,&dms));
  PetscCall(DMSetType(dms,DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject) dms, "Particles"));

  PetscCall(DMSwarmInitializeFieldRegister(dms));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms,"strain",1,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(dms));
  PetscCall(DMSwarmSetLocalSizes(dms,5+rank,4));
  PetscCall(DMView(dms,PETSC_VIEWER_STDOUT_WORLD));

  {
    PetscReal *array;
    PetscCall(DMSwarmGetField(dms,"viscosity",NULL,NULL,(void**)&array));
    for (p=0; p<5+rank; p++) {
      array[p] = 11.1 + p*0.1 + rank*100.0;
    }
    PetscCall(DMSwarmRestoreField(dms,"viscosity",NULL,NULL,(void**)&array));
  }

  {
    PetscReal *array;
    PetscCall(DMSwarmGetField(dms,"strain",NULL,NULL,(void**)&array));
    for (p=0; p<5+rank; p++) {
      array[p] = 2.0e-2 + p*0.001 + rank*1.0;
    }
    PetscCall(DMSwarmRestoreField(dms,"strain",NULL,NULL,(void**)&array));
  }

  PetscCall(DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x));

  PetscCall(DMSwarmVectorDefineField(dms,"strain"));
  PetscCall(DMCreateGlobalVector(dms,&x));
  PetscCall(VecDestroy(&x));

  {
    PetscInt    *rankval;
    PetscInt    npoints[2],npoints_orig[2];

    PetscCall(DMSwarmGetLocalSize(dms,&npoints_orig[0]));
    PetscCall(DMSwarmGetSize(dms,&npoints_orig[1]));
    PetscCall(DMSwarmGetField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval));
    if ((rank == 0) && (size > 1)) {
      rankval[0] = 1;
      rankval[3] = 1;
    }
    if (rank == 3) {
      rankval[2] = 1;
    }
    PetscCall(DMSwarmRestoreField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval));
    PetscCall(DMSwarmMigrate(dms,PETSC_TRUE));
    PetscCall(DMSwarmGetLocalSize(dms,&npoints[0]));
    PetscCall(DMSwarmGetSize(dms,&npoints[1]));
    PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] before(%" PetscInt_FMT ",%" PetscInt_FMT ") after(%" PetscInt_FMT ",%" PetscInt_FMT ")\n",rank,npoints_orig[0],npoints_orig[1],npoints[0],npoints[1]));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  }
  {
    PetscCall(DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x));
    PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x));
  }
  {
    PetscCall(DMSwarmCreateGlobalVectorFromField(dms,"strain",&x));
    PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(dms,"strain",&x));
  }

  PetscCall(DMDestroy(&dms));
  PetscFunctionReturn(0);
}

PetscErrorCode ex1_2(void)
{
  DM             dms;
  Vec            x;
  PetscMPIInt    rank,size;
  PetscInt       p;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(DMCreate(PETSC_COMM_WORLD,&dms));
  PetscCall(DMSetType(dms,DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject) dms, "Particles"));
  PetscCall(DMSwarmInitializeFieldRegister(dms));

  PetscCall(DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms,"strain",1,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(dms));
  PetscCall(DMSwarmSetLocalSizes(dms,5+rank,4));
  PetscCall(DMView(dms,PETSC_VIEWER_STDOUT_WORLD));
  {
    PetscReal *array;
    PetscCall(DMSwarmGetField(dms,"viscosity",NULL,NULL,(void**)&array));
    for (p=0; p<5+rank; p++) {
      array[p] = 11.1 + p*0.1 + rank*100.0;
    }
    PetscCall(DMSwarmRestoreField(dms,"viscosity",NULL,NULL,(void**)&array));
  }
  {
    PetscReal *array;
    PetscCall(DMSwarmGetField(dms,"strain",NULL,NULL,(void**)&array));
    for (p=0; p<5+rank; p++) {
      array[p] = 2.0e-2 + p*0.001 + rank*1.0;
    }
    PetscCall(DMSwarmRestoreField(dms,"strain",NULL,NULL,(void**)&array));
  }
  {
    PetscInt    *rankval;
    PetscInt    npoints[2],npoints_orig[2];

    PetscCall(DMSwarmGetLocalSize(dms,&npoints_orig[0]));
    PetscCall(DMSwarmGetSize(dms,&npoints_orig[1]));
    PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] before(%" PetscInt_FMT ",%" PetscInt_FMT ")\n",rank,npoints_orig[0],npoints_orig[1]));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(DMSwarmGetField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval));

    if (rank == 1) {
      rankval[0] = -1;
    }
    if (rank == 2) {
      rankval[1] = -1;
    }
    if (rank == 3) {
      rankval[3] = -1;
      rankval[4] = -1;
    }
    PetscCall(DMSwarmRestoreField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval));
    PetscCall(DMSwarmCollectViewCreate(dms));
    PetscCall(DMSwarmGetLocalSize(dms,&npoints[0]));
    PetscCall(DMSwarmGetSize(dms,&npoints[1]));
    PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] after(%" PetscInt_FMT ",%" PetscInt_FMT ")\n",rank,npoints[0],npoints[1]));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x));
    PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x));

    PetscCall(DMSwarmCollectViewDestroy(dms));
    PetscCall(DMSwarmGetLocalSize(dms,&npoints[0]));
    PetscCall(DMSwarmGetSize(dms,&npoints[1]));
    PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] after_v(%" PetscInt_FMT ",%" PetscInt_FMT ")\n",rank,npoints[0],npoints[1]));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x));
    PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x));
  }
  PetscCall(DMDestroy(&dms));
  PetscFunctionReturn(0);
}

/*
 splot "c-rank0.gp","c-rank1.gp","c-rank2.gp","c-rank3.gp"
*/
PetscErrorCode ex1_3(void)
{
  DM             dms;
  PetscMPIInt    rank,size;
  PetscInt       is,js,ni,nj,overlap;
  DM             dmcell;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  overlap = 2;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,13,13,PETSC_DECIDE,PETSC_DECIDE,1,overlap,NULL,NULL,&dmcell));
  PetscCall(DMSetFromOptions(dmcell));
  PetscCall(DMSetUp(dmcell));
  PetscCall(DMDASetUniformCoordinates(dmcell,-1.0,1.0,-1.0,1.0,-1.0,1.0));
  PetscCall(DMDAGetCorners(dmcell,&is,&js,NULL,&ni,&nj,NULL));
  PetscCall(DMCreate(PETSC_COMM_WORLD,&dms));
  PetscCall(DMSetType(dms,DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject) dms, "Particles"));
  PetscCall(DMSwarmSetCellDM(dms,dmcell));

  /* load in data types */
  PetscCall(DMSwarmInitializeFieldRegister(dms));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms,"coorx",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms,"coory",1,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(dms));
  PetscCall(DMSwarmSetLocalSizes(dms,ni*nj*4,4));
  PetscCall(DMView(dms,PETSC_VIEWER_STDOUT_WORLD));

  /* set values within the swarm */
  {
    PetscReal  *array_x,*array_y;
    PetscInt   npoints,i,j,cnt;
    DMDACoor2d **LA_coor;
    Vec        coor;
    DM         dmcellcdm;

    PetscCall(DMGetCoordinateDM(dmcell,&dmcellcdm));
    PetscCall(DMGetCoordinates(dmcell,&coor));
    PetscCall(DMDAVecGetArray(dmcellcdm,coor,&LA_coor));
    PetscCall(DMSwarmGetLocalSize(dms,&npoints));
    PetscCall(DMSwarmGetField(dms,"coorx",NULL,NULL,(void**)&array_x));
    PetscCall(DMSwarmGetField(dms,"coory",NULL,NULL,(void**)&array_y));
    cnt = 0;
    for (j=js; j<js+nj; j++) {
      for (i=is; i<is+ni; i++) {
        PetscReal xp,yp;
        xp = PetscRealPart(LA_coor[j][i].x);
        yp = PetscRealPart(LA_coor[j][i].y);
        array_x[4*cnt+0] = xp - 0.05; if (array_x[4*cnt+0] < -1.0) { array_x[4*cnt+0] = -1.0+1.0e-12; }
        array_x[4*cnt+1] = xp + 0.05; if (array_x[4*cnt+1] > 1.0)  { array_x[4*cnt+1] =  1.0-1.0e-12; }
        array_x[4*cnt+2] = xp - 0.05; if (array_x[4*cnt+2] < -1.0) { array_x[4*cnt+2] = -1.0+1.0e-12; }
        array_x[4*cnt+3] = xp + 0.05; if (array_x[4*cnt+3] > 1.0)  { array_x[4*cnt+3] =  1.0-1.0e-12; }

        array_y[4*cnt+0] = yp - 0.05; if (array_y[4*cnt+0] < -1.0) { array_y[4*cnt+0] = -1.0+1.0e-12; }
        array_y[4*cnt+1] = yp - 0.05; if (array_y[4*cnt+1] < -1.0) { array_y[4*cnt+1] = -1.0+1.0e-12; }
        array_y[4*cnt+2] = yp + 0.05; if (array_y[4*cnt+2] > 1.0)  { array_y[4*cnt+2] =  1.0-1.0e-12; }
        array_y[4*cnt+3] = yp + 0.05; if (array_y[4*cnt+3] > 1.0)  { array_y[4*cnt+3] =  1.0-1.0e-12; }
        cnt++;
      }
    }
    PetscCall(DMSwarmRestoreField(dms,"coory",NULL,NULL,(void**)&array_y));
    PetscCall(DMSwarmRestoreField(dms,"coorx",NULL,NULL,(void**)&array_x));
    PetscCall(DMDAVecRestoreArray(dmcellcdm,coor,&LA_coor));
  }
  {
    PetscInt    npoints[2],npoints_orig[2],ng;

    PetscCall(DMSwarmGetLocalSize(dms,&npoints_orig[0]));
    PetscCall(DMSwarmGetSize(dms,&npoints_orig[1]));
    PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] before(%" PetscInt_FMT ",%" PetscInt_FMT ")\n",rank,npoints_orig[0],npoints_orig[1]));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMSwarmCollect_DMDABoundingBox(dms,&ng));

    PetscCall(DMSwarmGetLocalSize(dms,&npoints[0]));
    PetscCall(DMSwarmGetSize(dms,&npoints[1]));
    PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] after(%" PetscInt_FMT ",%" PetscInt_FMT ")\n",rank,npoints[0],npoints[1]));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  }
  {
    PetscReal *array_x,*array_y;
    PetscInt  npoints,p;
    FILE      *fp = NULL;
    char      name[PETSC_MAX_PATH_LEN];

    PetscCall(PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"c-rank%d.gp",rank));
    fp = fopen(name,"w");
    PetscCheck(fp,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",name);
    PetscCall(DMSwarmGetLocalSize(dms,&npoints));
    PetscCall(DMSwarmGetField(dms,"coorx",NULL,NULL,(void**)&array_x));
    PetscCall(DMSwarmGetField(dms,"coory",NULL,NULL,(void**)&array_y));
    for (p=0; p<npoints; p++) {
      fprintf(fp,"%+1.4e %+1.4e %1.4e\n",array_x[p],array_y[p],(double)rank);
    }
    PetscCall(DMSwarmRestoreField(dms,"coory",NULL,NULL,(void**)&array_y));
    PetscCall(DMSwarmRestoreField(dms,"coorx",NULL,NULL,(void**)&array_x));
    fclose(fp);
  }
  PetscCall(DMDestroy(&dmcell));
  PetscCall(DMDestroy(&dms));
  PetscFunctionReturn(0);
}

typedef struct {
  PetscReal cx[2];
  PetscReal radius;
} CollectZoneCtx;

PetscErrorCode collect_zone(DM dm,void *ctx,PetscInt *nfound,PetscInt **foundlist)
{
  CollectZoneCtx *zone = (CollectZoneCtx*)ctx;
  PetscInt       p,npoints;
  PetscReal      *array_x,*array_y,r2;
  PetscInt       p2collect,*plist;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(DMSwarmGetLocalSize(dm,&npoints));
  PetscCall(DMSwarmGetField(dm,"coorx",NULL,NULL,(void**)&array_x));
  PetscCall(DMSwarmGetField(dm,"coory",NULL,NULL,(void**)&array_y));

  r2 = zone->radius * zone->radius;
  p2collect = 0;
  for (p=0; p<npoints; p++) {
    PetscReal sep2;

    sep2  = (array_x[p] - zone->cx[0])*(array_x[p] - zone->cx[0]);
    sep2 += (array_y[p] - zone->cx[1])*(array_y[p] - zone->cx[1]);
    if (sep2 < r2) {
      p2collect++;
    }
  }

  PetscCall(PetscMalloc1(p2collect+1,&plist));
  p2collect = 0;
  for (p=0; p<npoints; p++) {
    PetscReal sep2;

    sep2  = (array_x[p] - zone->cx[0])*(array_x[p] - zone->cx[0]);
    sep2 += (array_y[p] - zone->cx[1])*(array_y[p] - zone->cx[1]);
    if (sep2 < r2) {
      plist[p2collect] = p;
      p2collect++;
    }
  }
  PetscCall(DMSwarmRestoreField(dm,"coory",NULL,NULL,(void**)&array_y));
  PetscCall(DMSwarmRestoreField(dm,"coorx",NULL,NULL,(void**)&array_x));

  *nfound = p2collect;
  *foundlist = plist;
  PetscFunctionReturn(0);
}

PetscErrorCode ex1_4(void)
{
  DM             dms;
  PetscMPIInt    rank,size;
  PetscInt       is,js,ni,nj,overlap,nn;
  DM             dmcell;
  CollectZoneCtx *zone;
  PetscReal      dx;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  nn = 101;
  dx = 2.0/ (PetscReal)(nn-1);
  overlap = 0;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,nn,nn,PETSC_DECIDE,PETSC_DECIDE,1,overlap,NULL,NULL,&dmcell));
  PetscCall(DMSetFromOptions(dmcell));
  PetscCall(DMSetUp(dmcell));
  PetscCall(DMDASetUniformCoordinates(dmcell,-1.0,1.0,-1.0,1.0,-1.0,1.0));
  PetscCall(DMDAGetCorners(dmcell,&is,&js,NULL,&ni,&nj,NULL));
  PetscCall(DMCreate(PETSC_COMM_WORLD,&dms));
  PetscCall(DMSetType(dms,DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject) dms, "Particles"));

  /* load in data types */
  PetscCall(DMSwarmInitializeFieldRegister(dms));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms,"coorx",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms,"coory",1,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(dms));
  PetscCall(DMSwarmSetLocalSizes(dms,ni*nj*4,4));
  PetscCall(DMView(dms,PETSC_VIEWER_STDOUT_WORLD));

  /* set values within the swarm */
  {
    PetscReal  *array_x,*array_y;
    PetscInt   npoints,i,j,cnt;
    DMDACoor2d **LA_coor;
    Vec        coor;
    DM         dmcellcdm;

    PetscCall(DMGetCoordinateDM(dmcell,&dmcellcdm));
    PetscCall(DMGetCoordinates(dmcell,&coor));
    PetscCall(DMDAVecGetArray(dmcellcdm,coor,&LA_coor));
    PetscCall(DMSwarmGetLocalSize(dms,&npoints));
    PetscCall(DMSwarmGetField(dms,"coorx",NULL,NULL,(void**)&array_x));
    PetscCall(DMSwarmGetField(dms,"coory",NULL,NULL,(void**)&array_y));
    cnt = 0;
    for (j=js; j<js+nj; j++) {
      for (i=is; i<is+ni; i++) {
        PetscReal xp,yp;

        xp = PetscRealPart(LA_coor[j][i].x);
        yp = PetscRealPart(LA_coor[j][i].y);
        array_x[4*cnt+0] = xp - dx*0.1; /*if (array_x[4*cnt+0] < -1.0) { array_x[4*cnt+0] = -1.0+1.0e-12; }*/
        array_x[4*cnt+1] = xp + dx*0.1; /*if (array_x[4*cnt+1] > 1.0)  { array_x[4*cnt+1] =  1.0-1.0e-12; }*/
        array_x[4*cnt+2] = xp - dx*0.1; /*if (array_x[4*cnt+2] < -1.0) { array_x[4*cnt+2] = -1.0+1.0e-12; }*/
        array_x[4*cnt+3] = xp + dx*0.1; /*if (array_x[4*cnt+3] > 1.0)  { array_x[4*cnt+3] =  1.0-1.0e-12; }*/
        array_y[4*cnt+0] = yp - dx*0.1; /*if (array_y[4*cnt+0] < -1.0) { array_y[4*cnt+0] = -1.0+1.0e-12; }*/
        array_y[4*cnt+1] = yp - dx*0.1; /*if (array_y[4*cnt+1] < -1.0) { array_y[4*cnt+1] = -1.0+1.0e-12; }*/
        array_y[4*cnt+2] = yp + dx*0.1; /*if (array_y[4*cnt+2] > 1.0)  { array_y[4*cnt+2] =  1.0-1.0e-12; }*/
        array_y[4*cnt+3] = yp + dx*0.1; /*if (array_y[4*cnt+3] > 1.0)  { array_y[4*cnt+3] =  1.0-1.0e-12; }*/
        cnt++;
      }
    }
    PetscCall(DMSwarmRestoreField(dms,"coory",NULL,NULL,(void**)&array_y));
    PetscCall(DMSwarmRestoreField(dms,"coorx",NULL,NULL,(void**)&array_x));
    PetscCall(DMDAVecRestoreArray(dmcellcdm,coor,&LA_coor));
  }
  PetscCall(PetscMalloc1(1,&zone));
  if (size == 4) {
    if (rank == 0) {
      zone->cx[0] = 0.5;
      zone->cx[1] = 0.5;
      zone->radius = 0.3;
    }
    if (rank == 1) {
      zone->cx[0] = -0.5;
      zone->cx[1] = 0.5;
      zone->radius = 0.25;
    }
    if (rank == 2) {
      zone->cx[0] = 0.5;
      zone->cx[1] = -0.5;
      zone->radius = 0.2;
    }
    if (rank == 3) {
      zone->cx[0] = -0.5;
      zone->cx[1] = -0.5;
      zone->radius = 0.1;
    }
  } else {
    if (rank == 0) {
      zone->cx[0] = 0.5;
      zone->cx[1] = 0.5;
      zone->radius = 0.8;
    } else {
      zone->cx[0] = 10.0;
      zone->cx[1] = 10.0;
      zone->radius = 0.0;
    }
  }
  {
    PetscInt    npoints[2],npoints_orig[2],ng;

    PetscCall(DMSwarmGetLocalSize(dms,&npoints_orig[0]));
    PetscCall(DMSwarmGetSize(dms,&npoints_orig[1]));
    PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] before(%" PetscInt_FMT ",%" PetscInt_FMT ")\n",rank,npoints_orig[0],npoints_orig[1]));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMSwarmCollect_General(dms,collect_zone,sizeof(CollectZoneCtx),zone,&ng));
    PetscCall(DMSwarmGetLocalSize(dms,&npoints[0]));
    PetscCall(DMSwarmGetSize(dms,&npoints[1]));
    PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] after(%" PetscInt_FMT ",%" PetscInt_FMT ")\n",rank,npoints[0],npoints[1]));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  }
  {
    PetscReal *array_x,*array_y;
    PetscInt  npoints,p;
    FILE      *fp = NULL;
    char      name[PETSC_MAX_PATH_LEN];

    PetscCall(PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"c-rank%d.gp",rank));
    fp = fopen(name,"w");
    PetscCheck(fp,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",name);
    PetscCall(DMSwarmGetLocalSize(dms,&npoints));
    PetscCall(DMSwarmGetField(dms,"coorx",NULL,NULL,(void**)&array_x));
    PetscCall(DMSwarmGetField(dms,"coory",NULL,NULL,(void**)&array_y));
    for (p=0; p<npoints; p++) {
      fprintf(fp,"%+1.4e %+1.4e %1.4e\n",array_x[p],array_y[p],(double)rank);
    }
    PetscCall(DMSwarmRestoreField(dms,"coory",NULL,NULL,(void**)&array_y));
    PetscCall(DMSwarmRestoreField(dms,"coorx",NULL,NULL,(void**)&array_x));
    fclose(fp);
  }
  PetscCall(DMDestroy(&dmcell));
  PetscCall(DMDestroy(&dms));
  PetscCall(PetscFree(zone));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscInt       test_mode = 4;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-test_mode",&test_mode,NULL));
  if (test_mode == 1) {
    PetscCall(ex1_1());
  } else if (test_mode == 2) {
    PetscCall(ex1_2());
  } else if (test_mode == 3) {
    PetscCall(ex1_3());
  } else if (test_mode == 4) {
    PetscCall(ex1_4());
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown test_mode value, should be 1,2,3,4");
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex double

   test:
      args: -test_mode 1
      filter: grep -v atomic
      filter_output: grep -v atomic

   test:
      suffix: 2
      args: -test_mode 2
      filter: grep -v atomic
      filter_output: grep -v atomic

   test:
      suffix: 3
      args: -test_mode 3
      filter: grep -v atomic
      filter_output: grep -v atomic
      TODO: broken

   test:
      suffix: 4
      args: -test_mode 4
      filter: grep -v atomic
      filter_output: grep -v atomic

   test:
      suffix: 5
      nsize: 4
      args: -test_mode 1
      filter: grep -v atomic
      filter_output: grep -v atomic

   test:
      suffix: 6
      nsize: 4
      args: -test_mode 2
      filter: grep -v atomic
      filter_output: grep -v atomic

   test:
      suffix: 7
      nsize: 4
      args: -test_mode 3
      filter: grep -v atomic
      filter_output: grep -v atomic
      TODO: broken

   test:
      suffix: 8
      nsize: 4
      args: -test_mode 4
      filter: grep -v atomic
      filter_output: grep -v atomic

TEST*/
