
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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCheckFalse((size > 1) && (size != 4),PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must be run wuth 4 MPI ranks");

  CHKERRQ(DMCreate(PETSC_COMM_WORLD,&dms));
  CHKERRQ(DMSetType(dms,DMSWARM));

  CHKERRQ(DMSwarmInitializeFieldRegister(dms));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(dms,"strain",1,PETSC_REAL));
  CHKERRQ(DMSwarmFinalizeFieldRegister(dms));
  CHKERRQ(DMSwarmSetLocalSizes(dms,5+rank,4));
  CHKERRQ(DMView(dms,PETSC_VIEWER_STDOUT_WORLD));

  {
    PetscReal *array;
    CHKERRQ(DMSwarmGetField(dms,"viscosity",NULL,NULL,(void**)&array));
    for (p=0; p<5+rank; p++) {
      array[p] = 11.1 + p*0.1 + rank*100.0;
    }
    CHKERRQ(DMSwarmRestoreField(dms,"viscosity",NULL,NULL,(void**)&array));
  }

  {
    PetscReal *array;
    CHKERRQ(DMSwarmGetField(dms,"strain",NULL,NULL,(void**)&array));
    for (p=0; p<5+rank; p++) {
      array[p] = 2.0e-2 + p*0.001 + rank*1.0;
    }
    CHKERRQ(DMSwarmRestoreField(dms,"strain",NULL,NULL,(void**)&array));
  }

  CHKERRQ(DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x));

  CHKERRQ(DMSwarmVectorDefineField(dms,"strain"));
  CHKERRQ(DMCreateGlobalVector(dms,&x));
  CHKERRQ(VecDestroy(&x));

  {
    PetscInt    *rankval;
    PetscInt    npoints[2],npoints_orig[2];

    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints_orig[0]));
    CHKERRQ(DMSwarmGetSize(dms,&npoints_orig[1]));
    CHKERRQ(DMSwarmGetField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval));
    if ((rank == 0) && (size > 1)) {
      rankval[0] = 1;
      rankval[3] = 1;
    }
    if (rank == 3) {
      rankval[2] = 1;
    }
    CHKERRQ(DMSwarmRestoreField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval));
    CHKERRQ(DMSwarmMigrate(dms,PETSC_TRUE));
    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints[0]));
    CHKERRQ(DMSwarmGetSize(dms,&npoints[1]));
    CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] before(%D,%D) after(%D,%D)\n",rank,npoints_orig[0],npoints_orig[1],npoints[0],npoints[1]));
    CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  }
  {
    CHKERRQ(DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x));
    CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x));
  }
  {
    CHKERRQ(DMSwarmCreateGlobalVectorFromField(dms,"strain",&x));
    CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(DMSwarmDestroyGlobalVectorFromField(dms,"strain",&x));
  }

  CHKERRQ(DMDestroy(&dms));
  PetscFunctionReturn(0);
}

PetscErrorCode ex1_2(void)
{
  DM             dms;
  Vec            x;
  PetscMPIInt    rank,size;
  PetscInt       p;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(DMCreate(PETSC_COMM_WORLD,&dms));
  CHKERRQ(DMSetType(dms,DMSWARM));
  CHKERRQ(DMSwarmInitializeFieldRegister(dms));

  CHKERRQ(DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(dms,"strain",1,PETSC_REAL));
  CHKERRQ(DMSwarmFinalizeFieldRegister(dms));
  CHKERRQ(DMSwarmSetLocalSizes(dms,5+rank,4));
  CHKERRQ(DMView(dms,PETSC_VIEWER_STDOUT_WORLD));
  {
    PetscReal *array;
    CHKERRQ(DMSwarmGetField(dms,"viscosity",NULL,NULL,(void**)&array));
    for (p=0; p<5+rank; p++) {
      array[p] = 11.1 + p*0.1 + rank*100.0;
    }
    CHKERRQ(DMSwarmRestoreField(dms,"viscosity",NULL,NULL,(void**)&array));
  }
  {
    PetscReal *array;
    CHKERRQ(DMSwarmGetField(dms,"strain",NULL,NULL,(void**)&array));
    for (p=0; p<5+rank; p++) {
      array[p] = 2.0e-2 + p*0.001 + rank*1.0;
    }
    CHKERRQ(DMSwarmRestoreField(dms,"strain",NULL,NULL,(void**)&array));
  }
  {
    PetscInt    *rankval;
    PetscInt    npoints[2],npoints_orig[2];

    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints_orig[0]));
    CHKERRQ(DMSwarmGetSize(dms,&npoints_orig[1]));
    CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] before(%D,%D)\n",rank,npoints_orig[0],npoints_orig[1]));
    CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

    CHKERRQ(DMSwarmGetField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval));

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
    CHKERRQ(DMSwarmRestoreField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval));
    CHKERRQ(DMSwarmCollectViewCreate(dms));
    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints[0]));
    CHKERRQ(DMSwarmGetSize(dms,&npoints[1]));
    CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] after(%D,%D)\n",rank,npoints[0],npoints[1]));
    CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

    CHKERRQ(DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x));
    CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x));

    CHKERRQ(DMSwarmCollectViewDestroy(dms));
    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints[0]));
    CHKERRQ(DMSwarmGetSize(dms,&npoints[1]));
    CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] after_v(%D,%D)\n",rank,npoints[0],npoints[1]));
    CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

    CHKERRQ(DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x));
    CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x));
  }
  CHKERRQ(DMDestroy(&dms));
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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  overlap = 2;
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,13,13,PETSC_DECIDE,PETSC_DECIDE,1,overlap,NULL,NULL,&dmcell));
  CHKERRQ(DMSetFromOptions(dmcell));
  CHKERRQ(DMSetUp(dmcell));
  CHKERRQ(DMDASetUniformCoordinates(dmcell,-1.0,1.0,-1.0,1.0,-1.0,1.0));
  CHKERRQ(DMDAGetCorners(dmcell,&is,&js,NULL,&ni,&nj,NULL));
  CHKERRQ(DMCreate(PETSC_COMM_WORLD,&dms));
  CHKERRQ(DMSetType(dms,DMSWARM));
  CHKERRQ(DMSwarmSetCellDM(dms,dmcell));

  /* load in data types */
  CHKERRQ(DMSwarmInitializeFieldRegister(dms));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(dms,"coorx",1,PETSC_REAL));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(dms,"coory",1,PETSC_REAL));
  CHKERRQ(DMSwarmFinalizeFieldRegister(dms));
  CHKERRQ(DMSwarmSetLocalSizes(dms,ni*nj*4,4));
  CHKERRQ(DMView(dms,PETSC_VIEWER_STDOUT_WORLD));

  /* set values within the swarm */
  {
    PetscReal  *array_x,*array_y;
    PetscInt   npoints,i,j,cnt;
    DMDACoor2d **LA_coor;
    Vec        coor;
    DM         dmcellcdm;

    CHKERRQ(DMGetCoordinateDM(dmcell,&dmcellcdm));
    CHKERRQ(DMGetCoordinates(dmcell,&coor));
    CHKERRQ(DMDAVecGetArray(dmcellcdm,coor,&LA_coor));
    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints));
    CHKERRQ(DMSwarmGetField(dms,"coorx",NULL,NULL,(void**)&array_x));
    CHKERRQ(DMSwarmGetField(dms,"coory",NULL,NULL,(void**)&array_y));
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
    CHKERRQ(DMSwarmRestoreField(dms,"coory",NULL,NULL,(void**)&array_y));
    CHKERRQ(DMSwarmRestoreField(dms,"coorx",NULL,NULL,(void**)&array_x));
    CHKERRQ(DMDAVecRestoreArray(dmcellcdm,coor,&LA_coor));
  }
  {
    PetscInt    npoints[2],npoints_orig[2],ng;

    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints_orig[0]));
    CHKERRQ(DMSwarmGetSize(dms,&npoints_orig[1]));
    CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] before(%D,%D)\n",rank,npoints_orig[0],npoints_orig[1]));
    CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(DMSwarmCollect_DMDABoundingBox(dms,&ng));

    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints[0]));
    CHKERRQ(DMSwarmGetSize(dms,&npoints[1]));
    CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] after(%D,%D)\n",rank,npoints[0],npoints[1]));
    CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  }
  {
    PetscReal *array_x,*array_y;
    PetscInt  npoints,p;
    FILE      *fp = NULL;
    char      name[PETSC_MAX_PATH_LEN];

    CHKERRQ(PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"c-rank%d.gp",rank));
    fp = fopen(name,"w");
    PetscCheckFalse(!fp,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",name);
    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints));
    CHKERRQ(DMSwarmGetField(dms,"coorx",NULL,NULL,(void**)&array_x));
    CHKERRQ(DMSwarmGetField(dms,"coory",NULL,NULL,(void**)&array_y));
    for (p=0; p<npoints; p++) {
      fprintf(fp,"%+1.4e %+1.4e %1.4e\n",array_x[p],array_y[p],(double)rank);
    }
    CHKERRQ(DMSwarmRestoreField(dms,"coory",NULL,NULL,(void**)&array_y));
    CHKERRQ(DMSwarmRestoreField(dms,"coorx",NULL,NULL,(void**)&array_x));
    fclose(fp);
  }
  CHKERRQ(DMDestroy(&dmcell));
  CHKERRQ(DMDestroy(&dms));
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
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(DMSwarmGetLocalSize(dm,&npoints));
  CHKERRQ(DMSwarmGetField(dm,"coorx",NULL,NULL,(void**)&array_x));
  CHKERRQ(DMSwarmGetField(dm,"coory",NULL,NULL,(void**)&array_y));

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

  CHKERRQ(PetscMalloc1(p2collect+1,&plist));
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
  CHKERRQ(DMSwarmRestoreField(dm,"coory",NULL,NULL,(void**)&array_y));
  CHKERRQ(DMSwarmRestoreField(dm,"coorx",NULL,NULL,(void**)&array_x));

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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  nn = 101;
  dx = 2.0/ (PetscReal)(nn-1);
  overlap = 0;
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,nn,nn,PETSC_DECIDE,PETSC_DECIDE,1,overlap,NULL,NULL,&dmcell));
  CHKERRQ(DMSetFromOptions(dmcell));
  CHKERRQ(DMSetUp(dmcell));
  CHKERRQ(DMDASetUniformCoordinates(dmcell,-1.0,1.0,-1.0,1.0,-1.0,1.0));
  CHKERRQ(DMDAGetCorners(dmcell,&is,&js,NULL,&ni,&nj,NULL));
  CHKERRQ(DMCreate(PETSC_COMM_WORLD,&dms));
  CHKERRQ(DMSetType(dms,DMSWARM));

  /* load in data types */
  CHKERRQ(DMSwarmInitializeFieldRegister(dms));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(dms,"coorx",1,PETSC_REAL));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(dms,"coory",1,PETSC_REAL));
  CHKERRQ(DMSwarmFinalizeFieldRegister(dms));
  CHKERRQ(DMSwarmSetLocalSizes(dms,ni*nj*4,4));
  CHKERRQ(DMView(dms,PETSC_VIEWER_STDOUT_WORLD));

  /* set values within the swarm */
  {
    PetscReal  *array_x,*array_y;
    PetscInt   npoints,i,j,cnt;
    DMDACoor2d **LA_coor;
    Vec        coor;
    DM         dmcellcdm;

    CHKERRQ(DMGetCoordinateDM(dmcell,&dmcellcdm));
    CHKERRQ(DMGetCoordinates(dmcell,&coor));
    CHKERRQ(DMDAVecGetArray(dmcellcdm,coor,&LA_coor));
    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints));
    CHKERRQ(DMSwarmGetField(dms,"coorx",NULL,NULL,(void**)&array_x));
    CHKERRQ(DMSwarmGetField(dms,"coory",NULL,NULL,(void**)&array_y));
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
    CHKERRQ(DMSwarmRestoreField(dms,"coory",NULL,NULL,(void**)&array_y));
    CHKERRQ(DMSwarmRestoreField(dms,"coorx",NULL,NULL,(void**)&array_x));
    CHKERRQ(DMDAVecRestoreArray(dmcellcdm,coor,&LA_coor));
  }
  CHKERRQ(PetscMalloc1(1,&zone));
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

    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints_orig[0]));
    CHKERRQ(DMSwarmGetSize(dms,&npoints_orig[1]));
    CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] before(%D,%D)\n",rank,npoints_orig[0],npoints_orig[1]));
    CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(DMSwarmCollect_General(dms,collect_zone,sizeof(CollectZoneCtx),zone,&ng));
    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints[0]));
    CHKERRQ(DMSwarmGetSize(dms,&npoints[1]));
    CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"rank[%d] after(%D,%D)\n",rank,npoints[0],npoints[1]));
    CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  }
  {
    PetscReal *array_x,*array_y;
    PetscInt  npoints,p;
    FILE      *fp = NULL;
    char      name[PETSC_MAX_PATH_LEN];

    CHKERRQ(PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"c-rank%d.gp",rank));
    fp = fopen(name,"w");
    PetscCheckFalse(!fp,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",name);
    CHKERRQ(DMSwarmGetLocalSize(dms,&npoints));
    CHKERRQ(DMSwarmGetField(dms,"coorx",NULL,NULL,(void**)&array_x));
    CHKERRQ(DMSwarmGetField(dms,"coory",NULL,NULL,(void**)&array_y));
    for (p=0; p<npoints; p++) {
      fprintf(fp,"%+1.4e %+1.4e %1.4e\n",array_x[p],array_y[p],(double)rank);
    }
    CHKERRQ(DMSwarmRestoreField(dms,"coory",NULL,NULL,(void**)&array_y));
    CHKERRQ(DMSwarmRestoreField(dms,"coorx",NULL,NULL,(void**)&array_x));
    fclose(fp);
  }
  CHKERRQ(DMDestroy(&dmcell));
  CHKERRQ(DMDestroy(&dms));
  CHKERRQ(PetscFree(zone));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       test_mode = 4;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-test_mode",&test_mode,NULL));
  if (test_mode == 1) {
    CHKERRQ(ex1_1());
  } else if (test_mode == 2) {
    CHKERRQ(ex1_2());
  } else if (test_mode == 3) {
    CHKERRQ(ex1_3());
  } else if (test_mode == 4) {
    CHKERRQ(ex1_4());
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown test_mode value, should be 1,2,3,4");
  ierr = PetscFinalize();
  return ierr;
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
