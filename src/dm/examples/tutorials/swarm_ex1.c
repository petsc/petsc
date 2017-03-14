
static char help[] = "Tests DMSwarm\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>

PetscErrorCode DMSwarmCollect_General(DM dm,PetscErrorCode (*collect)(DM,void*,PetscInt*,PetscInt**),size_t ctx_size,void *ctx,PetscInt *globalsize);
PetscErrorCode DMSwarmCollect_DMDABoundingBox(DM dm,PetscInt *globalsize);

PetscErrorCode ex1_1(void)
{
  DM dms;
  PetscErrorCode ierr;
  Vec x;
  PetscMPIInt rank,commsize;
  PetscInt p;
  
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if ((commsize > 1) && (commsize != 4)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must be run wuth 4 MPI ranks");
  
  ierr = DMCreate(PETSC_COMM_WORLD,&dms);CHKERRQ(ierr);
  ierr = DMSetType(dms,DMSWARM);CHKERRQ(ierr);

  ierr = DMSwarmInitializeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"strain",1,PETSC_REAL);CHKERRQ(ierr);
  
  ierr = DMSwarmFinalizeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmSetLocalSizes(dms,5+rank,4);CHKERRQ(ierr);
  ierr = DMView(dms,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  {
    PetscReal *array;
    ierr = DMSwarmGetField(dms,"viscosity",NULL,NULL,(void**)&array);CHKERRQ(ierr);
    for (p=0; p<5+rank; p++) {
      array[p] = 11.1 + p*0.1 + rank*100.0;
    }
    ierr = DMSwarmRestoreField(dms,"viscosity",NULL,NULL,(void**)&array);CHKERRQ(ierr);
  }

  {
    PetscReal *array;
    ierr = DMSwarmGetField(dms,"strain",NULL,NULL,(void**)&array);CHKERRQ(ierr);
    for (p=0; p<5+rank; p++) {
      array[p] = 2.0e-2 + p*0.001 + rank*1.0;
    }
    ierr = DMSwarmRestoreField(dms,"strain",NULL,NULL,(void**)&array);CHKERRQ(ierr);
  }

  ierr = DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
  /*ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);*/
  ierr = DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
  
  ierr = DMSwarmVectorDefineField(dms,"strain");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dms,&x);CHKERRQ(ierr);
  /*ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);*/
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  
  {
    PetscInt *rankval;
    PetscInt npoints[2],npoints_orig[2];
    
    ierr = DMSwarmGetLocalSize(dms,&npoints_orig[0]);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(dms,&npoints_orig[1]);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
    if ((rank == 0) && (commsize > 1)) {
      rankval[0] = 1;
      rankval[3] = 1;
    }
    if (rank == 3) {
      rankval[2] = 1;
    }
    ierr = DMSwarmRestoreField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
    
    ierr = DMSwarmMigrate(dms,PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(dms,&npoints[0]);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(dms,&npoints[1]);CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_SELF,"rank[%d] before(%D,%D) after(%D,%D)\n",rank,npoints_orig[0],npoints_orig[1],npoints[0],npoints[1]);
  }
  
  {
    ierr = DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
  }
  {
    ierr = DMSwarmCreateGlobalVectorFromField(dms,"strain",&x);CHKERRQ(ierr);
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(dms,"strain",&x);CHKERRQ(ierr);
  }

  ierr = DMDestroy(&dms);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ex1_2(void)
{
  DM dms;
  PetscErrorCode ierr;
  Vec x;
  PetscMPIInt rank,commsize;
  PetscInt p;
  
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  
  ierr = DMCreate(PETSC_COMM_WORLD,&dms);CHKERRQ(ierr);
  ierr = DMSetType(dms,DMSWARM);CHKERRQ(ierr);
  
  ierr = DMSwarmInitializeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"strain",1,PETSC_REAL);CHKERRQ(ierr);
  
  ierr = DMSwarmFinalizeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmSetLocalSizes(dms,5+rank,4);CHKERRQ(ierr);
  ierr = DMView(dms,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  {
    PetscReal *array;
    ierr = DMSwarmGetField(dms,"viscosity",NULL,NULL,(void**)&array);CHKERRQ(ierr);
    for (p=0; p<5+rank; p++) {
      array[p] = 11.1 + p*0.1 + rank*100.0;
    }
    ierr = DMSwarmRestoreField(dms,"viscosity",NULL,NULL,(void**)&array);CHKERRQ(ierr);
  }
  
  {
    PetscReal *array;
    ierr = DMSwarmGetField(dms,"strain",NULL,NULL,(void**)&array);CHKERRQ(ierr);
    for (p=0; p<5+rank; p++) {
      array[p] = 2.0e-2 + p*0.001 + rank*1.0;
    }
    ierr = DMSwarmRestoreField(dms,"strain",NULL,NULL,(void**)&array);CHKERRQ(ierr);
  }
  
  {
    PetscInt *rankval;
    PetscInt npoints[2],npoints_orig[2];
    
    ierr = DMSwarmGetLocalSize(dms,&npoints_orig[0]);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(dms,&npoints_orig[1]);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF,"rank[%d] before(%D,%D)\n",rank,npoints_orig[0],npoints_orig[1]);

    ierr = DMSwarmGetField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);

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
    ierr = DMSwarmRestoreField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
    
    ierr = DMSwarmCollectViewCreate(dms);CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(dms,&npoints[0]);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(dms,&npoints[1]);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF,"rank[%d] after(%D,%D)\n",rank,npoints[0],npoints[1]);

    ierr = DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);

    ierr = DMSwarmCollectViewDestroy(dms);CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(dms,&npoints[0]);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(dms,&npoints[1]);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF,"rank[%d] after_v(%D,%D)\n",rank,npoints[0],npoints[1]);

    ierr = DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dms);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 splot "c-rank0.gp","c-rank1.gp","c-rank2.gp","c-rank3.gp"
*/
PetscErrorCode ex1_3(void)
{
  DM dms;
  PetscErrorCode ierr;
  PetscMPIInt rank,commsize;
  PetscInt is,js,ni,nj,overlap;
  DM dmcell;
  
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  
  overlap = 2;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,13,13,PETSC_DECIDE,PETSC_DECIDE,1,overlap,NULL,NULL,&dmcell);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmcell);CHKERRQ(ierr);
  ierr = DMSetUp(dmcell);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dmcell,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
  
  ierr = DMDAGetCorners(dmcell,&is,&js,NULL,&ni,&nj,NULL);CHKERRQ(ierr);
  
  ierr = DMCreate(PETSC_COMM_WORLD,&dms);CHKERRQ(ierr);
  ierr = DMSetType(dms,DMSWARM);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(dms,dmcell);CHKERRQ(ierr);
  
  /* load in data types */
  ierr = DMSwarmInitializeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"coorx",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"coory",1,PETSC_REAL);CHKERRQ(ierr);
  
  ierr = DMSwarmFinalizeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmSetLocalSizes(dms,ni*nj*4,4);CHKERRQ(ierr);
  ierr = DMView(dms,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  /* set values within the swarm */
  {
    PetscReal *array_x,*array_y;
    PetscInt npoints,i,j,cnt;
    DMDACoor2d **LA_coor;
    Vec coor;
    DM dmcellcdm;
    
    ierr = DMGetCoordinateDM(dmcell,&dmcellcdm);CHKERRQ(ierr);
    ierr = DMGetCoordinates(dmcell,&coor);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dmcellcdm,coor,&LA_coor);CHKERRQ(ierr);
    
    ierr = DMSwarmGetLocalSize(dms,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    cnt = 0;
    for (j=js; j<js+nj; j++) {
      for (i=is; i<is+ni; i++) {
        PetscReal xp,yp;
        
        xp = LA_coor[j][i].x;
        yp = LA_coor[j][i].y;
        
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
    
    ierr = DMSwarmRestoreField(dms,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dms,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dmcellcdm,coor,&LA_coor);CHKERRQ(ierr);
  }
  
  {
    PetscInt npoints[2],npoints_orig[2],ng;
    
    ierr = DMSwarmGetLocalSize(dms,&npoints_orig[0]);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(dms,&npoints_orig[1]);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF,"rank[%d] before(%D,%D)\n",rank,npoints_orig[0],npoints_orig[1]);
    
    ierr = DMSwarmCollect_DMDABoundingBox(dms,&ng);CHKERRQ(ierr);
    
    ierr = DMSwarmGetLocalSize(dms,&npoints[0]);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(dms,&npoints[1]);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF,"rank[%d] after(%D,%D)\n",rank,npoints[0],npoints[1]);
  }
  
  {
    PetscReal *array_x,*array_y;
    PetscInt npoints,p;
    FILE *fp = NULL;
    char name[PETSC_MAX_PATH_LEN];
    
    PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"c-rank%d.gp",rank);
    fp = fopen(name,"w");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",name);
    ierr = DMSwarmGetLocalSize(dms,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      fprintf(fp,"%+1.4e %+1.4e %1.4e\n",array_x[p],array_y[p],(double)rank);
    }
    ierr = DMSwarmRestoreField(dms,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dms,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
    fclose(fp);
  }
  
  ierr = DMDestroy(&dmcell);CHKERRQ(ierr);
  ierr = DMDestroy(&dms);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

typedef struct {
  PetscReal cx[2];
  PetscReal radius;
} CollectZoneCtx;

PetscErrorCode collect_zone(DM dm,void *ctx,PetscInt *nfound,PetscInt **foundlist)
{
  CollectZoneCtx *zone = (CollectZoneCtx*)ctx;
  PetscInt p,npoints;
  PetscReal *array_x,*array_y,r2;
  PetscInt p2collect,*plist;
  PetscMPIInt rank;
  PetscErrorCode ierr;
  
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /*PetscPrintf(PETSC_COMM_WORLD,"z %1.4e,%1.4e --> %1.4e\n",zone->cx[0],zone->cx[1],zone->radius);*/
  
  ierr = DMSwarmGetLocalSize(dm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);

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
  
  PetscMalloc1(p2collect+1,&plist);
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
  ierr = DMSwarmRestoreField(dm,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
  
  /*PetscPrintf(PETSC_COMM_WORLD,"rank[%d]: p2collect = %d\n",rank,p2collect);*/

  *nfound = p2collect;
  *foundlist = plist;
  PetscFunctionReturn(0);
}

PetscErrorCode ex1_4(void)
{
  DM dms;
  PetscErrorCode ierr;
  PetscMPIInt rank,commsize;
  PetscInt is,js,ni,nj,overlap,nn;
  DM dmcell;
  CollectZoneCtx *zone;
  PetscReal dx;
  
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  
  nn = 101;
  dx = 2.0/ (PetscReal)(nn-1);
  overlap = 0;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,nn,nn,PETSC_DECIDE,PETSC_DECIDE,1,overlap,NULL,NULL,&dmcell);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmcell);CHKERRQ(ierr);
  ierr = DMSetUp(dmcell);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dmcell,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
  
  ierr = DMDAGetCorners(dmcell,&is,&js,NULL,&ni,&nj,NULL);CHKERRQ(ierr);
  
  ierr = DMCreate(PETSC_COMM_WORLD,&dms);CHKERRQ(ierr);
  ierr = DMSetType(dms,DMSWARM);CHKERRQ(ierr);
  
  /* load in data types */
  ierr = DMSwarmInitializeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"coorx",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"coory",1,PETSC_REAL);CHKERRQ(ierr);
  
  ierr = DMSwarmFinalizeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmSetLocalSizes(dms,ni*nj*4,4);CHKERRQ(ierr);
  ierr = DMView(dms,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  /* set values within the swarm */
  {
    PetscReal *array_x,*array_y;
    PetscInt npoints,i,j,cnt;
    DMDACoor2d **LA_coor;
    Vec coor;
    DM dmcellcdm;
    
    ierr = DMGetCoordinateDM(dmcell,&dmcellcdm);CHKERRQ(ierr);
    ierr = DMGetCoordinates(dmcell,&coor);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dmcellcdm,coor,&LA_coor);CHKERRQ(ierr);
    
    ierr = DMSwarmGetLocalSize(dms,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    cnt = 0;
    for (j=js; j<js+nj; j++) {
      for (i=is; i<is+ni; i++) {
        PetscReal xp,yp;
        
        xp = LA_coor[j][i].x;
        yp = LA_coor[j][i].y;
        
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
    
    ierr = DMSwarmRestoreField(dms,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dms,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dmcellcdm,coor,&LA_coor);CHKERRQ(ierr);
  }
  
  ierr = PetscMalloc1(1,&zone);CHKERRQ(ierr);
  if (commsize == 4) {
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
    PetscInt npoints[2],npoints_orig[2],ng;
    
    ierr = DMSwarmGetLocalSize(dms,&npoints_orig[0]);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(dms,&npoints_orig[1]);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF,"rank[%d] before(%D,%D)\n",rank,npoints_orig[0],npoints_orig[1]);
    
    ierr = DMSwarmCollect_General(dms,collect_zone,sizeof(CollectZoneCtx),zone,&ng);CHKERRQ(ierr);
    
    ierr = DMSwarmGetLocalSize(dms,&npoints[0]);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(dms,&npoints[1]);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF,"rank[%d] after(%D,%D)\n",rank,npoints[0],npoints[1]);
  }
  
  {
    PetscReal *array_x,*array_y;
    PetscInt npoints,p;
    FILE *fp = NULL;
    char name[PETSC_MAX_PATH_LEN];
    
    PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"c-rank%d.gp",rank);
    fp = fopen(name,"w");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",name);
    ierr = DMSwarmGetLocalSize(dms,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      fprintf(fp,"%+1.4e %+1.4e %1.4e\n",array_x[p],array_y[p],(double)rank);
    }
    ierr = DMSwarmRestoreField(dms,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dms,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
    fclose(fp);
  }
  
  ierr = DMDestroy(&dmcell);CHKERRQ(ierr);
  ierr = DMDestroy(&dms);CHKERRQ(ierr);
  ierr = PetscFree(zone);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt test_mode = 4;
  
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-test_mode",&test_mode,NULL);CHKERRQ(ierr);
  if (test_mode == 1) {
    ierr = ex1_1();CHKERRQ(ierr);
  } else if (test_mode == 2) {
    ierr = ex1_2();CHKERRQ(ierr);
  } else if (test_mode == 3) {
    ierr = ex1_3();CHKERRQ(ierr);
  } else if (test_mode == 4) {
    ierr = ex1_4();CHKERRQ(ierr);
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"Unknown test_mode value, should be 1,2,3,4\n");
  }
  ierr = PetscFinalize();
  return 0;
}
