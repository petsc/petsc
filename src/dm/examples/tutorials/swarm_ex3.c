
static char help[] = "Tests DMSwarm with DMShell\n\n";

#include <petscdm.h>
#include <petscdmshell.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include <petsc/private/dmimpl.h>


#undef __FUNCT__
#define __FUNCT__ "DMLocatePoints_DMDARegular"
PetscErrorCode DMLocatePoints_DMDARegular(DM dm,Vec pos,IS *iscell)
{
  PetscInt p,n,bs,npoints,si,sj,milocal,mjlocal,mx,my;
  DM dmregular;
  PetscInt *cellidx;
  PetscScalar *coor;
  PetscReal dx,dy,x0,y0;
  PetscErrorCode ierr;
  PetscMPIInt rank;
  
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  ierr = VecGetLocalSize(pos,&n);CHKERRQ(ierr);
  ierr = VecGetBlockSize(pos,&bs);CHKERRQ(ierr);
  npoints = n/bs;
  
  PetscMalloc1(npoints,&cellidx);

  ierr = DMGetApplicationContext(dm,(void**)&dmregular);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dmregular,&si,&sj,NULL,&milocal,&mjlocal,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dmregular,NULL,&mx,&my,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  
  dx = 2.0/((PetscReal)mx);
  dy = 2.0/((PetscReal)my);

  x0 = -1.0 + si*dx - 0.5*dx;
  y0 = -1.0 + sj*dy - 0.5*dy;
  
  ierr = VecGetArray(pos,&coor);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    PetscReal coorx,coory;
    PetscInt mi,mj;
    
    coorx = coor[2*p];
    coory = coor[2*p+1];
    
    mi = (PetscInt)( (coorx - (-1.0))/dx );
    mj = (PetscInt)( (coory - (-1.0))/dy );
    
    
    cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    
    if ((mj >= sj) && (mj < sj + mjlocal)) {
      if ((mi >= si) && (mi < si + milocal)) {
        cellidx[p] = (mi-si) + (mj-sj) * milocal;
      }
    }
    if (coorx < -1.0) cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (coorx >  1.0) cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (coory < -1.0) cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (coory >  1.0) cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
  }
  ierr = VecRestoreArray(pos,&coor);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF,npoints,cellidx,PETSC_OWN_POINTER,iscell);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetNeighbors_DMDARegular"
PetscErrorCode DMGetNeighbors_DMDARegular(DM dm,PetscInt *nneighbors,const PetscMPIInt **neighbors)
{
  DM dmregular;
  PetscErrorCode ierr;
  
  ierr = DMGetApplicationContext(dm,(void**)&dmregular);CHKERRQ(ierr);
  ierr = DMGetNeighbors(dmregular,nneighbors,neighbors);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SwarmViewGP"
PetscErrorCode SwarmViewGP(DM dms,const char prefix[])
{
  PetscReal *array;
  PetscInt *iarray;
  PetscInt npoints,p,bs;
  FILE *fp;
  char name[100];
  PetscMPIInt rank;
  PetscErrorCode ierr;
  
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  sprintf(name,"%s-rank%d.gp",prefix,rank);
  fp = fopen(name,"w");
  ierr = DMSwarmGetLocalSize(dms,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dms,DMSwarmPICField_coor,&bs,NULL,(void**)&array);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dms,"itag",NULL,NULL,(void**)&iarray);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    fprintf(fp,"%+1.4e %+1.4e %1.4e\n",array[2*p],array[2*p+1],(double)iarray[p]);
  }
  ierr = DMSwarmRestoreField(dms,"itag",NULL,NULL,(void**)&iarray);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dms,DMSwarmPICField_coor,&bs,NULL,(void**)&array);CHKERRQ(ierr);
  fclose(fp);
  PetscFunctionReturn(0);
}

/*
 Create a DMShell and attach a regularly spaced DMDA for point location
 Override methods for point location
*/
#undef __FUNCT__
#define __FUNCT__ "ex3_1"
PetscErrorCode ex3_1(void)
{
  DM dms,dmcell,dmregular;
  PetscMPIInt rank,commsize;
  PetscInt p,bs,nlocal,overlap,mx,tk;
  PetscReal dx;
  PetscReal *array,dt;
  PetscInt *iarray;
  PetscErrorCode ierr;
  
  
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Create a regularly spaced DMDA */
  mx = 40;
  overlap = 0;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx,mx,PETSC_DECIDE,PETSC_DECIDE,1,overlap,NULL,NULL,&dmregular);CHKERRQ(ierr);

  dx = 2.0/((PetscReal)mx);
  ierr = DMDASetUniformCoordinates(dmregular,-1.0+0.5*dx,1.0-0.5*dx,-1.0+0.5*dx,1.0-0.5*dx,-1.0,1.0);CHKERRQ(ierr);
  
  /* Create a DMShell for point location purposes */
  ierr = DMShellCreate(PETSC_COMM_WORLD,&dmcell);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dmcell,(void*)dmregular);CHKERRQ(ierr);
  dmcell->ops->locatepoints = DMLocatePoints_DMDARegular;
  dmcell->ops->getneighbors = DMGetNeighbors_DMDARegular;
  
  /* Create the swarm */
  ierr = DMCreate(PETSC_COMM_WORLD,&dms);CHKERRQ(ierr);
  ierr = DMSetType(dms,DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(dms,2);CHKERRQ(ierr);

  ierr = DMSwarmSetType(dms,DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(dms,dmcell);CHKERRQ(ierr);
  
  /* init fields */
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"itag",1,PETSC_INT);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(dms);CHKERRQ(ierr);
  
  {
    PetscInt si,sj,milocal,mjlocal;
    PetscReal *LA_coors;
    Vec coors;
    PetscInt cnt;
    
    ierr = DMDAGetCorners(dmregular,&si,&sj,NULL,&milocal,&mjlocal,NULL);CHKERRQ(ierr);
    ierr = DMGetCoordinates(dmregular,&coors);CHKERRQ(ierr);
    //VecView(coors,PETSC_VIEWER_STDOUT_WORLD);
    ierr = VecGetArray(coors,&LA_coors);CHKERRQ(ierr);
    
    ierr = DMSwarmSetLocalSizes(dms,milocal*mjlocal,4);CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(dms,&nlocal);CHKERRQ(ierr);

    ierr = DMSwarmGetField(dms,DMSwarmPICField_coor,&bs,NULL,(void**)&array);CHKERRQ(ierr);
    cnt = 0;
    srand(0);
    for (p=0; p<nlocal; p++) {
      PetscReal px,py,rx,ry,r2;
      
      rx = 2.0*rand()/((PetscReal)RAND_MAX) - 1.0;
      ry = 2.0*rand()/((PetscReal)RAND_MAX) - 1.0;
      
      px = LA_coors[2*p+0] + 0.1*rx*dx;
      py = LA_coors[2*p+1] + 0.1*ry*dx;
      
      r2 = px*px + py*py;
      if (r2 < 0.75*0.75) {
        array[bs*cnt+0] = px;
        array[bs*cnt+1] = py;
        cnt++;
      }
      
    }
    ierr = DMSwarmRestoreField(dms,DMSwarmPICField_coor,&bs,NULL,(void**)&array);CHKERRQ(ierr);
    ierr = VecRestoreArray(coors,&LA_coors);CHKERRQ(ierr);
    ierr = DMSwarmSetLocalSizes(dms,cnt,4);CHKERRQ(ierr);

    ierr = DMSwarmGetLocalSize(dms,&nlocal);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,"itag",&bs,NULL,(void**)&iarray);CHKERRQ(ierr);
    for (p=0; p<nlocal; p++) {
      iarray[p] = (PetscInt)rank;
    }
    ierr = DMSwarmRestoreField(dms,"itag",&bs,NULL,(void**)&iarray);CHKERRQ(ierr);
  }

  ierr = DMView(dms,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  ierr = SwarmViewGP(dms,"step0");CHKERRQ(ierr);

  dt = 0.1;
  for (tk=1; tk<20; tk++) {
    char prefix[PETSC_MAX_PATH_LEN];
    PetscPrintf(PETSC_COMM_WORLD,"Step %D \n",tk);
    /* push points */
    ierr = DMSwarmGetLocalSize(dms,&nlocal);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,DMSwarmPICField_coor,&bs,NULL,(void**)&array);CHKERRQ(ierr);
    for (p=0; p<nlocal; p++) {
      PetscReal cx,cy,vx,vy;
      
      cx = array[2*p];
      cy = array[2*p+1];
      vx =  cy;
      vy = -cx;
      
      array[2*p  ] += dt * vx;
      array[2*p+1] += dt * vy;
    }
    ierr = DMSwarmRestoreField(dms,DMSwarmPICField_coor,&bs,NULL,(void**)&array);CHKERRQ(ierr);

    /* migrate points */
    ierr = DMSwarmMigrate(dms,PETSC_TRUE);CHKERRQ(ierr);
    /*
    {
      const PetscInt *LA_iscell;
      IS iscell;
      Vec pos;
      PetscInt npoints;

      ierr = DMSwarmCreateGlobalVectorFromField(dms,DMSwarmPICField_coor,&pos);CHKERRQ(ierr);
      ierr = DMLocatePoints(dmcell,pos,&iscell);CHKERRQ(ierr);
      ierr = DMSwarmDestroyGlobalVectorFromField(dms,DMSwarmPICField_coor,&pos);CHKERRQ(ierr);
      ISView(iscell,PETSC_VIEWER_STDOUT_SELF);
      
      ierr = DMSwarmGetLocalSize(dms,&npoints);CHKERRQ(ierr);
      ierr = DMSwarmGetField(dms,"itag",NULL,NULL,(void**)&iarray);CHKERRQ(ierr);
      ierr = ISGetIndices(iscell,&LA_iscell);CHKERRQ(ierr);
      for (p=0; p<npoints; p++) {
        iarray[p] = LA_iscell[p];
      }
      ierr = ISRestoreIndices(iscell,&LA_iscell);CHKERRQ(ierr);
      ierr = DMSwarmRestoreField(dms,"itag",NULL,NULL,(void**)&iarray);CHKERRQ(ierr);
      ierr = ISDestroy(&iscell);CHKERRQ(ierr);
    }
    */
    /* view points */
    PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"step%d",tk);
    ierr = SwarmViewGP(dms,prefix);CHKERRQ(ierr);
  }
  
  
  ierr = DMDestroy(&dmregular);CHKERRQ(ierr);
  ierr = DMDestroy(&dmcell);CHKERRQ(ierr);
  ierr = DMDestroy(&dms);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = ex3_1();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
