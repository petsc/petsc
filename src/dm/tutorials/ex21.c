
static char help[] = "DMSwarm-PIC demonstator of advecting points within cell DM defined by a DA or PLEX object \n\
Options: \n\
-ppcell   : Number of times to sub-divide the reference cell when layout the initial particle coordinates \n\
-meshtype : 0 ==> DA , 1 ==> PLEX \n\
-nt       : Number of timestep to perform \n\
-view     : Write out initial condition and time dependent data \n";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>

PetscErrorCode pic_advect(PetscInt ppcell,PetscInt meshtype)
{
  const PetscInt dim = 2;
  DM celldm,swarm;
  PetscInt tk,nt = 200;
  PetscBool view = PETSC_FALSE;
  Vec *pfields;
  PetscReal minradius;
  PetscReal dt;
  PetscReal vel[] = { 1.0, 0.16 };
  const char *fieldnames[] = { "phi" };
  PetscViewer viewer;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-view",&view,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL));

  /* Create the background cell DM */
  if (meshtype == 0) { /* DA */
    PetscInt nxy;
    PetscInt dof = 1;
    PetscInt stencil_width = 1;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Mesh type: DMDA\n"));
    nxy = 33;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,nxy,nxy,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,NULL,NULL,&celldm));

    PetscCall(DMDASetElementType(celldm,DMDA_ELEMENT_Q1));

    PetscCall(DMSetFromOptions(celldm));

    PetscCall(DMSetUp(celldm));

    PetscCall(DMDASetUniformCoordinates(celldm,0.0,1.0,0.0,1.0,0.0,1.5));

    minradius = 1.0/((PetscReal)(nxy-1));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"DA(minradius) %1.4e\n",(double)minradius));
  }

  if (meshtype == 1){ /* PLEX */
    DM distributedMesh = NULL;
    PetscInt numComp[] = {1};
    PetscInt numDof[] = {1,0,0}; /* vert, edge, cell */
    PetscInt faces[]  = {1,1,1};
    PetscInt numBC = 0;
    PetscSection section;
    Vec cellgeom = NULL;
    Vec facegeom = NULL;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Mesh type: DMPLEX\n"));
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_TRUE, faces, NULL, NULL, PETSC_TRUE, &celldm));

    /* Distribute mesh over processes */
    PetscCall(DMPlexDistribute(celldm,0,NULL,&distributedMesh));
    if (distributedMesh) {
      PetscCall(DMDestroy(&celldm));
      celldm = distributedMesh;
    }

    PetscCall(DMSetFromOptions(celldm));

    PetscCall(DMPlexCreateSection(celldm,NULL,numComp,numDof,numBC,NULL,NULL,NULL,NULL,&section));
    PetscCall(DMSetLocalSection(celldm,section));

    PetscCall(DMSetUp(celldm));

    /* Calling DMPlexComputeGeometryFVM() generates the value returned by DMPlexGetMinRadius() */
    PetscCall(DMPlexComputeGeometryFVM(celldm,&cellgeom,&facegeom));
    PetscCall(DMPlexGetMinRadius(celldm,&minradius));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"PLEX(minradius) %1.4e\n",(double)minradius));
    PetscCall(VecDestroy(&cellgeom));
    PetscCall(VecDestroy(&facegeom));
    PetscCall(PetscSectionDestroy(&section));
  }

  /* Create the DMSwarm */
  PetscCall(DMCreate(PETSC_COMM_WORLD,&swarm));
  PetscCall(DMSetType(swarm,DMSWARM));
  PetscCall(DMSetDimension(swarm,dim));

  /* Configure swarm to be of type PIC */
  PetscCall(DMSwarmSetType(swarm,DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(swarm,celldm));

  /* Register two scalar fields within the DMSwarm */
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"phi",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm,"region",1,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));

  /* Set initial local sizes of the DMSwarm with a buffer length of zero */
  PetscCall(DMSwarmSetLocalSizes(swarm,4,0));

  /* Insert swarm coordinates cell-wise */
  /*PetscCall(DMSwarmInsertPointsUsingCellDM(swarm,DMSWARMPIC_LAYOUT_REGULAR,ppcell));*/
  PetscCall(DMSwarmInsertPointsUsingCellDM(swarm,DMSWARMPIC_LAYOUT_SUBDIVISION,ppcell));

  /* Define initial conditions for th swarm fields "phi" and "region" */
  {
    PetscReal *s_coor,*s_phi,*s_region;
    PetscInt npoints,p;

    PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
    PetscCall(DMSwarmGetField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&s_coor));
    PetscCall(DMSwarmGetField(swarm,"phi",NULL,NULL,(void**)&s_phi));
    PetscCall(DMSwarmGetField(swarm,"region",NULL,NULL,(void**)&s_region));
    for (p=0; p<npoints; p++) {
      PetscReal pos[2];
      pos[0] = s_coor[2*p+0];
      pos[1] = s_coor[2*p+1];

      s_region[p] = 1.0;
      s_phi[p] = 1.0 + PetscExpReal(-200.0*((pos[0]-0.5)*(pos[0]-0.5) + (pos[1]-0.5)*(pos[1]-0.5)));
    }
    PetscCall(DMSwarmRestoreField(swarm,"region",NULL,NULL,(void**)&s_region));
    PetscCall(DMSwarmRestoreField(swarm,"phi",NULL,NULL,(void**)&s_phi));
    PetscCall(DMSwarmRestoreField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&s_coor));
  }

  /* Project initial value of phi onto the mesh */
  PetscCall(DMSwarmProjectFields(swarm,1,fieldnames,&pfields,PETSC_FALSE));

  if (view) {
    /* View swarm all swarm fields using data type PETSC_REAL */
    PetscCall(DMSwarmViewXDMF(swarm,"ic_dms.xmf"));

    /* View projected swarm field "phi" */
    PetscCall(PetscViewerCreate(PETSC_COMM_WORLD,&viewer));
    PetscCall(PetscViewerSetType(viewer,PETSCVIEWERVTK));
    PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
    if (meshtype == 0) { /* DA */
      PetscCall(PetscViewerFileSetName(viewer,"ic_dmda.vts"));
      PetscCall(VecView(pfields[0],viewer));
    }
    if (meshtype == 1) { /* PLEX */
      PetscCall(PetscViewerFileSetName(viewer,"ic_dmplex.vtk"));
      PetscCall(DMView(celldm,viewer));
      PetscCall(VecView(pfields[0],viewer));
    }
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(DMView(celldm,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMView(swarm,PETSC_VIEWER_STDOUT_WORLD));

  dt = 0.5 * minradius / PetscSqrtReal(vel[0]*vel[0] + vel[1]*vel[1]);
  for (tk=1; tk<=nt; tk++) {
    PetscReal *s_coor;
    PetscInt npoints,p;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[step %" PetscInt_FMT "]\n",tk));
    /* advect with analytic prescribed (constant) velocity field */
    PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
    PetscCall(DMSwarmGetField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&s_coor));
    for (p=0; p<npoints; p++) {
      s_coor[2*p+0] += dt * vel[0];
      s_coor[2*p+1] += dt * vel[1];
    }
    PetscCall(DMSwarmRestoreField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&s_coor));

    PetscCall(DMSwarmMigrate(swarm,PETSC_TRUE));

    /* Ad-hoc cell filling algorithm */
    /*
       The injection frequency is chosen for default DA case.
       They will likely not be appropriate for the general case using an unstructure PLEX mesh.
    */
    if (tk%10 == 0) {
      PetscReal dx = 1.0/32.0;
      PetscInt npoints_dir_x[] = { 32, 1 };
      PetscReal min[2],max[2];

      min[0] = 0.5 * dx;  max[0] = 0.5 * dx + 31.0 * dx;
      min[1] = 0.5 * dx;  max[1] = 0.5 * dx;
      PetscCall(DMSwarmSetPointsUniformCoordinates(swarm,min,max,npoints_dir_x,ADD_VALUES));
    }
    if (tk%2 == 0) {
      PetscReal dx = 1.0/32.0;
      PetscInt npoints_dir_y[] = { 2, 31 };
      PetscReal min[2],max[2];

      min[0] = 0.05 * dx; max[0] = 0.5 * dx;
      min[1] = 0.5 * dx;  max[1] = 0.5 * dx + 31.0 * dx;
      PetscCall(DMSwarmSetPointsUniformCoordinates(swarm,min,max,npoints_dir_y,ADD_VALUES));
    }

    /* Project swarm field "phi" onto the cell DM */
    PetscCall(DMSwarmProjectFields(swarm,1,fieldnames,&pfields,PETSC_TRUE));

    if (view) {
      PetscViewer viewer;
      char fname[PETSC_MAX_PATH_LEN];

      /* View swarm fields */
      PetscCall(PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"step%.4" PetscInt_FMT "_dms.xmf",tk));
      PetscCall(DMSwarmViewXDMF(swarm,fname));

      /* View projected field */
      PetscCall(PetscViewerCreate(PETSC_COMM_WORLD,&viewer));
      PetscCall(PetscViewerSetType(viewer,PETSCVIEWERVTK));
      PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));

      if (meshtype == 0) { /* DA */
        PetscCall(PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"step%.4" PetscInt_FMT "_dmda.vts",tk));
        PetscCall(PetscViewerFileSetName(viewer,fname));
        PetscCall(VecView(pfields[0],viewer));
      }
      if (meshtype == 1) { /* PLEX */
        PetscCall(PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"step%.4" PetscInt_FMT "_dmplex.vtk",tk));
        PetscCall(PetscViewerFileSetName(viewer,fname));
        PetscCall(DMView(celldm,viewer));
        PetscCall(VecView(pfields[0],viewer));
      }
      PetscCall(PetscViewerDestroy(&viewer));
    }

  }
  PetscCall(VecDestroy(&pfields[0]));
  PetscCall(PetscFree(pfields));
  PetscCall(DMDestroy(&celldm));
  PetscCall(DMDestroy(&swarm));

  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscInt ppcell = 1;
  PetscInt meshtype = 0;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ppcell",&ppcell,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-meshtype",&meshtype,NULL));
  PetscCheck(meshtype <= 1,PETSC_COMM_WORLD,PETSC_ERR_USER,"-meshtype <value> must be 0 or 1");

  PetscCall(pic_advect(ppcell,meshtype));

  PetscCall(PetscFinalize());
  return 0;
}
