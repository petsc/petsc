static char help[] = "Tests mesh adaptation with DMPlex and pragmatic.\n";

#include <petsc/private/dmpleximpl.h> 


typedef struct {
  DM            dm;
  /* Definition of the test case (mesh and metric field) */                           
  PetscInt      dim;                          /* The topological mesh dimension */
  char          mshNam[PETSC_MAX_PATH_LEN];   /* Name of the mesh filename if any */
  PetscInt      nbrVerEdge;                   /* Number of vertices per edge if unit square/cube generated */
  PetscBool     noBdyTags;                    /* Do not write boundary tags on generated unit square/cube */
  char          bdyLabel[PETSC_MAX_PATH_LEN]; /* Name of the label marking boundary facets */
  PetscInt      metOpt;                       /* Different choices of metric */
  PetscReal     hmax, hmin;                   /* Max and min sizes prescribed by the metric */
  PetscBool     vtkView;                      /* Write adapted mesh to vtk file */
} AppCtx;    



#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{ 
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim        = 2;
  ierr = PetscStrcpy(options->mshNam, "");CHKERRQ(ierr);
  options->nbrVerEdge = 5;
  options->noBdyTags  = PETSC_FALSE;
  ierr = PetscStrcpy(options->bdyLabel, "");CHKERRQ(ierr);
  options->metOpt     = 1;
  options->hmin       = 0.05;
  options->hmax       = 0.5;
  options->vtkView    = PETSC_FALSE;
  
  ierr = PetscOptionsBegin(comm, "", "Meshing Adaptation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex19.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-msh", "Name of the mesh filename if any", "ex19.c", options->mshNam, options->mshNam, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nbrVerEdge", "Number of vertices per edge if unit square/cube generated", "ex19.c", options->nbrVerEdge, &options->nbrVerEdge, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-noBdyTags", "Do not write boundary tags on generated unit square", "ex19.c", options->noBdyTags, &options->noBdyTags, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-bdyLabel", "Name of the label marking boundary facets", "ex19.c", options->bdyLabel, options->bdyLabel, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-met", "Different choices of metric", "ex19.c", options->metOpt, &options->metOpt, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hmax", "Max size prescribed by the metric", "ex19.c", options->hmax, &options->hmax, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hmin", "Min size prescribed by the metric", "ex19.c", options->hmin, &options->hmin, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-vtkView", "Write adapted mesh to vtk file", "ex19.c", options->vtkView, &options->vtkView, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  
  PetscFunctionReturn(0);
};



#undef __FUNCT__
#define __FUNCT__ "createBoxMesh"
/* same as DMPlexCreateBoxMesh but the number of vertices per edge is set byt the user */
PetscErrorCode createBoxMesh(MPI_Comm comm, AppCtx *user)  //PetscInt dim, PetscInt edge, PetscBool bdyMarkers, PetscBool interpolate, DM *dm)
{
  DM             boundary;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidPointer(&user->dm, 4);
  ierr = DMCreate(comm, &boundary);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(boundary, user->dim,2);
  ierr = DMSetType(boundary, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(boundary, user->dim-1);CHKERRQ(ierr);

  
  switch (user->dim) {
  case 2:
  {
    if (user->noBdyTags) {
      ierr = PetscOptionsSetValue(((PetscObject)(&user->dm))->options, "-dm_plex_separate_marker", "0");CHKERRQ(ierr);  
    } else {
      ierr = PetscOptionsSetValue(((PetscObject)(&user->dm))->options, "-dm_plex_separate_marker", "1");CHKERRQ(ierr);
    }
    
    PetscReal lower[2] = {0.0, 0.0};
    PetscReal upper[2] = {1.0, 1.0};
    PetscInt  edges[2] = {user->nbrVerEdge-1, user->nbrVerEdge-1};

    ierr = DMPlexCreateSquareBoundary(boundary, lower, upper, edges);CHKERRQ(ierr);
    break;
  }
  case 3:
  {
    PetscReal lower[3] = {0.0, 0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0, 1.0};
    PetscInt  faces[3] = {user->nbrVerEdge-1, user->nbrVerEdge-1, user->nbrVerEdge-1};

    ierr = DMPlexCreateCubeBoundary(boundary, lower, upper, faces);CHKERRQ(ierr);
    break;
  }
  default:
    SETERRQ1(comm, PETSC_ERR_SUP, "Dimension not supported: %d", user->dim);
  }
  ierr = DMPlexGenerate(boundary, NULL, PETSC_TRUE, &user->dm);CHKERRQ(ierr);
  ierr = DMDestroy(&boundary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user) 
{
  PetscBool      flag;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscStrcmp(user->mshNam, "", &flag);
  if (flag) {
    ierr = createBoxMesh(comm, user);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->mshNam, PETSC_TRUE, &user->dm);CHKERRQ(ierr);
    ierr = DMGetDimension(user->dm, &user->dim);;CHKERRQ(ierr); 
  }  
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "WriteMetric"
PetscErrorCode WriteMetric(MPI_Comm comm, AppCtx *user, Vec * metric) 
{  
  Vec               coordinates, met;
  const PetscScalar *coords;
  PetscReal         h, lambda1, lambda2, lambda3, lbd, lmax;
  PetscInt          dim, vStart, vEnd, numVertices, i;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  dim = user->dim;
  ierr = DMGetCoordinatesLocal(user->dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(user->dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  numVertices = vEnd - vStart;
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  
  ierr = VecCreate(comm, &met);CHKERRQ(ierr);
  ierr = VecSetSizes(met, PETSC_DECIDE, numVertices*dim*dim);CHKERRQ(ierr);
  ierr = VecSetFromOptions(met);CHKERRQ(ierr);
  
  for (i=0; i<numVertices; ++i) {
    switch (user->metOpt) {
    case 0:
      lbd = 1/(user->hmax*user->hmax);
      lambda1 = lambda2 = lambda3 = lbd;
      break;
    case 1:
      h = user->hmax - (user->hmax-user->hmin)*coords[dim*i];
      h = h*h;
      lmax = 1/(user->hmax*user->hmax);
      lambda1 = 1/h;
      lambda2 = lmax;
      lambda3 = lmax;
      break;
    case 2:
      h = user->hmax*fabs(1-exp(-fabs(coords[dim*i]-0.5))) + user->hmin;
      lbd = 1/(h*h);
      lmax = 1/(user->hmax*user->hmax);
      lambda1 = lbd;
      lambda2 = lmax;
      lambda3 = lmax;
      break;  
    default:
      SETERRQ1(PetscObjectComm((PetscObject) user->dm), PETSC_ERR_ARG_WRONG, "metOpt = 0, 1 or 2, cannot be %d", user->metOpt);
    } 
    if (dim == 2) {
      ierr = VecSetValue(met, 4*i  , lambda1, INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(met, 4*i+1, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(met, 4*i+2, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(met, 4*i+3, lambda2, INSERT_VALUES);CHKERRQ(ierr);
    }
    else {
      ierr = VecSetValue(met, 9*i  , lambda1, INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(met, 9*i+1, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(met, 9*i+2, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(met, 9*i+3, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(met, 9*i+4, lambda2, INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(met, 9*i+5, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(met, 9*i+6, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(met, 9*i+7, 0      , INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(met, 9*i+8, lambda3, INSERT_VALUES);CHKERRQ(ierr);
    }    
  }

  ierr = VecAssemblyBegin(met);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(met); CHKERRQ(ierr); 
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  *metric = met;
  PetscFunctionReturn(0);
}
                                 


#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc, char * argv[]) {
    
  AppCtx            user;                 /* user-defined work context */
  MPI_Comm          comm;
  DM                dma;
  Vec               metric;
  PetscErrorCode    ierr;
  PetscViewer       viewer;  

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
    
  ierr = CreateMesh(comm, &user);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user.dm, "DMinit");CHKERRQ(ierr);
  ierr = DMView(user.dm,0);CHKERRQ(ierr);
  ierr = WriteMetric(comm, &user, &metric);CHKERRQ(ierr);
  
  ierr = DMPlexAdapt(user.dm, metric, user.bdyLabel, &dma); CHKERRQ(ierr);
  
  ierr = PetscObjectSetName((PetscObject)dma, "DMadapt");CHKERRQ(ierr);
  ierr = DMView(dma,0);CHKERRQ(ierr);
  if (user.vtkView){
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "mesha.vtk");CHKERRQ(ierr);
    ierr = DMView(dma, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
   
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dma);CHKERRQ(ierr);
  ierr = VecDestroy(&metric);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
  
}