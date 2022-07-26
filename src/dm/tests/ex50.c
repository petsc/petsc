static char help[] = "Test GLVis high-order support with DMDAs\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmplex.h>
#include <petscdt.h>

static PetscErrorCode MapPoint(PetscScalar xyz[],PetscScalar mxyz[])
{
  PetscScalar x,y,z,x2,y2,z2;

  x = xyz[0];
  y = xyz[1];
  z = xyz[2];
  x2 = x*x;
  y2 = y*y;
  z2 = z*z;
  mxyz[0] = x*PetscSqrtScalar(1.0-y2/2.0-z2/2.0 + y2*z2/3.0);
  mxyz[1] = y*PetscSqrtScalar(1.0-z2/2.0-x2/2.0 + z2*x2/3.0);
  mxyz[2] = z*PetscSqrtScalar(1.0-x2/2.0-y2/2.0 + x2*y2/3.0);
  return 0;
}

static PetscErrorCode test_3d(PetscInt cells[], PetscBool plex, PetscBool ho)
{
  DM             dm;
  Vec            v;
  PetscScalar    *c;
  PetscInt       nl,i;
  PetscReal      u[3] = {1.0,1.0,1.0}, l[3] = {-1.0,-1.0,-1.0};

  PetscFunctionBeginUser;
  if (ho) {
    u[0] = 2.*cells[0];
    u[1] = 2.*cells[1];
    u[2] = 2.*cells[2];
    l[0] = 0.;
    l[1] = 0.;
    l[2] = 0.;
  }
  if (plex) {
    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMSetFromOptions(dm));
  } else {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,cells[0]+1,cells[1]+1,cells[2]+1,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&dm));
  }
  PetscCall(DMSetUp(dm));
  if (!plex) {
    PetscCall(DMDASetUniformCoordinates(dm,l[0],u[0],l[1],u[1],l[2],u[2]));
  }
  if (ho) { /* each element mapped to a sphere */
    DM           cdm;
    Vec          cv;
    PetscSection cSec;
    DMDACoor3d   ***_coords;
    PetscScalar  shift[3],*cptr;
    PetscInt     nel,dof = 3,nex,ney,nez,gx = 0,gy = 0,gz = 0;
    PetscInt     i,j,k,pi,pj,pk;
    PetscReal    *nodes,*weights;
    char         name[256];

    PetscCall(PetscOptionsGetInt(NULL,NULL,"-order",&dof,NULL));
    dof += 1;

    PetscCall(PetscMalloc1(dof,&nodes));
    PetscCall(PetscMalloc1(dof,&weights));
    PetscCall(PetscDTGaussLobattoLegendreQuadrature(dof,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,nodes,weights));
    PetscCall(DMGetCoordinatesLocal(dm,&cv));
    PetscCall(DMGetCoordinateDM(dm,&cdm));
    if (plex) {
      PetscInt cEnd,cStart;

      PetscCall(DMPlexGetHeightStratum(dm,0,&cStart,&cEnd));
      PetscCall(DMGetCoordinateSection(dm,&cSec));
      nel  = cEnd - cStart;
      nex  = nel;
      ney  = 1;
      nez  = 1;
    } else {
      PetscCall(DMDAVecGetArray(cdm,cv,&_coords));
      PetscCall(DMDAGetElementsCorners(dm,&gx,&gy,&gz));
      PetscCall(DMDAGetElementsSizes(dm,&nex,&ney,&nez));
      nel  = nex*ney*nez;
    }
    PetscCall(VecCreate(PETSC_COMM_WORLD,&v));
    PetscCall(VecSetSizes(v,3*dof*dof*dof*nel,PETSC_DECIDE));
    PetscCall(VecSetType(v,VECSTANDARD));
    PetscCall(VecGetArray(v,&c));
    cptr = c;
    for (k=gz;k<gz+nez;k++) {
      for (j=gy;j<gy+ney;j++) {
        for (i=gx;i<gx+nex;i++) {
          if (plex) {
            PetscScalar *t = NULL;

            PetscCall(DMPlexVecGetClosure(dm,cSec,cv,i,NULL,&t));
            shift[0] = t[0];
            shift[1] = t[1];
            shift[2] = t[2];
            PetscCall(DMPlexVecRestoreClosure(dm,cSec,cv,i,NULL,&t));
          } else {
            shift[0] = _coords[k][j][i].x;
            shift[1] = _coords[k][j][i].y;
            shift[2] = _coords[k][j][i].z;
          }
          for (pk=0;pk<dof;pk++) {
            PetscScalar xyz[3];

            xyz[2] = nodes[pk];
            for (pj=0;pj<dof;pj++) {
              xyz[1] = nodes[pj];
              for (pi=0;pi<dof;pi++) {
                xyz[0] = nodes[pi];
                PetscCall(MapPoint(xyz,cptr));
                cptr[0] += shift[0];
                cptr[1] += shift[1];
                cptr[2] += shift[2];
                cptr += 3;
              }
            }
          }
        }
      }
    }
    if (!plex) {
      PetscCall(DMDAVecRestoreArray(cdm,cv,&_coords));
    }
    PetscCall(VecRestoreArray(v,&c));
    PetscCall(PetscSNPrintf(name,sizeof(name),"FiniteElementCollection: L2_T1_3D_P%" PetscInt_FMT,dof-1));
    PetscCall(PetscObjectSetName((PetscObject)v,name));
    PetscCall(PetscObjectCompose((PetscObject)dm,"_glvis_mesh_coords",(PetscObject)v));
    PetscCall(VecDestroy(&v));
    PetscCall(PetscFree(nodes));
    PetscCall(PetscFree(weights));
    PetscCall(DMViewFromOptions(dm,NULL,"-view"));
  } else { /* map the whole domain to a sphere */
    PetscCall(DMGetCoordinates(dm,&v));
    PetscCall(VecGetLocalSize(v,&nl));
    PetscCall(VecGetArray(v,&c));
    for (i=0;i<nl/3;i++) {
      PetscCall(MapPoint(c+3*i,c+3*i));
    }
    PetscCall(VecRestoreArray(v,&c));
    PetscCall(DMSetCoordinates(dm,v));
    PetscCall(DMViewFromOptions(dm,NULL,"-view"));
  }
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscBool      ho = PETSC_TRUE;
  PetscBool      plex = PETSC_FALSE;
  PetscInt       cells[3] = {2,2,2};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-ho",&ho,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-plex",&plex,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nex",&cells[0],NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ney",&cells[1],NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nez",&cells[2],NULL));
  PetscCall(test_3d(cells,plex,ho));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
     nsize: 1
     args: -view glvis:

     test:
        suffix: dmda_glvis_ho
        args: -nex 1 -ney 1 -nez 1

     test:
        suffix: dmda_glvis_lo
        args: -ho 0

     test:
        suffix: dmplex_glvis_ho
        args: -nex 1 -ney 1 -nez 1

     test:
        suffix: dmplex_glvis_lo
        args: -ho 0

TEST*/
