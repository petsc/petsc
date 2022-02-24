static char help[] = "This example demonstrates the use of DMNetwork interface with subnetworks for solving a nonlinear electric power grid problem.\n\
                      The available solver options are in the poweroptions file and the data files are in the datafiles directory.\n\
                      The data file format used is from the MatPower package (http://www.pserc.cornell.edu//matpower/).\n\
                      This example shows the use of subnetwork feature in DMNetwork. It creates duplicates of the same network which are treated as subnetworks.\n\
                      Run this program: mpiexec -n <n> ./pf2\n\
                      mpiexec -n <n> ./pf2 \n";

/* T
   Concepts: DMNetwork
   Concepts: PETSc SNES solver
*/

#include "power.h"
#include <petscdmnetwork.h>

PetscErrorCode FormFunction_Subnet(DM networkdm,Vec localX, Vec localF,PetscInt nv,PetscInt ne,const PetscInt* vtx,const PetscInt* edges,void* appctx)
{
  UserCtx_Power     *User = (UserCtx_Power*)appctx;
  PetscInt          e,v,vfrom,vto;
  const PetscScalar *xarr;
  PetscScalar       *farr;
  PetscInt          offsetfrom,offsetto,offset;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(localX,&xarr));
  CHKERRQ(VecGetArray(localF,&farr));

  for (v=0; v<nv; v++) {
    PetscInt      i,j,key;
    PetscScalar   Vm;
    PetscScalar   Sbase = User->Sbase;
    VERTEX_Power  bus = NULL;
    GEN           gen;
    LOAD          load;
    PetscBool     ghostvtex;
    PetscInt      numComps;
    void*         component;

    CHKERRQ(DMNetworkIsGhostVertex(networkdm,vtx[v],&ghostvtex));
    CHKERRQ(DMNetworkGetNumComponents(networkdm,vtx[v],&numComps));
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vtx[v],ALL_COMPONENTS,&offset));
    for (j = 0; j < numComps; j++) {
      CHKERRQ(DMNetworkGetComponent(networkdm,vtx[v],j,&key,&component,NULL));
      if (key == 1) {
        PetscInt       nconnedges;
        const PetscInt *connedges;

        bus = (VERTEX_Power)(component);
        /* Handle reference bus constrained dofs */
        if (bus->ide == REF_BUS || bus->ide == ISOLATED_BUS) {
          farr[offset] = xarr[offset] - bus->va*PETSC_PI/180.0;
          farr[offset+1] = xarr[offset+1] - bus->vm;
          break;
        }

        if (!ghostvtex) {
          Vm = xarr[offset+1];

          /* Shunt injections */
          farr[offset] += Vm*Vm*bus->gl/Sbase;
          if (bus->ide != PV_BUS) farr[offset+1] += -Vm*Vm*bus->bl/Sbase;
        }

        CHKERRQ(DMNetworkGetSupportingEdges(networkdm,vtx[v],&nconnedges,&connedges));
        for (i=0; i < nconnedges; i++) {
          EDGE_Power     branch;
          PetscInt       keye;
          PetscScalar    Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt;
          const PetscInt *cone;
          PetscScalar    Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

          e = connedges[i];
          CHKERRQ(DMNetworkGetComponent(networkdm,e,0,&keye,(void**)&branch,NULL));
          if (!branch->status) continue;
          Gff = branch->yff[0];
          Bff = branch->yff[1];
          Gft = branch->yft[0];
          Bft = branch->yft[1];
          Gtf = branch->ytf[0];
          Btf = branch->ytf[1];
          Gtt = branch->ytt[0];
          Btt = branch->ytt[1];

          CHKERRQ(DMNetworkGetConnectedVertices(networkdm,e,&cone));
          vfrom = cone[0];
          vto   = cone[1];

          CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vfrom,ALL_COMPONENTS,&offsetfrom));
          CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vto,ALL_COMPONENTS,&offsetto));

          thetaf = xarr[offsetfrom];
          Vmf     = xarr[offsetfrom+1];
          thetat  = xarr[offsetto];
          Vmt     = xarr[offsetto+1];
          thetaft = thetaf - thetat;
          thetatf = thetat - thetaf;

          if (vfrom == vtx[v]) {
            farr[offsetfrom]   += Gff*Vmf*Vmf + Vmf*Vmt*(Gft*PetscCosScalar(thetaft) + Bft*PetscSinScalar(thetaft));
            farr[offsetfrom+1] += -Bff*Vmf*Vmf + Vmf*Vmt*(-Bft*PetscCosScalar(thetaft) + Gft*PetscSinScalar(thetaft));
          } else {
            farr[offsetto]   += Gtt*Vmt*Vmt + Vmt*Vmf*(Gtf*PetscCosScalar(thetatf) + Btf*PetscSinScalar(thetatf));
            farr[offsetto+1] += -Btt*Vmt*Vmt + Vmt*Vmf*(-Btf*PetscCosScalar(thetatf) + Gtf*PetscSinScalar(thetatf));
          }
        }
      } else if (key == 2) {
        if (!ghostvtex) {
          gen = (GEN)(component);
          if (!gen->status) continue;
          farr[offset] += -gen->pg/Sbase;
          farr[offset+1] += -gen->qg/Sbase;
        }
      } else if (key == 3) {
        if (!ghostvtex) {
          load = (LOAD)(component);
          farr[offset] += load->pl/Sbase;
          farr[offset+1] += load->ql/Sbase;
        }
      }
    }
    if (bus && bus->ide == PV_BUS) {
      farr[offset+1] = xarr[offset+1] - bus->vm;
    }
  }
  CHKERRQ(VecRestoreArrayRead(localX,&xarr));
  CHKERRQ(VecRestoreArray(localF,&farr));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction(SNES snes,Vec X, Vec F,void *appctx)
{
  DM             networkdm;
  Vec            localX,localF;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&networkdm));
  CHKERRQ(DMGetLocalVector(networkdm,&localX));
  CHKERRQ(DMGetLocalVector(networkdm,&localF));
  CHKERRQ(VecSet(F,0.0));

  CHKERRQ(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));

  CHKERRQ(DMGlobalToLocalBegin(networkdm,F,INSERT_VALUES,localF));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,F,INSERT_VALUES,localF));

  /* Form Function for first subnetwork */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));
  CHKERRQ(FormFunction_Subnet(networkdm,localX,localF,nv,ne,vtx,edges,appctx));

  /* Form Function for second subnetwork */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,1,&nv,&ne,&vtx,&edges));
  CHKERRQ(FormFunction_Subnet(networkdm,localX,localF,nv,ne,vtx,edges,appctx));

  CHKERRQ(DMRestoreLocalVector(networkdm,&localX));

  CHKERRQ(DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F));
  CHKERRQ(DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F));
  CHKERRQ(DMRestoreLocalVector(networkdm,&localF));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian_Subnet(DM networkdm,Vec localX, Mat J, Mat Jpre, PetscInt nv, PetscInt ne, const PetscInt *vtx, const PetscInt *edges, void *appctx)
{
  UserCtx_Power     *User=(UserCtx_Power*)appctx;
  PetscInt          e,v,vfrom,vto;
  const PetscScalar *xarr;
  PetscInt          offsetfrom,offsetto,goffsetfrom,goffsetto;
  PetscInt          row[2],col[8];
  PetscScalar       values[8];

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(localX,&xarr));

  for (v=0; v<nv; v++) {
    PetscInt    i,j,key;
    PetscInt    offset,goffset;
    PetscScalar Vm;
    PetscScalar Sbase=User->Sbase;
    VERTEX_Power bus;
    PetscBool   ghostvtex;
    PetscInt    numComps;
    void*       component;

    CHKERRQ(DMNetworkIsGhostVertex(networkdm,vtx[v],&ghostvtex));
    CHKERRQ(DMNetworkGetNumComponents(networkdm,vtx[v],&numComps));
    for (j = 0; j < numComps; j++) {
      CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vtx[v],ALL_COMPONENTS,&offset));
      CHKERRQ(DMNetworkGetGlobalVecOffset(networkdm,vtx[v],ALL_COMPONENTS,&goffset));
      CHKERRQ(DMNetworkGetComponent(networkdm,vtx[v],j,&key,&component,NULL));
      if (key == 1) {
        PetscInt       nconnedges;
        const PetscInt *connedges;

        bus = (VERTEX_Power)(component);
        if (!ghostvtex) {
          /* Handle reference bus constrained dofs */
          if (bus->ide == REF_BUS || bus->ide == ISOLATED_BUS) {
            row[0] = goffset; row[1] = goffset+1;
            col[0] = goffset; col[1] = goffset+1; col[2] = goffset; col[3] = goffset+1;
            values[0] = 1.0; values[1] = 0.0; values[2] = 0.0; values[3] = 1.0;
            CHKERRQ(MatSetValues(J,2,row,2,col,values,ADD_VALUES));
            break;
          }

          Vm = xarr[offset+1];

          /* Shunt injections */
          row[0] = goffset; row[1] = goffset+1;
          col[0] = goffset; col[1] = goffset+1;
          values[0] = values[1] = values[2] = values[3] = 0.0;
          if (bus->ide != PV_BUS) {
            values[1] = 2.0*Vm*bus->gl/Sbase;
            values[3] = -2.0*Vm*bus->bl/Sbase;
          }
          CHKERRQ(MatSetValues(J,2,row,2,col,values,ADD_VALUES));
        }

        CHKERRQ(DMNetworkGetSupportingEdges(networkdm,vtx[v],&nconnedges,&connedges));
        for (i=0; i < nconnedges; i++) {
          EDGE_Power       branch;
          VERTEX_Power     busf,bust;
          PetscInt       keyf,keyt;
          PetscScalar    Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt;
          const PetscInt *cone;
          PetscScalar    Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

          e = connedges[i];
          CHKERRQ(DMNetworkGetComponent(networkdm,e,0,&key,(void**)&branch,NULL));
          if (!branch->status) continue;

          Gff = branch->yff[0];
          Bff = branch->yff[1];
          Gft = branch->yft[0];
          Bft = branch->yft[1];
          Gtf = branch->ytf[0];
          Btf = branch->ytf[1];
          Gtt = branch->ytt[0];
          Btt = branch->ytt[1];

          CHKERRQ(DMNetworkGetConnectedVertices(networkdm,e,&cone));
          vfrom = cone[0];
          vto   = cone[1];

          CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vfrom,ALL_COMPONENTS,&offsetfrom));
          CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vto,ALL_COMPONENTS,&offsetto));
          CHKERRQ(DMNetworkGetGlobalVecOffset(networkdm,vfrom,ALL_COMPONENTS,&goffsetfrom));
          CHKERRQ(DMNetworkGetGlobalVecOffset(networkdm,vto,ALL_COMPONENTS,&goffsetto));

          if (goffsetto < 0) goffsetto = -goffsetto - 1;

          thetaf = xarr[offsetfrom];
          Vmf     = xarr[offsetfrom+1];
          thetat = xarr[offsetto];
          Vmt     = xarr[offsetto+1];
          thetaft = thetaf - thetat;
          thetatf = thetat - thetaf;

          CHKERRQ(DMNetworkGetComponent(networkdm,vfrom,0,&keyf,(void**)&busf,NULL));
          CHKERRQ(DMNetworkGetComponent(networkdm,vto,0,&keyt,(void**)&bust,NULL));

          if (vfrom == vtx[v]) {
            if (busf->ide != REF_BUS) {
              /*    farr[offsetfrom]   += Gff*Vmf*Vmf + Vmf*Vmt*(Gft*PetscCosScalar(thetaft) + Bft*PetscSinScalar(thetaft));  */
              row[0]  = goffsetfrom;
              col[0]  = goffsetfrom; col[1] = goffsetfrom+1; col[2] = goffsetto; col[3] = goffsetto+1;
              values[0] =  Vmf*Vmt*(Gft*-PetscSinScalar(thetaft) + Bft*PetscCosScalar(thetaft)); /* df_dthetaf */
              values[1] =  2.0*Gff*Vmf + Vmt*(Gft*PetscCosScalar(thetaft) + Bft*PetscSinScalar(thetaft)); /* df_dVmf */
              values[2] =  Vmf*Vmt*(Gft*PetscSinScalar(thetaft) + Bft*-PetscCosScalar(thetaft)); /* df_dthetat */
              values[3] =  Vmf*(Gft*PetscCosScalar(thetaft) + Bft*PetscSinScalar(thetaft)); /* df_dVmt */

              CHKERRQ(MatSetValues(J,1,row,4,col,values,ADD_VALUES));
            }
            if (busf->ide != PV_BUS && busf->ide != REF_BUS) {
              row[0] = goffsetfrom+1;
              col[0]  = goffsetfrom; col[1] = goffsetfrom+1; col[2] = goffsetto; col[3] = goffsetto+1;
              /*    farr[offsetfrom+1] += -Bff*Vmf*Vmf + Vmf*Vmt*(-Bft*PetscCosScalar(thetaft) + Gft*PetscSinScalar(thetaft)); */
              values[0] =  Vmf*Vmt*(Bft*PetscSinScalar(thetaft) + Gft*PetscCosScalar(thetaft));
              values[1] =  -2.0*Bff*Vmf + Vmt*(-Bft*PetscCosScalar(thetaft) + Gft*PetscSinScalar(thetaft));
              values[2] =  Vmf*Vmt*(-Bft*PetscSinScalar(thetaft) + Gft*-PetscCosScalar(thetaft));
              values[3] =  Vmf*(-Bft*PetscCosScalar(thetaft) + Gft*PetscSinScalar(thetaft));

              CHKERRQ(MatSetValues(J,1,row,4,col,values,ADD_VALUES));
            }
          } else {
            if (bust->ide != REF_BUS) {
              row[0] = goffsetto;
              col[0] = goffsetto; col[1] = goffsetto+1; col[2] = goffsetfrom; col[3] = goffsetfrom+1;
              /*    farr[offsetto]   += Gtt*Vmt*Vmt + Vmt*Vmf*(Gtf*PetscCosScalar(thetatf) + Btf*PetscSinScalar(thetatf)); */
              values[0] =  Vmt*Vmf*(Gtf*-PetscSinScalar(thetatf) + Btf*PetscCosScalar(thetaft)); /* df_dthetat */
              values[1] =  2.0*Gtt*Vmt + Vmf*(Gtf*PetscCosScalar(thetatf) + Btf*PetscSinScalar(thetatf)); /* df_dVmt */
              values[2] =  Vmt*Vmf*(Gtf*PetscSinScalar(thetatf) + Btf*-PetscCosScalar(thetatf)); /* df_dthetaf */
              values[3] =  Vmt*(Gtf*PetscCosScalar(thetatf) + Btf*PetscSinScalar(thetatf)); /* df_dVmf */

              CHKERRQ(MatSetValues(J,1,row,4,col,values,ADD_VALUES));
            }
            if (bust->ide != PV_BUS && bust->ide != REF_BUS) {
              row[0] = goffsetto+1;
              col[0] = goffsetto; col[1] = goffsetto+1; col[2] = goffsetfrom; col[3] = goffsetfrom+1;
              /*    farr[offsetto+1] += -Btt*Vmt*Vmt + Vmt*Vmf*(-Btf*PetscCosScalar(thetatf) + Gtf*PetscSinScalar(thetatf)); */
              values[0] =  Vmt*Vmf*(Btf*PetscSinScalar(thetatf) + Gtf*PetscCosScalar(thetatf));
              values[1] =  -2.0*Btt*Vmt + Vmf*(-Btf*PetscCosScalar(thetatf) + Gtf*PetscSinScalar(thetatf));
              values[2] =  Vmt*Vmf*(-Btf*PetscSinScalar(thetatf) + Gtf*-PetscCosScalar(thetatf));
              values[3] =  Vmt*(-Btf*PetscCosScalar(thetatf) + Gtf*PetscSinScalar(thetatf));

              CHKERRQ(MatSetValues(J,1,row,4,col,values,ADD_VALUES));
            }
          }
        }
        if (!ghostvtex && bus->ide == PV_BUS) {
          row[0] = goffset+1; col[0] = goffset+1;
          values[0]  = 1.0;
          CHKERRQ(MatSetValues(J,1,row,1,col,values,ADD_VALUES));
        }
      }
    }
  }
  CHKERRQ(VecRestoreArrayRead(localX,&xarr));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian(SNES snes,Vec X, Mat J,Mat Jpre,void *appctx)
{
  DM             networkdm;
  Vec            localX;
  PetscInt       ne,nv;
  const PetscInt *vtx,*edges;

  PetscFunctionBegin;
  CHKERRQ(MatZeroEntries(J));

  CHKERRQ(SNESGetDM(snes,&networkdm));
  CHKERRQ(DMGetLocalVector(networkdm,&localX));

  CHKERRQ(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));

  /* Form Jacobian for first subnetwork */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));
  CHKERRQ(FormJacobian_Subnet(networkdm,localX,J,Jpre,nv,ne,vtx,edges,appctx));

  /* Form Jacobian for second subnetwork */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,1,&nv,&ne,&vtx,&edges));
  CHKERRQ(FormJacobian_Subnet(networkdm,localX,J,Jpre,nv,ne,vtx,edges,appctx));

  CHKERRQ(DMRestoreLocalVector(networkdm,&localX));

  CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialValues_Subnet(DM networkdm,Vec localX,PetscInt nv,PetscInt ne, const PetscInt *vtx, const PetscInt *edges,void* appctx)
{
  VERTEX_Power   bus;
  PetscInt       i;
  GEN            gen;
  PetscBool      ghostvtex;
  PetscScalar    *xarr;
  PetscInt       key,numComps,j,offset;
  void*          component;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(localX,&xarr));
  for (i = 0; i < nv; i++) {
    CHKERRQ(DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex));
    if (ghostvtex) continue;

    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vtx[i],ALL_COMPONENTS,&offset));
    CHKERRQ(DMNetworkGetNumComponents(networkdm,vtx[i],&numComps));
    for (j=0; j < numComps; j++) {
      CHKERRQ(DMNetworkGetComponent(networkdm,vtx[i],j,&key,&component,NULL));
      if (key == 1) {
        bus = (VERTEX_Power)(component);
        xarr[offset] = bus->va*PETSC_PI/180.0;
        xarr[offset+1] = bus->vm;
      } else if (key == 2) {
        gen = (GEN)(component);
        if (!gen->status) continue;
        xarr[offset+1] = gen->vs;
        break;
      }
    }
  }
  CHKERRQ(VecRestoreArray(localX,&xarr));
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialValues(DM networkdm, Vec X,void* appctx)
{
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;
  Vec            localX;

  PetscFunctionBegin;
  CHKERRQ(DMGetLocalVector(networkdm,&localX));

  CHKERRQ(VecSet(X,0.0));
  CHKERRQ(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));

  /* Set initial guess for first subnetwork */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));
  CHKERRQ(SetInitialValues_Subnet(networkdm,localX,nv,ne,vtx,edges,appctx));

  /* Set initial guess for second subnetwork */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,1,&nv,&ne,&vtx,&edges));
  CHKERRQ(SetInitialValues_Subnet(networkdm,localX,nv,ne,vtx,edges,appctx));

  CHKERRQ(DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X));
  CHKERRQ(DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X));
  CHKERRQ(DMRestoreLocalVector(networkdm,&localX));
  PetscFunctionReturn(0);
}

int main(int argc,char ** argv)
{
  PetscErrorCode   ierr;
  char             pfdata_file[PETSC_MAX_PATH_LEN]="case9.m";
  PFDATA           *pfdata1,*pfdata2;
  PetscInt         numEdges1=0,numEdges2=0;
  PetscInt         *edgelist1 = NULL,*edgelist2 = NULL,componentkey[4];
  DM               networkdm;
  UserCtx_Power    User;
#if defined(PETSC_USE_LOG)
  PetscLogStage    stage1,stage2;
#endif
  PetscMPIInt      rank;
  PetscInt         nsubnet = 2,nv,ne,i,j,genj,loadj;
  const PetscInt   *vtx,*edges;
  Vec              X,F;
  Mat              J;
  SNES             snes;

  ierr = PetscInitialize(&argc,&argv,"poweroptions",help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  {
    /* introduce the const crank so the clang static analyzer realizes that if it enters any of the if (crank) then it must have entered the first */
    /* this is an experiment to see how the analyzer reacts */
    const PetscMPIInt crank = rank;

    /* Create an empty network object */
    CHKERRQ(DMNetworkCreate(PETSC_COMM_WORLD,&networkdm));

    /* Register the components in the network */
    CHKERRQ(DMNetworkRegisterComponent(networkdm,"branchstruct",sizeof(struct _p_EDGE_Power),&componentkey[0]));
    CHKERRQ(DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEX_Power),&componentkey[1]));
    CHKERRQ(DMNetworkRegisterComponent(networkdm,"genstruct",sizeof(struct _p_GEN),&componentkey[2]));
    CHKERRQ(DMNetworkRegisterComponent(networkdm,"loadstruct",sizeof(struct _p_LOAD),&componentkey[3]));

    CHKERRQ(PetscLogStageRegister("Read Data",&stage1));
    PetscLogStagePush(stage1);
    /* READ THE DATA */
    if (!crank) {
      /* Only rank 0 reads the data */
      CHKERRQ(PetscOptionsGetString(NULL,NULL,"-pfdata",pfdata_file,sizeof(pfdata_file),NULL));
      /* HERE WE CREATE COPIES OF THE SAME NETWORK THAT WILL BE TREATED AS SUBNETWORKS */

      /*    READ DATA FOR THE FIRST SUBNETWORK */
      CHKERRQ(PetscNew(&pfdata1));
      CHKERRQ(PFReadMatPowerData(pfdata1,pfdata_file));
      User.Sbase = pfdata1->sbase;

      numEdges1 = pfdata1->nbranch;
      CHKERRQ(PetscMalloc1(2*numEdges1,&edgelist1));
      CHKERRQ(GetListofEdges_Power(pfdata1,edgelist1));

      /*    READ DATA FOR THE SECOND SUBNETWORK */
      CHKERRQ(PetscNew(&pfdata2));
      CHKERRQ(PFReadMatPowerData(pfdata2,pfdata_file));
      User.Sbase = pfdata2->sbase;

      numEdges2 = pfdata2->nbranch;
      CHKERRQ(PetscMalloc1(2*numEdges2,&edgelist2));
      CHKERRQ(GetListofEdges_Power(pfdata2,edgelist2));
    }

    PetscLogStagePop();
    CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
    CHKERRQ(PetscLogStageRegister("Create network",&stage2));
    PetscLogStagePush(stage2);

    /* Set number of nodes/edges and edge connectivity */
    CHKERRQ(DMNetworkSetNumSubNetworks(networkdm,PETSC_DECIDE,nsubnet));
    CHKERRQ(DMNetworkAddSubnetwork(networkdm,"",numEdges1,edgelist1,NULL));
    CHKERRQ(DMNetworkAddSubnetwork(networkdm,"",numEdges2,edgelist2,NULL));

    /* Set up the network layout */
    CHKERRQ(DMNetworkLayoutSetUp(networkdm));

    /* Add network components only process 0 has any data to add*/
    if (!crank) {
      genj=0; loadj=0;

      /* ADD VARIABLES AND COMPONENTS FOR THE FIRST SUBNETWORK */
      CHKERRQ(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));

      for (i = 0; i < ne; i++) {
        CHKERRQ(DMNetworkAddComponent(networkdm,edges[i],componentkey[0],&pfdata1->branch[i],0));
      }

      for (i = 0; i < nv; i++) {
        CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],componentkey[1],&pfdata1->bus[i],2));
        if (pfdata1->bus[i].ngen) {
          for (j = 0; j < pfdata1->bus[i].ngen; j++) {
            CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],componentkey[2],&pfdata1->gen[genj++],0));
          }
        }
        if (pfdata1->bus[i].nload) {
          for (j=0; j < pfdata1->bus[i].nload; j++) {
            CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],componentkey[3],&pfdata1->load[loadj++],0));
          }
        }
      }

      genj=0; loadj=0;

      /* ADD VARIABLES AND COMPONENTS FOR THE SECOND SUBNETWORK */
      CHKERRQ(DMNetworkGetSubnetwork(networkdm,1,&nv,&ne,&vtx,&edges));

      for (i = 0; i < ne; i++) {
        CHKERRQ(DMNetworkAddComponent(networkdm,edges[i],componentkey[0],&pfdata2->branch[i],0));
      }

      for (i = 0; i < nv; i++) {
        CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],componentkey[1],&pfdata2->bus[i],2));
        if (pfdata2->bus[i].ngen) {
          for (j = 0; j < pfdata2->bus[i].ngen; j++) {
            CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],componentkey[2],&pfdata2->gen[genj++],0));
          }
        }
        if (pfdata2->bus[i].nload) {
          for (j=0; j < pfdata2->bus[i].nload; j++) {
            CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],componentkey[3],&pfdata2->load[loadj++],0));
          }
        }
      }
    }

    /* Set up DM for use */
    CHKERRQ(DMSetUp(networkdm));

    if (!crank) {
      CHKERRQ(PetscFree(edgelist1));
      CHKERRQ(PetscFree(edgelist2));
    }

    if (!crank) {
      CHKERRQ(PetscFree(pfdata1->bus));
      CHKERRQ(PetscFree(pfdata1->gen));
      CHKERRQ(PetscFree(pfdata1->branch));
      CHKERRQ(PetscFree(pfdata1->load));
      CHKERRQ(PetscFree(pfdata1));

      CHKERRQ(PetscFree(pfdata2->bus));
      CHKERRQ(PetscFree(pfdata2->gen));
      CHKERRQ(PetscFree(pfdata2->branch));
      CHKERRQ(PetscFree(pfdata2->load));
      CHKERRQ(PetscFree(pfdata2));
    }

    /* Distribute networkdm to multiple processes */
    CHKERRQ(DMNetworkDistribute(&networkdm,0));

    PetscLogStagePop();

    /* Broadcast Sbase to all processors */
    CHKERRMPI(MPI_Bcast(&User.Sbase,1,MPIU_SCALAR,0,PETSC_COMM_WORLD));

    CHKERRQ(DMCreateGlobalVector(networkdm,&X));
    CHKERRQ(VecDuplicate(X,&F));

    CHKERRQ(DMCreateMatrix(networkdm,&J));
    CHKERRQ(MatSetOption(J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

    CHKERRQ(SetInitialValues(networkdm,X,&User));

    /* HOOK UP SOLVER */
    CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));
    CHKERRQ(SNESSetDM(snes,networkdm));
    CHKERRQ(SNESSetFunction(snes,F,FormFunction,&User));
    CHKERRQ(SNESSetJacobian(snes,J,J,FormJacobian,&User));
    CHKERRQ(SNESSetFromOptions(snes));

    CHKERRQ(SNESSolve(snes,NULL,X));
    /* CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD)); */

    CHKERRQ(VecDestroy(&X));
    CHKERRQ(VecDestroy(&F));
    CHKERRQ(MatDestroy(&J));

    CHKERRQ(SNESDestroy(&snes));
    CHKERRQ(DMDestroy(&networkdm));
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     depends: PFReadData.c pffunctions.c
     requires: !complex double defined(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
     args: -snes_rtol 1.e-3
     localrunfiles: poweroptions case9.m
     output_file: output/power_1.out

   test:
     suffix: 2
     args: -snes_rtol 1.e-3 -petscpartitioner_type simple
     nsize: 4
     localrunfiles: poweroptions case9.m
     output_file: output/power_1.out

TEST*/
