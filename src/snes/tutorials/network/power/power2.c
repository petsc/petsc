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
  PetscErrorCode    ierr;
  UserCtx_Power     *User = (UserCtx_Power*)appctx;
  PetscInt          e,v,vfrom,vto;
  const PetscScalar *xarr;
  PetscScalar       *farr;
  PetscInt          offsetfrom,offsetto,offset;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

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

    ierr = DMNetworkIsGhostVertex(networkdm,vtx[v],&ghostvtex);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,vtx[v],&numComps);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm,vtx[v],&offset);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = DMNetworkGetComponent(networkdm,vtx[v],j,&key,&component);CHKERRQ(ierr);
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

        ierr = DMNetworkGetSupportingEdges(networkdm,vtx[v],&nconnedges,&connedges);CHKERRQ(ierr);
        for (i=0; i < nconnedges; i++) {
          EDGE_Power     branch;
          PetscInt       keye;
          PetscScalar    Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt;
          const PetscInt *cone;
          PetscScalar    Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

          e = connedges[i];
          ierr = DMNetworkGetComponent(networkdm,e,0,&keye,(void**)&branch);CHKERRQ(ierr);
          if (!branch->status) continue;
          Gff = branch->yff[0];
          Bff = branch->yff[1];
          Gft = branch->yft[0];
          Bft = branch->yft[1];
          Gtf = branch->ytf[0];
          Btf = branch->ytf[1];
          Gtt = branch->ytt[0];
          Btt = branch->ytt[1];

          ierr = DMNetworkGetConnectedVertices(networkdm,e,&cone);CHKERRQ(ierr);
          vfrom = cone[0];
          vto   = cone[1];

          ierr = DMNetworkGetVariableOffset(networkdm,vfrom,&offsetfrom);CHKERRQ(ierr);
          ierr = DMNetworkGetVariableOffset(networkdm,vto,&offsetto);CHKERRQ(ierr);

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
  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode FormFunction(SNES snes,Vec X, Vec F,void *appctx)
{
  PetscErrorCode ierr;
  DM             networkdm;
  Vec            localX,localF;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localF);CHKERRQ(ierr);
  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,F,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,F,INSERT_VALUES,localF);CHKERRQ(ierr);

  /* Form Function for first subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = FormFunction_Subnet(networkdm,localX,localF,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  /* Form Function for second subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = FormFunction_Subnet(networkdm,localX,localF,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian_Subnet(DM networkdm,Vec localX, Mat J, Mat Jpre, PetscInt nv, PetscInt ne, const PetscInt *vtx, const PetscInt *edges, void *appctx)
{
  PetscErrorCode    ierr;
  UserCtx_Power     *User=(UserCtx_Power*)appctx;
  PetscInt          e,v,vfrom,vto;
  const PetscScalar *xarr;
  PetscInt          offsetfrom,offsetto,goffsetfrom,goffsetto;
  PetscInt          row[2],col[8];
  PetscScalar       values[8];

  PetscFunctionBegin;
  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);

  for (v=0; v<nv; v++) {
    PetscInt    i,j,key;
    PetscInt    offset,goffset;
    PetscScalar Vm;
    PetscScalar Sbase=User->Sbase;
    VERTEX_Power bus;
    PetscBool   ghostvtex;
    PetscInt    numComps;
    void*       component;

    ierr = DMNetworkIsGhostVertex(networkdm,vtx[v],&ghostvtex);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,vtx[v],&numComps);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = DMNetworkGetVariableOffset(networkdm,vtx[v],&offset);CHKERRQ(ierr);
      ierr = DMNetworkGetVariableGlobalOffset(networkdm,vtx[v],&goffset);CHKERRQ(ierr);
      ierr = DMNetworkGetComponent(networkdm,vtx[v],j,&key,&component);CHKERRQ(ierr);
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
            ierr = MatSetValues(J,2,row,2,col,values,ADD_VALUES);CHKERRQ(ierr);
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
          ierr = MatSetValues(J,2,row,2,col,values,ADD_VALUES);CHKERRQ(ierr);
        }

        ierr = DMNetworkGetSupportingEdges(networkdm,vtx[v],&nconnedges,&connedges);CHKERRQ(ierr);
        for (i=0; i < nconnedges; i++) {
          EDGE_Power       branch;
          VERTEX_Power     busf,bust;
          PetscInt       keyf,keyt;
          PetscScalar    Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt;
          const PetscInt *cone;
          PetscScalar    Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

          e = connedges[i];
          ierr = DMNetworkGetComponent(networkdm,e,0,&key,(void**)&branch);CHKERRQ(ierr);
          if (!branch->status) continue;

          Gff = branch->yff[0];
          Bff = branch->yff[1];
          Gft = branch->yft[0];
          Bft = branch->yft[1];
          Gtf = branch->ytf[0];
          Btf = branch->ytf[1];
          Gtt = branch->ytt[0];
          Btt = branch->ytt[1];

          ierr = DMNetworkGetConnectedVertices(networkdm,e,&cone);CHKERRQ(ierr);
          vfrom = cone[0];
          vto   = cone[1];

          ierr = DMNetworkGetVariableOffset(networkdm,vfrom,&offsetfrom);CHKERRQ(ierr);
          ierr = DMNetworkGetVariableOffset(networkdm,vto,&offsetto);CHKERRQ(ierr);
          ierr = DMNetworkGetVariableGlobalOffset(networkdm,vfrom,&goffsetfrom);CHKERRQ(ierr);
          ierr = DMNetworkGetVariableGlobalOffset(networkdm,vto,&goffsetto);CHKERRQ(ierr);

          if (goffsetto < 0) goffsetto = -goffsetto - 1;

          thetaf = xarr[offsetfrom];
          Vmf     = xarr[offsetfrom+1];
          thetat = xarr[offsetto];
          Vmt     = xarr[offsetto+1];
          thetaft = thetaf - thetat;
          thetatf = thetat - thetaf;

          ierr = DMNetworkGetComponent(networkdm,vfrom,0,&keyf,(void**)&busf);CHKERRQ(ierr);
          ierr = DMNetworkGetComponent(networkdm,vto,0,&keyt,(void**)&bust);CHKERRQ(ierr);

          if (vfrom == vtx[v]) {
            if (busf->ide != REF_BUS) {
              /*    farr[offsetfrom]   += Gff*Vmf*Vmf + Vmf*Vmt*(Gft*PetscCosScalar(thetaft) + Bft*PetscSinScalar(thetaft));  */
              row[0]  = goffsetfrom;
              col[0]  = goffsetfrom; col[1] = goffsetfrom+1; col[2] = goffsetto; col[3] = goffsetto+1;
              values[0] =  Vmf*Vmt*(Gft*-PetscSinScalar(thetaft) + Bft*PetscCosScalar(thetaft)); /* df_dthetaf */
              values[1] =  2.0*Gff*Vmf + Vmt*(Gft*PetscCosScalar(thetaft) + Bft*PetscSinScalar(thetaft)); /* df_dVmf */
              values[2] =  Vmf*Vmt*(Gft*PetscSinScalar(thetaft) + Bft*-PetscCosScalar(thetaft)); /* df_dthetat */
              values[3] =  Vmf*(Gft*PetscCosScalar(thetaft) + Bft*PetscSinScalar(thetaft)); /* df_dVmt */

              ierr = MatSetValues(J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
            }
            if (busf->ide != PV_BUS && busf->ide != REF_BUS) {
              row[0] = goffsetfrom+1;
              col[0]  = goffsetfrom; col[1] = goffsetfrom+1; col[2] = goffsetto; col[3] = goffsetto+1;
              /*    farr[offsetfrom+1] += -Bff*Vmf*Vmf + Vmf*Vmt*(-Bft*PetscCosScalar(thetaft) + Gft*PetscSinScalar(thetaft)); */
              values[0] =  Vmf*Vmt*(Bft*PetscSinScalar(thetaft) + Gft*PetscCosScalar(thetaft));
              values[1] =  -2.0*Bff*Vmf + Vmt*(-Bft*PetscCosScalar(thetaft) + Gft*PetscSinScalar(thetaft));
              values[2] =  Vmf*Vmt*(-Bft*PetscSinScalar(thetaft) + Gft*-PetscCosScalar(thetaft));
              values[3] =  Vmf*(-Bft*PetscCosScalar(thetaft) + Gft*PetscSinScalar(thetaft));

              ierr = MatSetValues(J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
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

              ierr = MatSetValues(J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
            }
            if (bust->ide != PV_BUS && bust->ide != REF_BUS) {
              row[0] = goffsetto+1;
              col[0] = goffsetto; col[1] = goffsetto+1; col[2] = goffsetfrom; col[3] = goffsetfrom+1;
              /*    farr[offsetto+1] += -Btt*Vmt*Vmt + Vmt*Vmf*(-Btf*PetscCosScalar(thetatf) + Gtf*PetscSinScalar(thetatf)); */
              values[0] =  Vmt*Vmf*(Btf*PetscSinScalar(thetatf) + Gtf*PetscCosScalar(thetatf));
              values[1] =  -2.0*Btt*Vmt + Vmf*(-Btf*PetscCosScalar(thetatf) + Gtf*PetscSinScalar(thetatf));
              values[2] =  Vmt*Vmf*(-Btf*PetscSinScalar(thetatf) + Gtf*-PetscCosScalar(thetatf));
              values[3] =  Vmt*(-Btf*PetscCosScalar(thetatf) + Gtf*PetscSinScalar(thetatf));

              ierr = MatSetValues(J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
            }
          }
        }
        if (!ghostvtex && bus->ide == PV_BUS) {
          row[0] = goffset+1; col[0] = goffset+1;
          values[0]  = 1.0;
          ierr = MatSetValues(J,1,row,1,col,values,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian(SNES snes,Vec X, Mat J,Mat Jpre,void *appctx)
{
  PetscErrorCode ierr;
  DM             networkdm;
  Vec            localX;
  PetscInt       ne,nv;
  const PetscInt *vtx,*edges;

  PetscFunctionBegin;
  ierr = MatZeroEntries(J);CHKERRQ(ierr);

  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Form Jacobian for first subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = FormJacobian_Subnet(networkdm,localX,J,Jpre,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  /* Form Jacobian for second subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = FormJacobian_Subnet(networkdm,localX,J,Jpre,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialValues_Subnet(DM networkdm,Vec localX,PetscInt nv,PetscInt ne, const PetscInt *vtx, const PetscInt *edges,void* appctx)
{
  PetscErrorCode ierr;
  VERTEX_Power   bus;
  PetscInt       i;
  GEN            gen;
  PetscBool      ghostvtex;
  PetscScalar    *xarr;
  PetscInt       key,numComps,j,offset;
  void*          component;

  PetscFunctionBegin;
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  for (i = 0; i < nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;

    ierr = DMNetworkGetVariableOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,vtx[i],&numComps);CHKERRQ(ierr);
    for (j=0; j < numComps; j++) {
      ierr = DMNetworkGetComponent(networkdm,vtx[i],j,&key,&component);CHKERRQ(ierr);
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
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialValues(DM networkdm, Vec X,void* appctx)
{
  PetscErrorCode ierr;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Set initial guess for first subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = SetInitialValues_Subnet(networkdm,localX,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  /* Set initial guess for second subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = SetInitialValues_Subnet(networkdm,localX,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char ** argv)
{
  PetscErrorCode   ierr;
  char             pfdata_file[PETSC_MAX_PATH_LEN]="case9.m";
  PFDATA           *pfdata1,*pfdata2;
  PetscInt         numEdges1=0,numVertices1=0,numEdges2=0,numVertices2=0;
  PetscInt         *edgelist1 = NULL,*edgelist2 = NULL;
  DM               networkdm;
  PetscInt         componentkey[4];
  UserCtx_Power    User;
#if defined(PETSC_USE_LOG)
  PetscLogStage    stage1,stage2;
#endif
  PetscMPIInt      rank;
  PetscInt         nsubnet = 2;
  PetscInt         numVertices[2],numEdges[2];
  PetscInt         *edgelist[2];
  PetscInt         nv,ne;
  const PetscInt   *vtx;
  const PetscInt   *edges;
  PetscInt         i,j;
  PetscInt         genj,loadj;
  Vec              X,F;
  Mat              J;
  SNES             snes;


  ierr = PetscInitialize(&argc,&argv,"poweroptions",help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  {
    /* introduce the const crank so the clang static analyzer realizes that if it enters any of the if (crank) then it must have entered the first */
    /* this is an experiment to see how the analyzer reacts */
    const PetscMPIInt crank = rank;

    /* Create an empty network object */
    ierr = DMNetworkCreate(PETSC_COMM_WORLD,&networkdm);CHKERRQ(ierr);

    /* Register the components in the network */
    ierr = DMNetworkRegisterComponent(networkdm,"branchstruct",sizeof(struct _p_EDGE_Power),&componentkey[0]);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEX_Power),&componentkey[1]);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(networkdm,"genstruct",sizeof(struct _p_GEN),&componentkey[2]);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(networkdm,"loadstruct",sizeof(struct _p_LOAD),&componentkey[3]);CHKERRQ(ierr);

    ierr = PetscLogStageRegister("Read Data",&stage1);CHKERRQ(ierr);
    PetscLogStagePush(stage1);
    /* READ THE DATA */
    if (!crank) {
      /* Only rank 0 reads the data */
      ierr = PetscOptionsGetString(NULL,NULL,"-pfdata",pfdata_file,sizeof(pfdata_file),NULL);CHKERRQ(ierr);
      /* HERE WE CREATE COPIES OF THE SAME NETWORK THAT WILL BE TREATED AS SUBNETWORKS */

      /*    READ DATA FOR THE FIRST SUBNETWORK */
      ierr = PetscNew(&pfdata1);CHKERRQ(ierr);
      ierr = PFReadMatPowerData(pfdata1,pfdata_file);CHKERRQ(ierr);
      User.Sbase = pfdata1->sbase;

      numEdges1 = pfdata1->nbranch;
      numVertices1 = pfdata1->nbus;

      ierr = PetscMalloc1(2*numEdges1,&edgelist1);CHKERRQ(ierr);
      ierr = GetListofEdges_Power(pfdata1,edgelist1);CHKERRQ(ierr);

      /*    READ DATA FOR THE SECOND SUBNETWORK */
      ierr = PetscNew(&pfdata2);CHKERRQ(ierr);
      ierr = PFReadMatPowerData(pfdata2,pfdata_file);CHKERRQ(ierr);
      User.Sbase = pfdata2->sbase;

      numEdges2 = pfdata2->nbranch;
      numVertices2 = pfdata2->nbus;

      ierr = PetscMalloc1(2*numEdges2,&edgelist2);CHKERRQ(ierr);
      ierr = GetListofEdges_Power(pfdata2,edgelist2);CHKERRQ(ierr);
    }

    PetscLogStagePop();
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Create network",&stage2);CHKERRQ(ierr);
    PetscLogStagePush(stage2);

    /* Set number of nodes/edges */
    numVertices[0] = numVertices1; numVertices[1] = numVertices2;
    numEdges[0] = numEdges1; numEdges[1] = numEdges2;
    ierr = DMNetworkSetSizes(networkdm,nsubnet,numVertices,numEdges,0,NULL);CHKERRQ(ierr);

    edgelist[0] = edgelist1; edgelist[1] = edgelist2;

    /* Add edge connectivity */
    ierr = DMNetworkSetEdgeList(networkdm,edgelist,NULL);CHKERRQ(ierr);

    /* Set up the network layout */
    ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

    /* Add network components only process 0 has any data to add*/
    if (!crank) {
      genj=0; loadj=0;

      /* ADD VARIABLES AND COMPONENTS FOR THE FIRST SUBNETWORK */
      ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);

      for (i = 0; i < ne; i++) {
        ierr = DMNetworkAddComponent(networkdm,edges[i],componentkey[0],&pfdata1->branch[i]);CHKERRQ(ierr);
      }

      for (i = 0; i < nv; i++) {
        ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[1],&pfdata1->bus[i]);CHKERRQ(ierr);
        if (pfdata1->bus[i].ngen) {
          for (j = 0; j < pfdata1->bus[i].ngen; j++) {
            ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[2],&pfdata1->gen[genj++]);CHKERRQ(ierr);
          }
        }
        if (pfdata1->bus[i].nload) {
          for (j=0; j < pfdata1->bus[i].nload; j++) {
            ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[3],&pfdata1->load[loadj++]);CHKERRQ(ierr);
          }
        }
        /* Add number of variables */
        ierr = DMNetworkAddNumVariables(networkdm,vtx[i],2);CHKERRQ(ierr);
      }

      genj=0; loadj=0;

      /* ADD VARIABLES AND COMPONENTS FOR THE SECOND SUBNETWORK */
      ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);

      for (i = 0; i < ne; i++) {
        ierr = DMNetworkAddComponent(networkdm,edges[i],componentkey[0],&pfdata2->branch[i]);CHKERRQ(ierr);
      }

      for (i = 0; i < nv; i++) {
        ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[1],&pfdata2->bus[i]);CHKERRQ(ierr);
        if (pfdata2->bus[i].ngen) {
          for (j = 0; j < pfdata2->bus[i].ngen; j++) {
            ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[2],&pfdata2->gen[genj++]);CHKERRQ(ierr);
          }
        }
        if (pfdata2->bus[i].nload) {
          for (j=0; j < pfdata2->bus[i].nload; j++) {
            ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[3],&pfdata2->load[loadj++]);CHKERRQ(ierr);
          }
        }
        /* Add number of variables */
        ierr = DMNetworkAddNumVariables(networkdm,vtx[i],2);CHKERRQ(ierr);
      }
    }

    /* Set up DM for use */
    ierr = DMSetUp(networkdm);CHKERRQ(ierr);

    if (!crank) {
      ierr = PetscFree(edgelist1);CHKERRQ(ierr);
      ierr = PetscFree(edgelist2);CHKERRQ(ierr);
    }

    if (!crank) {
      ierr = PetscFree(pfdata1->bus);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1->gen);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1->branch);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1->load);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1);CHKERRQ(ierr);

      ierr = PetscFree(pfdata2->bus);CHKERRQ(ierr);
      ierr = PetscFree(pfdata2->gen);CHKERRQ(ierr);
      ierr = PetscFree(pfdata2->branch);CHKERRQ(ierr);
      ierr = PetscFree(pfdata2->load);CHKERRQ(ierr);
      ierr = PetscFree(pfdata2);CHKERRQ(ierr);
    }

    /* Distribute networkdm to multiple processes */
    ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);

    PetscLogStagePop();

    /* Broadcast Sbase to all processors */
    ierr = MPI_Bcast(&User.Sbase,1,MPIU_SCALAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(networkdm,&X);CHKERRQ(ierr);
    ierr = VecDuplicate(X,&F);CHKERRQ(ierr);

    ierr = DMCreateMatrix(networkdm,&J);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

    ierr = SetInitialValues(networkdm,X,&User);CHKERRQ(ierr);

    /* HOOK UP SOLVER */
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,networkdm);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,F,FormFunction,&User);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian,&User);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
    /* ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

    ierr = VecDestroy(&X);CHKERRQ(ierr);
    ierr = VecDestroy(&F);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);

    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
    ierr = DMDestroy(&networkdm);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     depends: PFReadData.c pffunctions.c
     requires: !complex double define(PETSC_HAVE_ATTRIBUTEALIGNED)


   test:
     args: -snes_rtol 1.e-3
     localrunfiles: poweroptions case9.m
     output_file: output/power2_1.out
     requires: double !complex

   test:
     suffix: 2
     args: -snes_rtol 1.e-3 -petscpartitioner_type simple
     nsize: 4
     localrunfiles: poweroptions case9.m
     output_file: output/power2_1.out
     requires: double !complex

TEST*/
