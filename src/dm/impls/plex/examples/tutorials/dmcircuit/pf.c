static char help[] = "Main routine for the per phase steady state power in power balance form.\n\
Run this program: mpiexec -n <n> ./PF\n\
                  mpiexec -n <n> ./PF -pfdata datafiles/<filename>\n";

/* T
   Concepts: Per-phase steady state power flow
   Concepts: PETSc SNES solver
*/

#include "pf.h"

PetscMPIInt rank;

typedef struct{
  PFDATA *pfdata;
  PetscSection EdgeSection;
  PetscSection VertexSection;
  PetscSection GenSection;
  PetscSection LoadSection;
  PetscSection DofSection;
}UserCtx;

/* Returns PETSC_TRUE if the topological point is a ghost point */
#undef __FUNCT__
#define __FUNCT__ "DMPlexIsGhostPoint"
PetscErrorCode DMPlexIsGhostPoint(DM dm,PetscInt p,PetscBool *isghost)
{
  PetscErrorCode ierr;
  PetscInt       offsetg;
  PetscSection   sectiong;

  PetscFunctionBegin;
  *isghost = PETSC_FALSE;
  ierr = DMGetDefaultGlobalSection(dm,&sectiong);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(sectiong,p,&offsetg);CHKERRQ(ierr);
  if (offsetg < 0) *isghost = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetInitialValues"
PetscErrorCode SetInitialValues(DM dm,Vec X,void* appctx)
{
  PetscErrorCode ierr;
  UserCtx        *User=(UserCtx*)appctx;
  PFDATA         *pfdata=(PFDATA*)User->pfdata;
  VERTEXDATA     bus;
  GEN            gen;
  PetscInt       v,vstart,vend,offset,i;
  PetscBool      ghostvtex;
  Vec            localX;
  PetscScalar    *xarr;
  PetscSection  section;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum(dm,1,&vstart,&vend);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);
  ierr = VecSet(X,0.0);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  for (v = vstart; v < vend; v++) {
    ierr = DMPlexIsGhostPoint(dm,v,&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;

    ierr = PetscSectionGetOffset(section,v,&offset);CHKERRQ(ierr);
    bus = &pfdata->bus[v-vstart];
    xarr[offset] = bus->va*PETSC_PI/180.0;
    xarr[offset+1] = bus->vm;
    PetscInt dofg,offsetg;
    ierr = PetscSectionGetDof(User->GenSection,v,&dofg);CHKERRQ(ierr);
    if (dofg) {
      ierr = PetscSectionGetOffset(User->GenSection,v,&offsetg);CHKERRQ(ierr);
      offsetg = offsetg*sizeof(PetscInt)/sizeof(struct _p_GEN);
      for (i = 0; i < bus->ngen;i++) {
	gen = &pfdata->gen[offsetg+i];
	if (!gen->status) continue;
	xarr[offset+1] = gen->vs; 
	break;
      }
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec X, Vec F,void *appctx)
{
  PetscErrorCode ierr;
  DM             dm;
  UserCtx       *User=(UserCtx*)appctx;
  PFDATA        *pfdata=(PFDATA*)User->pfdata;
  Vec           localX,localF;
  PetscInt      e;
  PetscInt      v,vStart,vEnd,vfrom,vto;
  PetscSection  section;
  const PetscScalar *xarr;
  PetscScalar   *farr;
  PetscInt      offsetfrom,offsetto;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&localF);CHKERRQ(ierr);
  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dm,F,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,F,INSERT_VALUES,localF);CHKERRQ(ierr);

  ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm,1,&vStart,&vEnd);CHKERRQ(ierr);
  for (v=vStart; v < vEnd; v++) {
    PetscInt    offset,i;
    PetscScalar Vm;
    PetscScalar Sbase=pfdata->sbase;
    VERTEXDATA  bus;
    GEN         gen;
    LOAD        load;
    PetscBool   ghostvtex;

    ierr = DMPlexIsGhostPoint(dm,v,&ghostvtex);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section,v,&offset);CHKERRQ(ierr);

    bus = &pfdata->bus[v-vStart];
    /* Handle reference bus constrained dofs */
    if (bus->ide == REF_BUS || bus->ide == ISOLATED_BUS) {
      farr[offset] = 0.0;
      farr[offset+1] = 0.0;
      continue;
    }

    if (!ghostvtex) {
      Vm = xarr[offset+1];

      /* Shunt injections */
      farr[offset] += Vm*Vm*bus->gl/Sbase;
      farr[offset+1] += -Vm*Vm*bus->bl/Sbase;

      PetscInt dofg, offsetg;
      /* Generators */
      ierr = PetscSectionGetDof(User->GenSection,v,&dofg);CHKERRQ(ierr);
      if (dofg) {
	ierr = PetscSectionGetOffset(User->GenSection,v,&offsetg);CHKERRQ(ierr);
	offsetg = offsetg*sizeof(PetscInt)/sizeof(struct _p_GEN);
	for (i = 0; i < bus->ngen;i++) {
	  gen = &pfdata->gen[offsetg+i];
	  if (!gen->status) continue;
	  farr[offset] += -gen->pg/Sbase;
	  farr[offset+1] += -gen->qg/Sbase;
	}
      }
      /* Const power loads */
      PetscInt dofl,offsetl;
      ierr = PetscSectionGetDof(User->LoadSection,v,&dofl);CHKERRQ(ierr);
      if (dofl) {
	ierr = PetscSectionGetOffset(User->LoadSection,v,&offsetl);CHKERRQ(ierr);
	offsetl = offsetl*sizeof(PetscInt)/sizeof(struct _p_LOAD);
	for (i = 0; i < bus->nload; i++) {
	  load = &pfdata->load[offsetl+i];
	  farr[offset] += load->pl/Sbase;
	  farr[offset+1] += load->ql/Sbase;
	}
      }
    }
    PetscInt nconnedges;
    const PetscInt *connedges;

    ierr = DMPlexGetSupportSize(dm,v,&nconnedges);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm,v,&connedges);CHKERRQ(ierr);

    for (i=0; i < nconnedges; i++) {
      EDGEDATA branch;
      e = connedges[i];
      branch = &pfdata->branch[e];
      if (!branch->status) continue;
      PetscScalar Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt;
      Gff = branch->yff[0];
      Bff = branch->yff[1];
      Gft = branch->yft[0];
      Bft = branch->yft[1];
      Gtf = branch->ytf[0];
      Btf = branch->ytf[1];
      Gtt = branch->ytt[0];
      Btt = branch->ytt[1];

      const PetscInt *cone;
      ierr = DMPlexGetCone(dm,e,&cone);CHKERRQ(ierr);
      vfrom = cone[0];
      vto   = cone[1];

      ierr = PetscSectionGetOffset(section,vfrom,&offsetfrom);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section,vto,&offsetto);CHKERRQ(ierr);

      PetscScalar Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

      thetaf = xarr[offsetfrom];
      Vmf     = xarr[offsetfrom+1];
      thetat = xarr[offsetto];
      Vmt     = xarr[offsetto+1];
      thetaft = thetaf - thetat;
      thetatf = thetat - thetaf;

      if (vfrom == v) {
	farr[offsetfrom]   += Gff*Vmf*Vmf + Vmf*Vmt*(Gft*cos(thetaft) + Bft*sin(thetaft));
	farr[offsetfrom+1] += -Bff*Vmf*Vmf + Vmf*Vmt*(-Bft*cos(thetaft) + Gft*sin(thetaft));
      } else {
	farr[offsetto]   += Gtt*Vmt*Vmt + Vmt*Vmf*(Gtf*cos(thetatf) + Btf*sin(thetatf));
	farr[offsetto+1] += -Btt*Vmt*Vmt + Vmt*Vmf*(-Btf*cos(thetatf) + Gtf*sin(thetatf));
      }
    }

    /* Handle PV bus constrained dofs */
    if (bus->ide == PV_BUS) {
      farr[offset+1] = 0.0;
    }						    
  }

  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(dm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
PetscErrorCode FormJacobian(SNES snes,Vec X, Mat *J,Mat *Jpre,MatStructure *flg,void *appctx)
{
  PetscErrorCode ierr;
  DM             dm;
  UserCtx       *User=(UserCtx*)appctx;
  PFDATA        *pfdata=(PFDATA*)User->pfdata;
  Vec           localX;
  PetscInt      e;
  PetscInt      v,vStart,vEnd,vfrom,vto;
  PetscSection  section,gsection;
  const PetscScalar *xarr;
  PetscInt      offsetfrom,offsetto,goffsetfrom,goffsetto;
  PetscInt      row[2],col[8];
  PetscScalar   values[8];

  PetscFunctionBegin;
  *flg = SAME_NONZERO_PATTERN;
  ierr = MatZeroEntries(*J);CHKERRQ(ierr);

  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm,&gsection);CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm,1,&vStart,&vEnd);CHKERRQ(ierr);
  for (v=vStart; v < vEnd; v++) {
    PetscInt offset,goffset,i;
    PetscScalar Vm;
    PetscScalar Sbase=pfdata->sbase;
    VERTEXDATA         bus;
    PetscBool   ghostvtex;

    ierr = DMPlexIsGhostPoint(dm,v,&ghostvtex);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section,v,&offset);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(gsection,v,&goffset);CHKERRQ(ierr);

    bus = &pfdata->bus[v-vStart];
    if (!ghostvtex) {
      /* Handle reference bus constrained dofs */
      if (bus->ide == REF_BUS || bus->ide == ISOLATED_BUS) {
	row[0] = goffset; row[1] = goffset+1;
	col[0] = goffset; col[1] = goffset+1; col[2] = goffset; col[3] = goffset+1;
	values[0] = 1.0; values[1] = 0.0; values[2] = 0.0; values[3] = 1.0;
	ierr = MatSetValues(*J,2,row,2,col,values,ADD_VALUES);CHKERRQ(ierr);
	continue;
      }

      Vm = xarr[offset+1];

      /* Shunt injections */
      row[0] = goffset; row[1] = goffset+1;
      col[0] = goffset; col[1] = goffset+1; col[2] = goffset; col[3] = goffset+1;
      values[0] = 2*Vm*bus->gl/Sbase;
      values[1] = values[2] = 0.0;
      values[3] = -2*Vm*bus->bl/Sbase;

      ierr = MatSetValues(*J,2,row,2,col,values,ADD_VALUES);CHKERRQ(ierr);

    }
    PetscInt nconnedges;
    const PetscInt *connedges;

    ierr = DMPlexGetSupportSize(dm,v,&nconnedges);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm,v,&connedges);CHKERRQ(ierr);

    for (i=0; i < nconnedges; i++) {
      EDGEDATA branch;
      e = connedges[i];
      branch = &pfdata->branch[e];
      if (!branch->status) continue;
      PetscScalar Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt;
      Gff = branch->yff[0];
      Bff = branch->yff[1];
      Gft = branch->yft[0];
      Bft = branch->yft[1];
      Gtf = branch->ytf[0];
      Btf = branch->ytf[1];
      Gtt = branch->ytt[0];
      Btt = branch->ytt[1];

      const PetscInt *cone;
      ierr = DMPlexGetCone(dm,e,&cone);CHKERRQ(ierr);
      vfrom = cone[0];
      vto   = cone[1];

      ierr = PetscSectionGetOffset(section,vfrom,&offsetfrom);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section,vto,&offsetto);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(gsection,vfrom,&goffsetfrom);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(gsection,vto,&goffsetto);CHKERRQ(ierr);

      if (goffsetfrom < 0) goffsetfrom = -goffsetfrom - 1; /* Convert to actual global offset for ghost nodes, global offset is -(gstart+1) */
      if (goffsetto < 0) goffsetto = -goffsetto - 1;
      PetscScalar Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

      thetaf = xarr[offsetfrom];
      Vmf     = xarr[offsetfrom+1];
      thetat = xarr[offsetto];
      Vmt     = xarr[offsetto+1];
      thetaft = thetaf - thetat;
      thetatf = thetat - thetaf;

      VERTEXDATA busf,bust;
      busf = &pfdata->bus[vfrom-vStart];
      bust = &pfdata->bus[vto-vStart];

      if (vfrom == v) {
	/*    farr[offsetfrom]   += Gff*Vmf*Vmf + Vmf*Vmt*(Gft*cos(thetaft) + Bft*sin(thetaft));  */
	row[0]  = goffsetfrom;
	col[0]  = goffsetfrom; col[1] = goffsetfrom+1; col[2] = goffsetto; col[3] = goffsetto+1;
	values[0] =  Vmf*Vmt*(Gft*-sin(thetaft) + Bft*cos(thetaft)); /* df_dthetaf */    
	values[1] =  2*Gff*Vmf + Vmt*(Gft*cos(thetaft) + Bft*sin(thetaft)); /* df_dVmf */
	values[2] =  Vmf*Vmt*(Gft*sin(thetaft) + Bft*-cos(thetaft)); /* df_dthetat */
	values[3] =  Vmf*(Gft*cos(thetaft) + Bft*sin(thetaft)); /* df_dVmt */

	ierr = MatSetValues(*J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);

	if (busf->ide != PV_BUS) {
	  row[0] = goffsetfrom+1;
	  col[0]  = goffsetfrom; col[1] = goffsetfrom+1; col[2] = goffsetto; col[3] = goffsetto+1;
	  /*    farr[offsetfrom+1] += -Bff*Vmf*Vmf + Vmf*Vmt*(-Bft*cos(thetaft) + Gft*sin(thetaft)); */
	  values[0] =  Vmf*Vmt*(Bft*sin(thetaft) + Gft*cos(thetaft));
	  values[1] =  -2*Bff*Vmf + Vmt*(-Bft*cos(thetaft) + Gft*sin(thetaft));
	  values[2] =  Vmf*Vmt*(-Bft*sin(thetaft) + Gft*-cos(thetaft));
	  values[3] =  Vmf*(-Bft*cos(thetaft) + Gft*sin(thetaft));
	  
	  ierr = MatSetValues(*J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
	}

      } else {
	row[0] = goffsetto;
	col[0] = goffsetto; col[1] = goffsetto+1; col[2] = goffsetfrom; col[3] = goffsetfrom+1;
	/*    farr[offsetto]   += Gtt*Vmt*Vmt + Vmt*Vmf*(Gtf*cos(thetatf) + Btf*sin(thetatf)); */
	values[0] =  Vmt*Vmf*(Gtf*-sin(thetatf) + Btf*cos(thetaft)); /* df_dthetat */
        values[1] =  2*Gtt*Vmt + Vmf*(Gtf*cos(thetatf) + Btf*sin(thetatf)); /* df_dVmt */
	values[2] =  Vmt*Vmf*(Gtf*sin(thetatf) + Btf*-cos(thetatf)); /* df_dthetaf */
	values[3] =  Vmt*(Gtf*cos(thetatf) + Btf*sin(thetatf)); /* df_dVmf */

	ierr = MatSetValues(*J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);

	if (bust->ide != PV_BUS) {
	  row[0] = goffsetto+1;
	  col[0] = goffsetto; col[1] = goffsetto+1; col[2] = goffsetfrom; col[3] = goffsetfrom+1;
	  /*    farr[offsetto+1] += -Btt*Vmt*Vmt + Vmt*Vmf*(-Btf*cos(thetatf) + Gtf*sin(thetatf)); */
	  values[0] =  Vmt*Vmf*(Btf*sin(thetatf) + Gtf*cos(thetatf));
	  values[1] =  -2*Btt*Vmt + Vmf*(-Btf*cos(thetatf) + Gtf*sin(thetatf));
	  values[2] =  Vmt*Vmf*(-Btf*sin(thetatf) + Gtf*-cos(thetatf));
	  values[3] =  Vmt*(-Btf*cos(thetatf) + Gtf*sin(thetatf));
	  
	  ierr = MatSetValues(*J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
	}
      }
    }

    if (bus->ide == PV_BUS) {
      row[0] = goffset+1; col[0] = goffset+1;
      values[0]  = 1.0;
      ierr = MatSetValues(*J,1,row,1,col,values,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetListofEdges"
PetscErrorCode GetListofEdges(PetscInt nbranches, EDGEDATA branch,PetscInt *edges)
{
  PetscInt       i, fbus,tbus;

  PetscFunctionBegin;
  
  for (i=0; i < nbranches; i++) {
    fbus = branch[i].internal_i;
    tbus = branch[i].internal_j;
    edges[2*i]   = fbus;
    edges[2*i+1] = tbus;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc,char ** argv)
{
  PetscErrorCode       ierr;
  char                 pfdata_file[PETSC_MAX_PATH_LEN]="datafiles/case9.m";
  UserCtx              User;
  PFDATA               pfdata;
  DM                   dm;
  DM                   distributedmesh;
  const char *partitioner = "chaco";
  PetscInt             dim=1,numEdges=0,numVertices=0,numCorners=0,spaceDim=2;
  PetscInt             *edges = NULL;
  double               *vertexCoords = NULL;
  PetscInt             i;  
  SNES                 snes;

  PetscInitialize(&argc,&argv,"pfoptions",help);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if (!rank) {
    /*    READ PARAMETERS */
    /* Only rank 0 reads the data */
    ierr = PetscOptionsGetString(PETSC_NULL,"-pfdata",pfdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
    ierr = PFReadMatPowerData(&pfdata,pfdata_file);CHKERRQ(ierr);
    User.pfdata = &pfdata;

    /* Reorder the generator data structure according to bus numbers */
    GEN  newgen;
    LOAD newload;
    PetscInt genj=0,loadj=0,j;
    ierr = PetscMalloc(pfdata.ngen*sizeof(struct _p_GEN),&newgen);CHKERRQ(ierr);
    ierr = PetscMalloc(pfdata.nload*sizeof(struct _p_LOAD),&newload);CHKERRQ(ierr);
    for (i = 0; i < pfdata.nbus; i++) {
      for (j = 0; j < pfdata.bus[i].ngen; j++) {
	ierr = PetscMemcpy(&newgen[genj++],&pfdata.gen[pfdata.bus[i].gidx[j]],sizeof(struct _p_GEN));
      }
      for (j = 0; j < pfdata.bus[i].nload; j++) {
	ierr = PetscMemcpy(&newload[loadj++],&pfdata.load[pfdata.bus[i].lidx[j]],sizeof(struct _p_LOAD));
      }
      
    }
    ierr = PetscFree(pfdata.gen);CHKERRQ(ierr);
    ierr = PetscFree(pfdata.load);CHKERRQ(ierr);
    pfdata.gen = newgen;
    pfdata.load = newload;
    numEdges = pfdata.nbranch;
    
    numVertices = pfdata.nbus;
    numCorners = 2; /* Each edge is connected to two vertices */
    ierr = PetscMalloc(numCorners*numEdges*sizeof(PetscInt),&edges);CHKERRQ(ierr);
    ierr = GetListofEdges(pfdata.nbranch,pfdata.branch,edges);CHKERRQ(ierr);

    ierr = PetscMalloc(numCorners*numVertices*sizeof(PetscInt),&vertexCoords);CHKERRQ(ierr);
  }

  /* ENTIRE MESH CREATED ON PROCESS 0 */
  ierr = DMPlexCreateFromCellList(PETSC_COMM_WORLD,dim,numEdges,numVertices,numCorners,PETSC_FALSE,edges,spaceDim,vertexCoords,&dm);CHKERRQ(ierr);

  PetscInt eStart,eEnd,vStart,vEnd,pStart,pEnd;
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&vStart,&vEnd);CHKERRQ(ierr);

  PetscSection oldEdgeSection, oldVertexSection,oldGenSection,oldLoadSection,oldDofSection;
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&oldEdgeSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&oldDofSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(oldDofSection,pStart,pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(oldEdgeSection,eStart,eEnd);CHKERRQ(ierr);

  for (i = eStart; i < eEnd; i++) {
    ierr = PetscSectionSetDof(oldEdgeSection,i,sizeof(struct _p_EDGEDATA)/sizeof(PetscInt));CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(oldEdgeSection);CHKERRQ(ierr);

  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&oldVertexSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&oldGenSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&oldLoadSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(oldVertexSection,vStart,vEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(oldGenSection,vStart,vEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(oldLoadSection,vStart,vEnd);CHKERRQ(ierr);

  for (i = vStart; i < vEnd; i++) {
    ierr = PetscSectionSetDof(oldVertexSection,i,sizeof(struct _p_VERTEXDATA)/sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscSectionSetDof(oldGenSection,i,pfdata.bus[i-vStart].ngen*sizeof(struct _p_GEN)/sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscSectionSetDof(oldLoadSection,i,pfdata.bus[i-vStart].nload*sizeof(struct _p_LOAD)/sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscSectionSetDof(oldDofSection,i,2);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(oldVertexSection);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(oldGenSection);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(oldLoadSection);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(oldDofSection);CHKERRQ(ierr);


  PetscSF pointsf;
  EDGEDATA  localedgedata;
  VERTEXDATA localvertexdata;
  GEN        localgendata;
  LOAD       localloaddata;

  /* DISTRIBUTE MESH */
  ierr = DMPlexDistribute(dm,partitioner,0,&pointsf,&distributedmesh);CHKERRQ(ierr);
  if (distributedmesh) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm = distributedmesh;
    ierr = PetscSectionCreate(PETSC_COMM_WORLD,&User.EdgeSection);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_WORLD,&User.VertexSection);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_WORLD,&User.GenSection);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_WORLD,&User.LoadSection);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_WORLD,&User.DofSection);CHKERRQ(ierr);

    ierr = DMPlexDistributeData(dm,pointsf,oldEdgeSection,MPI_INT,(void*)pfdata.branch,User.EdgeSection,(void**)&localedgedata);CHKERRQ(ierr);
    ierr = DMPlexDistributeData(dm,pointsf,oldVertexSection,MPI_INT,(void*)pfdata.bus,User.VertexSection,(void**)&localvertexdata);CHKERRQ(ierr);
    ierr = DMPlexDistributeData(dm,pointsf,oldGenSection,MPI_INT,(void*)pfdata.gen,User.GenSection,(void**)&localgendata);CHKERRQ(ierr);
    ierr = DMPlexDistributeData(dm,pointsf,oldLoadSection,MPI_INT,(void*)pfdata.load,User.LoadSection,(void**)&localloaddata);CHKERRQ(ierr);

    ierr = PetscSFDistributeSection(pointsf,oldDofSection,NULL,User.DofSection);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscFree(pfdata.branch);CHKERRQ(ierr);
      ierr = PetscFree(pfdata.bus);CHKERRQ(ierr);
      ierr = PetscFree(pfdata.gen);CHKERRQ(ierr);
      ierr = PetscFree(pfdata.load);CHKERRQ(ierr);
    }
    pfdata.branch = localedgedata;
    pfdata.bus    = localvertexdata;
    pfdata.gen    = localgendata;
    pfdata.load   = localloaddata;
  } else {
    User.EdgeSection = oldEdgeSection;
    User.VertexSection = oldVertexSection;
    User.GenSection    = oldGenSection;
    User.LoadSection  = oldLoadSection;
    User.DofSection   = oldDofSection;
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  /* Broadcast Sbase to all processors */
  ierr = MPI_Bcast(&pfdata.sbase,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = DMSetDefaultSection(dm,User.DofSection);CHKERRQ(ierr);

  User.pfdata = &pfdata;
  Vec X,F;
  ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
  ierr = SetInitialValues(dm,X,&User);CHKERRQ(ierr);

  Mat J;
  ierr = DMCreateMatrix(dm,MATAIJ,&J);CHKERRQ(ierr);

  /* HOOK UP SOLVER */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,F,FormFunction,&User);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,&User);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
