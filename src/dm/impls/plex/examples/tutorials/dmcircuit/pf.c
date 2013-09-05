static char help[] = "Main routine for the per phase steady state power in power balance form.\n\
Run this program: mpiexec -n <n> ./PF\n\
                  mpiexec -n <n> ./PF -pfdata datafiles/<filename>\n";

/* T
   Concepts: Per-phase steady state power flow
   Concepts: PETSc SNES solver
*/

#include "pf.h"

/* Labels for indexing parameters from global array of structs */
const char *lineappnum="lineappnum",*busappnum="busappnum";
const char *localvertex="localvertex";
/* Label for user provided degrees of freedom */
const char *doflabel="dof";

PetscMPIInt rank;

typedef struct{
  PFDATA *pfdata;
  PetscSection datasection;
  void         **data;
  PetscBool    *ghostpoint;
}UserCtx;

PetscLogEvent GetLabel;

#undef __FUNCT__
#define __FUNCT__ "GetLabelValue"
PetscErrorCode GetLabelValue(DM dm,const char *labelname,PetscInt v,PetscInt *labelvalue)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscLogEventBegin(GetLabel,0,0,0,0);
  ierr = DMPlexGetLabelValue(dm,labelname,v,labelvalue);CHKERRQ(ierr);
  PetscLogEventEnd(GetLabel,0,0,0,0);
  PetscFunctionReturn(0);
}

/* Returns the parameter structure associated with the topological point from the "data" input structure */
#undef __FUNCT__
#define __FUNCT__ "DMPlexGetParameterStructure"
PetscErrorCode DMPlexGetParameterStructure(DM dm,PetscSection section,PetscInt p,void **data,void **ctx)
{
  PetscErrorCode ierr;
  PetscInt       offset;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(section,p,&offset);CHKERRQ(ierr);
  *ctx = data[offset];

  PetscFunctionReturn(0);
}

/* Returns PETSC_TRUE if the topological point is a ghost point */
#undef __FUNCT__
#define __FUNCT__ "DMPlexIsGhostPoint"
PetscErrorCode DMPlexIsGhostPoint(DM dm,PetscSection section,PetscInt p,PetscBool *ghostlist,PetscBool *isghost)
{
  PetscErrorCode ierr;
  PetscInt       offset;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(section,p,&offset);CHKERRQ(ierr);
  *isghost = ghostlist[offset];

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
  GEN           gen=pfdata->gen;
  LOAD          load=pfdata->load;
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
    PetscInt gidx,lidx,offset,i;
    PetscScalar Vm;
    PetscScalar Sbase=pfdata->sbase;
    VERTEXDATA         bus;
    PetscBool   ghostvtex;

    ierr = DMPlexIsGhostPoint(dm,User->datasection,v,User->ghostpoint,&ghostvtex);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section,v,&offset);CHKERRQ(ierr);

    ierr = DMPlexGetParameterStructure(dm,User->datasection,v,User->data,(void**)&bus);CHKERRQ(ierr);
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
      
      /* Generators */
      for (i = 0; i < bus->ngen;i++) {
	gidx = bus->gidx[i];
	if (!gen[gidx].status) continue;
	
	farr[offset] += -gen[gidx].pg/Sbase;
	farr[offset+1] += -gen[gidx].qg/Sbase;
      }

      /* Const power loads */
      for (i = 0; i < bus->nload; i++) {
	lidx = bus->lidx[i];
	farr[offset] += load[lidx].pl/Sbase;
	farr[offset+1] += load[lidx].ql/Sbase;
      }
    }
    PetscInt nconnedges;
    const PetscInt *connedges;

    ierr = DMPlexGetSupportSize(dm,v,&nconnedges);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm,v,&connedges);CHKERRQ(ierr);

    for (i=0; i < nconnedges; i++) {
      EDGEDATA branch;
      e = connedges[i];
      ierr = DMPlexGetParameterStructure(dm,User->datasection,e,User->data,(void**)&branch);CHKERRQ(ierr);
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

    ierr = DMPlexIsGhostPoint(dm,User->datasection,v,User->ghostpoint,&ghostvtex);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section,v,&offset);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(gsection,v,&goffset);CHKERRQ(ierr);
    ierr = DMPlexGetParameterStructure(dm,User->datasection,v,User->data,(void**)&bus);CHKERRQ(ierr);

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
      ierr = DMPlexGetParameterStructure(dm,User->datasection,e,User->data,(void**)&branch);CHKERRQ(ierr);
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
      ierr = DMPlexGetParameterStructure(dm,User->datasection,vfrom,User->data,(void**)&busf);CHKERRQ(ierr);
      ierr = DMPlexGetParameterStructure(dm,User->datasection,vto,User->data,(void**)&bust);CHKERRQ(ierr);

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
#define __FUNCT__ "SetInitialValues"
PetscErrorCode SetInitialValues(DM dm,Vec X,void* appctx)
{
  PetscErrorCode ierr;
  UserCtx        *User=(UserCtx*)appctx;
  PFDATA         *pfdata=(PFDATA*)User->pfdata;
  GEN            gen=pfdata->gen;
  VERTEXDATA     bus;
  PetscInt       v,vstart,vend,gidx,offset,i;
  PetscSection   gsection;
  PetscInt       idx[2];
  PetscScalar    values[2];
  PetscBool      ghostvtex;

  PetscFunctionBegin;

  ierr = DMGetDefaultGlobalSection(dm,&gsection);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&vstart,&vend);CHKERRQ(ierr);

  for (v = vstart; v < vend; v++) {
    ierr = DMPlexIsGhostPoint(dm,User->datasection,v,User->ghostpoint,&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;

    ierr = DMPlexGetParameterStructure(dm,User->datasection,v,User->data,(void**)&bus);CHKERRQ(ierr);

    ierr = PetscSectionGetOffset(gsection,v,&offset);CHKERRQ(ierr);
    idx[0]  = offset;
    idx[1] = offset+1;
    values[0] = bus->va*PETSC_PI/180.0;
    values[1] = bus->vm;
    for (i = 0; i < bus->ngen;i++) {
      gidx = bus->gidx[i];
      if (!gen[gidx].status) continue;
      values[1] = gen[gidx].vs; 
      break;
    }
    ierr = VecSetValues(X,2,idx,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
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
  PetscInt             i,busnum,linenum;  
  SNES                 snes;

  PetscInitialize(&argc,&argv,"pfoptions",help);

  PetscLogEventRegister("DMPlexGetLabelValue",0,&GetLabel);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /*    READ PARAMETERS */
  /* Each processor reads the data file and has access to the entire file. Eventually, only processor 0 should read the entire data and scatter the data to the processor that needs it 
   */
  ierr = PetscOptionsGetString(PETSC_NULL,"-pfdata",pfdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
  ierr = PFReadMatPowerData(&pfdata,pfdata_file);CHKERRQ(ierr);
  User.pfdata = &pfdata;

  if (!rank) {
    numEdges = pfdata.nbranch;
    numVertices = pfdata.nbus;
    numCorners = 2; /* Each edge is connected to two vertices */
    ierr = PetscMalloc(numCorners*numEdges*sizeof(PetscInt),&edges);CHKERRQ(ierr);
    ierr = GetListofEdges(pfdata.nbranch,pfdata.branch,edges);CHKERRQ(ierr);

    ierr = PetscMalloc(numCorners*numVertices*sizeof(PetscInt),&vertexCoords);CHKERRQ(ierr);
  }

  /* ENTIRE MESH CREATED ON PROCESS 0 */
  ierr = DMPlexCreateFromCellList(PETSC_COMM_WORLD,dim,numEdges,numVertices,numCorners,PETSC_FALSE,edges,spaceDim,vertexCoords,&dm);CHKERRQ(ierr);

  /* INSERT INDICES FOR ACCESSING VERTEX/EDGE PARAMETERS IN LABEL VALUES */
  ierr = DMPlexCreateLabel(dm,busappnum);CHKERRQ(ierr);
  ierr = DMPlexCreateLabel(dm,lineappnum);CHKERRQ(ierr);
  ierr = DMPlexCreateLabel(dm,doflabel);CHKERRQ(ierr); /* label for user provided dofs */

  PetscInt eStart,eEnd,vStart,vEnd,pStart,pEnd;
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&vStart,&vEnd);CHKERRQ(ierr);

  for (i = eStart; i < eEnd; i++) {
    ierr = DMPlexSetLabelValue(dm,lineappnum,i,i-eStart);CHKERRQ(ierr);
  }

  for (i = vStart; i < vEnd; i++) {
    ierr = DMPlexSetLabelValue(dm,busappnum,i,i-vStart);CHKERRQ(ierr);
    ierr = DMPlexSetLabelValue(dm,doflabel,i,2);CHKERRQ(ierr);
  }

  /* DISTRIBUTE MESH ... LABELS ALSO DISTRIBUTED */
  ierr = DMPlexDistribute(dm,partitioner,0,&distributedmesh);CHKERRQ(ierr);
  if (distributedmesh) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm = distributedmesh;
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&vStart,&vEnd);CHKERRQ(ierr);

  /* CREATE A SECTION FOR THE USER PROVIDED DEGREES OF FREEDOM */
  PetscSection section;
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section,pStart,pEnd);CHKERRQ(ierr);

  PetscInt dof;
  for (i = vStart; i < vEnd; i++) {
    ierr = GetLabelValue(dm,doflabel,i,&dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(section,i,dof);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);

  ierr = DMSetDefaultSection(dm,section);CHKERRQ(ierr);
 
  numEdges = eEnd - eStart;
  numVertices = vEnd - vStart;
  ierr = PetscMalloc((numEdges+numVertices)*sizeof(void*),&User.data);CHKERRQ(ierr);
  /* Create a section for managing the data */
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&User.datasection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(User.datasection,pStart,pEnd);CHKERRQ(ierr);

  for (i = eStart; i < eEnd; i++) {
    ierr = PetscSectionSetDof(User.datasection,i,1);CHKERRQ(ierr);
    ierr = GetLabelValue(dm,lineappnum,i,&linenum);CHKERRQ(ierr);
    User.data[i] = &pfdata.branch[linenum];
  }
  for (i = vStart; i < vEnd; i++) {
    ierr = PetscSectionSetDof(User.datasection,i,1);CHKERRQ(ierr);
    ierr = GetLabelValue(dm,busappnum,i,&busnum);CHKERRQ(ierr);
    User.data[i] = &pfdata.bus[busnum];
  }
  ierr = PetscSectionSetUp(User.datasection);CHKERRQ(ierr);

  PetscSection sectiong;
  ierr = DMGetDefaultGlobalSection(dm,&sectiong);CHKERRQ(ierr);

  ierr = PetscMalloc((numEdges+numVertices)*sizeof(PetscBool),&User.ghostpoint);CHKERRQ(ierr);
  PetscInt offset;
  for (i = pStart; i < pEnd; i++) {
    ierr = PetscSectionGetOffset(sectiong,i,&offset);CHKERRQ(ierr);
    if (offset >= 0) User.ghostpoint[i] = 0;
    else User.ghostpoint[i] = 1;
  }

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

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&User.datasection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
