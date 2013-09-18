static char help[] = "Main routine for the per phase steady state power in power balance form.\n\
Run this program: mpiexec -n <n> ./PF\n\
                  mpiexec -n <n> ./PF -pfdata datafiles/<filename>\n";

/* T
   Concepts: Per-phase steady state power flow
   Concepts: PETSc SNES solver
*/

#include "pf.h"

PetscMPIInt rank;

#define MAX_DATA_AT_POINT 14

typedef PetscInt ComponentDataArrayType;

typedef struct _p_PetscCircuitComponentHeader *PetscCircuitComponentHeader;
struct _p_PetscCircuitComponentHeader {
  PetscInt ndata; 
  PetscInt size[MAX_DATA_AT_POINT];
  PetscInt key[MAX_DATA_AT_POINT];
  PetscInt offset[MAX_DATA_AT_POINT];
};

typedef struct _p_PetscCircuitComponentValue *PetscCircuitComponentValue;
struct _p_PetscCircuitComponentValue {
  void* data[MAX_DATA_AT_POINT];
};

typedef struct {
  char name[20];
  PetscInt size;
}PetscCircuitComponent;

struct _p_PetscCircuit{
  PetscInt              NEdges; /* Number of global edges */
  PetscInt              NNodes; /* Number of global nodes */
  PetscInt              nEdges; /* Number of local edges */
  PetscInt              nNodes; /* Number of local nodes */
  PetscInt              *edges; /* Edge list */
  PetscInt              pStart,pEnd,vStart,vEnd,eStart,eEnd;
  DM                    dm;     /* DM */
  PetscSection          DataSection; /* Section for managing parameter distribution */
  PetscSection          DofSection;  /* Section for managing data distribution */
  PetscSection          GlobalDofSection; /* Global Dof section */
  PetscInt              ncomponent;
  PetscCircuitComponent  component[10];
  PetscCircuitComponentHeader header;  
  PetscCircuitComponentValue  cvalue;
  PetscInt               dataheadersize;
  ComponentDataArrayType         *componentdataarray; /* Array to hold the data */
};
typedef struct _p_PetscCircuit *PetscCircuit;

/* The interface needs a major rehaul. This is a quick implementation of the PetscCircuit interface */
/* Creates an empty PetscCircuit object */
#undef __FUNCT__ 
#define __FUNCT__ "PetscCircuitCreate"
PetscErrorCode PetscCircuitCreate(PetscCircuit *circuit)
{
  PetscErrorCode ierr;
  PetscCircuit   circ;
  PetscFunctionBegin;
  ierr = PetscNew(struct _p_PetscCircuit,&circ);CHKERRQ(ierr);
  circ->NNodes = 0;
  circ->nNodes = 0;
  circ->NEdges = 0;
  circ->nEdges = 0;
  *circuit = circ;
  PetscFunctionReturn(0);
}

/* Set the number of local/global edges and nodes */
#undef __FUNCT__
#define __FUNCT__ "PetscCircuitSetSizes"
PetscErrorCode PetscCircuitSetSizes(PetscCircuit circuit,PetscInt nNodes,PetscInt nEdges,PetscInt NNodes,PetscInt NEdges)
{

  PetscFunctionBegin;
  circuit->nNodes = nNodes;
  circuit->nEdges = nEdges;
  circuit->NNodes = NNodes;
  circuit->NEdges = NEdges;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitSetEdges"
PetscErrorCode PetscCircuitSetEdges(PetscCircuit circuit,PetscInt *edges)
{
  PetscFunctionBegin;
  circuit->edges = edges;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitLayoutSetUp"
PetscErrorCode PetscCircuitLayoutSetUp(PetscCircuit circuit)
{
  PetscErrorCode ierr;
  PetscInt       dim = 1; /* One dimensional circuit */
  PetscInt       numCorners=2;
  PetscInt       spacedim=2;
  double         *vertexcoords=NULL;
  PetscInt       i;

  PetscFunctionBegin;
  if (circuit->nNodes) {
    ierr = PetscMalloc(numCorners*circuit->nNodes*sizeof(PetscInt),&vertexcoords);CHKERRQ(ierr);
  }
  ierr = DMPlexCreateFromCellList(PETSC_COMM_WORLD,dim,circuit->nEdges,circuit->nNodes,numCorners,PETSC_FALSE,circuit->edges,spacedim,vertexcoords,&circuit->dm);CHKERRQ(ierr);
  ierr = DMPlexGetChart(circuit->dm,&circuit->pStart,&circuit->pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(circuit->dm,0,&circuit->eStart,&circuit->eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(circuit->dm,1,&circuit->vStart,&circuit->vEnd);CHKERRQ(ierr);
  
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&circuit->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&circuit->DofSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(circuit->DataSection,circuit->pStart,circuit->pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(circuit->DofSection,circuit->pStart,circuit->pEnd);CHKERRQ(ierr);

  circuit->dataheadersize = sizeof(struct _p_PetscCircuitComponentHeader)/sizeof(ComponentDataArrayType);
  ierr = PetscMalloc((circuit->pEnd-circuit->pStart)*sizeof(struct _p_PetscCircuitComponentHeader),&circuit->header);CHKERRQ(ierr);
  for (i = circuit->pStart; i < circuit->pEnd; i++) {
    PetscInt ndata;
    circuit->header[i].ndata = 0;
    ndata = circuit->header[i].ndata;
    ierr = PetscSectionAddDof(circuit->DataSection,i,circuit->dataheadersize);CHKERRQ(ierr);
    circuit->header[i].offset[ndata] = 0;
  }
  ierr = PetscMalloc((circuit->pEnd-circuit->pStart)*sizeof(struct _p_PetscCircuitComponentValue),&circuit->cvalue);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitRegisterComponent"
PetscErrorCode PetscCircuitRegisterComponent(PetscCircuit circuit,const char *name,PetscInt size,PetscInt *componentkey)
{
  PetscErrorCode ierr;
  PetscCircuitComponent *component=&circuit->component[circuit->ncomponent];
  PetscFunctionBegin;

  /* Skipping string comparison for now to check if the parameter already exists */
  ierr = PetscStrcpy(component->name,name);CHKERRQ(ierr);
  component->size = size/sizeof(ComponentDataArrayType);
  *componentkey = circuit->ncomponent;
  circuit->ncomponent++;
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetVertexRange"
PetscErrorCode PetscCircuitGetVertexRange(PetscCircuit circuit,PetscInt *vStart,PetscInt *vEnd)
{

  PetscFunctionBegin;
  if (vStart) *vStart = circuit->vStart;
  if (vEnd) *vEnd = circuit->vEnd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetEdgeRange"
PetscErrorCode PetscCircuitGetEdgeRange(PetscCircuit circuit,PetscInt *eStart,PetscInt *eEnd)
{

  PetscFunctionBegin;
  if (eStart) *eStart = circuit->eStart;
  if (eEnd) *eEnd = circuit->eEnd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitAddComponent"
PetscErrorCode PetscCircuitAddComponent(PetscCircuit circuit, PetscInt p,PetscInt componentkey,void* compvalue)
{
  PetscCircuitComponent component=circuit->component[componentkey];
  PetscCircuitComponentHeader header=&circuit->header[p];
  PetscCircuitComponentValue  cvalue=&circuit->cvalue[p];
  PetscErrorCode         ierr;
  
  PetscFunctionBegin;
  header->size[header->ndata] = component.size;
  ierr = PetscSectionAddDof(circuit->DataSection,p,component.size);CHKERRQ(ierr);
  header->key[header->ndata] = componentkey;
  if (header->ndata != 0) header->offset[header->ndata] = header->offset[header->ndata-1] + header->size[header->ndata-1]; 

  cvalue->data[header->ndata] = (void*)compvalue;
  header->ndata++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetNumComponents"
PetscErrorCode PetscCircuitGetNumComponents(PetscCircuit circuit,PetscInt p,PetscInt *numcomponents)
{
  PetscErrorCode ierr;
  PetscInt       offset;
  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(circuit->DataSection,p,&offset);CHKERRQ(ierr);
  *numcomponents = ((PetscCircuitComponentHeader)(circuit->componentdataarray+offset))->ndata;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetComponentTypeOffset"
PetscErrorCode PetscCircuitGetComponentTypeOffset(PetscCircuit circuit,PetscInt p, PetscInt compnum, PetscInt *compkey, PetscInt *offset)
{
  PetscErrorCode ierr;
  PetscInt       offsetp;
  PetscCircuitComponentHeader header;
  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(circuit->DataSection,p,&offsetp);CHKERRQ(ierr);
  header = (PetscCircuitComponentHeader)(circuit->componentdataarray+offsetp);
  *compkey = header->key[compnum];
  *offset  = offsetp+circuit->dataheadersize+header->offset[compnum];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetVariableOffset"
PetscErrorCode PetscCircuitGetVariableOffset(PetscCircuit circuit,PetscInt p,PetscInt *offset)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(circuit->DofSection,p,offset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetVariableGlobalOffset"
PetscErrorCode PetscCircuitGetVariableGlobalOffset(PetscCircuit circuit,PetscInt p,PetscInt *offsetg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(circuit->GlobalDofSection,p,offsetg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitAddNumVariables"
PetscErrorCode PetscCircuitAddNumVariables(PetscCircuit circuit,PetscInt p,PetscInt nvar)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscSectionAddDof(circuit->DofSection,p,nvar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This can be removed later */
#undef __FUNCT__
#define __FUNCT__ "PetscCircuitComponentSetUp"
PetscErrorCode PetscCircuitComponentSetUp(PetscCircuit circuit)
{
  PetscErrorCode              ierr;
  PetscInt                    arr_size;
  PetscInt                    p,offset,offsetp;
  PetscCircuitComponentHeader header;
  PetscCircuitComponentValue  cvalue;
  ComponentDataArrayType      *componentdataarray;
  PetscFunctionBegin;
  ierr = PetscSectionSetUp(circuit->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(circuit->DataSection,&arr_size);CHKERRQ(ierr);
  ierr = PetscMalloc(arr_size*sizeof(ComponentDataArrayType),&circuit->componentdataarray);CHKERRQ(ierr);
  componentdataarray = circuit->componentdataarray;
  for (p = circuit->pStart; p < circuit->pEnd; p++) {
    ierr = PetscSectionGetOffset(circuit->DataSection,p,&offsetp);CHKERRQ(ierr);
    /* Copy header */
    header = &circuit->header[p];
    ierr = PetscMemcpy(componentdataarray+offsetp,header,circuit->dataheadersize*sizeof(ComponentDataArrayType));
    /* Copy data */
    cvalue = &circuit->cvalue[p];
    PetscInt ncomp=header->ndata,i;
    for (i = 0; i < ncomp; i++) {
      offset = offsetp + circuit->dataheadersize + header->offset[i];
      ierr = PetscMemcpy(componentdataarray+offset,cvalue->data[i],header->size[i]*sizeof(ComponentDataArrayType));
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitVariablesSetUp"
PetscErrorCode PetscCircuitVariablesSetUp(PetscCircuit circuit)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSectionSetUp(circuit->DofSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetComponentDataArray"
PetscErrorCode PetscCircuitGetComponentDataArray(PetscCircuit circuit,ComponentDataArrayType **componentdataarray)
{
  PetscFunctionBegin;
  *componentdataarray = circuit->componentdataarray;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitDistribute"
PetscErrorCode PetscCircuitDistribute(PetscCircuit oldCircuit,PetscCircuit *distCircuit)
{
  PetscErrorCode ierr;
  PetscInt       size;
  const char*    partitioner="chaco";
  PetscSF        pointsf;
  PetscCircuit Circuitout;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  if (size == 1) {
    Circuitout = oldCircuit;
  } else {
    ierr = PetscCircuitCreate(&Circuitout);CHKERRQ(ierr);
    Circuitout->dataheadersize = sizeof(struct _p_PetscCircuitComponentHeader)/sizeof(ComponentDataArrayType);
    /* Distribute dm */
    ierr = DMPlexDistribute(oldCircuit->dm,partitioner,0,&pointsf,&Circuitout->dm);CHKERRQ(ierr);
    /* Distribute dof section */
    ierr = PetscSectionCreate(PETSC_COMM_WORLD,&Circuitout->DofSection);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(pointsf,oldCircuit->DofSection,NULL,Circuitout->DofSection);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_WORLD,&Circuitout->DataSection);CHKERRQ(ierr);
    /* Distribute data */
    ierr = DMPlexDistributeData(Circuitout->dm,pointsf,oldCircuit->DataSection,MPI_INT,(void*)oldCircuit->componentdataarray,Circuitout->DataSection,(void**)&Circuitout->componentdataarray);CHKERRQ(ierr);
    
    ierr = PetscSectionGetChart(Circuitout->DataSection,&Circuitout->pStart,&Circuitout->pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(Circuitout->dm,0, &Circuitout->eStart,&Circuitout->eEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(Circuitout->dm,1,&Circuitout->vStart,&Circuitout->vEnd);CHKERRQ(ierr);
    Circuitout->nEdges = Circuitout->eEnd - Circuitout->eStart;
    Circuitout->nNodes = Circuitout->vEnd - Circuitout->vStart;
  }
  /* Set Dof section as the default section for dm */
  ierr = DMSetDefaultSection(Circuitout->dm,Circuitout->DofSection);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(Circuitout->dm,&Circuitout->GlobalDofSection);CHKERRQ(ierr);
  *distCircuit = Circuitout;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetSupportingEdges"
PetscErrorCode PetscCircuitGetSupportingEdges(PetscCircuit circuit,PetscInt vertex,PetscInt *nedges,const PetscInt **edges)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetSupportSize(circuit->dm,vertex,nedges);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(circuit->dm,vertex,edges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetConnectedNodes"
PetscErrorCode PetscCircuitGetConnectedNodes(PetscCircuit circuit,PetscInt edge,const PetscInt **vertices)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMPlexGetCone(circuit->dm,edge,vertices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Returns PETSC_TRUE if the vertex is a ghost point */
#undef __FUNCT__
#define __FUNCT__ "PetscCircuitIsGhostVertex"
PetscErrorCode PetscCircuitIsGhostVertex(PetscCircuit circuit,PetscInt p,PetscBool *isghost)
{
  PetscErrorCode ierr;
  PetscInt       offsetg;
  PetscSection   sectiong;

  PetscFunctionBegin;
  *isghost = PETSC_FALSE;
  ierr = DMGetDefaultGlobalSection(circuit->dm,&sectiong);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(sectiong,p,&offsetg);CHKERRQ(ierr);
  if (offsetg < 0) *isghost = PETSC_TRUE;
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

typedef struct{
  PetscCircuit circuit;
  PetscScalar  Sbase;
}UserCtx;

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec X, Vec F,void *appctx)
{
  PetscErrorCode ierr;
  DM             dm;
  UserCtx       *User=(UserCtx*)appctx;
  PetscCircuit  circuit=User->circuit;
  Vec           localX,localF;
  PetscInt      e;
  PetscInt      v,vStart,vEnd,vfrom,vto;
  const PetscScalar *xarr;
  PetscScalar   *farr;
  PetscInt      offsetfrom,offsetto,offset;
  ComponentDataArrayType *arr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&localF);CHKERRQ(ierr);
  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dm,F,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,F,INSERT_VALUES,localF);CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

  ierr = PetscCircuitGetVertexRange(circuit,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = PetscCircuitGetComponentDataArray(circuit,&arr);CHKERRQ(ierr);

  for (v=vStart; v < vEnd; v++) {
    PetscInt    i,j,offsetd,key;
    PetscScalar Vm;
    PetscScalar Sbase=User->Sbase;
    VERTEXDATA  bus;
    GEN         gen;
    LOAD        load;
    PetscBool   ghostvtex;
    PetscInt    numComps;

    ierr = PetscCircuitIsGhostVertex(circuit,v,&ghostvtex);CHKERRQ(ierr);
    ierr = PetscCircuitGetNumComponents(circuit,v,&numComps);CHKERRQ(ierr);
    ierr = PetscCircuitGetVariableOffset(circuit,v,&offset);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = PetscCircuitGetComponentTypeOffset(circuit,v,j,&key,&offsetd);CHKERRQ(ierr);
      if (key == 1) {
	bus = (VERTEXDATA)(arr+offsetd);
	/* Handle reference bus constrained dofs */
	if (bus->ide == REF_BUS || bus->ide == ISOLATED_BUS) {
	  farr[offset] = 0.0;
	  farr[offset+1] = 0.0;
	  break;
	}

	if (!ghostvtex) {
	  Vm = xarr[offset+1];

	  /* Shunt injections */
	  farr[offset] += Vm*Vm*bus->gl/Sbase;
	  farr[offset+1] += -Vm*Vm*bus->bl/Sbase;
	}
	PetscInt nconnedges;
	const PetscInt *connedges;

	ierr = PetscCircuitGetSupportingEdges(circuit,v,&nconnedges,&connedges);CHKERRQ(ierr);
	for (i=0; i < nconnedges; i++) {
	  EDGEDATA branch;
	  PetscInt keye;
	  e = connedges[i];
	  ierr = PetscCircuitGetComponentTypeOffset(circuit,e,0,&keye,&offsetd);CHKERRQ(ierr);
	  branch = (EDGEDATA)(arr+offsetd);
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
	  ierr = PetscCircuitGetConnectedNodes(circuit,e,&cone);CHKERRQ(ierr);
	  vfrom = cone[0];
	  vto   = cone[1];

	  ierr = PetscCircuitGetVariableOffset(circuit,vfrom,&offsetfrom);CHKERRQ(ierr);
	  ierr = PetscCircuitGetVariableOffset(circuit,vto,&offsetto);CHKERRQ(ierr);

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
      } else if (key == 2) {
	if (!ghostvtex) {
	  gen = (GEN)(arr+offsetd);
	  if (!gen->status) continue;
	  farr[offset] += -gen->pg/Sbase;
	  farr[offset+1] += -gen->qg/Sbase;
	}
      } else if (key == 3) {
	if (!ghostvtex) {
	  load = (LOAD)(arr+offsetd);
	  farr[offset] += load->pl/Sbase;
	  farr[offset+1] += load->ql/Sbase;
	}
      }
    }
    if (bus->ide == PV_BUS) {
      farr[offset+1] = 0.0;
    }
    //    ierr = PetscPrintf(PETSC_COMM_SELF,"%d, %4d, %d, %10f, %10f\n",rank,bus->internal_i,(PetscInt)ghostvtex,farr[offset],farr[offset+1]);CHKERRQ(ierr);
  }
  //  exit(1);
  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(circuit->dm,&localX);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(circuit->dm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(circuit->dm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(circuit->dm,&localF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
PetscErrorCode FormJacobian(SNES snes,Vec X, Mat *J,Mat *Jpre,MatStructure *flg,void *appctx)
{
  PetscErrorCode ierr;
  DM            dm;
  UserCtx       *User=(UserCtx*)appctx;
  PetscCircuit  circuit=User->circuit;
  Vec           localX;
  PetscInt      e;
  PetscInt      v,vStart,vEnd,vfrom,vto;
  const PetscScalar *xarr;
  PetscInt      offsetfrom,offsetto,goffsetfrom,goffsetto;
  ComponentDataArrayType *arr;
  PetscInt      row[2],col[8];
  PetscScalar   values[8];

  PetscFunctionBegin;
  *flg = SAME_NONZERO_PATTERN;
  ierr = MatZeroEntries(*J);CHKERRQ(ierr);

  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);

  ierr = PetscCircuitGetVertexRange(circuit,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = PetscCircuitGetComponentDataArray(circuit,&arr);CHKERRQ(ierr);

  for (v=vStart; v < vEnd; v++) {
    PetscInt    i,j,offsetd,key;
    PetscInt    offset,goffset;
    PetscScalar Vm;
    PetscScalar Sbase=User->Sbase;
    VERTEXDATA  bus;
    PetscBool   ghostvtex;
    PetscInt    numComps;

    ierr = PetscCircuitIsGhostVertex(circuit,v,&ghostvtex);CHKERRQ(ierr);
    ierr = PetscCircuitGetNumComponents(circuit,v,&numComps);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = PetscCircuitGetVariableOffset(circuit,v,&offset);CHKERRQ(ierr);
      ierr = PetscCircuitGetVariableGlobalOffset(circuit,v,&goffset);CHKERRQ(ierr);
      ierr = PetscCircuitGetComponentTypeOffset(circuit,v,j,&key,&offsetd);CHKERRQ(ierr);
      if (key == 1) {
	bus = (VERTEXDATA)(arr+offsetd);
	if (!ghostvtex) {
	  /* Handle reference bus constrained dofs */
	  if (bus->ide == REF_BUS || bus->ide == ISOLATED_BUS) {
	    row[0] = goffset; row[1] = goffset+1;
	    col[0] = goffset; col[1] = goffset+1; col[2] = goffset; col[3] = goffset+1;
	    values[0] = 1.0; values[1] = 0.0; values[2] = 0.0; values[3] = 1.0;
	    ierr = MatSetValues(*J,2,row,2,col,values,ADD_VALUES);CHKERRQ(ierr);
	    break;
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

	ierr = PetscCircuitGetSupportingEdges(circuit,v,&nconnedges,&connedges);CHKERRQ(ierr);
	for (i=0; i < nconnedges; i++) {
	  EDGEDATA branch;
	  VERTEXDATA busf,bust;
	  PetscInt   offsetfd,offsettd,keyf,keyt;
	  e = connedges[i];
	  ierr = PetscCircuitGetComponentTypeOffset(circuit,e,0,&key,&offsetd);CHKERRQ(ierr);
	  branch = (EDGEDATA)(arr+offsetd);
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
	  ierr = PetscCircuitGetConnectedNodes(circuit,e,&cone);CHKERRQ(ierr);
	  vfrom = cone[0];
	  vto   = cone[1];

	  ierr = PetscCircuitGetVariableOffset(circuit,vfrom,&offsetfrom);CHKERRQ(ierr);
	  ierr = PetscCircuitGetVariableOffset(circuit,vto,&offsetto);CHKERRQ(ierr);
	  ierr = PetscCircuitGetVariableGlobalOffset(circuit,vfrom,&goffsetfrom);CHKERRQ(ierr);
	  ierr = PetscCircuitGetVariableGlobalOffset(circuit,vto,&goffsetto);CHKERRQ(ierr);

	  if (goffsetfrom < 0) goffsetfrom = -goffsetfrom - 1; /* Convert to actual global offset for ghost nodes, global offset is -(gstart+1) */
	  if (goffsetto < 0) goffsetto = -goffsetto - 1;
	  PetscScalar Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

	  thetaf = xarr[offsetfrom];
	  Vmf     = xarr[offsetfrom+1];
	  thetat = xarr[offsetto];
	  Vmt     = xarr[offsetto+1];
	  thetaft = thetaf - thetat;
	  thetatf = thetat - thetaf;

	  ierr = PetscCircuitGetComponentTypeOffset(circuit,vfrom,0,&keyf,&offsetfd);CHKERRQ(ierr);
	  ierr = PetscCircuitGetComponentTypeOffset(circuit,vto,0,&keyt,&offsettd);CHKERRQ(ierr);
	  busf = (VERTEXDATA)(arr+offsetfd);
	  bust = (VERTEXDATA)(arr+offsettd);

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
	if (!ghostvtex && bus->ide == PV_BUS) {
	  row[0] = goffset+1; col[0] = goffset+1;
	  values[0]  = 1.0;
	  ierr = MatSetValues(*J,1,row,1,col,values,ADD_VALUES);CHKERRQ(ierr);
	}
      }
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
PetscErrorCode SetInitialValues(PetscCircuit circuit,Vec X,void* appctx)
{
  PetscErrorCode ierr;
  VERTEXDATA     bus;
  GEN            gen;
  PetscInt       v, vStart, vEnd, offset;
  PetscBool      ghostvtex;
  Vec            localX;
  PetscScalar    *xarr;
  PetscInt       key;
  ComponentDataArrayType *arr;
  
  PetscFunctionBegin;
  ierr = PetscCircuitGetVertexRange(circuit,&vStart, &vEnd);CHKERRQ(ierr);

  ierr = DMGetLocalVector(circuit->dm,&localX);CHKERRQ(ierr);

  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(circuit->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(circuit->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = PetscCircuitGetComponentDataArray(circuit,&arr);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; v++) {
    ierr = PetscCircuitIsGhostVertex(circuit,v,&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;
    PetscInt numComps;
    PetscInt j;
    PetscInt offsetd;
    ierr = PetscCircuitGetVariableOffset(circuit,v,&offset);CHKERRQ(ierr);
    ierr = PetscCircuitGetNumComponents(circuit,v,&numComps);CHKERRQ(ierr);
    for (j=0; j < numComps; j++) {
      ierr = PetscCircuitGetComponentTypeOffset(circuit,v,j,&key,&offsetd);CHKERRQ(ierr);
      if (key == 1) {
	bus = (VERTEXDATA)(arr+offsetd);
	xarr[offset] = bus->va*PETSC_PI/180.0;
	xarr[offset+1] = bus->vm;
      } else if(key == 2) {
	gen = (GEN)(arr+offsetd);
	if (!gen->status) continue;
	xarr[offset+1] = gen->vs; 
	break;
      }
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(circuit->dm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(circuit->dm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(circuit->dm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char ** argv)
{
  PetscErrorCode       ierr;
  char                 pfdata_file[PETSC_MAX_PATH_LEN]="datafiles/case9.m";
  PFDATA               pfdata;
  PetscInt             numEdges=0,numVertices=0;
  PetscInt             *edges = NULL;
  PetscInt             i;  
  PetscCircuit         circuit;
  PetscInt             componentkey[4];
  UserCtx              User;
  PetscLogStage        stage1,stage2;

  PetscInitialize(&argc,&argv,"pfoptions",help);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Create an empty circuit object */
  ierr = PetscCircuitCreate(&circuit);CHKERRQ(ierr);
  /* Register the components in the circuit */
  ierr = PetscCircuitRegisterComponent(circuit,"branchstruct",sizeof(struct _p_EDGEDATA),&componentkey[0]);CHKERRQ(ierr);
  ierr = PetscCircuitRegisterComponent(circuit,"busstruct",sizeof(struct _p_VERTEXDATA),&componentkey[1]);CHKERRQ(ierr);
  ierr = PetscCircuitRegisterComponent(circuit,"genstruct",sizeof(struct _p_GEN),&componentkey[2]);CHKERRQ(ierr);
  ierr = PetscCircuitRegisterComponent(circuit,"loadstruct",sizeof(struct _p_LOAD),&componentkey[3]);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Read Data",&stage1);CHKERRQ(ierr);
  PetscLogStagePush(stage1);
  /* READ THE DATA */
  if (!rank) {
    /*    READ DATA */
    /* Only rank 0 reads the data */
    ierr = PetscOptionsGetString(PETSC_NULL,"-pfdata",pfdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
    ierr = PFReadMatPowerData(&pfdata,pfdata_file);CHKERRQ(ierr);
    User.Sbase = pfdata.sbase;

    numEdges = pfdata.nbranch;
    numVertices = pfdata.nbus;

    ierr = PetscMalloc(2*numEdges*sizeof(PetscInt),&edges);CHKERRQ(ierr);
    ierr = GetListofEdges(pfdata.nbranch,pfdata.branch,edges);CHKERRQ(ierr);
  }
  PetscLogStagePop();
  ierr = PetscLogStageRegister("Create circuit",&stage2);CHKERRQ(ierr);
  PetscLogStagePush(stage2);
  /* Set number of nodes/edges */
  ierr = PetscCircuitSetSizes(circuit,numVertices,numEdges,numVertices,numEdges);CHKERRQ(ierr);
  /* Add edge connectivity */
  ierr = PetscCircuitSetEdges(circuit,edges);CHKERRQ(ierr);
  /* Set up the circuit layout */
  ierr = PetscCircuitLayoutSetUp(circuit);CHKERRQ(ierr);

  /* Add circuit components */
  PetscInt eStart, eEnd, vStart, vEnd,j;
  PetscInt genj=0,loadj=0;
  ierr = PetscCircuitGetEdgeRange(circuit,&eStart,&eEnd);CHKERRQ(ierr);
  for (i = eStart; i < eEnd; i++) {
    ierr = PetscCircuitAddComponent(circuit,i,componentkey[0],&pfdata.branch[i-eStart]);CHKERRQ(ierr);
  }
  ierr = PetscCircuitGetVertexRange(circuit,&vStart,&vEnd);CHKERRQ(ierr);
  for (i = vStart; i < vEnd; i++) {
    ierr = PetscCircuitAddComponent(circuit,i,componentkey[1],&pfdata.bus[i-vStart]);CHKERRQ(ierr);
    if (pfdata.bus[i-vStart].ngen) {
      for (j = 0; j < pfdata.bus[i-vStart].ngen; j++) {
	ierr = PetscCircuitAddComponent(circuit,i,componentkey[2],&pfdata.gen[genj++]);CHKERRQ(ierr);
      }
    }
    if (pfdata.bus[i-vStart].nload) {
      for (j=0; j < pfdata.bus[i-vStart].nload; j++) {
	ierr = PetscCircuitAddComponent(circuit,i,componentkey[3],&pfdata.load[loadj++]);CHKERRQ(ierr);
      }
    }
    /* Add number of variables */
    ierr = PetscCircuitAddNumVariables(circuit,i,2);CHKERRQ(ierr);
  }
  /* Set up components and variables */
  ierr = PetscCircuitComponentSetUp(circuit);CHKERRQ(ierr);
  ierr = PetscCircuitVariablesSetUp(circuit);CHKERRQ(ierr);

  /* Circuit partitioning and distribution of data */
  ierr = PetscCircuitDistribute(circuit,&User.circuit);CHKERRQ(ierr);
  circuit = User.circuit;

  PetscLogStagePop();
  ierr = PetscCircuitGetEdgeRange(circuit,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = PetscCircuitGetVertexRange(circuit,&vStart,&vEnd);CHKERRQ(ierr);
  
#if 0
  PetscInt numComponents;
  EDGEDATA edge;
  PetscInt offset,key;
  ComponentDataArrayType *arr;
  for (i = eStart; i < eEnd; i++) {
    ierr = PetscCircuitGetComponentDataArray(circuit,&arr);CHKERRQ(ierr);
    ierr = PetscCircuitGetComponentTypeOffset(circuit,i,0,&key,&offset);CHKERRQ(ierr);
    edge = (EDGEDATA)(arr+offset);
    ierr = PetscCircuitGetNumComponents(circuit,i,&numComponents);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Line %d ---- %d\n",rank,numComponents,edge->internal_i,edge->internal_j);CHKERRQ(ierr);
  }    

  VERTEXDATA bus;
  GEN        gen;
  LOAD       load;
  PetscInt   kk;
  for (i = vStart; i < vEnd; i++) {
    ierr = PetscCircuitGetComponentDataArray(circuit,&arr);CHKERRQ(ierr);
    ierr = PetscCircuitGetNumComponents(circuit,i,&numComponents);CHKERRQ(ierr);
    for (kk=0; kk < numComponents; kk++) {
      ierr = PetscCircuitGetComponentTypeOffset(circuit,i,kk,&key,&offset);CHKERRQ(ierr);
      if (key == 1) {
	bus = (VERTEXDATA)(arr+offset);
	ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Bus %d\n",rank,numComponents,bus->internal_i);CHKERRQ(ierr);
      } else if (key == 2) {
	gen = (GEN)(arr+offset);
	ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d Gen pg = %f qg = %f\n",rank,gen->pg,gen->qg);CHKERRQ(ierr);
      } else if (key == 3) {
	load = (LOAD)(arr+offset);
	ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d Load pd = %f qd = %f\n",rank,load->pl,load->ql);CHKERRQ(ierr);
      }
    }
  }  
#endif  
  /* Broadcast Sbase to all processors */
  ierr = MPI_Bcast(&User.Sbase,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  Vec X,F;
  ierr = DMCreateGlobalVector(circuit->dm,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
  ierr = SetInitialValues(circuit,X,&User);CHKERRQ(ierr);

  Mat J;
  ierr = DMCreateMatrix(circuit->dm,MATAIJ,&J);CHKERRQ(ierr);
  ierr = MatSetOption(J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

  SNES snes;
  /* HOOK UP SOLVER */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,circuit->dm);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,F,FormFunction,&User);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,&User);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
