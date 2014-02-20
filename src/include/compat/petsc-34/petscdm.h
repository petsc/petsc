#define DMBoundaryType DMDABoundaryType
#define DM_BOUNDARY_NONE DMDA_BOUNDARY_NONE
#define DM_BOUNDARY_GHOSTED DMDA_BOUNDARY_GHOSTED
#define DM_BOUNDARY_MIRROR DMDA_BOUNDARY_MIRROR
#define DM_BOUNDARY_PERIODIC DMDA_BOUNDARY_PERIODIC

#undef  __FUNCT__
#define __FUNCT__ "DMClone"
static PetscErrorCode
DMClone(DM dm, DM *newdm)
{
  PetscBool      match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidType(dm,1);
  PetscValidPointer(newdm,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMDA,&match);CHKERRQ(ierr);
  if (match) {
    DM da;                    
    MPI_Comm comm=MPI_COMM_NULL;
    PetscInt dim=0,dof=0,swidth=0;
    PetscInt M=1,N=1,P=1;
    PetscInt m=1,n=1,p=1;
    const PetscInt *lx=NULL,*ly=NULL,*lz=NULL;
    DMBoundaryType btx,bty,btz;
    DMDAStencilType  stype = DMDA_STENCIL_STAR;
    /**/
    ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
    ierr = DMDAGetInfo(dm,&dim,&M,&N,&P,&m,&n,&p,&dof,&swidth,
                       &btx,&bty,&btz,&stype);CHKERRQ(ierr);
    ierr = DMDAGetOwnershipRanges(dm,&lx,&ly,&lz);CHKERRQ(ierr);
    /**/
    ierr = DMDACreate(comm,&da);CHKERRQ(ierr);
    ierr = DMDASetDim(da,dim);CHKERRQ(ierr);
    ierr = DMDASetDof(da,dof);CHKERRQ(ierr);
    ierr = DMDASetSizes(da,M,N,P);CHKERRQ(ierr);
    ierr = DMDASetNumProcs(da,m,n,p);CHKERRQ(ierr);
    ierr = DMDASetOwnershipRanges(da,lx,ly,lz);CHKERRQ(ierr);
    ierr = DMDASetBoundaryType(da,btx,bty,btz);CHKERRQ(ierr);
    ierr = DMDASetStencilType(da,stype);CHKERRQ(ierr);
    ierr = DMDASetStencilWidth(da,swidth);CHKERRQ(ierr);
    ierr = DMSetUp(da);CHKERRQ(ierr);
    /**/
    *newdm = da;
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&match);CHKERRQ(ierr);
  if (match) {
    ierr = DMPlexClone(dm,newdm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}

#undef  __FUNCT__
#define __FUNCT__ "DMLocalToLocalBegin"
static PetscErrorCode
DMLocalToLocalBegin(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  PetscBool match;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidType(dm,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMDA,&match);CHKERRQ(ierr);
  if (!match) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
    PetscFunctionReturn(PETSC_ERR_SUP);
  }
  ierr = DMDALocalToLocalBegin(dm,g,mode,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMLocalToLocalEnd"
static PetscErrorCode
DMLocalToLocalEnd(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscBool      match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidType(dm,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMDA,&match);CHKERRQ(ierr);
  if (!match) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
    PetscFunctionReturn(PETSC_ERR_SUP);
  }
  ierr = DMDALocalToLocalEnd(dm,g,mode,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_Compat"
static PetscErrorCode DMCreateMatrix_Compat(DM dm,Mat *A)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMCreateMatrix(dm,NULL,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define DMCreateMatrix DMCreateMatrix_Compat

#define DMGetCoordinateSection DMPlexGetCoordinateSection

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateCohesiveSubmesh_Compat"
static PetscErrorCode DMPlexCreateCohesiveSubmesh_Compat(DM dm, PetscBool hasLagrange, const char label[], PetscInt value, DM *subdm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMPlexCreateCohesiveSubmesh(dm, hasLagrange, subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define DMPlexCreateCohesiveSubmesh DMPlexCreateCohesiveSubmesh_Compat

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistribute_Compat"
static PetscErrorCode DMPlexDistribute_Compat(DM dm, const char partitioner[], PetscInt overlap, PetscSF *pointsf, DM *dmParallel)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMPlexDistribute(dm, partitioner, overlap, dmParallel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define DMPlexDistribute DMPlexDistribute_Compat

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateGmsh_Compat"
static PetscErrorCode DMPlexCreateGmsh_Compat(MPI_Comm comm, PetscViewer viewer, PetscBool interpolate, DM *dm)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
  PetscFunctionReturn(0);
}
#define DMPlexCreateGmsh DMPlexCreateGmsh_Compat

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSquareBoundary_Compat"
static PetscErrorCode DMPlexCreateSquareBoundary_Compat(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt edges[])
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
  PetscFunctionReturn(0);
}
#define DMPlexCreateSquareBoundary DMPlexCreateSquareBoundary_Compat
