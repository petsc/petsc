/* ----------------------------------------------------------------------
 * Ethan Coon <ecoon@ldeo.columbia.edu> and Richard Katz <katz@ldeo.columbia.edu>
 *
 *	This is a library of functions to write .info files with matlab code
 *      for interpreting various PETSc binary files.
 *
 *	Note all "name" and "DAFieldName" variables must be Matlab-Kosher
 *	i.e. no whitespace or illegal characters such as grouping 
 *	operators, quotations, math/boolean operators, etc. 
 * ----------------------------------------------------------------------*/
#include <petscviewer.h>
#include <petscda.h>

/* ---------------------------------------------------------------------
 *  PetscViewerBinaryMatlabOpen
 *	
 *		Input
 *		------------------------------
 *		comm	| mpi communicator
 *		fname	| filename
 *		viewer	| petsc viewer object
 *		
 * ----------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "PetscViewerBinaryMatlabOpen"
/*@C
  PetscViewerBinaryMatlabOpen - Open a binary viewer and write matlab info file initialization.

  Collective on MPI_Comm

  Input Parameters:
+ comm - The communicator
- fname - The output filename

  Output Parameter:
. viewer - The viewer object

  Level: beginner

  .seealso: PetscViewerBinaryMatlabDestroy()
@*/
PetscErrorCode PetscViewerBinaryMatlabOpen(MPI_Comm comm, const char fname[], PetscViewer *viewer)
{
  FILE          *info;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryOpen(comm,fname,FILE_MODE_WRITE,viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetInfoPointer(*viewer,&info);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%--- begin code written by PetscWriteOutputInitialize ---%\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%$$ Set.filename = '%s';\n",fname);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%$$ fd = fopen(Set.filename, 'r', 'ieee-be');\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%$$ if (fd < 0) error('Cannot open %s, check for existence of file'); end\n",fname);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%--- end code written by PetscWriteOutputInitialize ---%\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerBinaryMatlabDestroy - Write matlab info file finalization and destroy viewer.

  Not Collective

  Input Parameter:
. viewer - The viewer object

  Level: beginner

  .seealso PetscViewerBinaryMatlabOpen(), PetscViewerBinaryMatlabOutputVec(), 
           PetscViewerBinaryMatlabOutputVecDA(), PetscViewerBinaryMatlabOutputBag()
@*/
#undef __FUNCT__
#define __FUNCT__ "PetscViewerBinaryMatlabDestroy"
PetscErrorCode PetscViewerBinaryMatlabDestroy(PetscViewer viewer)
{
  FILE          *info;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetInfoPointer(viewer,&info);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%--- begin code written by PetscWriteOutputFinalize ---%\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%$$ fclose(fd);\n");
  ierr = PetscFPrintf(comm,info,"%%--- end code written by PetscWriteOutputFinalize ---%\n\n");CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerBinaryMatlabOutputBag - Write matlab code to info file to read a PetscBag from binary.

  Input Parameters:
+ viewer - The viewer object
. name - The bag name
- bag - The bag object containing data to output

  Level: intermediate

  .seealso: PetscViewerBinaryMatlabOpen(), PetscViewerBinaryMatlabOutputVec(), PetscViewerBinaryMatlabOutputVecDA()
@*/
#undef __FUNCT__
#define __FUNCT__ "PetscViewerBinaryMatlabOutputBag"
PetscErrorCode PetscViewerBinaryMatlabOutputBag(PetscViewer viewer, const char name[], PetscBag bag)
{
  FILE          *info;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetInfoPointer(viewer,&info);CHKERRQ(ierr);
  ierr = PetscBagView(bag,viewer);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%--- begin code written by PetscWriteOutputBag ---%\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%$$ Set.%s = PetscBinaryRead(fd);\n",name);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%--- end code written by PetscWriteOutputBag ---%\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
/*@C
  PetscViewerBinaryMatlabOutputVec - Write matlab code to info file to read a (non DA) Vec from binary.

  Input Parameters:
+ viewer - The viewer object
. name - The name of the field variable to be written
- vec -The Vec containing the field data

  Level: intermediate

  .seealso: PetscViewerBinaryMatlabOpen(), PetscViewerBinaryMatlabOutputBag(), PetscViewerBinaryMatlabOutputVecDA()
@*/
#undef __FUNCT__
#define __FUNCT__ "PetscViewerBinaryMatlabOutputVec"
PetscErrorCode PetscViewerBinaryMatlabOutputVec(PetscViewer viewer, const char name[], Vec vec)
{
  FILE          *info;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetInfoPointer(viewer,&info);CHKERRQ(ierr);
  ierr = VecView(vec,viewer);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%--- begin code written by PetscWriteOutputVec ---%\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%$$ Set.%s = PetscBinaryRead(fd);\n",name);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%--- end code written by PetscWriteOutputVec ---%\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerBinaryMatlabOutputVecDA - Write matlab code to info file to read a DA's Vec from binary.

  Input Parameters:
+ viewer - The viewer object
. name - The name of the field variable to be written
. vec - The Vec containing the field data to output
- da - The DA governing layout of Vec

  Level: intermediate

  Note: This method requires dof names have been set using DASetFieldName().

  .seealso: PetscViewerBinaryMatlabOpen(), PetscViewerBinaryMatlabOutputBag(), PetscViewerBinaryMatlabOutputVec(), DASetFieldName()
@*/
#undef __FUNCT__
#define __FUNCT__ "PetscViewerBinaryMatlabOutputVecDA"
PetscErrorCode PetscViewerBinaryMatlabOutputVecDA(PetscViewer viewer, const char name[], Vec vec, DA da)
{
  MPI_Comm       comm;
  FILE          *info;
  char          *fieldname;
  PetscInt       dim,ni,nj,nk,pi,pj,pk,dof,n;
  PetscTruth     flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetInfoPointer(viewer,&info);CHKERRQ(ierr);
  ierr = DAGetInfo(da,&dim,&ni,&nj,&nk,&pi,&pj,&pk,&dof,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecView(vec,viewer);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%--- begin code written by PetscWriteOutputVecDA ---%\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%$$ tmp = PetscBinaryRead(fd); \n");CHKERRQ(ierr);
  if (dim == 1) { ierr = PetscFPrintf(comm,info,"%%$$ tmp = reshape(tmp,%d,%d);\n",dof,ni);CHKERRQ(ierr); }
  if (dim == 2) { ierr = PetscFPrintf(comm,info,"%%$$ tmp = reshape(tmp,%d,%d,%d);\n",dof,ni,nj);CHKERRQ(ierr); }
  if (dim == 3) { ierr = PetscFPrintf(comm,info,"%%$$ tmp = reshape(tmp,%d,%d,%d,%d);\n",dof,ni,nj,nk);CHKERRQ(ierr); }

  for(n=0; n<dof; n++) {
    ierr = DAGetFieldName(da,n,&fieldname); CHKERRQ(ierr);
    ierr = PetscStrcmp(fieldname,"",&flg);CHKERRQ(ierr);
    if (!flg) {
      if (dim == 1) { ierr = PetscFPrintf(comm,info,"%%$$ Set.%s.%s = squeeze(tmp(%d,:))';\n",name,fieldname,n+1);CHKERRQ(ierr); }
      if (dim == 2) { ierr = PetscFPrintf(comm,info,"%%$$ Set.%s.%s = squeeze(tmp(%d,:,:))';\n",name,fieldname,n+1);CHKERRQ(ierr); } 
      if (dim == 3) { ierr = PetscFPrintf(comm,info,"%%$$ Set.%s.%s = permute(squeeze(tmp(%d,:,:,:)),[2 1 3]);\n",name,fieldname,n+1);CHKERRQ(ierr);}
    }
  }
  ierr = PetscFPrintf(comm,info,"%%$$ clear tmp; \n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,info,"%%--- end code written by PetscWriteOutputVecDA ---%\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
