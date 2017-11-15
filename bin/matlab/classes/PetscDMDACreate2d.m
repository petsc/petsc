function [da,err] = PetscDMDACreate2d(wrapx,wrapy,stentype,M,N,mm,nn,dof,s)
  da = PetscDM();
  err = da.SetType('da'); 
  da.SetBoundaryType(wrapx,wrapy,PetscDM.BOUNDARY_NONE);
  da.SetSizes([M N 1]);
  da.SetDof(dof);
  da.SetStencilType(stentype);
  da.SetStencilWidth(s);
  da.SetDim(2);
  da.SetFromOptions();
  err = da.SetUp(); 

