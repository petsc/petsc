function [da,err] = PetscDMDACreate1d(wrap,M,dof,s,lx)
  if (nargin == 4) 
    lx = 0;
  end
  da = PetscDM();
  err = da.SetType('da'); 
  da.SetPeriodicity(wrap);
  da.SetSizes([M 1 1]);
  da.SetDof(dof);
  da.SetStencilWidth(s);
  da.SetDim(1);
  err = da.SetUp(); 

