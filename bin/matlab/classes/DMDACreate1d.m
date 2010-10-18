function [da,err] = DMDACreate1d(wrap,M,dof,s,lx)
  da = DM();
  err = da.SetType('da'); 
  da.SetPeriodicity(wrap);
  da.SetSizes([M 1 1]);
  da.SetDof(dof);
  da.SetStencilWidth(s);
  da.SetDim(1);
  err = da.SetUp(); 

