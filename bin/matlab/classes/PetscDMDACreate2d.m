function [da,err] = DMDACreate2d(wrap,stentype,M,N,mm,nn,dof,s,lx,ly)
  da = DM();
  err = da.SetType('da'); 
  da.SetPeriodicity(wrap);
  da.SetSizes([M N 1]);
  da.SetDof(dof);
  da.SetStencilType(stentype);
  da.SetStencilWidth(s);
  da.SetDim(2);
  err = da.SetUp(); 

