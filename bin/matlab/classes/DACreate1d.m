function [da,err] = DACreate1d(wrap,M,dof,s,lx)
  da = DA();
  da.SetPeriodicity(wrap);
  da.SetSizes([M 1 1]);
  da.SetDof(dof);
  da.SetStencilWidth(s);
  err = da.SetType('da1d'); 

