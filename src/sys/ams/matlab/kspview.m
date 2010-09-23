function kspview(comm,memory)
%
%  
%
[it,changed1,step1] = ams_get_variable(comm,memory,'Iteration','changed');
[residual] = ams_get_variable(comm,memory,'Residual');
['Current KSP iteration ' int2str(it) ' Residual norm ' num2str(residual)]