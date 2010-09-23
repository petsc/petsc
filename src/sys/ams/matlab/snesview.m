function kspview(comm,memory)
%
%  
%
[it,changed,step] = ams_get_variable(comm,memory,'Iteration','changed');
[residual] = ams_get_variable(comm,memory,'Residual');
['Current SNES iteration ' int2str(it) ' Residual norm ' num2str(residual)]