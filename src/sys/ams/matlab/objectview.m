function objectview(comm,memory)
%
%  Calls the viewer appropriate for the particular object
%
class = ams_get_variable(comm,memory,'Class');
if (strncmp(class,'Vec',3))
  vecview(comm,memory);
elseif (strncmp(class,'KSP',3))
  kspview(comm,memory);
elseif (strncmp(class,'SNES',4))
  snesview(comm,memory);
else
  'Cannot view class yet ',class
end

