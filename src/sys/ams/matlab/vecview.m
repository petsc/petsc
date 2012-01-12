function vecview(comm,memory)
%
%  Displays a two dimensional vector with a surface plot
%
[v,changed,step] = ams_get_variable(comm,memory,'values','changed');
if (~changed) 
  return
end

[n,m] = size(v);
if (n > 1 & m > 1)
  figure(2);
  surf(v);
else 
  figure(2);
  plot(v);
end
Name = ams_get_variable(comm,memory,'Name');
if (~isempty(Name))
  title(Name);
end
