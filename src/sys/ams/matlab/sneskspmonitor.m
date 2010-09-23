function it_monitor(hostname,port)
%
%   Crude example of monitoring a series of 
% KSP in SNES solves. Just publish the SNES object in 
% the PETSc program.
%
if (nargin == 0) 
  commlist = ams_connect;
elseif (nargin == 1)
  commlist = ams_connect(hostname);
else
  commlist = ams_connect(hostname,port);
end

comm    = ams_comm_attach(commlist(1,:));                  
memlist = ams_comm_get_memory_list(comm); 
% Find the memory that is associated with the KSP
ind = strmatch('KSP',memlist);              
memory = ams_memory_attach(comm,memlist(ind,:));  
% We are interested in the Iteration and Residual fields
its     = ams_memory_get_field_info(memory,'Iteration');
rnorm   = ams_memory_get_field_info(memory,'Residual');


changed = 0;
b       = 0;
while (1)
  % wait until the residual norm has been updated
  while (changed == 0) 
    ams_memory_update_recv_begin(memory);                    
    [changed,step] = ams_memory_update_recv_end(memory); 
  end
  its     = [its  ams_memory_get_field_info(memory,'Iteration')];
  rnorm   = [rnorm ams_memory_get_field_info(memory,'Residual')];

  [mits,nits] = size(its);
  cnt = 0;
  g(1) = its(1);
  for i=2:nits,
   if (its(i) < its(i-1)) 
     cnt = cnt + its(i-1);
   end
   g(i) = its(i) + cnt;
  end

  if (b ~= 0) 
    delete(b)
  end
  clf
  semilogy(g,rnorm,'b*-'); 
  axis tight;
  ylabel('Residual Norm');
  xlabel('Total Linear Iterations');

  h = figure(1);
  a = get(h,'CurrentAxes');
  f = str2num(get(a,'XTickLabel'))';
  [mlabel,nlabel] = size(f);

  labels = interp1(g,its,f); 
  a1 = get(a,'XTick');
  a2 = get(a,'XLim');
  b = axes('YTick',0,'Color','none','XAxis','top','XLim',a2,'XTick',a1,'XTickLabel',labels);
  c = get(b,'XLabel');
  set(c,'String','Current Linear Iteration');
  pause(1);
  
  changed = 0;
end

