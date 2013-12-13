% test minq.m, minqdef.m, minqsep.m

disp('test of minq.m')

n=7;
m=10;
p=0.8;		% approx. fraction of activities and equalities
newdata=1;	% new data?

if newdata,
  % create random data satisfying the KKT conditions
  A=rand(m,n);
  c=rand(n,1);
  d=rand(n,1);
  ydes=rand(m,1)-p;
  act=( ydes>p );		% indices of nonactive inequalities
  ydes(act)=0*ydes(act);
  eq=( ydes<0 );		% indices of equations
  xdes=(A'*ydes-c)./d;
  res=rand(m,1);
  res(~act)=0*res(~act);
  b=A*xdes-res;
  save minq_data
 prt=0;
else
  load minq_data
  warning debug
  prt=1;
end;


for cas=1:2,
  if cas==1, 
    disp('test of minqsep.m');
    [x,y,ier]=minqsep(c,d,A,b,eq,prt);
  else        
    disp('test of minqdef.m');
    [x,y,ier]=minqdef(c,diag(d),A,b,eq,prt);
  end;

  compldes=[ydes,A*xdes-b];
  compl=[y,A*x-b];
  act_eq=[act,eq]'
  ydif=[ydes,y]
  disp('usually two equal columns');
  disp('but sometimes the dual is not unique')
  xdif=[xdes,x]
  disp('xdif should have two equal columns')
  ier
  if cas==2, break; end;
  cont=input('enter return (next test) or 0 (quit)>');
  if isempty(cont),	% continue
  elseif cont==0,      return; 
  end;
end;
