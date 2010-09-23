function petscview(host,port)
%
%  petscview(host,port)
%
%   Displays all the published PETSc objects
%
%  Input Parameters:
%   host - name of publisher machine
%   port - port number on publisher machine (optional)
%
%  Output Parameter:
%   
%
%seealso: sneskspmonitor()
%
%  Notes: this Matlab routine does not have a corresponding routine 
%         in the underlying AMS API.
%
if (nargin == 0) 
  host = ams_get_servername;
  if (isempty(host))
    exit
  end
end
if (nargin < 2) 
  port = -1;
end

figure(1);
%
% Keeps the graphics handles associated with the buttons
%
global PetscViewButtons

[name,comm] = ams_view_memory(host,port,'PETSc');
%

%  make sure that PETSc is one of the communicators
I = strmatch('PETSc',name);
if isempty(I)
  'Error: not a PETSc publisher'
  return;
end
%
tname = name;
[m,n] = size(name);
name  = [];
for i=1:m,
 ntmp = [tname(i,8:n) blanks(16)];
 name = [name ; ntmp(1,1:16)];
end
%
% Get all Id and Parent ID values
%
Id       = [];
ParentId = [];
Type     = [];
Class    = [];
Name     = [];

%
%  Put list of memories into format for ams_memory_attach()
%
mname = [];
pname = [];
cnt   = 0;
for i=1:m
  [mem,dummy,ierr] = ams_memory_attach(comm,deblank(name(i,:)));
  if (ierr == 0) 
    [list,ierr] = ams_memory_get_field_list(mem); 
    if (ierr == 0)
      f = strmatch('ParentId',list);
      if (~isempty(f))    % it is a PETSc object
       if (cnt > 0) 
          mname = strcat(mname,'|');
          mname = strcat(mname, deblank(name(i,:)));
        else
          mname = deblank(name(i,:));
        end
        cnt = cnt + 1;
        tmpp  = [name(i,:) blanks(16)];
        pname = [pname ; tmpp(1:16)];
      end
    end
  end
end

if isempty(mname)
  'No PETSc objects published; did you forget -ams_publish_objects'
  return 
end

[memory,dummy,ierr] = ams_memory_attach(comm,mname);
if (ierr ~= 0) 
  ['petscview: one of the memories no longer exists']
end

for i=1:cnt,
  Id          = [ Id ; ams_get_variable(comm,memory(i),'Id')];
  ParentId    = [ ParentId ; ams_get_variable(comm,memory(i),'ParentId')];
  stype       = [ams_get_variable(comm,memory(i),'Type') blanks(9)];
  Type        = [ Type; stype(1:8)];
  sclass      = [ams_get_variable(comm,memory(i),'Class') blanks(9)];
  Class       = [ Class; sclass(1:8)];
%
% For objects with names use them instead of memory name
%
  zname       = [ams_get_variable(comm,memory(i),'Name') blanks(16)];
  if (zname(1) ~= ' ') 
    pname(i,:) = zname(1:16);
  end;
end
sname = cellstr(pname(:,1:12));

%
%  For each object i, Parents(i) is the object parent
%
MaxId     = max(Id);
Parents   = zeros(MaxId,1);
IdToShort = zeros(MaxId,1);
for i=1:cnt,
  IdToShort(Id(i)) = i;
end


for i=1:cnt,
  if (ParentId(i) > 0)
    if ~IdToShort(ParentId(i))
%      parent was mistakenly not published hence we do not keep reference
      ParentId(i) = 0;
    else 
      Parents(Id(i)) = ParentId(i);
    end
  end
end

%
%  Compute the number of generations
maxlevel = 1;
for i=1:cnt,
  cnt1 = 1;
  j   = ParentId(i);
  while (j > 0)
    cnt1 = cnt1 + 1; 
    j   = Parents(j);
  end
  maxlevel = max(maxlevel,cnt1);
end

%
% Determine generation of each object
level = zeros(m,1);
for i=1:cnt,
  cnt1 = 1;
  j   = ParentId(i);
  while (j > 0)
    cnt1 = cnt1 + 1; 
    j   = Parents(j);
  end
  level(i) = cnt1;
end

%
%  Generate a list of objects on each level
levels  = zeros(m,maxlevel);
plevels = zeros(m,maxlevel);
nlevels = zeros(maxlevel,1);
for i=1:cnt,
  nlevels(level(i)) = nlevels(level(i)) + 1;
  levels(level(i),nlevels(level(i)))  = Id(i);
  plevels(level(i),nlevels(level(i))) = ParentId(i);
end     
maxx = max(nlevels);
%
%  Sort objects on each level by parent
for i=2:maxlevel,
  [ss,ii] = sort(plevels(i,1:nlevels(i)));
  levels(i,1:nlevels(i))  = levels(i,ii);
  plevels(i,1:nlevels(i)) = ss;
end
%
%
% Create an array of x y locations for icons
xes = zeros(maxlevel,maxx);
yes = zeros(maxlevel,maxx);
dy = .45/maxlevel; dx = .45/maxx;
y = .95 - dy;
for i=1:maxlevel
  x = .05;
  for j=1:nlevels(i,1);
    xes(i,j) = x;
    yes(i,j) = y;
    X(levels(i,j)) = x;
    Y(levels(i,j)) = y;
    x = x + 2*dx;
  end
  y = y - 2*dy;
end
%
PetscViewButtons = [];
clf reset
axis off
ax = gca;
set(ax,'position',[0 0 1 1])
axis manual
axis([0 1 0 1])

%

% Create a button for each object
buttons = [];
for i=1:maxlevel
  for j=1:nlevels(i,1);
    x = xes(i,j);
    y = yes(i,j);
    ll    = max(length(char(sname(IdToShort(levels(i,j))))),...
                length(deblank(Type(IdToShort(levels(i,j),:)))));
    ll    = max(ll,length(char(Class(IdToShort(levels(i,j))))));
    l1    = [char(sname(IdToShort(levels(i,j)))) blanks(ll)];
    l2    = [Class(IdToShort(levels(i,j)),:)     blanks(ll)];
    l3    = [Type(IdToShort(levels(i,j)),:)     blanks(ll)];
    label = ['text(.5,.5,[ ''' l1(1:ll) ''' ;  ''' l2(1:ll) ''' ;  ''' l3(1:ll) '''] ,''Horiz'',''center'')'];

    callback   = ['objectview(' int2str(comm) ',' int2str(memory(IdToShort(levels(i,j)))) ')'];
    buttons = [buttons  btngroup('GroupID', 'TestGroup',...
                                 'ButtonID', 'B1',...
                                 'Position', [x y dx dy],...
                                 'Callbacks',callback,...
                                 'IconFunctions', label)];
%   draw line to parent
    if (i > 1) 
      line([x+.5*dx X(Parents(levels(i,j)))+.5*dx],[y+dy,Y(Parents(levels(i,j)))]);
    end
  end
end
PetscViewButtons = buttons;





