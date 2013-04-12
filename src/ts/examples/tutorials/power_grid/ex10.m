function[] = ex11(xmin,xmax,ymin,ymax,nx,ny)
% ex11 - Plots the contour of the PDF solution
Pm = 0.9;
Pmax2 = 1.1358/0.745;
delta_s = asin(Pm/Pmax2); % Stable equilibrium point
delta_u = pi - delta_s; % Unstable equilibrium point

%%% 3rd order upwinding scheme
A = PetscBinaryRead('ex10output','cell',10000);
l = size(A); l = l(2);
%n = size(A{1}); n = sqrt(n(1));
for i=1:l; A{i} = reshape(A{i},nx,ny)'; end
B = A; C = A;

xvec = linspace(xmin,xmax,nx);
yvec = linspace(ymin,ymax,ny);

for i=1:l
    for j=1:ny; A{i}(j,:) = min(A{i}(j,:)); end
    for j=1:ny; B{i}(j,:) = B{i}(j,:) - min(B{i}(j,:)); end
end

% figure(1)
% i = 1;
% for k = 1:1:l
%     [maxi,yi] = max(C{k});
%     [maxi,xi] = max(maxi);
%     yi = yi(xi);
%     x0max(i) = xvec(xi); 
%     y0max(i) = yvec(yi);
%     i = i+1;
% end
% plot(x0max,y0max,'ko','Markersize',10); hold on; grid on;
% set(gca,'FontSize',20);
% xlabel('\theta','FontSize',20);
% ylabel('\omega','FontSize',20);
% axis([xmin xmax ymin ymax]);

x = repmat(xvec,ny,1);
y = repmat(yvec',1,nx);

dx = (xmax-xmin)/(nx-1); 
dy = yvec(2)-yvec(1);

% Prepare the new file for Peng
% vidObj = VideoWriter('swing_pdf_fault');
% vidObj.FrameRate = 5;
% open(vidObj);
%axis tight
f1 = figure(2),clf;
set(f1,'Position',[0,0,1440,900]);
h2 = gca;
set(h2,'nextplot','replacechildren');

for k = 1:1:l
   max_p = max(max(C{k})); 
   sum_p = sum(sum(C{k}));
   th = title(['Sum(p(\Theta,\Omega;t))*dx*dy =',num2str(sum_p)],'FontSize',20);
   c1 = contourf(x,y,C{k}/max_p);
   colormap jet;
   colorbar;
   set(gca,'FontSize',20);
   axis([xmin xmax ymin ymax]);
   yh = ylabel('\Omega','FontSize',20);
   xh = xlabel('\Theta','FontSize',20,'Rotation',90.0);
   % Plot stable equilibrium point
   h1 = text(delta_s,1.0,'\delta^s','FontSize',20,'Color','Red');
   % Plot unstable equilibrium point
   h2 = text(delta_u,1.0,'\delta^u','FontSize',20,'Color','Red');
   % Plot point having the largest probability
   [maxi,yi] = max(C{k});
   [maxi,xi] = max(maxi);
   yi = yi(xi);
   x0max = xvec(xi); 
   y0max = yvec(yi);
   h3 = text(x0max,y0max,'X','FontSize',20,'Color','Red');
   currFrame = getframe(gcf);
%   Write each frame to the file.
%   writeVideo(vidObj,currFrame);
%   mov(:,k) = getframe(gcf);
end