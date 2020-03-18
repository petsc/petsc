clear all;
close all;

fontsize_labels = 14;
fontsize_grid   = 12;
fontname = 'Times';

%the files optimize** contain
%Grad    --- gradient, 
%Init_ts --- initial condition of forward problem
%Init_adj--- initial condition of backward problem

%additionally 
%xg - the grid
%obj- the objective function
%ic - initial condition (the one to be optimized)


% run('PDEadjoint/optimize06.m')
% figure(2)
% plot(xg,obj,'k-','Markersize',10,'Linewidth',2); drawnow
% hold on
% plot(xg,fwd,'b-','Markersize',10); drawnow
% plot(xg,Init_ts,'r*-','Markersize',10); drawnow
% plot(xg,Grad,'bs','Markersize',10); drawnow
% plot(xg,ic,'g-','LineWidth',2,'Markersize',12);

% figure(29)
% run('tss.m')
% plot(xg,init)
% hold on
% plot(xg,fin,'ro-')
% break
% figure(3)
% plot(xg,Init_adj,'k*-','Markersize',10); drawnow
% hold on
% plot(xg,Grad,'go-','Markersize',10); drawnow
% plot(xg,Init_adj,'r*-','Markersize',10); drawnow
% plot(xg,obj,'r*-','Markersize',10); drawnow
% figure(55)
% plot(xg,Init_adj,'k*-','Markersize',10); drawnow
% hold on
% plot(xg,-2*(obj-fwd),'ro','Markersize',10); drawnow

figure(15)
for ii=11:13
file=sprintf('PDEadjoint/optimize%02d.m',ii);  
run(file)
plot(grid,Init_ts,'ko-','Markersize',10); drawnow;
hold on
%plot(grid,temp,'c*','Markersize',10); drawnow;
plot(grid,exact,'r*','Markersize',10); drawnow;
%plot(grid,Curr_sol,'g-','Markersize',10); drawnow;
%plot(grid,Init_adj,'bo-','Markersize',10); drawnow;
end
xlabel('x (GLL grid)');
ylabel('f(x)- objective');
% 
% break
% tt=senmask.*obj;
% tt(abs(tt)==0)=NaN;
% plot(xg,tt,'ks-','Markersize',10); drawnow;
% break
% Err1=Err; TAO1=TAO; 
% 
% %break
% figure(15)
% for ii=35:40
%     file=sprintf('PDEadjoint_hc/optimize%02d.m',ii);  
% run(file)
% plot(xg,Init_ts,'go-','Markersize',10); drawnow;
% hold on
% plot(xg,obj,'r-','Markersize',10); drawnow;
% plot(xg,fwd,'bo-','Markersize',10); drawnow;
% end
% xlabel('x (GLL grid)');
% ylabel('f(x)- objective');
% 
% 
% figure(99)
% semilogy(Err1,'k-','Markersize',6,'LineWidth',2); drawnow;
% hold on
% semilogy(Err,'r-','Markersize',6,'LineWidth',2); drawnow;
% 
% 
% set(gca,'FontName',fontname)
% set(gca,'FontSize',fontsize_grid)
% set(gca,'FontSize',fontsize_labels)
% 
% legend('Discrete Objective','Continous Objective')
% xlabel('Iterations');
% ylabel('Error solution');
% grid on
% legend boxoff;
% axis tight; axis square
% 
% break
% 
% figure(12)
% plot(xg,objk,'b*-','Markersize',6,'LineWidth',2); drawnow;
% 
% hold on
% plot(xg,Init_ts,'ro-','Markersize',6,'LineWidth',2); drawnow;
% plot(xg,ic,'ks-','Markersize',6,'LineWidth',2); drawnow;
% legend('Objective','Optimal','Starting')
% legend boxoff
% xlabel('GLL grid');
% ylabel('Diffusion solution (Data assimilation)');
% set(gca,'FontName',fontname)
% set(gca,'FontSize',fontsize_grid)
% set(gca,'FontSize',fontsize_labels)
% figure(95)
% t=0.6; mu=0.001;x=xg;
% plot(xg,2.0*mu*pi*sin(pi*x).*exp(-pi^2*t*mu)./(2.0+exp(-pi^2*t*mu).*cos(pi*x)));
% 
% break
% figure(1)
% plot(xg,Init_ts,'ro-','Markersize',8,'LineWidth',2); drawnow;
% %break
% figure(2);set(gca,'FontSize',18);hold on
% 
% run('PDEadjoint/optimize00.m')
% plot(xg,Grad,'k*-');
% run('PDEadjoint/optimize04.m')
% plot(xg,Grad,'ro-');
% 
% set(gca,'FontName',fontname)
% set(gca,'FontSize',fontsize_grid)
% set(gca,'FontSize',fontsize_labels)
% 
% xlabel('x (GLL grid)');
% ylabel('f(x)- objective');
% 
% legend('Grad at it=0','Grad at it=1')
% 
% figure(10)
% run('fd.m')
% %plot(gradj)
% plot(xg,gradj./Mass,'ro-','LineWidth',2,'Markersize',12);
% hold on
% run('PDEadjoint/optimize01.m')
% plot(xg,Grad,'k*-','LineWidth',2,'Markersize',10);
% 
% set(gca,'FontName',fontname)
% set(gca,'FontSize',fontsize_grid)
% set(gca,'FontSize',fontsize_labels)
% 
% legend('Gradient FD','Gradient Adjoint')
% xlabel('x (GLL grid)');
% ylabel('Gradient');
% axis tight; axis square
% 
% errgrad=max(abs(gradj./Mass-Grad))
% 
% 
% figure(21)
% semilogy(1:21,TAO,'r','LineWidth',2)
% % hold on
% % semilogy(1:31,L2,'r','LineWidth',2)
% grid on
% xlabel('No iterations');
% ylabel('Cost function');
% set(gca,'FontName',fontname)
% set(gca,'FontSize',fontsize_grid)
% set(gca,'FontSize',fontsize_labels)
% % 
% % legend('TAO','User')
