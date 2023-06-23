% Clear command window and figure
clc;
close all;
clear;

simSize = [128 128];
r = rr(simSize);

% Define groundtruth object
if(1)       % spokes-target
    obj = readim('resolution_512');
    obj = obj(373+(-64:64-1),248+(-64:64-1));

elseif(0)   % Actin filaments
    obj = readtimeseries('Zeiss_Actin_525nm_crop.tif'); 
    obj = squeeze(sum(obj,[],3)); obj = extract(obj,[128 128],[329 64]);

elseif(1)   % cameraman
    obj = readim('cameraman'); obj = resample(obj,[0.5 0.5]);
elseif(0)   % rectangles      
    obj = readim('srect3'); obj = extract(obj,[128 128]);
elseif(0)
    obj = newim(simSize); obj(64-2,64) = 100; obj(64+2,64) = 100;
else
    obj = 0.5*(1+cos((0.010*r^2)));
end
obj = DampEdge(obj);
% [p,s_f]=PS_decomposition(obj)
obj = obj - min(obj)+0;
obj = obj/max(obj) * 1e+2;


% Create PSF & OTF
R = 20; % R = 30; % R = 10;
psf = abssqr(ift2d(r<R)); psf = psf/sum(psf) * sqrt(prod(size(psf)));
otf = ft2d(psf);

myMask = r<=2*R;

% Define forward and backwards model
Fwd = @(x) real(ift2d(ft2d(x) * otf));
Bwd = @(x) real(ift2d(ft2d(x) * conj(otf)));
Ansc = @(x) 2*real(sqrt(x+3/8));
iAnsc = @(x) (x/2)^2 - 3/8;
% iAnsc = @(x) 1/4*x^2 - 1/8 + 1/4*sqrt(3/2)*1/(x+1e-12) -11/8*1/(x^2+1e-12) +5/8*sqrt(3/2)*1/(x^3+1e-12);


% Simulate image formation & shot noise
img = Fwd(obj); img(img<0) = 0;
img = noise(img,'poisson');
% img = noise(img,'gaussian',2);  % additive Gaussian makes problem

% Inverse Anscombe transform
% img = iAnsc(Ansc(img)); img(img<0) = 0;

% Perform binomial splitting!
p = 0.90; 
% p = 0.50;
img_T = Random_Binomial(img,p);
img_V = img - img_T;


%% Training & Validation

% Create initial estimate & number of iterations
est = newim(img) + mean(img_T);
N = 100; N = 40; %N = 20;

% Initialize performance parameters
NCC_T = zeros(1,N);
L_T = zeros(1,N);
L_V = zeros(1,N);

% Perform RL-deconvolution, while calculating the performance parameter for
% the split case
Est_T = newim([size2d(est) N]);
for l=1:N
    convEst = Fwd(est); convEst(convEst<0) = 0;
    
    ratio = img_T / (convEst + 1e-12);
    convRatio = Bwd(ratio); convRatio(convRatio<0) = 0;
    convRatio = convRatio / Bwd(newim(est)+1.0);
    
    est = est * convRatio;
    Est_T(:,:,l-1) = est;
    
    L_T(1,l) = sum(-Fwd(est) + img_T * log(Fwd(est)+1e-12));
    L_V(1,l) = sum(-Fwd(est) + p/(1-p) * img_V * log(Fwd(est)+1e-12));

    NCC_T(1,l) = mean((obj-mean(obj)) * (est-mean(est))) / (std(obj) * std(est));

end

% Plot the results
figure
sp1=subplot(1,3,1);
plot(L_T,'b');
set(gca,'YTickLabels',{})
xlabel('Iteration')
ylabel('- Log-Likelihood')
title('Training')
[val,I_T] = max(L_T);
hold on
plot(I_T,val,'bo')

sp2=subplot(1,3,2);
plot(L_V,'k');
set(gca,'YTickLabels',{})
xlabel('Iteration')
ylabel('- Log-Likelihood')
title('Validation')
[val,I_V] = max(L_V);
hold on
plot(I_V,val,'ko')


%% Deconvolve non-split image for comparison

% Make RL-deconvolution of non-split image as a reference
est = newim(img) + mean(img);
NCC = zeros(1,N);
Est = newim([size2d(est) N]);
for l=1:N
    convEst = Fwd(est); convEst(convEst<0) = 0;
    ratio = img / (convEst + 1e-6);
    convRatio = Bwd(ratio); convRatio(convRatio<0) = 0;
    convRatio = convRatio * Bwd(newim(est)+1.0);
    
    est = est * convRatio;
    Est(:,:,l-1) = est;
    
    NCC(1,l) =  mean( (obj - mean(obj)) * (est - mean(est)) ) / (std(obj) * std(est));
end

% Plot new results
sp3=subplot(1,3,3);
plot(NCC,'r-')
hold on
plot(NCC_T,'k--')
set(gca,'YTickLabels',{})
xlabel('Iteration')
ylabel('NCC-value')

[val,I] = max(NCC);
hold on
plot(I,val,'ro')

title(['Full: ' num2str(round(NCC(:,I)*100)/100) '; Split: ' num2str(round(NCC_T(:,I_V)*100)/100)],'FontSize',10)
plot(I_V,NCC_T(:,I_V),'ko')



%% Visually compare results
cat(3,img_T,img_V,img,Est_T(:,:,I_V-1),Est(:,:,I-1),obj) 
% cat(4,Est_T,Est)

disp('NCC-values:')
[NCC_T(:,I_V) NCC(:,I)]

%% Here is the unsupervised-version 
% Replace the for-loop with a while-loop
if(0)
    est = newim(img) + mean(img_T);

    j = 0; est_old = est;
    lv_old = sum(-Bwd(est) + img_V * log(Bwd(est)+1e-12)); lv = lv_old + 1; 
    while lv - lv_old >0 
        j=j+1;
        disp(['Iteration #  :  ' num2str(j)])
        
        convEst = Fwd(est); convEst(convEst<0) = 0;
        ratio = img_T / (convEst + 1e-6);
        convRatio = Bwd(ratio); convRatio(convRatio<0) = 0;
        convRatio = convRatio / Bwd(newim(est)+1.0);

        est_old = est;
        est = est * convRatio;

        lv_old = lv;
        lv = sum(-Bwd(est) + p/(1-p) * img_V * log(Bwd(est)+1e-12));

    end

    est = est_old;
    cat(3,img,est,obj)
end

%% Fourier space: error vs spatial frequencies
if(1)

FTError_T = abssqr(ft2d(Est_T*1/p) - ft2d(obj)) * myMask;
FTError = abssqr(ft2d(Est) - ft2d(obj)) * myMask;

FTErrLine_T = newim([64 N]);
FTErrLine = newim([64 N]);
for l=0:1:N-1
    FTErrLine_T(:,l) = radialProj(FTError_T(:,:,l),'mean');
    FTErrLine(:,l) = radialProj(FTError(:,:,l),'mean');
end

cmap = colormap(hot(N));
figure; j = 0;
for l=linspace(0,size(Est_T,3)-1,size(cmap,1))
    subplot(1,2,1); plot(FTErrLine_T(:,l),'Color',cmap(j+1,:)); hold on
    subplot(1,2,2); plot(FTErrLine(:,l),'Color',cmap(j+1,:)); hold on
    
    j = j +1;
end
sp1=subplot(1,2,1);
colormap(hot(N))
h=colorbar; title(['Training: ' num2str(p*100) ' %'])
set(h,'YTickLabel',{0 20 40 60 80 100})
% ylabel(h,'Iteration #')
xlabel('Spatial frequencies / (a.u.)')
ylabel('MSE / (a.u.)')
sp2=subplot(1,2,2);
colormap(hot(N))
h=colorbar; title('No splitting')
set(h,'YTickLabel',{0 20 40 60 80 100})
ylabel(h,'Iteration #')
xlabel('Spatial frequencies / (a.u.)')
ylabel('MSE / (a.u.)')

subplot(1,2,1); plot(FTErrLine_T(:,I_V-1),'c--','LineWidth',1)
plot(FTErrLine(:,I-1),'g--','LineWidth',1)
subplot(1,2,2); plot(FTErrLine(:,I-1),'g--','LineWidth',1)
plot(FTErrLine_T(:,I_V-1),'c--','LineWidth',1)

linkaxes([sp1 sp2],'xy')

% axes('Position',[0.35 0.45 0.4 0.4]);
% box on
% plot(FTErrLine_T(:,I_V-1),'b-','LineWidth',1)
% hold on
% plot(FTErrLine_T(:,I-1),'g--','LineWidth',1)


end

%% Relative spectral energy regain 
if(0)
Delta_ER_T = newim([64 N]); 
Delta_ER = newim([64 N]); 

myMask = newim(size2d(obj))+1.0;

tempA = ft2d(Est_T*1/p); tempB = ft2d(obj); tempC = ft2d(Est);
for l=0:N-1
    Delta_ER_T(:,l) = radialProj(abssqr(tempA(:,:,l) - tempB)*myMask,'mean');
    Delta_ER(:,l) = radialProj(abssqr(tempC(:,:,l) - tempB)*myMask,'mean');
end
ER = radialProj(abssqr(ft2d(obj))*myMask);

GR_T = (ER - Delta_ER_T) / (ER + 1e-12);
GR = (ER - Delta_ER) / (ER + 1e-12);

cmap = colormap(hot(N));
figure; j = 0;
for l=linspace(0,size(Est_T,3)-1,size(cmap,1))
    subplot(1,2,1); plot(GR_T(:,l),'Color',cmap(j+1,:)); hold on
    subplot(1,2,2); plot(GR(:,l),'Color',cmap(j+1,:)); hold on
    
    j = j +1;
end
sp1=subplot(1,2,1);
colormap(hot(N))
h=colorbar; title(['Training: ' num2str(p*100) ' %'])
set(h,'YTickLabel',{0 20 40 60 80 100})
% ylabel(h,'Iteration #')
xlabel('Spatial frequencies / (a.u.)')
ylabel('Relative energy regain')
sp2=subplot(1,2,2);
colormap(hot(N))
h=colorbar; title('No splitting')
set(h,'YTickLabel',{0 20 40 60 80 100})
ylabel(h,'Iteration #')
xlabel('Spatial frequencies / (a.u.)')
ylabel('Relative energy regain')

subplot(1,2,1); plot(GR_T(:,I_V-1),'c--','LineWidth',1)
plot(GR(:,I-1),'g--','LineWidth',1)
subplot(1,2,2); plot(GR(:,I-1),'g--','LineWidth',1)
plot(GR_T(:,I_V-1),'c--','LineWidth',1)

linkaxes([sp1 sp2],'xy')
set(gca,'YLim',[-0.5 1])

end

