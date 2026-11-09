%% Exp5_ANFIS_SoC.m
% Experiment 5 : ANFIS based battery SoC prediction using Z-score data normalization
%
%   Inputs : terminal voltage V (V), load current I (A), temperature T (deg C)
%   Output : state of charge SoC (%)
%   Model  : first-order Sugeno ANFIS, Gaussian membership functions, grid
%            partition, hybrid learning (least squares + back-propagation)
%
% The script splits the data, computes the Z-score parameters from the training
% set, selects the ANFIS structure on the validation set, tests the model, runs
% it as a real-time BMS routine and saves the trained model for the Simulink
% model (build_Exp5_Simulink.m).
%
% Requires: MATLAB with the Fuzzy Logic Toolbox.   Run:  >> Exp5_ANFIS_SoC

clear; clc; close all;
here   = fileparts(mfilename('fullpath'));   if isempty(here), here = pwd; end
figDir = fullfile(here,'figures_matlab');
if ~exist(figDir,'dir'), mkdir(figDir); end
logFile = fullfile(here,'Exp5_run.log');
if exist(logFile,'file'), delete(logFile); end
diary(logFile);

epochs    = 200;                              % training epochs
cands     = [2 2 2; 3 2 2; 4 2 2; 5 2 2; 3 3 3];   % membership functions for [V I T]
socWarn   = 20;   socCutoff = 5;              % low-SoC warning and load cut-off (%)
inNames   = {'Voltage','Current','Temperature'};
cBlue = [42 120 214]/255;  cOrange = [235 104 52]/255;  cAqua = [27 175 122]/255;  cGrey = [82 81 78]/255;
noMsg = evalfisOptions('OutOfRangeInputValueMessage','none','NoRuleFiredMessage','none', ...
                       'EmptyOutputFuzzySetMessage','none');

%% 1. Data, split and Z-score normalization
D = readmatrix(fullfile(here,'bms_dataset.csv'));
X = D(:,1:3);   y = D(:,4);   N = numel(y);

k  = mod((0:N-1)',20);                        % systematic split: of every 20 rows, 14 / 3 / 3
tr = k < 14;    va = k >= 14 & k < 17;    te = k >= 17;

mu    = mean(X(tr,:));                        % Z-score parameters from the TRAINING set only
sigma = std(X(tr,:));
Z     = (X - mu) ./ sigma;

fprintf('Samples: train %d, validation %d, test %d\n',sum(tr),sum(va),sum(te));
fprintf('Z-score parameters   mu = [%.4f %.4f %.4f]   sigma = [%.4f %.4f %.4f]\n',mu,sigma);

%% 2. Structure selection on the validation set
nC = size(cands,1);
fisC = cell(nC,1);  fis0C = cell(nC,1);  trC = cell(nC,1);  vaC = cell(nC,1);
sel  = zeros(nC,5);                           % rules, train, validation, test RMSE, best epoch
fprintf('\nStructure selection (Z-score inputs)\n');
for c = 1:nC
    [fisC{c},fis0C{c},trC{c},vaC{c}] = trainAnfis(Z(tr,:),y(tr),Z(va,:),y(va),cands(c,:),epochs);
    [~,bestEp] = min(vaC{c});
    sel(c,:) = [numel(fisC{c}.Rules), rmse(evalfis(fisC{c},Z(tr,:),noMsg),y(tr)), ...
                rmse(evalfis(fisC{c},Z(va,:),noMsg),y(va)), rmse(evalfis(fisC{c},Z(te,:),noMsg),y(te)), bestEp];
    fprintf('  MFs [%d %d %d]  rules %2d  train %.4f  validation %.4f  test %.4f\n',cands(c,:),sel(c,1:4));
end
[~,best] = min(sel(:,3));
fis = fisC{best};   fis0 = fis0C{best};   nMF = cands(best,:);
fis.Name = 'anfis_soc';
fprintf('Selected structure: [%d %d %d] -> %d rules (best epoch %d)\n',nMF,sel(best,1),sel(best,5));

%% 3. Reference models
beta = [Z(tr,:) ones(sum(tr),1)] \ y(tr);                                   % linear regression
[fisRaw,~,trRaw,vaRaw] = trainAnfis(X(tr,:),y(tr),X(va,:),y(va),nMF,epochs);    % same ANFIS, raw inputs
fisV = trainAnfis(Z(tr,1),y(tr),Z(va,1),y(va),nMF(1),epochs);              % voltage only

pred = struct('name',{'Linear regression','ANFIS, voltage only','ANFIS, raw inputs','ANFIS, Z-score inputs'}, ...
              'yhat',{[Z(te,:) ones(sum(te),1)]*beta, evalfis(fisV,Z(te,1),noMsg), ...
                      evalfis(fisRaw,X(te,:),noMsg), evalfis(fis,Z(te,:),noMsg)});
fprintf('\nTest-set performance (%d unseen samples)\n',sum(te));
res = zeros(numel(pred),4);
for m = 1:numel(pred)
    e = pred(m).yhat - y(te);
    res(m,:) = [sqrt(mean(e.^2)), mean(abs(e)), max(abs(e)), 1 - sum(e.^2)/sum((y(te)-mean(y(te))).^2)];
    fprintf('  %-24s RMSE %.4f %%  MAE %.4f %%  max %.4f %%  R2 %.6f\n',pred(m).name,res(m,:));
end

%% 4. Real-time BMS routine: one call per sampling instant
[ys,order] = sort(y(te),'descend');           % replay the test samples from full to empty
Xte = X(te,:);   Xs = Xte(order,:);
live = zeros(numel(ys),3);
for n = 1:numel(ys)
    [live(n,1),live(n,2),live(n,3)] = bmsStep(fis,mu,sigma,Xs(n,1),Xs(n,2),Xs(n,3),socWarn,socCutoff,noMsg);
end
fprintf('\nBMS replay of %d test samples: RMSE %.4f %%, warning flag on %d samples, cut-off flag on %d samples\n', ...
        numel(ys),rmse(live(:,1),ys),sum(live(:,2)),sum(live(:,3)));
fprintf('   V (V)    I (A)   T (C)   SoC actual  SoC ANFIS  warn cut\n');
show = round(linspace(1,numel(ys),8));
for n = show
    fprintf('  %6.3f  %7.3f  %6.2f   %8.3f   %8.3f    %d    %d\n',Xs(n,:),ys(n),live(n,:));
end

%% 5. Save the model for Simulink and the results for the report
writeFIS(fis,fullfile(here,'anfis_soc.fis'));
save(fullfile(here,'anfis_soc_model.mat'),'fis','mu','sigma','Xs','ys','socWarn','socCutoff','nMF');
R = struct('n',[sum(tr) sum(va) sum(te)],'mu',mu,'sigma',sigma,'cands',cands,'selection',sel, ...
           'selected',nMF,'models',{{pred.name}},'results',res, ...
           'replay',struct('rmse',rmse(live(:,1),ys),'warn',sum(live(:,2)),'cut',sum(live(:,3))), ...
           'sample',[Xs(show,:) ys(show) live(show,:)],'mfTrained',{mfParams(fis)},'mfInitial',{mfParams(fis0)}, ...
           'rawBestEpoch',find(vaRaw==min(vaRaw),1),'finalTrain',[trC{best}(end) trRaw(end)]);
fid = fopen(fullfile(here,'matlab_results.json'),'w');  fprintf(fid,'%s',jsonencode(R,'PrettyPrint',true));  fclose(fid);

%% 6. Figures
% (a) data set and the effect of Z-score normalization
f = newFig([60 60 1150 340]);  tiledlayout(f,1,3,'TileSpacing','compact','Padding','loose');
nexttile;  scatter(X(:,1),y,10,cBlue,'filled','MarkerFaceAlpha',0.6);  grid on;  box off;
xlabel('Terminal voltage (V)');  ylabel('SoC (%)');  title('(a) SoC against voltage');
nexttile;  boxchart(X(tr,:),'BoxFaceColor',cOrange,'MarkerColor',cOrange);  grid on;  box off;
set(gca,'XTickLabel',{'V (V)','I (A)','T (°C)'});  ylabel('Raw value');  title('(b) Raw inputs');
nexttile;  boxchart(Z(tr,:),'BoxFaceColor',cBlue,'MarkerColor',cBlue);  grid on;  box off;
set(gca,'XTickLabel',{'V','I','T'});  ylabel('Z-score');  ylim([-2.5 2.5]);  title('(c) Z-score normalized inputs');
saveFig(f,fullfile(figDir,'m_dataset_normalization.png'));

% (b) structure selection
f = newFig([60 60 720 340]);
b = bar(sel(:,3),0.55,'FaceColor',cBlue,'EdgeColor','none');  grid on;  box off;
b.FaceColor = 'flat';  b.CData = repmat(0.75*[1 1 1],nC,1);  b.CData(best,:) = cBlue;
set(gca,'XTickLabel',arrayfun(@(c) sprintf('[%d %d %d]\\newline%d rules',cands(c,:),sel(c,1)),1:nC,'UniformOutput',false));
text(1:nC,sel(:,3),compose('%.4f',sel(:,3)),'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',10);
xlabel('Membership functions for [V I T]');  ylabel('Validation RMSE (% SoC)');  ylim([0 1.18*max(sel(:,3))]);
title('ANFIS structure selection (selected structure in blue)');
saveFig(f,fullfile(figDir,'m_structure_selection.png'));

% (c) hybrid-learning error curves
f = newFig([60 60 720 360]);
semilogy(1:epochs,trC{best},'-','Color',cBlue,'LineWidth',2);  hold on;
semilogy(1:epochs,vaC{best},'--','Color',cBlue,'LineWidth',2);
semilogy(1:epochs,trRaw,'-','Color',cOrange,'LineWidth',2);
semilogy(1:epochs,vaRaw,'--','Color',cOrange,'LineWidth',2);  grid on;  box off;
xlabel('Epoch');  ylabel('RMSE (% SoC)');
legend({'Z-score inputs, training','Z-score inputs, validation','Raw inputs, training','Raw inputs, validation'}, ...
       'Location','northeast','Box','off');
title(sprintf('Hybrid-learning error of the [%d %d %d] ANFIS',nMF));
saveFig(f,fullfile(figDir,'m_error_curves.png'));

% (d) membership functions before and after training
P0 = mfParams(fis0);  P1 = mfParams(fis);
f = newFig([60 60 1150 360]);  tiledlayout(f,1,3,'TileSpacing','compact','Padding','loose');
for i = 1:3
    nexttile;  hold on;  zz = linspace(min(Z(tr,i)),max(Z(tr,i)),300);
    for j = 1:size(P0{i},1)
        h0 = plot(zz,gaussmf(zz,P0{i}(j,:)),':','Color',cGrey,'LineWidth',1.2);
        h1 = plot(zz,gaussmf(zz,P1{i}(j,:)),'-','Color',cBlue,'LineWidth',2);
    end
    grid on;  ylim([0 1.05]);  xlim([zz(1) zz(end)]);  xlabel(['Z-score of ' lower(inNames{i})]);
    title(sprintf('(%s) %s',char(96+i),inNames{i}));
    if i == 1, ylabel('Degree of membership'); end
end
lg = legend([h0 h1],{'Initial (grid partition)','Trained'},'Orientation','horizontal','Box','off');
lg.Layout.Tile = 'south';
saveFig(f,fullfile(figDir,'m_membership_functions.png'));

% (e) predicted against actual SoC and the test error
yhat = pred(4).yhat;
f = newFig([60 60 1150 380]);  tiledlayout(f,1,2,'TileSpacing','compact','Padding','loose');
nexttile;  plot([0 100],[0 100],'-','Color',cGrey,'LineWidth',1);  hold on;
scatter(y(te),yhat,26,cBlue,'filled','MarkerEdgeColor','w');  grid on;  box off;
xlabel('Actual SoC (%)');  ylabel('Predicted SoC (%)');  title('(a) Predicted against actual SoC');
legend({'Ideal (y = x)','Test samples'},'Location','northwest','Box','off');
nexttile;  yline(0,'-','Color',cGrey);  hold on;
scatter(y(te),1000*(yhat - y(te)),26,cOrange,'filled','MarkerEdgeColor','w');  grid on;  box off;
xlabel('Actual SoC (%)');  ylabel('Prediction error (\times10^{-3} % SoC)');
title(sprintf('(b) Test error (RMSE = %.4f %%)',res(4,1)));
saveFig(f,fullfile(figDir,'m_predicted_vs_actual.png'));

% (f) input-output surfaces of the trained ANFIS (raw engineering units)
[vg,ig] = meshgrid(linspace(3.0,4.2,45),linspace(-10,10,45));
s1 = reshape(evalfis(fis,([vg(:) ig(:) 40*ones(numel(vg),1)] - mu)./sigma,noMsg),size(vg));
[ig2,tg] = meshgrid(linspace(-10,10,45),linspace(20,60,45));
s2 = reshape(evalfis(fis,([3.6*ones(numel(ig2),1) ig2(:) tg(:)] - mu)./sigma,noMsg),size(ig2));
f = newFig([60 60 1150 400]);  tiledlayout(f,1,2,'TileSpacing','compact','Padding','loose');
nexttile;  surf(vg,ig,s1,'EdgeAlpha',0.25);  view(-38,28);
xlim([3 4.2]);  xlabel('Voltage (V)');  ylabel('Current (A)');  zlabel('SoC (%)');  title('(a) SoC surface at T = 40 °C');
nexttile;  surf(ig2,tg,s2,'EdgeAlpha',0.25);  view(-38,28);
xlabel('Current (A)');  ylabel('Temperature (°C)');  zlabel('SoC (%)');  title('(b) SoC surface at V = 3.6 V');
colormap(f,[linspace(0.86,0.05,64)' linspace(0.92,0.25,64)' linspace(0.99,0.55,64)']);
saveFig(f,fullfile(figDir,'m_anfis_surfaces.png'));

fprintf('\nModel saved to anfis_soc.fis / anfis_soc_model.mat, figures saved in %s\n',figDir);
diary off;

%% ------------------------------------------------------------------ local functions
function [fis,fis0,trErr,vaErr] = trainAnfis(Xtr,ytr,Xva,yva,nMF,epochs)
% Grid-partition Sugeno FIS with Gaussian MFs, trained by hybrid learning.
% The returned FIS is the one with the lowest validation error.
gopt = genfisOptions('GridPartition','NumMembershipFunctions',nMF, ...
                     'InputMembershipFunctionType','gaussmf','OutputMembershipFunctionType','linear');
fis0 = genfis(Xtr,ytr,gopt);
aopt = anfisOptions('InitialFIS',fis0,'EpochNumber',epochs,'InitialStepSize',0.01, ...
                    'ValidationData',[Xva yva],'DisplayANFISInformation',0,'DisplayErrorValues',0, ...
                    'DisplayStepSize',0,'DisplayFinalResults',0);
[~,trErr,~,fis,vaErr] = anfis([Xtr ytr],aopt);
end

function [soc,warn,cutoff] = bmsStep(fis,mu,sigma,v,i,t,socWarn,socCutoff,opt)
% One BMS sampling instant: raw reading -> Z-score -> ANFIS -> 0..100 % limit -> flags
z      = ([v i t] - mu) ./ sigma;
soc    = min(max(evalfis(fis,z,opt),0),100);
warn   = soc < socWarn;
cutoff = soc < socCutoff;
end

function P = mfParams(fis)
% Gaussian MF parameters [sigma centre] of every input, one row per MF
P = cell(1,numel(fis.Inputs));
for i = 1:numel(fis.Inputs)
    P{i} = vertcat(fis.Inputs(i).MembershipFunctions.Parameters);
end
end

function f = newFig(pos)
% White-background figure in Times New Roman (also when MATLAB uses the dark theme)
f = figure('Position',pos,'Color','w','DefaultAxesFontName','Times New Roman', ...
           'DefaultTextFontName','Times New Roman','DefaultAxesFontSize',11);
try, theme(f,'light'); catch, end
end

function saveFig(f,file)
try
    exportgraphics(f,file,'Resolution',300,'Padding',12);
catch
    exportgraphics(f,file,'Resolution',300);
end
end

function e = rmse(a,b)
e = sqrt(mean((a(:)-b(:)).^2));
end
