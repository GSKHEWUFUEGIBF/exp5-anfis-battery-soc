%% build_Exp5_Simulink.m
% Experiment 5 : ANFIS based battery SoC prediction using Z-score data normalization
% Builds the Simulink model Exp5_ANFIS_SoC_BMS.slx, which runs the trained ANFIS
% as the real-time SoC estimator of a battery management system (BMS):
%
%   sensor readings V, I, T -> Z-score normalization -> ANFIS (Fuzzy Logic Controller
%   block) -> 0..100 % limiter -> low-SoC warning and cut-off flags -> display / logging
%
% One simulation step = one BMS sampling instant. The 150 unseen test samples are
% replayed in order of decreasing SoC, so the run looks like a discharge from full
% to empty. All data and the trained FIS are stored in the model workspace, so the
% saved model runs on its own.
%
% Requires: MATLAB, Simulink and the Fuzzy Logic Toolbox.
% Run Exp5_ANFIS_SoC first (it creates anfis_soc_model.mat), then  >> build_Exp5_Simulink

clc; close all;
here   = fileparts(mfilename('fullpath'));   if isempty(here), here = pwd; end
mdl    = 'Exp5_ANFIS_SoC_BMS';
figDir = fullfile(here,'figures_simulink');
if ~exist(figDir,'dir'), mkdir(figDir); end

S = load(fullfile(here,'anfis_soc_model.mat'));      % fis, mu, sigma, Xs, ys, socWarn, socCutoff
N = numel(S.ys);    t = (0:N-1)';

%% 1. Create the model and fill its workspace
if bdIsLoaded(mdl), close_system(mdl,0); end
slxFile = fullfile(here,[mdl '.slx']);
if exist(slxFile,'file'), delete(slxFile); end
new_system(mdl);   open_system(mdl);
set_param(mdl,'Solver','FixedStepDiscrete','FixedStep','1','StartTime','0','StopTime',num2str(N-1));

ws = get_param(mdl,'ModelWorkspace');
assignin(ws,'Exp5_fis',S.fis);            assignin(ws,'Exp5_mu',S.mu);        assignin(ws,'Exp5_sigma',S.sigma);
assignin(ws,'Exp5_V',[t S.Xs(:,1)]);      assignin(ws,'Exp5_I',[t S.Xs(:,2)]);
assignin(ws,'Exp5_T',[t S.Xs(:,3)]);      assignin(ws,'Exp5_SoC',[t S.ys]);
assignin(ws,'Exp5_warn',S.socWarn);       assignin(ws,'Exp5_cutoff',S.socCutoff);

fuzzyBlk = ['fuzblock/Fuzzy Logic ' newline 'Controller'];
load_system('fuzblock');

%% 2. Blocks
% ---- step 1: sensor data acquisition (one sample per step) ----
src = {'Voltage_Sensor','Exp5_V',40; 'Current_Sensor','Exp5_I',120; 'Temperature_Sensor','Exp5_T',200; ...
       'SoC_Reference','Exp5_SoC',400};
for k = 1:size(src,1)
    addb(mdl,'simulink/Sources/From Workspace',src{k,1},[40 src{k,3} 150 src{k,3}+40], ...
         'VariableName',src{k,2},'SampleTime','1','Interpolate','off','OutputAfterFinalValue','Holding final value');
end

% ---- step 2: Z-score normalization  z = (x - mu) / sigma ----
sub = [mdl '/Zscore_Normalization'];
addb(mdl,'simulink/Ports & Subsystems/Subsystem','Zscore_Normalization',[230 30 370 250]);
Simulink.SubSystem.deleteContents(sub);
inP = {'V','I','T'};
for k = 1:3
    add_block('simulink/Sources/In1',[sub '/' inP{k}],'Position',[30 40+60*(k-1) 60 54+60*(k-1)]);
end
add_block('simulink/Signal Routing/Mux',      [sub '/Mux_VIT'],'Inputs','3','Position',[120 35 125 180]);
add_block('simulink/Sources/Constant',        [sub '/mu'],'Value','Exp5_mu','Position',[110 220 190 250]);
add_block('simulink/Math Operations/Sum',     [sub '/Subtract_mean'],'Inputs','+-','IconShape','rectangular','Position',[230 95 260 135]);
add_block('simulink/Sources/Constant',        [sub '/sigma'],'Value','Exp5_sigma','Position',[230 220 310 250]);
add_block('simulink/Math Operations/Divide',  [sub '/Divide_by_std'],'Position',[350 100 380 140]);
add_block('simulink/Sinks/Out1',              [sub '/z'],'Position',[440 113 470 127]);
for k = 1:3, add_line(sub,[inP{k} '/1'],sprintf('Mux_VIT/%d',k),'autorouting','smart'); end
add_line(sub,'Mux_VIT/1','Subtract_mean/1','autorouting','smart');
add_line(sub,'mu/1','Subtract_mean/2','autorouting','smart');
add_line(sub,'Subtract_mean/1','Divide_by_std/1','autorouting','smart');
add_line(sub,'sigma/1','Divide_by_std/2','autorouting','smart');
add_line(sub,'Divide_by_std/1','z/1','autorouting','smart');

% ---- step 4: ANFIS inference ----
addb(mdl,fuzzyBlk,'ANFIS_SoC_Estimator',[440 105 560 175],'FIS','Exp5_fis', ...
     'OutOfRangeInputValueMessage','none','NoRuleFiredMessage','none','EmptyOutputFuzzySetMessage','none');

% ---- step 5: boundary check and safety flags ----
addb(mdl,'simulink/Discontinuities/Saturation','SoC_Limiter_0_100',[630 120 680 160],'UpperLimit','100','LowerLimit','0');
addb(mdl,'simulink/Logic and Bit Operations/Compare To Constant','Low_SoC_Warning',[780 235 880 275], ...
     'relop','<','const','Exp5_warn');
addb(mdl,'simulink/Logic and Bit Operations/Compare To Constant','Cutoff_Flag',[780 315 880 355], ...
     'relop','<','const','Exp5_cutoff');
addb(mdl,'simulink/Logic and Bit Operations/Logical Operator','NOT',[940 395 980 425],'Operator','NOT');
addb(mdl,'simulink/Math Operations/Sum','SoC_Error',[790 470 820 500],'Inputs','+-','IconShape','rectangular');

% ---- step 6: display and logging ----
addb(mdl,'simulink/Sinks/Display','SoC_Display_percent',[780 35 900 75]);
addb(mdl,'simulink/Sinks/Display','Load_Contactor_Closed',[1050 392 1130 428]);
addb(mdl,'simulink/Signal Routing/Mux','Mux_SoC',[940 120 945 180],'Inputs','2');
addb(mdl,'simulink/Signal Routing/Mux','Mux_Flags',[940 250 945 340],'Inputs','2');
addb(mdl,'simulink/Sinks/Scope','Scope_SoC',[1050 132 1090 168]);
addb(mdl,'simulink/Sinks/Scope','Scope_Flags',[1050 277 1090 313]);
addb(mdl,'simulink/Sinks/Scope','Scope_Error',[1050 467 1090 503]);
addb(mdl,'simulink/Sinks/To Workspace','Log_SoC',[1050 195 1130 225],'VariableName','log_soc','SaveFormat','Array');
addb(mdl,'simulink/Sinks/To Workspace','Log_Flags',[1050 335 1130 365],'VariableName','log_flags','SaveFormat','Array');
addb(mdl,'simulink/Sinks/To Workspace','Log_Error',[1050 525 1130 555],'VariableName','log_err','SaveFormat','Array');

%% 3. Connections
ln(mdl,'Voltage_Sensor/1','Zscore_Normalization/1');
ln(mdl,'Current_Sensor/1','Zscore_Normalization/2');
ln(mdl,'Temperature_Sensor/1','Zscore_Normalization/3');
ln(mdl,'Zscore_Normalization/1','ANFIS_SoC_Estimator/1');
ln(mdl,'ANFIS_SoC_Estimator/1','SoC_Limiter_0_100/1');
ln(mdl,'SoC_Limiter_0_100/1','SoC_Display_percent/1');
ln(mdl,'SoC_Limiter_0_100/1','Mux_SoC/1');
ln(mdl,'SoC_Reference/1','Mux_SoC/2');
ln(mdl,'SoC_Limiter_0_100/1','Low_SoC_Warning/1');
ln(mdl,'SoC_Limiter_0_100/1','Cutoff_Flag/1');
ln(mdl,'SoC_Limiter_0_100/1','SoC_Error/1');
ln(mdl,'SoC_Reference/1','SoC_Error/2');
ln(mdl,'Low_SoC_Warning/1','Mux_Flags/1');
ln(mdl,'Cutoff_Flag/1','Mux_Flags/2');
ln(mdl,'Cutoff_Flag/1','NOT/1');
ln(mdl,'NOT/1','Load_Contactor_Closed/1');
ln(mdl,'Mux_SoC/1','Scope_SoC/1');        ln(mdl,'Mux_SoC/1','Log_SoC/1');
ln(mdl,'Mux_Flags/1','Scope_Flags/1');    ln(mdl,'Mux_Flags/1','Log_Flags/1');
ln(mdl,'SoC_Error/1','Scope_Error/1');    ln(mdl,'SoC_Error/1','Log_Error/1');

note(mdl,'Experiment 5 - ANFIS SoC estimator for a BMS  (1 step = 1 sampling instant)',[40 -25]);
note(mdl,'Sensors',[60 5]);                 note(mdl,'Pre-processing',[250 5]);
note(mdl,'ANFIS inference',[450 80]);       note(mdl,'Limit',[635 95]);
note(mdl,'Safety flags, display and logging',[850 5]);

save_system(mdl,slxFile);
fprintf('Created %s\n',slxFile);

%% 4. Model images
try
    set_param(mdl,'ZoomFactor','FitSystem');
    print(['-s' mdl],'-dpng','-r300',fullfile(figDir,'model_top_level.png'));
    open_system(sub);   set_param(sub,'ZoomFactor','FitSystem');
    print(['-s' sub],'-dpng','-r300',fullfile(figDir,'model_zscore_subsystem.png'));
    close_system(sub);
catch ME
    warning('Model image export failed: %s',ME.message);
end

%% 5. Simulate and check against the MATLAB result
simOut = sim(mdl,'ReturnWorkspaceOutputs','on');
soc  = simOut.get('log_soc');   flg = double(simOut.get('log_flags'));   err = simOut.get('log_err');
assert(size(soc,1) == N,'Expected %d samples, got %d',N,size(soc,1));
est  = soc(:,1);   ref = soc(:,2);

opt  = evalfisOptions('OutOfRangeInputValueMessage','none','NoRuleFiredMessage','none','EmptyOutputFuzzySetMessage','none');
offl = min(max(evalfis(S.fis,(S.Xs - S.mu)./S.sigma,opt),0),100);       % same samples in plain MATLAB
R = struct('samples',N,'rmse',sqrt(mean(err.^2)),'mae',mean(abs(err)),'maxErr',max(abs(err)), ...
           'warn',sum(flg(:,1)),'cut',sum(flg(:,2)),'firstWarn',find(flg(:,1),1),'firstCut',find(flg(:,2),1), ...
           'maxDiffVsMatlab',max(abs(est - offl)),'socStart',est(1),'socEnd',est(end));
fprintf('\nSimulink replay of %d test samples\n',N);
fprintf('  RMSE %.4f %% SoC, MAE %.4f %%, maximum error %.4f %%\n',R.rmse,R.mae,R.maxErr);
fprintf('  Low-SoC warning (< %g %%) on %d samples, first at sample %d\n',S.socWarn,R.warn,R.firstWarn);
fprintf('  Cut-off flag    (< %g %%) on %d samples, first at sample %d\n',S.socCutoff,R.cut,R.firstCut);
fprintf('  Maximum difference between Simulink and MATLAB evalfis: %.2e %% SoC\n',R.maxDiffVsMatlab);
fid = fopen(fullfile(here,'simulink_results.json'),'w');  fprintf(fid,'%s',jsonencode(R,'PrettyPrint',true));  fclose(fid);

%% 6. Result figure
cBlue = [42 120 214]/255;  cOrange = [235 104 52]/255;  cAqua = [27 175 122]/255;  cGrey = [82 81 78]/255;
f = figure('Position',[60 60 900 760],'Color','w','DefaultAxesFontName','Times New Roman', ...
           'DefaultTextFontName','Times New Roman','DefaultAxesFontSize',11);
try, theme(f,'light'); catch, end
tiledlayout(f,3,1,'TileSpacing','compact','Padding','loose');
n = (1:N)';
nexttile;  plot(n,ref,'-','Color',cGrey,'LineWidth',3);  hold on;  plot(n,est,'--','Color',cBlue,'LineWidth',1.6);
yline(S.socWarn,':','Color',cOrange,'LineWidth',1.2);   yline(S.socCutoff,':','Color',cOrange,'LineWidth',1.2);
text(3,S.socWarn+5,sprintf('Warning level %g %%',S.socWarn),'Color',cGrey,'FontSize',10);
text(3,S.socCutoff+5,sprintf('Cut-off level %g %%',S.socCutoff),'Color',cGrey,'FontSize',10);
grid on;  box off;  ylim([0 100]);  xlim([1 N]);  ylabel('SoC (%)');
legend({'Actual SoC','ANFIS estimate (Simulink)'},'Location','northeast','Box','off');
title('(a) Actual and estimated SoC');
nexttile;  stem(n,1000*err,'Color',cBlue,'Marker','none','LineWidth',1);  grid on;  box off;  xlim([1 N]);
ylabel('Error (\times10^{-3} % SoC)');  title(sprintf('(b) Estimation error (RMSE = %.4f %% SoC)',R.rmse));
nexttile;  stairs(n,flg(:,1),'-','Color',cOrange,'LineWidth',2);  hold on;
stairs(n,flg(:,2)-1.5,'-','Color',cAqua,'LineWidth',2);  grid on;  box off;  xlim([1 N]);  ylim([-1.8 1.3]);
set(gca,'YTick',[-1.5 -0.5 0 1],'YTickLabel',{'0','1','0','1'});
text(3,0.55,'Low-SoC warning flag','Color',cGrey,'FontSize',10);
text(3,-0.95,'Cut-off flag','Color',cGrey,'FontSize',10);
xlabel('Sample index (test samples in order of decreasing SoC)');  ylabel('Flag state');  title('(c) Safety flags');
try
    exportgraphics(f,fullfile(figDir,'sim_results.png'),'Resolution',300,'Padding',12);
catch
    exportgraphics(f,fullfile(figDir,'sim_results.png'),'Resolution',300);
end
save_system(mdl);
fprintf('Figures saved in %s\n',figDir);

%% ------------------------------------------------------------------ local functions
function addb(mdl,lib,name,pos,varargin)
add_block(lib,[mdl '/' name],'Position',pos,varargin{:});
end

function ln(mdl,src,dst)
add_line(mdl,src,dst,'autorouting','smart');
end

function note(mdl,txt,pos)
a = Simulink.Annotation(mdl,txt);
a.Position = pos;   a.FontName = 'Times New Roman';   a.FontSize = 12;
end
