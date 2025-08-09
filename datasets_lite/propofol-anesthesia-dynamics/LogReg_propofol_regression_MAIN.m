%% Logistic Regression on Propofol Data

%% SET PARAMETERS - THIS IS THE ONLY SECTION TO BE MODIFIED BY THE USER

metric = 'mean'; %options: mean or median, what metric to use to consolidate information within each window
history = 0; %0 for no history, 1 for including history

Y_mode = 4; %1 for unconscious vs conscious, 2 for before vs after LOC, 3 for before vs after ROC, 4 for baseline at beginning vs end
X_mode = 1; %1 for HRV and EDA with amplitudes, 2 for HRV and EDA without pulse amplitudes, 3 for HRV only, 4 for EDA only, 5 for EDA without pulse amplitudes

time_oneside = 20; %in minutes, on one side of LOC or ROC for Y_mode = 2 or 3, not used in Y_mode = 1 or 4
time_window = 30; %in seconds, window length

%For history, get lags for each signal 
HRV_lag = 60; %in seconds, duration of history to include for HRV indices
EDA_lag = 60; %in seconds, duration of history to include for EDA indices

%% Initialize X and Y arrays

Y_all = [];
X1_all = [];
subj_lengths = [];

col_labels_X1 = {'muRR','sigmaRR','muHR','sigmaHR','totalpower','LF','HF','ratio','LFnu','HFnu','eda_tonic','muPR','sigmaPR','mu_amp','sigma_amp'};

col_labels_Y = {'Prop_stage'};

%% If to include history, initialize history columns

if history == 1
    HRV_lagcols = ceil(HRV_lag/time_window);
    EDA_lagcols = ceil(EDA_lag/time_window);
    
    if X_mode == 1
        X1_lagcols = [repmat(HRV_lagcols,10,1); repmat(EDA_lagcols,5,1)];
    elseif X_mode == 2
        X1_lagcols = [repmat(HRV_lagcols,10,1); repmat(EDA_lagcols,3,1)];
    elseif X_mode == 3
        X1_lagcols = repmat(HRV_lagcols,10,1);
    elseif X_mode == 4
        X1_lagcols = repmat(EDA_lagcols,5,1);
    elseif X_mode == 5
        X1_lagcols = repmat(EDA_lagcols,3,1);
    end
end

%% Go through all subjects and fill X and Y arrays

subjs = 1:9;

for s = 1:length(subjs)
    s
    load(sprintf('Data/subj_data_%d.mat',s));

    %% Pull all indices to be used as features
    loc = subj_data.LOC;
    roc = subj_data.ROC;
    events = subj_data.events;
    t_HRV = subj_data.t_HRV;
    t_EDA = subj_data.t_EDA;
    t_EDA_tonic = subj_data.t_EDA_tonic;
    muRR = subj_data.muRR; %in msec
    muHR = subj_data.muHR;
    sigmaRR = subj_data.sigmaRR;
    sigmaHR = subj_data.sigmaHR;
    pow_tot = subj_data.pow_tot;
    LF = subj_data.LF;
    HF = subj_data.HF;
    LFnu = subj_data.LFnu;
    HFnu = subj_data.HFnu;
    ratio = subj_data.ratio;
    eda_tonic = subj_data.eda_tonic;
    muPR = subj_data.muPR;
    sigmaPR = subj_data.sigmaPR;
    mu_amp = subj_data.mu_amp;
    sigma_amp = subj_data.sigma_amp;

    signals = {muRR,sigmaRR,muHR,sigmaHR,pow_tot,LF,HF,ratio,LFnu,HFnu,eda_tonic,muPR,sigmaPR,mu_amp,sigma_amp};
    t_s = {t_HRV,t_HRV,t_HRV,t_HRV,t_HRV,t_HRV,t_HRV,t_HRV,t_HRV,t_HRV,t_EDA_tonic,t_EDA,t_EDA,t_EDA,t_EDA};

    if X_mode == 1
        signals = signals(1:15);
        t_s = t_s(1:15);
    elseif X_mode == 2
        signals = signals(1:13);
        t_s = t_s(1:13);
    elseif X_mode == 3
        signals = signals(1:10);
        t_s = t_s(1:10);
    elseif X_mode == 4
        signals = signals(11:15);
        t_s = t_s(11:15);
    elseif X_mode == 5
        signals = signals(11:13);
        t_s = t_s(11:13);
    end
    
    dur = max([t_HRV(end) t_EDA(end) t_EDA_tonic(end)]);

    %% Modify X and Y arrays based on Y_mode

    if Y_mode == 1
        num_bins = ceil(dur/time_window);
    elseif Y_mode < 4
        num_bins = 2*ceil(time_oneside*60/time_window) + history*ceil(HRV_lag/time_window); %15 minutes before and after
    else
        num_bins = ceil(dur/time_window);
    end
    X1 = zeros(num_bins,length(signals));

    Y = zeros(num_bins,1);
    if Y_mode == 1
        %Conscious = 0, unconscious = 1
        bin_loc = round(loc/time_window);
        bin_roc = round(roc/time_window);
        Y((bin_loc+1):bin_roc) = 1;

        %Pull the indices for each time window
        for j = 1:length(signals)
            X1(:,j) = return_y_vec_prop(t_s{j},signals{j},metric,time_window,num_bins);
        end
    elseif Y_mode < 4
        %before LOC or ROC = 0, after LOC or ROC = 1;
        one_side = ceil(time_oneside*60/time_window); 
        Y((end-one_side+1):end) = 1;

        %Isolate the right chunk of signal
        if Y_mode == 2
            end_point = loc + one_side*time_window;
            start_point = loc - one_side*time_window - history*time_window*ceil(HRV_lag/time_window);
        else
            end_point = min(roc + one_side*time_window,dur);
            start_point = roc - one_side*time_window - history*time_window*ceil(HRV_lag/time_window);
        end

        %Pull the indices for each time window
        for j = 1:length(signals)
            inds = (t_s{j} > start_point) & (t_s{j} <= end_point);
            t_seg = t_s{j}(inds);
            signal_seg = signals{j}(inds);
            X1(:,j) = return_y_vec_prop(t_seg,signal_seg,metric,time_window,num_bins);
        end
        ind = find(isnan(X1(:,1)),1);
        X1(ind:end,:) = [];
        Y(ind:end) = [];

    else %Y_mode = 4
        bin_inds = round(events/time_window); 

        ind_start = 1;
        for j = 1:length(bin_inds)
            Y(ind_start:bin_inds(j)) = j;
            ind_start = bin_inds(j) + 1;
        end
        %Baseline at the beginning = 0, baseline at the end = 1
        Y(ind_start:end) = length(bin_inds)+1;
        Y2 = Y(Y == 2 | Y == 11);
        Y2 = (Y2 == 11);
        
        %Pull the indices for each time window
        for j = 1:length(signals)
            X1(:,j) = return_y_vec_prop(t_s{j},signals{j},metric,time_window,num_bins);
        end
    end

    %% End task-specific

    %If history, add the correct number of lagged columns for each index
    if history == 1
        X1 = addLagCols(X1,X1_lagcols);
    end
    
    %Concatenate with other subjects' data after normalizing within subject
    if Y_mode == 4
        X1 = X1(Y == 2 | Y == 11,:);
        Y_all = [Y_all; Y2];
        X1_all = [X1_all; normalize(X1)];
        subj_lengths(s) = size(X1,1);
    else
        Y_all = [Y_all; Y];
        X1_all = [X1_all; normalize(X1)];
        subj_lengths(s) = size(X1,1);
    end    
    
    clear subj_data
end

%% Perform Logistic Regresssion using Leave-one-subject-out Cross-Validation

%Fill in all the NaNs with 0's so that the regression still uses those data
%points but just treats them as the mean value

X1_all_use = X1_all;
X1_all_use(isnan(X1_all)) = 0;

preds = zeros(size(Y_all,1),1);
best_coeffs = zeros(size(X1_all_use,2)+1,9);
predictors = zeros(size(X1_all_use,2),9);

subj_ends = cumsum(subj_lengths);
subj_starts = [1 subj_ends(1:end-1)+1];

for s = 1:length(subj_lengths)
    %Fit model on other 8 subjects
    inds_subj = [subj_starts(s):subj_ends(s)];
    inds_notsubj = setdiff(1:length(preds),inds_subj);

    [B_temp,FitInfo_temp] = lassoglm(X1_all_use(inds_notsubj,:),Y_all(inds_notsubj),'binomial');

    %Find best fit model from AIC
    %AIC = Dev + 2*k
    AIC_temp = FitInfo_temp.Deviance + 2*FitInfo_temp.DF;

    %Pull the minimum AIC model
    [best_AIC_temp,best_model_temp] = min(AIC_temp(1:end-1));
    best_coeffs_temp = [FitInfo_temp.Intercept(best_model_temp); B_temp(:,best_model_temp)];
    if s == 10
        best_coeffs(:,s-1) = best_coeffs_temp;
    else
        best_coeffs(:,s) = best_coeffs_temp;
    end

    predictors_temp = find(B_temp(:,best_model_temp)); % indices of nonzero predictors
    predictors_X1_temp = find(B_temp(1:size(X1_all_use,2),best_model_temp));
    if s == 10
        predictors(:,s-1) = (best_coeffs_temp(2:end) ~= 0);
    else
        predictors(:,s) = (best_coeffs_temp(2:end) ~= 0);
    end

    %Compute predictions on left out subject
    preds_temp = glmval(best_coeffs_temp,X1_all_use(inds_subj,:),'logit','Constant','on');
    preds(inds_subj) = preds_temp;
end
best_coeffs = mean(best_coeffs,2);

%Sort predictions
[preds_sorted,order] = sort(preds);
length_keep = find(isnan(preds_sorted),1,'first')-1;
if isempty(length_keep)
    length_keep = length(order);
end
order = order(1:length_keep);
true = Y_all(order);

%Show the results
figure;
plot(1:length_keep,true,'m*','LineWidth',1.5)
hold on;
plot(1:length_keep,preds_sorted(1:length_keep),'b.','LineWidth',2)
axis tight

%Show the results by subject
[p,~] = numSubplots(length(subj_lengths));
figure;
for i = 1:length(subj_lengths)
    i
    subplot(p(1),p(2),i)
    if i > 1
        [temp,temp_order] = sort(preds((sum(subj_lengths(1:i-1))+1):sum(subj_lengths(1:i))));
        temp_Y = Y_all((sum(subj_lengths(1:i-1))+1):sum(subj_lengths(1:i)));
        temp_Y = temp_Y(temp_order);
    else
        [temp,temp_order] = sort(preds(1:subj_lengths(i)));
        temp_Y = Y_all(1:subj_lengths(i));
        temp_Y = temp_Y(temp_order);
    end
    plot(1:subj_lengths(i),temp_Y,'m*','LineWidth',1.5)
    hold on;
    plot(1:subj_lengths(i),temp,'b.','LineWidth',2)
    axis tight
    hold off;
    title(sprintf('%s',subjs(i)))
end

%Plot ROC curve
[X,Y,T,AUC] = perfcurve(true,preds_sorted(1:length_keep),1);
figure;
plot(X,Y,'b','LineWidth',3);
hold on;
plot([0 1],[0 1],'k','LineWidth',2)
title('ROC')
text(0.5,0.2,sprintf('AUROC: %f',AUC),'FontSize',32)

%Plot results by subject across time
figure;
for i = 1:length(subj_lengths)
    subplot(p(1),p(2),i)
    if i > 1
        temp = preds((sum(subj_lengths(1:i-1))+1):sum(subj_lengths(1:i)));
        temp_Y = Y_all((sum(subj_lengths(1:i-1))+1):sum(subj_lengths(1:i)));        
    else
        temp = preds(1:subj_lengths(i));
        temp_Y = Y_all(1:subj_lengths(i));
    end
    plot((1/60)*(time_window.*(1:subj_lengths(i))-(time_window/2)),temp_Y,'m*','LineWidth',1.5)
    hold on;
    plot((1/60)*(time_window.*(1:subj_lengths(i))-(time_window/2)),temp,'b','LineWidth',1)    
    axis tight
    hold off;
    title(sprintf('%s',subjs(i)))
    xlabel('Minutes')
end