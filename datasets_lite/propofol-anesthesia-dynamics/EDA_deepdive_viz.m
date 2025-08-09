%This script visualizes the EDA specific information for all 9 subjects to
%replicate Figs 2-4 and figures in the S2 Appendix
%Figures 1-9 are for subjects 1-9 respectively

load('Data/EDA_deepdive.mat');

subjs = 1:9;

for s = 1:length(subjs)
    %Identify indices
    t = (1:size(EDA_temp_amp(s).X,1))*20/60;
    mu_amp = EDA_temp_amp(s).X(:,4);
    sigma_amp = EDA_temp_amp(s).X(:,5);
    mu_BR = EDA_temp_amp(s).X(:,2);
    sigma_BR = EDA_temp_amp(s).X(:,3);
    
    %Smooth signals before computing correlation
    mu_amp_smoothed = smooth(mu_amp,9,'rlowess');
    sigma_amp_smoothed = smooth(sigma_amp,9,'rlowess');
    mu_BR_smoothed = smooth(mu_BR,9,'rlowess');
    sigma_BR_smoothed = smooth(sigma_BR,9,'rlowess');
    
    %Replace NaNs
    mu_amp_smoothed(isnan(mu_amp_smoothed)) = nanmean(mu_amp_smoothed);
    sigma_amp_smoothed(isnan(sigma_amp_smoothed)) = nanmean(sigma_amp_smoothed);
    mu_BR_smoothed(isnan(mu_BR_smoothed)) = nanmean(mu_BR_smoothed);
    sigma_BR_smoothed(isnan(sigma_BR_smoothed)) = nanmean(sigma_BR_smoothed);
    
    mu_amp(isnan(mu_amp)) = nanmean(mu_amp);
    sigma_amp(isnan(sigma_amp)) = nanmean(sigma_amp);
    mu_BR(isnan(mu_BR)) = nanmean(mu_BR);
    sigma_BR(isnan(sigma_BR)) = nanmean(sigma_BR);

    %Compute correlation
    [r_mean,p_mean] = movcorr(mu_amp_smoothed,mu_BR_smoothed,30);
    [r_std,p_std] = movcorr(sigma_amp_smoothed,sigma_BR_smoothed,30);
     
    %Plot
    figure;

    subplot(6,1,1)
    plot(t, mu_BR,'LineWidth',1)
    ylabel('Pulses/min')
    title('Mean pulse rate')
    xlabel('Time (min)')
    ax1 = gca;
    
    subplot(6,1,2)
    plot(t, mu_amp,'LineWidth',1)
    ylabel('Pulse prominence') 
    title('Mean pulse amplitude')
    xlabel('Time (min)')
    ax2 = gca;
    
    subplot(6,1,3)
    plot(t, r_mean,'LineWidth',2)
    ylabel('Correlation') 
    title('Correlation between Mean Pulse Rate and Amplitude')
    xlabel('Time (min)')
    ax3 = gca;

    subplot(6,1,4)
    plot(t, sigma_BR,'LineWidth',1)
    ylabel('Pulses/min')
    title('Standard Deviation of Pulse Rate')
    xlabel('Time (min)')
    ax4 = gca;
    
    subplot(6,1,5)
    plot(t, sigma_amp,'LineWidth',1)
    ylabel('Pulse prominence') 
    title('Standard Deviation of Pulse Amplitude')
    xlabel('Time (min)')
    ax5 = gca;
    
    subplot(6,1,6)
    plot(t, r_std,'LineWidth',2)
    ylabel('Correlation')
    title('Correlation between Standard Deviation of Pulse Rate and Amplitude')
    xlabel('Time (min)')
    ax6 = gca;
    
    linkaxes([ax1,ax2,ax3,ax4,ax5,ax6],'x')
end