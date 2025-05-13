
load('monkeydata_training.mat');

% one trial and direction
trial_number = 1;
angle_number = 1;
trial_data = trial(trial_number, angle_number);

neurons_to_plot = [1, 2, 3, 96, 97, 98];
time_step = size(trial_data.spikes, 2);

figure;
hold on;

for i = 1:length(neurons_to_plot)
    neuron_index = neurons_to_plot(i);
    spike_times = find(trial_data.spikes(neuron_index, :) == 1);
    
    for t = spike_times
        line([t t], [i-0.4 i+0.4], 'Color', 'k');
    end
end

yticks(1:length(neurons_to_plot));
yticklabels(["Neuron 1", "Neuron 2", "Neuron 3", "", "Neuron 97", "Neuron 98"]);

xlabel('Time (ms)');
ylabel('Selected Neural Units');
title('Example Spike Trains from Distributed Neural Units (Trial 1, Direction 1)');
xlim([0 time_step]);
ylim([0.5 length(neurons_to_plot)+0.5])
