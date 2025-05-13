function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
    trialId = test_data.trialId;
    direction = test_data.decodedHandPos; % Using decodedHandPos to help key by direction
    if isempty(direction)
        directionId = 1; % Fallback if empty
    else
        directionId = size(direction, 2); % Proxy for how many time steps we've done
    end

    spikes = test_data.spikes;
    numNeurons = modelParameters.numNeurons;
    windowSize = 20;
    
    % Compute firing rates in 20ms bins
    numWindows = floor(size(spikes, 2) / windowSize);
    firingRates = zeros(numNeurons, numWindows);
    for w = 1:numWindows
        firingRates(:, w) = sum(spikes(:, (w - 1) * windowSize + 1 : w * windowSize), 2) / windowSize;
    end

    % Get state
    key = sprintf('trial%d_dir%d', trialId, directionId);
    if isfield(modelParameters.trialStates, key)
        h0 = modelParameters.trialStates.(key).h;
        c0 = modelParameters.trialStates.(key).c;
        prev_x = modelParameters.trialStates.(key).x;
        prev_y = modelParameters.trialStates.(key).y;
    else
        hiddenSize = size(modelParameters.lstmParams.Wf, 1);
        h0 = zeros(hiddenSize, 1);
        c0 = zeros(hiddenSize, 1);
        prev_x = test_data.startHandPos(1);
        prev_y = test_data.startHandPos(2);
    end

    % Forward Pass
    [h_seq, c_seq, velocity] = lstmForward(firingRates, h0, c0, ...
        modelParameters.lstmParams.Wf, modelParameters.lstmParams.Wi, ...
        modelParameters.lstmParams.Wo, modelParameters.lstmParams.Wc, ...
        modelParameters.lstmParams.bf, modelParameters.lstmParams.bi, ...
        modelParameters.lstmParams.bo, modelParameters.lstmParams.bc, ...
        modelParameters.lstmParams.Wy, modelParameters.lstmParams.by);

    % integrate velocity
    delta_x = cumsum(velocity(1, :));
    delta_y = cumsum(velocity(2, :));
    x = prev_x + delta_x(end);
    y = prev_y + delta_y(end);

    % store state
    modelParameters.trialStates.(key).h = h_seq(:, end);
    modelParameters.trialStates.(key).c = c_seq(:, end);
    modelParameters.trialStates.(key).x = x;
    modelParameters.trialStates.(key).y = y;
end


function [h_seq, c_seq, y_pred] = lstmForward(x_seq, h0, c0, Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wy, by)
    numSteps = size(x_seq, 2);
    numHiddenUnits = size(Wf, 1);
    
    h_seq = zeros(numHiddenUnits, numSteps);
    c_seq = zeros(numHiddenUnits, numSteps);
    y_pred = zeros(2, numSteps);
    
    h_t = h0;
    c_t = c0;
    
    for t = 1:numSteps
        x_t = x_seq(:, t);
        concat = [h_t; x_t];
        
        f_t = sigmoid(Wf * concat + bf);
        i_t = sigmoid(Wi * concat + bi);
        o_t = sigmoid(Wo * concat + bo);
        c_tilde = tanh(Wc * concat + bc);
        
        c_t = f_t .* c_t + i_t .* c_tilde;
        h_t = o_t .* tanh(c_t);
        
        h_seq(:, t) = h_t;
        c_seq(:, t) = c_t;
        
        y_pred(:, t) = Wy * h_t + by;
    end
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end
