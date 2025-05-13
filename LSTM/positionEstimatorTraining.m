
function [modelParameters] = positionEstimatorTraining(training_data)
  % Arguments:
  
  % - training_data:
  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  
  % ... train your model
  
  % Return Value:
  
  % - modelParameters:
  %     single structure containing all the learned parameters of your
  %     model and which can be used by the "positionEstimator" function.
  
    numAngles  = size(training_data, 2);  
    numTrials  = size(training_data, 1);  
    numNeurons = size(training_data(1,1).spikes, 1);
    
    maxTrialLength = 0;
    for k = 1:numAngles
        for n = 1:numTrials
            maxTrialLength = max(maxTrialLength, size(training_data(n, k).spikes, 2));
        end
    end
    
    first300ms = 300;
    windowSize = 20;
    startTime = 1;
    endTime = maxTrialLength-150;
    numWindows = floor((endTime - startTime) / windowSize);
    
    neuralData = struct();
    allFiringRates = zeros(numTrials, numNeurons, numWindows);
    allVelocities = zeros(numTrials, 2, numWindows);
    for k = 1:numAngles
        for n = 1:numTrials
            for neuron = 1:numNeurons
                spikes = training_data(n, k).spikes(neuron, :);
                T = size(spikes, 2);
                endTimeTrial = min(T-50, startTime + numWindows * windowSize);
                
                for w = 1:numWindows
                    windowStart = startTime + (w - 1) * windowSize;
                    windowEnd = windowStart + windowSize - 1;
                    
                    if windowEnd <= endTimeTrial
                        allFiringRates(n, neuron, w) = sum(spikes(windowStart:windowEnd)) / windowSize;
                    end
                end
            end
            
            handPos = training_data(n, k).handPos(1:2, :);
            T = size(handPos, 2);
            endTimeTrial = min(T, startTime + numWindows * windowSize - 1);
            
            for w = 1:numWindows
                windowStart = startTime + (w - 1) * windowSize;
                windowEnd = windowStart + windowSize - 1;
                
                if windowEnd <= endTimeTrial
                    startPos = handPos(:, windowStart);
                    endPos = handPos(:, windowEnd);
                    velocity = (endPos - startPos) / windowSize;
                    allVelocities(n, :, w) = velocity;
                end
            end
        end
        neuralData(k).firingRates = allFiringRates;
        neuralData(k).handVelocities = allVelocities;
    end
    
    numHiddenUnits = 50; 
    inputSize = numNeurons; 
    outputSize = 2; 
    
    Wf = randn(numHiddenUnits, inputSize + numHiddenUnits) * 0.01;
    Wi = randn(numHiddenUnits, inputSize + numHiddenUnits) * 0.01;
    Wo = randn(numHiddenUnits, inputSize + numHiddenUnits) * 0.01;
    Wc = randn(numHiddenUnits, inputSize + numHiddenUnits) * 0.01;
    
    bf = zeros(numHiddenUnits, 1);
    bi = zeros(numHiddenUnits, 1);
    bo = zeros(numHiddenUnits, 1);
    bc = zeros(numHiddenUnits, 1);
    
    Wy = randn(outputSize, numHiddenUnits) * 0.01;
    by = zeros(outputSize, 1);
    
    h0 = zeros(numHiddenUnits, 1);
    c0 = zeros(numHiddenUnits, 1);
    
    learningRate = 0.001;
    numEpochs = 200;
    X = permute(neuralData(k).firingRates, [2, 1, 3]); 
    Y = permute(neuralData(k).handVelocities, [2, 1, 3]); 
    for k = 1:numAngles

        
        for epoch = 1:numEpochs
            for i = 1:size(X, 2)
                x_seq = squeeze(X(:, i, :));
                y_seq = squeeze(Y(:, i, :));
                
                [h_seq, c_seq, y_pred] = lstmForward(x_seq, h0, c0, Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wy, by);
                
                loss = mean((y_pred - y_seq).^2, 'all');
                
                [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWy, dby] = lstmBackward(x_seq, y_seq, y_pred, h_seq, c_seq, Wf, Wi, Wo, Wc, Wy, bc, bi, bo, bf);
                
                Wf = Wf - learningRate * dWf;
                Wi = Wi - learningRate * dWi;
                Wo = Wo - learningRate * dWo;
                Wc = Wc - learningRate * dWc;
                Wy = Wy - learningRate * dWy;
                bf = bf - learningRate * dbf;
                bi = bi - learningRate * dbi;
                bo = bo - learningRate * dbo;
                bc = bc - learningRate * dbc;
                by = by - learningRate * dby;
            end
        end
    end
    
    modelParameters.numAngles = numAngles;
    modelParameters.numNeurons = numNeurons;
    modelParameters.neuralData = neuralData;
    modelParameters.lstmParams.Wf = Wf;
    modelParameters.lstmParams.Wi = Wi;
    modelParameters.lstmParams.Wo = Wo;
    modelParameters.lstmParams.Wc = Wc;
    modelParameters.lstmParams.bf = bf;
    modelParameters.lstmParams.bi = bi;
    modelParameters.lstmParams.bo = bo;
    modelParameters.lstmParams.bc = bc;
    modelParameters.lstmParams.Wy = Wy;
    modelParameters.lstmParams.by = by;
    modelParameters.trialStates = struct();

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

function [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWy, dby] = lstmBackward(x_seq, y_seq, y_pred, h_seq, c_seq, Wf, Wi, Wo, Wc, Wy, bc, bi, bo, bf)
    numSteps = size(x_seq, 2);
    numHiddenUnits = size(Wf, 1);
    
    dWf = zeros(size(Wf));
    dWi = zeros(size(Wi));
    dWo = zeros(size(Wo));
    dWc = zeros(size(Wc));
    dWy = zeros(size(Wy));
    
    dbf = zeros(size(Wf, 1), 1);
    dbi = zeros(size(Wi, 1), 1);
    dbo = zeros(size(Wo, 1), 1);
    dbc = zeros(size(Wc, 1), 1);
    dby = zeros(size(Wy, 1), 1);
    
    dht_next = zeros(numHiddenUnits, 1);
    dct_next = zeros(numHiddenUnits, 1);
    
    for t = numSteps:-1:1
        dy = y_pred(:, t) - y_seq(:, t);
        dWy = dWy + dy * h_seq(:, t)';
        dby = dby + dy;
        
        dh = Wy' * dy + dht_next;
        do = dh .* tanh(c_seq(:, t));
        dct = dh .* sigmoid(c_seq(:, t)) .* (1 - tanh(c_seq(:, t)).^2) + dct_next;
        
        di = dct .* tanh(Wc * [h_seq(:, max(1, t-1)); x_seq(:, t)] + bc);
        df = dct .* c_seq(:, max(1, t-1));
        dc = dct .* sigmoid(Wi * [h_seq(:, max(1, t-1)); x_seq(:, t)] + bi);
        
        dWo = dWo + (do .* sigmoid(Wo * [h_seq(:, max(1, t-1)); x_seq(:, t)] + bo) .* (1 - sigmoid(Wo * [h_seq(:, max(1, t-1)); x_seq(:, t)] + bo))) * [h_seq(:, max(1, t-1)); x_seq(:, t)]';
        dWi = dWi + (di .* sigmoid(Wi * [h_seq(:, max(1, t-1)); x_seq(:, t)] + bi) .* (1 - sigmoid(Wi * [h_seq(:, max(1, t-1)); x_seq(:, t)] + bi))) * [h_seq(:, max(1, t-1)); x_seq(:, t)]';
        dWf = dWf + (df .* sigmoid(Wf * [h_seq(:, max(1, t-1)); x_seq(:, t)] + bf) .* (1 - sigmoid(Wf * [h_seq(:, max(1, t-1)); x_seq(:, t)] + bf))) * [h_seq(:, max(1, t-1)); x_seq(:, t)]';
        dWc = dWc + (dc .* (1 - tanh(Wc * [h_seq(:, max(1, t-1)); x_seq(:, t)] + bc).^2)) * [h_seq(:, max(1, t-1)); x_seq(:, t)]';
        
        dbo = dbo + do;
        dbi = dbi + di;
        dbf = dbf + df;
        dbc = dbc + dc;
        
        dht_next = Wf(:, 1:numHiddenUnits)' * df + Wi(:, 1:numHiddenUnits)' * di + Wo(:, 1:numHiddenUnits)' * do + Wc(:, 1:numHiddenUnits)' * dc;
        dct_next = dct .* sigmoid(Wf * [h_seq(:, max(1, t-1)); x_seq(:, t)] + bf);
    end
end