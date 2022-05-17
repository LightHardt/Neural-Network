#include "neuralNet.hpp"

NeuralNet::NeuralNet(std::vector<unsigned> &topology)
{
    unsigned int numLayers = topology.size();

    for(unsigned int i = 0; i < numLayers; i++)
    {
        layers.push_back(Layer());

        // If not last layer get num outputs if last set 0
        unsigned int numOutputs = i == topology.size() - 1 ? 0 : topology[i + 1];

        // Add neurons then a bias
        for(unsigned int neuronNum = 0; neuronNum <= topology[i]; neuronNum++)
        {
            layers.back().push_back(Neuron(numOutputs,neuronNum));
        }
        // Set bias node
        layers.back().back().setOutputVal(1.0);
    }
}

void NeuralNet::feedForward(std::vector<double> &inputVals)
{
    assert(inputVals.size() == layers[0].size() - 1);

    for(unsigned int i = 0; i < inputVals.size(); i++)
        layers[0][i].setOutputVal(inputVals[i]);
    
    // Forward Propagate
    for(unsigned int layerNum = 1; layerNum < layers.size(); layerNum++)
    {
        Layer &prevLayer = layers[layerNum - 1];
        for(unsigned int n = 0; n < layers[layerNum].size() - 1; n++)
        {
            layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void NeuralNet::backProp(std::vector<double> &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = layers.back();
    error = 0.0;
    
    for(unsigned int n = 0; n < outputLayer.size() - 1; n++)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        error += delta * delta;
    }
    error /= outputLayer.size() - 1; // get average error squared
    error = sqrt(error); //RMS

    // Implement Recent average measurement
    recentAverageError =
            (recentAverageError * recentAverageSmoothingFactor + error) 
            / (recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for(unsigned int n = 0; n < outputLayer.size() - 1; n++)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers
    for(unsigned int layerNum = layers.size() - 2; layerNum > 0; layerNum--)
    {
        Layer &hiddenLayer = layers[layerNum];
        Layer &nextLayer = layers[layerNum + 1];

        for(unsigned int n = 0; n < hiddenLayer.size(); n++)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }

    }
    // Update connection weights
    for(unsigned int layerNum = layers.size() - 1; layerNum > 0; layerNum--)
    {
        Layer &layer = layers[layerNum];
        Layer &prevLayer = layers[layerNum - 1];

        for(unsigned int n = 0; n < layer.size() - 1; n++)
        {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void NeuralNet::getResults(std::vector<double> &resultVals)
{
    resultVals.clear();

    for(unsigned int n = 0; n < layers.back().size() - 1; n++)
    {
        resultVals.push_back(layers.back()[n].getOutputVal());
    }
}
