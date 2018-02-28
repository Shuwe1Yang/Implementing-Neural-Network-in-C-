#ifndef Net_hpp
#define Net_hpp
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include "type.hpp"
#include "Neuron.hpp"
using namespace std;






//Class Net is here*************************
class Net{
public:
    Net(const vector<unsigned int> &topology);
    void feedForward(const vector<double> &inputData);
    void backProp(const vector<double> &targetData);
    void getResults(vector<double> &resultData) const;
    double getRecentAvgError()const;
    
private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAvgError;
    double m_recentAvgSmoothingFactor;
};

//Body of Net constructor
Net::Net(const vector<unsigned int> &topology){
    unsigned int numLayers = topology.size();
    for (unsigned int layerNum = 0; layerNum < numLayers; ++layerNum){
        m_layers.push_back(Layer());
        unsigned int numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        
        //we have made a new Layer, now fill it ith neurons, and add a bias neuron to the layer:
        for (unsigned int neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            
            //cout << "Made a Neuron!" << endl;
        }
    }
    
}

//Body of FeedForward
void Net::feedForward(const vector<double> &inputData){
    
    assert(inputData.size() == m_layers[0].size() - 1);
    
    // Assign (latch) the input values into the input neurons
    for (unsigned int i = 0; i < inputData.size(); ++i){
        m_layers[0][i].setOutputData(inputData[i]);
    }
    
    //Forward propagation
    for (unsigned int layerNum = 1; layerNum < m_layers.size(); ++layerNum){
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned int i = 0; i < m_layers[layerNum].size() - 1; ++i){
            m_layers[layerNum][i].feedForward(prevLayer);
        }
        //Forcing the bias node's output value to 1.0. It's the last neuron created above
        m_layers.back().back().setOutputData(1.0);
    }
}

//Body of backpropagation
void Net::backProp(const vector<double> &targetData){
    //Calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = m_layers.back();
    double m_error = 0.0;
    
    for (unsigned int i =0; i < outputLayer.size() - 1; ++i){
        double delta = targetData[i] - outputLayer[i].getOutputData();
        m_error += delta * delta;
        
    }
    m_error /= outputLayer.size() - 1;//get avg error squared
    m_error = sqrt(m_error);//RMS
    
    //Implement a recent avg measurement;
    m_recentAvgError = (m_recentAvgError * m_recentAvgSmoothingFactor + m_error)/(m_recentAvgSmoothingFactor + 1.0);
    
    //Calculate output layer gradients
    for (unsigned int i = 0; i < outputLayer.size() - 1; ++i){
        outputLayer[i].calcOutputGrad(targetData[i]);
    }
    
    //Calculate gradients on hidden layers
    for (unsigned int layerNum = m_layers.size() - 2; layerNum > 0; --layerNum){
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
        
        for (unsigned int i = 0; i < hiddenLayer.size(); ++i){
            hiddenLayer[i].calcHiddenGrad(nextLayer);
        }
    }
    //For all layers from outputs to first hidden layer
    //update connection weights
    for (unsigned int layerNum = m_layers.size() - 1; layerNum > 0; --layerNum){
        Layer &layer = m_layers[layerNum];
        Layer prevLayer = m_layers[layerNum - 1];
        
        for (unsigned int i = 0; i < layer.size() - 1; ++i){
            layer[i].updateInputWeights(prevLayer);
        }
    }
}


//Body of get results
void Net::getResults(vector<double> &resultData) const{
    resultData.clear();
    
    for(unsigned int i = 0; i < m_layers.back().size() - 1; ++i){
        resultData.push_back(m_layers.back()[i].getOutputData());
    
    }
}

//Body of get recent avg error
double Net::getRecentAvgError()const{
    return m_recentAvgError;
}


#endif /* Net_hpp */
