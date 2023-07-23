#pragma once

#include <iostream>
#include <vector>

class Neuron {
public:
    Neuron(int n_weights);
    ~Neuron();

    void activate(std::vector<float> inputs);
    void transfer();
    float transfer_derivate() { return static_cast<float>(m_output * (1.0 - m_output)); };

    // return mutable reference to the neuron weights
    std::vector<float>& get_weights() { return m_weights; };

    float get_output() { return m_output; };
    float get_activation() { return m_activation; };
    float get_delta() { return m_delta; };

    void set_delta(float delta) { m_delta = delta; };

private:
    size_t m_nWeights;
    std::vector<float> m_weights;
    float m_activation;
    float m_output;
    float m_delta;

    void initWeights(int n_weights);
};


class Layer {
public:
    Layer(int n_neurons, int n_weights);
    ~Layer();

    // return mutable reference to the neurons
    std::vector<Neuron>& get_neurons() { return m_neurons; };

private:
    void initNeurons(int n_neurons, int n_weights);
    
    std::vector<Neuron> m_neurons;
};

class Network {
public:
    Network();
    ~Network();

    void initialize_network(int n_inputs, int n_hidden, int n_outputs);

    void add_layer(int n_neurons, int n_weights);
    std::vector<float> forward_propagate(std::vector<float> inputs);
    void backward_propagate_error(std::vector<float> expected);
    void update_weights(std::vector<float> inputs, float lr);
    void train(std::vector<std::vector<float>>training_data, float lr, size_t n_epoch, size_t n_outputs);
    int predict(std::vector<float> input);

    void display_human();

private:
    size_t m_nLayers;
    std::vector<Layer> m_layers;
};