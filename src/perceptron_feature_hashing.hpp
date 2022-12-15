#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronFeatureHashing : public BaseClf<PerceptronFeatureHashing> {
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    std::vector<double> weights_;

    int seed_;

public:
    PerceptronFeatureHashing(int log_num_buckets, double learning_rate)
        : log_num_buckets_(log_num_buckets)
        , learning_rate_(learning_rate)
        , bias_(0.0)
        , seed_(0x9748cd)
    {
        // set all weights to zero
        weights_.resize(1 << log_num_buckets_, 0.0);
    }

    void update_(const Email& email)
    {
        EmailIter iter = EmailIter(email, this->ngram_k);
        int o_d = transformTargetValue(this->classify(email));
        int t_d = transformTargetValue(email.is_spam());
        while(!iter.is_done()){
            std::string_view n_gram = iter.next();
            int bucketIndex = this->hash(n_gram,this->seed_) % (1 << log_num_buckets_);
            // Calculate Delta w_i
            double x_id = t_d;
            double delta_w_i = learning_rate_ * (t_d - o_d) * x_id;
            // Update weights
            weights_[bucketIndex] += delta_w_i;
        }
    }

    double predict_(const Email& email) const
    {
        double activationValue = 0;

        EmailIter iter = EmailIter(email, this->ngram_k);
        int targetValue = transformTargetValue(email.is_spam());
        while(!iter.is_done()){
            // Parse n_gram
            std::string_view n_gram = iter.next();
            int bucketIndex = this->hash(n_gram,this->seed_) % (1 << log_num_buckets_);
            // Get weight corresponding to n_gram
            double w_i = weights_[bucketIndex];

            // Update Activation Value
            activationValue += w_i * targetValue;

        }


        return activationValue - bias_;
    }

    void print_weights() const
    {
        std::cout << "bias " << bias_ << std::endl;
        for (size_t i = 0; i < weights_.size(); ++i)
        {
            std::cout << "w" <<i << " " << weights_[i] << std::endl;
        }
    }

private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_) % (1 << log_num_buckets_)); }

    size_t get_bucket(size_t hash) const
    {
        return hash;
    }

    int transformTargetValue(bool isSpam) const{
        int targetValue;
        (isSpam) ? (targetValue = 1) : (targetValue = -1);
        return targetValue;
    }

};

} // namespace bdap
