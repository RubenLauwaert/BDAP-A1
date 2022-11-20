#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesFeatureHashing : public BaseClf<NaiveBayesFeatureHashing> {
    int log_num_buckets_;
    int amt_buckets_;
    std::vector<int> buckets_spam_;
    std::vector<int> buckets_ham_;

    // TODO add fields here

    int seed_;

public:
    NaiveBayesFeatureHashing(int log_num_buckets, double threshold)
        : log_num_buckets_(log_num_buckets)
        , amt_buckets_(std::pow(2,log_num_buckets_))
        , buckets_spam_(std::vector(amt_buckets_,1))
        , buckets_ham_(std::vector<int>(amt_buckets_,1))
        , seed_(0x249cd)
    {
        // TODO initialize the data structures here
        this->threshold = threshold;
    }

    void update_(const Email& email)
    {
        // TODO implement this

        EmailIter iter = EmailIter(email, this->ngram_k);
        while(!iter.is_done()){
            size_t index = this->hash(iter.next(),seed_) % amt_buckets_;
            (email.is_spam()) ? (buckets_spam_[index] = buckets_spam_[index] + 1) :
                (buckets_ham_[index] = buckets_ham_[index] + 1);            
        }
        for(int bucket : buckets_ham_){
            //std::cout << bucket << "\n";
        }
    }

    double predict_(const Email& email) const
    {
        // https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering#Combining_individual_probabilities
        size_t amt_of_spam_ngrams = std::accumulate(buckets_spam_.begin(), buckets_spam_.end(),0);
        size_t amt_of_ham_ngrams = std::accumulate(buckets_ham_.begin(), buckets_ham_.end(),0);
        double p_S = (double) amt_of_spam_ngrams / (amt_of_ham_ngrams + amt_of_spam_ngrams);
        double p_H = 1 - p_S;

        double log_x = std::log(p_S);
        double log_y = std::log(p_H);
        
        EmailIter iter = EmailIter(email, this->ngram_k);
        while(!iter.is_done()){
            size_t index = this->hash(iter.next(),seed_) % amt_buckets_;
            double p_ngram_given_S = (double) buckets_spam_[index] / amt_of_spam_ngrams;
            double p_ngram_given_H = (double) buckets_ham_[index] / amt_of_ham_ngrams;

            // Update log_x and log_y

            log_x += std::log(p_ngram_given_S);
            log_y += std::log(p_ngram_given_H);
        }
        double inter = std::log(1 + std::exp(log_y - log_x));
        //std::cout << inter << "\n";
        return std::exp(inter);
        
    }

    void print_weights() const
    {
        size_t n = (1 << log_num_buckets_);
        for (size_t i = 0; i < n; ++i)
        {
            std::cout << "w" <<i << " " << buckets_spam_[i] << ", " << buckets_spam_[n + i] << std::endl;
        }
    }

private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const
    {
        // TODO limit the range of the hash values here
        return hash;
    }
};

} // namespace bdap
