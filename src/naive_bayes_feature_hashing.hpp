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
    int amt_ngrams_spam;
    int amt_ngrams_ham;
    std::vector<int> buckets_spam_;
    std::vector<int> buckets_ham_;

    // TODO add fields here

    int seed_;

public:
    NaiveBayesFeatureHashing(int log_num_buckets, double threshold)
        : log_num_buckets_(log_num_buckets)
        , amt_buckets_(std::pow(2,log_num_buckets_))
        , amt_ngrams_spam(std::pow(2,log_num_buckets_))
        , amt_ngrams_ham(std::pow(2,log_num_buckets_))
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
            (email.is_spam()) ? ({buckets_spam_[index] += 1;
            amt_ngrams_spam += 1;}) :
                ({buckets_ham_[index] += 1;
                amt_ngrams_ham += 1;});            
        }
    }

    double predict_(const Email& email) const
    {
        double p_S = (double) amt_ngrams_spam / (amt_ngrams_ham + amt_ngrams_spam);
        double p_H = 1 - p_S;
        std::vector<double> vec_p_spam_given_n_gram = std::vector<double>();
        double sum_vec = 0;
        EmailIter iter = EmailIter(email, this->ngram_k);
        while(!iter.is_done()){
            size_t index = get_bucket(iter.next());
            double p_ngram_given_S = (double) buckets_spam_[index] / amt_ngrams_spam;
            double p_ngram_given_H = (double) buckets_ham_[index] / amt_ngrams_ham;
            double p_spam_given_n_gram = (p_S * p_ngram_given_S) / (p_S * p_ngram_given_S + p_H * p_ngram_given_H);
            vec_p_spam_given_n_gram.push_back(p_spam_given_n_gram);
            sum_vec += p_spam_given_n_gram;
        }
        
        double average = (double) sum_vec / vec_p_spam_given_n_gram.size();
        return average;
        
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
        
        return hash % amt_buckets_;
    }
};

} // namespace bdap
