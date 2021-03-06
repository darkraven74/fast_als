#ifndef FAST_ALS_H_
#define FAST_ALS_H_

#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

class fast_als {
public:
	///
	/// Definition of features vector
	///
	typedef std::vector<float> features_vector;
	typedef std::vector< std::vector<int> >	 likes_vector;
	typedef std::vector< std::vector<float> >   likes_weights_vector;
	typedef std::vector<int> likes_vector_item;
	typedef std::vector<float> likes_weights_vector_item;
	///
	/// Ctor
	/// Inputs are:
	/// stream with triplets:
	/// count_features - count latent features
	/// format of likes
	/// 0 - old
	/// 1 - simple
	/// <user> <item> <weight>
	///
	fast_als(std::istream& tuples_stream,
			int count_features,
			float alfa,
			float gamma,
			int count_samples,
			int likes_format,
			int count_error_samples_for_users,
			int count_error_samples_for_items,
			int max_likes);

	virtual ~fast_als();

	///
	/// Calculate als (Matrix Factorization)
	/// in
	/// count_iterations - count iterations
	///
	virtual void calculate(int count_iterations);

	virtual float hit_rate_cpu();

	///
	/// Get Items features vector
	///
	const features_vector& get_features_items() const { return _features_items; }
	int get_count_items() const { return _count_items; }

	///
	/// Get Users features vector
	///
	const features_vector& get_features_users() const { return _features_users; }
	int get_count_users() const { return _count_users; }

	void serialize_map(std::ostream& out, std::map<unsigned long, int>& out_map);
	void serialize_matrix(std::ostream& out, const float* mat, int crow, int ccol, bool id = false);
	void serialize_users(std::ostream& out);
	void serialize_items(std::ostream& out);
	void serialize_users_map(std::ostream& out);
	void serialize_items_map(std::ostream& out);

protected:
	///
	/// Read likes from stream
	/// if format == 0
	/// user group item
	/// if format == 1
	/// user item weight
	///
	void read_likes(std::istream& tuples_stream, int count_simples, int format);

	///
	/// fill random values to features matrix
	///
	void fill_rnd(features_vector& in_v, int in_size);

	///
	/// solve one iteration of als
	///
	void solve(
			const likes_vector::const_iterator& likes,
			const likes_weights_vector::const_iterator& weights,
			const features_vector& in_v,
			int in_size,
			features_vector& out_v,
			int out_size,
			int _count_features);

	fast_als::features_vector calc_g(const features_vector& in_v, int in_size, int _count_features);

	void calc_ridge_regression(
			const likes_vector_item& likes,
			const likes_weights_vector_item& weights,
			const features_vector& in_v,
			int in_size,
			features_vector& out_v,
			int _count_features,
			features_vector& g,
			int id);

	void generate_test_set();

private:
	///
	/// features vectors, for users and items
	///
	features_vector _features_users;
	int _count_users;
	features_vector _features_items;
	int _count_items;

	int _count_features;

	///
	/// Internal data
	///
	std::map<unsigned long, int> _users_map;
	std::map<unsigned long, int> _items_map;
	likes_vector                 _user_likes;
	likes_weights_vector         _user_likes_weights;
	likes_vector                 _item_likes;
	likes_weights_vector         _item_likes_weights;

	float _als_alfa;
	float _als_gamma;

	int _max_likes;

	///
	/// Count samples for calculate error
	///
	int _count_error_samples_for_users;
	std::vector<int>   users_for_error;
	int _count_error_samples_for_items;
	std::vector<int>   items_for_error;

	std::vector<std::pair<int, int> > test_set;
	likes_weights_vector         _user_likes_weights_temp;
};

#endif /* FAST_ALS_H_ */
