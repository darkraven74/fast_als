#ifndef FAST_ALS_H_
#define FAST_ALS_H_

class fast_als {
public:
	fast_als();
	virtual ~fast_als();

	///
	/// Definition of features vector
	///
	typedef std::vector<float> features_vector;
	typedef thrust::device_vector<float> features_vector_device;
	typedef std::vector< std::vector<int> >	 likes_vector;
	typedef std::vector< std::vector<float> >   likes_weights_vector;
	///
	/// Ctor
	/// Inputs are:
	/// stream with triplets:
	/// count_users - count users in stream
	/// count_items - count items
	/// count_features - count latent features
	/// format of likes
	/// 0 - old
	/// 1 - simple
	/// <user> <item> <weight>
	///
	als( std::istream& tuples_stream,
		 int count_features,
		 float alfa,
		 float gamma,
		 int count_samples,
		 int count_error_samples_for_users,
		 int count_error_samples_for_items,
		 int likes_format,
		 int count_gpus = 1);

	virtual ~als();

	///
	/// Calculate als (Matrix Factorization)
	/// in
	/// count_iterations - count iterations
	///
	virtual void calculate(int count_iterations);

};

#endif /* FAST_ALS_H_ */
