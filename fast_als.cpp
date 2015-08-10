#include "fast_als.h"
#include <armadillo>


fast_als::fast_als(std::istream& tuples_stream,
		int count_features,
		float alfa,
		float gamma,
		int count_samples,
		int likes_format) :
		_count_users(0),
		_count_items(0),
		_count_features(count_features),
		_als_alfa(alfa),
		_als_gamma(gamma)
{
	read_likes(tuples_stream, count_samples, likes_format);

	_features_users.assign(_count_users * _count_features, 0 );
	_features_items.assign(_count_items * _count_features, 0 );
}

fast_als::~fast_als()
{

}

void fast_als::read_likes(std::istream& tuples_stream, int count_simples, int format)
{
	std::string line;
	char const tab_delim = '\t';
	int i = 0;

	while(getline(tuples_stream, line))
	{
		std::istringstream line_stream(line);
		std::string value;
		getline(line_stream, value, tab_delim);
		unsigned long uid = atol(value.c_str());
		if (_users_map.find(uid) == _users_map.end())
		{
			_users_map[uid] = _count_users;
			_count_users++;
			_user_likes.push_back(std::vector<int>());
			_user_likes_weights.push_back(std::vector<float>());
		}

		int user = _users_map[uid];

		if( format == 0 )
		{
			getline(line_stream, value, tab_delim);
//			unsigned long gid = atol(value.c_str());
		}

		getline(line_stream, value, tab_delim);
		unsigned long iid = atol(value.c_str());
		float weight = 1;

		if(format == 1)
		{
			getline(line_stream, value, tab_delim);
			weight = atof( value.c_str() );
		}

		if (_items_map.find(iid) == _items_map.end())
		{
			_items_map[iid] = _count_items;
			_item_likes.push_back(std::vector<int>());
			_item_likes_weights.push_back(std::vector<float>());
			_count_items++;
		}

		int item = _items_map[iid];
		///
		/// adding data to user likes
		/// and to item likes
		///
		_user_likes[user].push_back( item );
		_user_likes_weights[user].push_back( weight );
		_item_likes[item].push_back( user );
		_item_likes_weights[item].push_back( weight );

		if (i % 10000 == 0) std::cerr << i << " u: " << _count_users << " i: " << _count_items << "\r";

		///std::cout << "u:" << user << " -> " << item << std::endl;
		///std::cout << "i:" << item << " -> " << user << std::endl;

		i++;
		if(count_simples && i > count_simples) break;
	}

	std::cerr << " u: " << _count_users << " i: " << _count_items << std::endl;
}

void fast_als::fill_rnd(features_vector& in_v, int in_size)
{
	std::cerr << "Generate random features.. ";
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0, 1);

	for (int i = 0; i < in_size * _count_features; i++)
	{
		in_v[i] = distribution(generator);
	}

	std::cerr << "done" << std::endl;
}

void fast_als::calculate(int count_iterations)
{
	fill_rnd(_features_users, _count_users);
	fill_rnd(_features_items, _count_items);

	for(int i = 0; i < count_iterations; i++)
	{
		time_t start =  time(0);
		std::cerr << "ALS Iteration: " << i << std::endl;

		std::cerr << "Items." << std::endl;
		solve(_item_likes.begin(), _item_likes_weights.begin(), _features_users, _count_users, _features_items, _count_items, _count_features);
		std::cerr << "Users." << std::endl;
		solve(_user_likes.begin(), _user_likes_weights.begin(), _features_items, _count_items, _features_users, _count_users, _count_features);

		time_t end =  time(0);
		std::cerr << "==== Iteration time : " << end - start << std::endl;

//		calc_error();
	}

	/// serialize(std::cout);
	/// calc_error();
}

void fast_als::solve(
		const likes_vector::const_iterator& likes,
		const likes_weights_vector::const_iterator& weights,
		const features_vector& in_v,
		int in_size,
		features_vector& out_v,
		int out_size,
		int _count_features)
{
	fast_als::features_vector g = calc_g(in_v, in_size, _count_features);

	for (int i = 0; i < out_size; i++)
	{
		calc_ridge_regression(*(likes + i), *(weights + i), in_v, in_size, out_v, out_size, _count_features, g, i);
	}
}

fast_als::features_vector fast_als::calc_g(const features_vector& in_v, int in_size, int _count_features)
{
	arma::fmat A(in_v);
	A.reshape(in_size, _count_features);
	A = A.t() * A;

	arma::fvec eigval;
	arma::fmat eigvec;

	arma::eig_sym(eigval, eigvec, A);

	arma::fmat lam_sqrt(arma::diagmat(arma::sqrt(eigval)));
	arma::fmat G = lam_sqrt * eigvec.t();

	return arma::conv_to<fast_als::features_vector>::from(arma::vectorise(G));
}

void fast_als::calc_ridge_regression(
		const likes_vector_item& likes,
		const likes_weights_vector_item& weights,
		const features_vector& in_v,
		int in_size,
		features_vector& out_v,
		int out_size,
		int _count_features,
		features_vector& g,
		int id)
{
	int count_samples = in_size + _count_features;
	std::vector<float> errors(count_samples);

	for (int i = 0; i < in_size; i++)
	{
		float sum = 0;
		for (int j = 0; j < _count_features; j++)
		{
			sum += out_v[id * _count_features + j] * in_v[i * _count_features + j];
		}
		errors[i] = likes[i] - sum;
	}

	for (int i = 0; i < _count_features; i++)
	{
		float sum = 0;
		for (int j = 0; j < _count_features; j++)
		{
			sum += out_v[id * _count_features + j] * g[i * _count_features + j];
		}
		errors[in_size + i] = -sum;
	}

	for (int k = 0; k < _count_features; k++)
	{
		for (int i = 0; i < in_size; i++)
		{
			errors[i] += out_v[id * _count_features + k] * in_v[i * _count_features + k];
		}
		for (int i = 0; i < _count_features; i++)
		{
			errors[in_size + i] += out_v[id * _count_features + k] * g[i * _count_features + k];
		}

		out_v[id * _count_features + k] = 0;

		float a = 0;
		float d = 0;
		for (int i = 0; i < in_size; i++)
		{
//			a += (1 + alpha * weights[i]) * in_v[i * _count_features + k] * in_v[i * _count_features + k];
//			d += (1 + alpha * weights[i]) * in_v[i * _count_features + k] * errors[i];
		}


	}

}


void fast_als::serialize_users_map(std::ostream& out)
{
	serialize_map(out, _users_map);
}

void fast_als::serialize_items_map(std::ostream& out)
{
	serialize_map(out, _items_map);
}

void fast_als::serialize_map(std::ostream& out, std::map<unsigned long, int>& out_map)
{
	std::map<unsigned long, int>::iterator it = out_map.begin();
	for( ; it != out_map.end(); it++)
	{
		out << it->first << "\t" << it->second << std::endl;
	}
}

void fast_als::serialize_items(std::ostream& out)
{
	const fast_als::features_vector& items = get_features_items();
	serialize_matrix(out, &items.front(),  _count_items, _count_features, true);
}

void fast_als::serialize_users(std::ostream& out)
{
	const fast_als::features_vector& users = get_features_users();
	serialize_matrix(out, &users.front(),  _count_users, _count_features, true);
}

void fast_als::serialize_matrix(std::ostream& out, const float* mat, int crow, int ccol, bool id)
{
	for(int i = 0; i < ccol; i++)
	{
		if(id) out << i << "\t";

		for(int j = 0; j < crow;  j++)
		{
			out << mat[i * crow + j] << (( j == crow-1)? "" : "\t");
		}
		out << std::endl;
	}
}
