#include "fast_als.h"
#include <armadillo>
#include <cmath>
#include <algorithm>
#include <set>
#include <map>
#include <ctime>
#include <limits>

fast_als::fast_als(std::istream& tuples_stream,
		int count_features,
		float alfa,
		float gamma,
		int count_samples,
		int likes_format,
		int count_error_samples_for_users,
		int count_error_samples_for_items,
		int max_likes) :
		_count_users(0),
		_count_items(0),
		_count_features(count_features),
		_als_alfa(alfa),
		_als_gamma(gamma),
		_count_error_samples_for_users(count_error_samples_for_users),
		_count_error_samples_for_items(count_error_samples_for_items),
		_max_likes(max_likes)
{
	//srand(time(NULL));
	srand(34);

	if (!_max_likes)
	{
		_max_likes = std::numeric_limits<int>::max();
	}

	read_likes(tuples_stream, count_samples, likes_format);

	//generate_test_set();

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
			_user_likes_weights_temp.push_back(std::vector<float>());
		}

		int user = _users_map[uid];

		if( format == 0 )
		{
			//getline(line_stream, value, tab_delim);
		}

		getline(line_stream, value, tab_delim);
		unsigned long iid = atol(value.c_str());
		float weight = 1;

		float weight_temp = 1;

		if(format == 1)
		{
			getline(line_stream, value, tab_delim);
			weight_temp = atof( value.c_str() );
			//weight = weight_temp;
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
		_user_likes_weights_temp[user].push_back( weight_temp );
		_item_likes[item].push_back( user );
		_item_likes_weights[item].push_back( weight );

		if (i % 10000 == 0) std::cout << i << " u: " << _count_users << " i: " << _count_items << "\r";

		i++;
		if(count_simples && i > count_simples) break;
	}

	std::cout.flush();
	std::cout << "\ntotal:\n u: " << _count_users << " i: " << _count_items << std::endl;
}

void fast_als::generate_test_set()
{
	int total_size = 0;
	//for (int idx = 0; idx < 10000; idx++)
	for (int i = 0; i < _count_users; i++)
	{
		//int i = rand() % _count_users;
		total_size += _user_likes[i].size();
		int coin = 1;
		if (coin == 1)
		{
			int size = _user_likes[i].size();
			for (int j = 0; j < size / 2;)
			{
				int id = rand() % _user_likes[i].size();

				if (_user_likes_weights_temp[i][id] < 4)
				{
					continue;
				}
				test_set.push_back(std::make_pair(i, _user_likes[i][id]));

				for (unsigned int k = 0; k < _item_likes[_user_likes[i][id]].size(); k++)
				{
					if (_item_likes[_user_likes[i][id]][k] == i)
					{
						_item_likes[_user_likes[i][id]].erase(_item_likes[_user_likes[i][id]].begin() + k);
						_item_likes_weights[_user_likes[i][id]].erase(_item_likes_weights[_user_likes[i][id]].begin() + k);
					}
				}

				_user_likes[i].erase(_user_likes[i].begin() + id);
				_user_likes_weights[i].erase(_user_likes_weights[i].begin() + id);
				_user_likes_weights_temp[i].erase(_user_likes_weights_temp[i].begin() + id);
				break;
			}
		}
	}
}

void fast_als::fill_rnd(features_vector& in_v, int in_size)
{
	std::cerr << "Generate random features.. ";
	for (int i = 0; i < in_size * _count_features; i++)
	{
		in_v[i] = ((float)rand()/(float)RAND_MAX);
	}

	std::cerr << "done" << std::endl;
}

void fast_als::calculate(int count_iterations)
{
	fill_rnd(_features_users, _count_users);

	std::ofstream hr10("hr10.txt");

	omp_set_num_threads(24);
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

		//MSE();
		hr10 << hit_rate_cpu() << std::endl;

	}

	hr10.close();

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
	time_t start =  time(0);
	fast_als::features_vector g = calc_g(in_v, in_size, _count_features);
	//fast_als::features_vector g(_count_features * _count_features);
	//fill_rnd(g, _count_features);
	time_t end =  time(0) - start;
	std::cerr << "calc g: " << end << std::endl;

	start = time(0);
	#pragma omp parallel for num_threads(24)
	for (int i = 0; i < out_size; i++)
	{
		calc_ridge_regression(*(likes + i), *(weights + i), in_v, (*(likes + i)).size(), out_v, _count_features, g, i);
	}
	end = time(0) - start;
	std::cerr << "calc regression: " << end << std::endl;

}

fast_als::features_vector fast_als::calc_g(const features_vector& in_v, int in_size, int _count_features)
{
	arma::fmat A(in_v);
	A.reshape(_count_features, in_size);
	A = A.t();
	A = A.t() * A;

	arma::fmat U;
	arma::fvec s;
	arma::fmat V;

	arma::svd(U,s,V,A);

	arma::fmat lam_sqrt2(arma::diagmat(arma::sqrt(s)));
	arma::fmat G2 = lam_sqrt2 * U.t();

	return arma::conv_to<fast_als::features_vector>::from(arma::vectorise(G2.t()));
}

void fast_als::calc_ridge_regression(
		const likes_vector_item& likes,
		const likes_weights_vector_item& weights,
		const features_vector& in_v,
		int in_size,
		features_vector& out_v,
		int _count_features,
		features_vector& g,
		int id)
{
	int random_offset = 0;
	if (in_size > _max_likes)
	{
		random_offset = rand() % (in_size - _max_likes + 1);
		in_size = _max_likes;
	}
	int count_samples = in_size + _count_features;
	std::vector<float> errors(count_samples);

	int out_offset = id * _count_features;

	for (int i = 0; i < in_size; i++)
	{
		int in_id_off = likes[i + random_offset] * _count_features;
		float sum = 0;
		for (int j = 0; j < _count_features; j++)
		{
			sum += out_v[out_offset + j] * in_v[in_id_off + j];
		}
		float c = 1 + _als_alfa * weights[i + random_offset];
		errors[i] = (c / (c - 1)) - sum;
	}

	for (int i = 0; i < _count_features; i++)
	{
		float sum = 0;
		int g_off = i * _count_features;
		for (int j = 0; j < _count_features; j++)
		{
			sum += out_v[out_offset + j] * g[g_off + j];
		}
		errors[in_size + i] = -sum;
	}

	for (int k = 0; k < _count_features; k++)
	{
		float out_v_cur = out_v[out_offset + k];

		float a = 0;
		float d = 0;
		for (int i = 0; i < in_size; i++)
		{
			int in_id = likes[i + random_offset];
			float c = _als_alfa * weights[i + random_offset];
			float in_v_cur = in_v[in_id * _count_features + k];
			a += c * in_v_cur * in_v_cur;
			d += c * in_v_cur * (errors[i] + out_v_cur * in_v_cur);
		}
		for (int i = 0; i < _count_features; i++)
		{
			float g_cur = g[i * _count_features + k];
			a += g_cur * g_cur;
			d += g_cur * (errors[in_size + i] + out_v_cur * g_cur);
		}

		float out_v_cur_new = d / (_als_gamma + a);

		out_v[out_offset + k] = out_v_cur_new;

		float out_diff = out_v_cur - out_v_cur_new;

		for (int i = 0; i < in_size; i++)
		{
			errors[i] += out_diff * in_v[likes[i + random_offset] * _count_features + k];
		}
		for (int i = 0; i < _count_features; i++)
		{
			errors[in_size + i] += out_diff * g[i * _count_features + k];
		}
	}
}

float fast_als::hit_rate_cpu()
{
	if (!test_set.size())
	{
		return 0;
	}
	float tp = 0;
	for (int i = 0; i < test_set.size(); i++)
	{
		int user = test_set[i].first;
		int item = test_set[i].second;
		std::vector<float> predict(_count_items);
		#pragma omp parallel for num_threads(24)
		for (int j = 0; j < _count_items; j++)
		{
			float sum = 0;
			for (int k = 0; k < _count_features; k++)
			{
				sum += _features_users[user * _count_features + k]
					* _features_items[j * _count_features + k];
			}
			predict[j] = sum;
		}

		for (unsigned int j = 0; j < _user_likes[user].size(); j++)
		{
			int item_id = _user_likes[user][j];
			predict[item_id] = -1000000;
		}

		for (int j = 0; j < 10; j++)
		{
			std::vector<float>::iterator it = std::max_element(predict.begin(), predict.end());
			int top_item = std::distance(predict.begin(), it);
			predict[top_item] = -1000000;
			if (top_item == item)
			{
				tp++;
				break;
			}
		}
	}

	float hr10 = tp * 1.0 / test_set.size();

	std::cout << hr10 << std::endl;

	return hr10;
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
	serialize_matrix(out, &items.front(), _count_features, _count_items, true);
}

void fast_als::serialize_users(std::ostream& out)
{
	const fast_als::features_vector& users = get_features_users();
	serialize_matrix(out, &users.front(), _count_features, _count_users, true);
}

void fast_als::serialize_matrix(std::ostream& out, const float* mat, int crow, int ccol, bool id)
{
	char* buf = (char*)malloc(10 * sizeof(char));
	for(int i = 0; i < ccol; i++)
	{
		if(id) out << i << "\t";

		for(int j = 0; j < crow;  j++)
		{
			sprintf(buf, "%.1f", mat[i * crow + j]);
			out << buf << (( j == crow-1)? "" : "\t");
		}
		out << std::endl;
	}
}
