#include "fast_als.h"
#include <armadillo>
#include <cmath>
//#include <chrono>
#include <algorithm>
#include <set>
#include <map>
#include <ctime>

fast_als::fast_als(std::istream& tuples_stream,
		int count_features,
		float alfa,
		float gamma,
		int count_samples,
		int likes_format,
		int count_error_samples_for_users,
		int count_error_samples_for_items) :
		_count_users(0),
		_count_items(0),
		_count_features(count_features),
		_als_alfa(alfa),
		_als_gamma(gamma),
		_count_error_samples_for_users(count_error_samples_for_users),
		_count_error_samples_for_items(count_error_samples_for_items)
{
	srand(time(NULL));
	//srand(34);

	read_likes(tuples_stream, count_samples, likes_format);

	/*std::map<unsigned long, int> m;

	for (int i = 0; i < _count_users; i++)
	{
		m.insert(std::make_pair(i, _user_likes[i].size()));
	}

	for (std::map<unsigned long, int>::iterator it = _users_map.begin(); it != _users_map.end(); it++)
	{
		unsigned long uid = it->first;
		int array_id = it->second;
		int size = m[array_id];

		if (size > 500)
			std::cout << uid << " " << size << std::endl;
	}
	*/

	generate_test_set();

	_features_users.assign(_count_users * _count_features, 0 );
	_features_items.assign(_count_items * _count_features, 0 );
}

fast_als::~fast_als()
{

}

void fast_als::read_likes(std::istream& tuples_stream, int count_simples, int format)
{

	/*
	std::ifstream f_stream("/home/darkraven/Prog/Mail.ru/als/als_data/500map.txt");
	std::istream& in_good_id(f_stream);


	std::ofstream small_data("small_data.txt");


	std::set<unsigned long> ss;

	std::string gg;
	while(getline(in_good_id, gg))
	{
		std::istringstream line_stream(gg);
		std::string value;
		getline(line_stream, value);
		ss.insert(atol(value.c_str()));
	}
	 */

//	std::ofstream small_ml("ml_new");



	std::string line;
	char const tab_delim = '\t';
	int i = 0;

	while(getline(tuples_stream, line))
	{
		std::istringstream line_stream(line);
		std::string value;
		getline(line_stream, value, tab_delim);
		unsigned long uid = atol(value.c_str());

		/*if (ss.count(uid) != 0)
		{
			small_data << line << std::endl;
		}*/

		if (_users_map.find(uid) == _users_map.end())
		{
			/*if (_count_users == 677)
			{
				std::cout << "bad user id: " << uid << std::endl;
			}*/
			_users_map[uid] = _count_users;
			_count_users++;
			_user_likes.push_back(std::vector<int>());
			_user_likes_weights.push_back(std::vector<float>());
			_user_likes_weights_temp.push_back(std::vector<float>());
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

		float weight_temp = 1;

		if(format == 1)
		{
			getline(line_stream, value, tab_delim);
			weight_temp = atof( value.c_str() );
			//weight = weight_temp;
		}

		// discard ratings 3 and below

		/*if (weight_temp > 3)
		{
			small_ml << line << std::endl;
		}*/


		// discard bad users
		/*if (uid != 685)
		{
			small_ml << line << std::endl;
		}*/


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

		///std::cout << "u:" << user << " -> " << item << std::endl;
		///std::cout << "i:" << item << " -> " << user << std::endl;

		i++;
		if(count_simples && i > count_simples) break;
	}


//	small_ml.close();


	//small_data.close();

	std::cout.flush();
	std::cout << "\ntotal:\n u: " << _count_users << " i: " << _count_items << std::endl;
}

void fast_als::generate_test_set()
{
//	std::ofstream out_test("out_test.txt");
//	std::ofstream out_train("out_train.txt");

	//out_test << "u: " << _count_users << std::endl;
	//out_test << "i: " << _count_items << std::endl;


	int total_size = 0;
	for (int idx = 0; idx < 10000; idx++)
	//for (int i = 0; i < _count_users; i++)
	{
		int i = rand() % _count_users;
		total_size += _user_likes[i].size();
		int coin = 1;
		if (coin == 1)
		{
			int size = _user_likes[i].size();
			for (int j = 0; j < size / 2;)
			{
				int id = rand() % _user_likes[i].size();
//				int id = j;

				/*if (_user_likes_weights_temp[i][id] < 4)
				{
					continue;
				}*/
				test_set.push_back(std::make_pair(i, _user_likes[i][id]));
//				out_test << i << "," << _user_likes[i][id] << "," << "1" << std::endl;

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
//	std::cout << "test_set size" << test_set.size() << std::endl;
//	std::cout << "test_set %: " << test_set.size() * 1.0 / total_size << std::endl;
//	std::cout << "reco in user " << total_size* 1.0 / _count_users << std::endl;


	/*out_train << "user,item,rating" << std::endl;

	for (int i = 0; i < _count_users; i++)
	{
		int size = _user_likes[i].size();
		for (int j = 0; j < size; j++)
		{
			out_train << i << "," << _user_likes[i][j] << "," << "1" << std::endl;
		}
	}


	out_test.close();
	out_train.close();
	*/


}

void fast_als::fill_rnd(features_vector& in_v, int in_size)
{
	std::cerr << "Generate random features.. ";
//	std::default_random_engine generator(time(NULL));
//	std::default_random_engine generator(34);
//	std::normal_distribution<float> distribution(0, 1);
//	std::uniform_real_distribution<float> distribution(0, 1);

	for (int i = 0; i < in_size * _count_features; i++)
	{
//		in_v[i] = distribution(generator);
		in_v[i] = ((float)rand()/(float)RAND_MAX);
//		in_v[i] = sqrt(1.0 / _count_features) * distribution(generator);
	}

	std::cerr << "done" << std::endl;
}

void fast_als::calculate(int count_iterations)
{
	fill_rnd(_features_users, _count_users);
//	fill_rnd(_features_items, _count_items);

	std::ofstream hr10("hr10.txt");

	//openblas_set_num_threads(24);
	//goto_set_num_threads(24);
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
		calc_ridge_regression(*(likes + i), *(weights + i), in_v, (*(likes + i)).size(), out_v, out_size, _count_features, g, i);
	}
	end = time(0) - start;
	std::cerr << "calc regression: " << end << std::endl;

}

fast_als::features_vector fast_als::calc_g(const features_vector& in_v, int in_size, int _count_features)
{
	arma::fmat A(in_v);
	A.reshape(_count_features, in_size);
	A = A.t();

//	std::cerr << "\nA arma matrix: " << std::endl;
//	A.print();

	A = A.t() * A;



//	std::cout << "\nAt*A matrix: " << std::endl;
//	A.print();

	/*arma::fvec eigval;
	arma::fmat eigvec;

	arma::eig_sym(eigval, eigvec, A);
	*/
//	std::cout << "\neigval matrix: \n" << std::endl;
//	eigval.print();

//	std::cout << "\neigvec matrix: " << std::endl;
//	eigvec.print();


	arma::fmat U;
	arma::fvec s;
	arma::fmat V;

	arma::svd(U,s,V,A);

//	std::cout << "\nU matrix (eigvec): \n" << std::endl;
//	U.print();

//	std::cout << "\ns vector (eigval): \n" << std::endl;
//	s.print();

//	std::cout << "\nV matrix: \n" << std::endl;
//	V.print();


	arma::fmat lam_sqrt2(arma::diagmat(arma::sqrt(s)));

//	arma::fmat lam_sqrt(arma::diagmat(arma::sqrt(eigval)));

//	std::cerr << "\nlam matrix: " << std::endl;
//	lam_sqrt.print();

//	arma::fmat G = lam_sqrt * eigvec.t();

	arma::fmat G2 = lam_sqrt2 * U.t();

//	std::cout << "\nG matrix: " << std::endl;
//	G.print();


//	std::cout << "\nG2 matrix: " << std::endl;
//	G2.print();

	return arma::conv_to<fast_als::features_vector>::from(arma::vectorise(G2.t()));
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

	int out_offset = id * _count_features;

	for (int i = 0; i < in_size; i++)
	{
		int in_id_off = likes[i] * _count_features;
		float sum = 0;
		for (int j = 0; j < _count_features; j++)
		{
			sum += out_v[out_offset + j] * in_v[in_id_off + j];
		}
		float c = 1 + _als_alfa * weights[i];
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
			int in_id = likes[i];
			float c = _als_alfa * weights[i];
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
			errors[i] += out_diff * in_v[likes[i] * _count_features + k];
		}
		for (int i = 0; i < _count_features; i++)
		{
			errors[in_size + i] += out_diff * g[i * _count_features + k];
		}
	}
}

void fast_als::MSE()
{
	std::vector<float> r(_count_error_samples_for_users * _count_error_samples_for_items, 0);
	features_vector users;
	features_vector items;
	srand(time(NULL));
	users.assign(_count_error_samples_for_users * _count_features, 0);
	if (users_for_error.size() == 0)
	{
		for(int i = 0; i < _count_error_samples_for_users; i++)
		{
//			const int r1 = rand() % _count_users;
			const int r1 = i;
			users_for_error.push_back(r1);
		}
	}

	for(unsigned int i = 0;  i < users_for_error.size(); i++)
	{
		const int r1 = users_for_error[i];
		for( int c = 0; c < _count_features; c++)
			users[i * _count_features + c] = _features_users[r1 * _count_features + c];
	}

	items.assign(_count_error_samples_for_items * _count_features, 0);

	if (items_for_error.size() == 0)
	{
		for(int i = 0; i < _count_error_samples_for_items; i++)
		{
//			const int r1 = rand() % _count_items;
			const int r1 = i;
			items_for_error.push_back(r1);
		}
	}

	for(unsigned int i = 0; i < items_for_error.size(); i++)
	{
		const int r1 = items_for_error[i];
		for( int c = 0; c < _count_features; c++)
			items[i * _count_features + c] = _features_items[r1 * _count_features + c];
	}

	/*for (int i = 0; i < _count_error_samples_for_users; i++)
	{
		for (int j = 0; j < _count_error_samples_for_items; j++)
		{
			int user_id = users_for_error[i];
			int item_id = items_for_error[j];

			for (unsigned int k = 0; k < _user_likes[user_id].size(); k++)
			{
				if (_user_likes[user_id][k] == item_id)
				{
					r[i * _count_error_samples_for_items + j] = 1;
				}
			}
		}
	}*/

	for (int i = 0; i < _count_error_samples_for_users; i++)
	{
		for (unsigned int k = 0; k < _user_likes[i].size(); k++)
		{
			if (_user_likes[i][k] < _count_error_samples_for_items)
				r[i * _count_error_samples_for_items + _user_likes[i][k]] = 1;
		}
	}

	/*for (unsigned int i = 0; i < test_set.size(); i++)
	{
		int user = test_set[i].first;
		int item = test_set[i].second;

		if (item < _count_error_samples_for_items)
			r[user * _count_error_samples_for_items + item] = 1;
	}*/



	arma::fmat P(users);
	P.reshape(_count_features, _count_error_samples_for_users);
	P = P.t();

	arma::fmat Q(items);
	Q.reshape(_count_features, _count_error_samples_for_items);


	arma::fmat predict = P * Q;

	float mse = 0;
	float size = 0;
	for (int i = 0; i < _count_error_samples_for_users; i++)
	{
		for (int j = 0; j < _count_error_samples_for_items; j++)
		{
			if (r[i * _count_error_samples_for_items + j] == 1)
			{
				size++;
				mse += (r[i * _count_error_samples_for_items + j] - predict.at(i, j)) * (r[i * _count_error_samples_for_items + j] - predict.at(i, j));
			}
		}
	}
	mse /= size;

//	std::cerr << " MSE: " << mse << std::endl;
	std::cout << mse << std::endl;
}

float fast_als::hit_rate_cpu()
{
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

float fast_als::hit_rate()
{

	/*std::ifstream train("out_train.txt");
	std::istream& str(train);
	std::set<std::pair<int, int> > set_tr;

	char const delim = ',';


	std::string gg;
	getline(str, gg);
	while(getline(str, gg))
	{
		std::istringstream line_stream(gg);
		std::string value;
		getline(line_stream, value, delim);
		int user = atoi(value.c_str());
		getline(line_stream, value, delim);
		set_tr.insert(std::make_pair(user, atoi(value.c_str())));
	}


	std::ifstream test("out_test.txt");
	std::istream& str2(test);
	std::set<std::pair<int, int> > set_te;


	while(getline(str2, gg))
	{
		std::istringstream line_stream(gg);
		std::string value;
		getline(line_stream, value, delim);
		int user = atoi(value.c_str());
		getline(line_stream, value, delim);
		set_te.insert(std::make_pair(user, atoi(value.c_str())));
	}


//	std::vector<float> p(942 * 1682);

	std::ifstream pred("/home/darkraven/Downloads/recs.csv");
	std::istream& str3(pred);

	std::set<std::pair<int, int> > rec;

	getline(str3, gg);
	float sum = 0;
	std::set<std::pair<int, int> > test_set_set(set_te.begin(), set_te.end());

	int cur_user = 0;

	int rank = 1;

	while(getline(str3, gg))
	{
		std::istringstream line_stream(gg);
		std::string value;
		getline(line_stream, value, delim);
		int user = atoi(value.c_str());
		getline(line_stream, value, delim);
		int item = atoi(value.c_str());
		getline(line_stream, value, delim);
		getline(line_stream, value, delim);
		rank = atoi(value.c_str());

		if (test_set_set.count(std::make_pair(user, item)))
		{
			sum += 1.0 / rank;
			cur_user++;
		}
		rec.insert(std::make_pair(user, item));
	}

//	float mrr = sum / _count_users;
//	std::cout << mrr << std::endl;



	float tp = 0;
	for (std::set<std::pair<int, int> >::iterator it = rec.begin(); it != rec.end(); it++)
	{
		if (test_set_set.count(*it))
		{
			tp++;
		}
	}
	float p = tp * 1.0 / rec.size();
	std::cout << p << std::endl;
*/
	/*int u = 0;
	while(getline(str3, gg))
	{
		std::istringstream line_stream(gg);
		std::string value;
		for (int i = 0; i < 1682; i++)
		{
			getline(line_stream, value, delim);
//			std::cout << value << " ";
			p[u * 1682 + i] = atof(value.c_str());
		}
		u++;
	}


	for (std::set<std::pair<int, int> >::iterator it = set_tr.begin(); it != set_tr.end(); it++)
	{
		p[it->first * 1682 + it->second] = -1000000;
	}


//	std::cout << "recs for 12: ";



	for (int i = 0; i < _count_users; i++)
	{

		std::vector<float> v(p.begin() + i * 1682, p.begin() + (i + 1) * 1682);


		for (int j = 0; j < 10; j++)
		{
			std::vector<float>::iterator it = std::max_element(v.begin(), v.end());
			int item = std::distance(v.begin(), it);
			if (i == 15)
			{
				//std::cout << "val: " << v[j] << "\n";
				printf("%d\n", item);
			}
			v[item] = -1000000;
			rec.insert(std::make_pair(i, item));
		}
	}*/

	//hit-rate10 calc
	/*float hit = 0;
	for (std::set<std::pair<int, int> >::iterator it = set_te.begin(); it != set_te.end(); it++)
	{
		if (rec.count(*it))
		{
			hit++;
		}
	}
	float hr = hit * 1.0 / set_te.size();

	std::cout << "rec size: " << rec.size() << " test size: " << set_te.size() << std::endl;
	std::cout << "Hit-rate: " << hr << std::endl;


	return;
*/
//*******************************************************

	arma::fmat P(_features_users);
	P.reshape(_count_features, _count_users);
	P = P.t();

	arma::fmat Q(_features_items);
	Q.reshape(_count_features, _count_items);

	arma::fmat predict = P * Q;

	for (int i = 0; i < _count_users; i++)
	{
		for (unsigned int j = 0; j < _user_likes[i].size(); j++)
		{
			int item_id = _user_likes[i][j];
			predict.at(i, item_id) = -1000000;
		}
	}

	std::set<std::pair<int, int> > test_set_set(test_set.begin(), test_set.end());

	std::set<std::pair<int, int> > recs;
//	float sum = 0;
	for (int i = 0; i < _count_users; i++)
	{
		arma::frowvec cur = predict.row(i);
		std::vector<float> v = arma::conv_to<std::vector<float> >::from(cur);

		for (int j = 0; j < 10; j++)
		{
			std::vector<float>::iterator it = std::max_element(v.begin(), v.end());
			int item = std::distance(v.begin(), it);
			v[item] = -1000000;
			recs.insert(std::make_pair(i, item));
			/*if (test_set_set.count(std::make_pair(i, item)))
			{
				sum += 1.0 / (j + 1);
				break;
			}*/
		}
	}

//	float mrr = sum / _count_users;

//	float sum = 0;
//	std::set<int> test_u;

	/*for (int u = 0; u < _count_users; u++)
	{

		float tp = 0;
		float size = 0;

		for (unsigned int i = 0; i < test_set.size(); i++)
		{
			int user = test_set[i].first;
			int item = test_set[i].second;

			test_u.insert(user);
			if (user == u)
			{
				size++;
			}

			if (user == u && recs.count(std::make_pair(user, item)))
			{
				tp++;
			}
		}

		if (size != 0)
			sum += tp / size;

	}*/

	//hit-rate10 calc
	float tp = 0;
	for (std::set<std::pair<int, int> >::iterator it = test_set_set.begin(); it != test_set_set.end(); it++)
	{
		if (recs.count(*it))
		{
			tp++;
		}
	}
	float hr10 = tp * 1.0 / test_set_set.size();


	//prec calc
	/*std::set<std::pair<int, int> > test_set_set(test_set.begin(), test_set.end());
	float tp = 0;
	for (std::set<std::pair<int, int> >::iterator it = recs.begin(); it != recs.end(); it++)
	{
		if (test_set_set.count(*it))
		{
			tp++;
		}
	}
	float p = tp * 1.0 / recs.size();
*/
//	float res = sum * 1.0 / test_u.size();

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
