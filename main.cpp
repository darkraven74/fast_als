#include "fast_als.h"
#include <string>
#include <cstring>
#include <iostream>
#include <sys/time.h>

using namespace std;

int main(int argc, char *argv[])
{
	string output_file_name; // = "out.txt";
	string likes_file_name;
	int features_size = 50;
	int csimples = 0;
	int cit = 10;
	int likes_format = 0;
	float als_alfa = 5;

	for(int i = 1; i <  argc; i++)
	{
		std::string sarg = argv[i];
		if( sarg == "--likes")
		{
			i++;
			likes_file_name = argv[i];
		}
		else
		if( sarg == "--f_size")
		{
			i++;
			features_size = atoi(argv[i]);
			std::cerr << " Count features:  " << features_size << std::endl;
		}
		else
		if( sarg == "--csamples")
		{
			i++;
			csimples = atoi(argv[i]);
		}
		else
		if( sarg == "--it")
		{
			i++;
			cit = atoi(argv[i]);
		}else
		if( sarg == "--out")
		{
			i++;
			output_file_name = argv[i];
		}else
		if( sarg == "--likes-format")
		{
			i++;
			likes_format = atoi(argv[i]);
		}else
		if( sarg == "--als-alfa")
		{
			i++;
			als_alfa = atof(argv[i]);
		}
	}

	std::ifstream f_stream(likes_file_name.c_str() );
	std::istream& in((likes_file_name.length() == 0) ? std::cin : f_stream);

	std::cerr << " Count ALS iteration " << cit << std::endl;
	std::cerr << " Start Matrix Factorization - ALS " << std::endl;
	std::cerr << " Input file format -  " << likes_format << std::endl;
	std::cerr << " ALS alfa -  " << als_alfa << std::endl;

	fast_als als_alg(in, features_size, als_alfa, 0.01, csimples, likes_format);

	struct timeval t1;
	struct timeval t2;

	gettimeofday(&t1, NULL);
	als_alg.calculate(cit);
	gettimeofday(&t2, NULL);

	std::cerr << "als calc time: " << t2.tv_sec - t1.tv_sec << std::endl;

	std::ofstream fout_users((output_file_name+".ufea").c_str());
	als_alg.serialize_users(fout_users);
	fout_users.close();

	std::ofstream fout_items((output_file_name+".ifea").c_str());
	als_alg.serialize_items(fout_items);
	fout_items.close();

	std::ofstream fout_umap((output_file_name+".umap").c_str());
	als_alg.serialize_users_map(fout_umap);
	fout_umap.close();

	std::ofstream fout_imap((output_file_name+".imap").c_str());
	als_alg.serialize_items_map(fout_imap);
	fout_imap.close();

	return 0;
}
