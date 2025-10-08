#include <iostream>
#include <iomanip>
#include "include/pricer.h"

int main(){
	constexpr double S = 100.0;
	constexpr double K = 100.0;
	constexpr double T = 1.0;
	constexpr double r = 0.05;
	constexpr double v = 0.20;

	double call_price = black_scholes_price(S, K, T, r, v, true);
	double put_price = black_scholes_price(S, K, T, r, v,false);
	
	std :: cout << std :: fixed << std :: setprecision(4);
	std :: cout << call_price << " " << put_price << "\n";
}
