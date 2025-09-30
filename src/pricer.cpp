#include "pricer.h"
#include "utils.h"
#include <cmath>

double black_scholes_price(double S, double K, double T, double r, double v, bool is_call){
	double d1 = (std::log(S / K) + (r + 0.5 * v * v) * T) / (v * std::sqrt(T));
	double d2 = d1 - v * sqrt(T);

	if(is_call)
		return S * normal_cdf(d1) - K * std :: exp(-r * T) * normal_cdf(d2);
	return K * std::exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1);
}
