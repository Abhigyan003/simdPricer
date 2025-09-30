#include "utils.h"

double normal_cdf(double value)
{
	return 0.5 * std::erfc(-value * M_SQRT1_2);
}
