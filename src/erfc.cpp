#include<iostream>
#include<cmath>
#include<cassert>
#include<iomanip>

double erfcApporx(double x){
    double p = 0.3275911;
    double a[5] = {0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429};
    double t = 1 / (1 + x * p);
    double s = 0;
    for(int i=4;i>=0;i--){
        s = t * (s + a[i]); 
    }
    s *= std::exp(-x * x);
    return s;
}

int main(){
    std :: cout << std :: fixed << std :: setprecision(15) << '\n';
    double x = 0.9182634;
    std :: cout << std::erfc(x) << '\n';
    std :: cout << erfcApporx(x) << '\n';
}
