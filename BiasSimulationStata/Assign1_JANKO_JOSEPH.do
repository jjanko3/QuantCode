clear
capture cd "M:\stata"
if c(username)=="mayamikdash" {
cd "C:/Users/mayamikdash/Documents/EconometricsII/Assignment1"
}



capture prog drop endog
program endog, rclass
syntax, beta2(real) zmult(real) umult(real)
drop _all


scalar beta0 = 0.7
scalar beta1 = 0.5

set obs 5000
gen Z = rnormal(0,1)
gen u = rnormal(0,1)
gen e = rnormal(0,1)
gen X = `zmult'*Z+`umult'*u
gen Y = beta0 + beta1*X + `beta2'*u + e
reg Y X
return scalar b1 = _b[X]
return scalar b2 = `beta2'
end

local i = 0
forvalues i = -1(0.2)1{
	simulate estbeta1=r(b1) estbeta2=r(b2), saving(data`i'.dta,replace) reps(60): endog, beta2(`i') zmult(.8) umult(.5)
}

clear
forvalues i = -1(0.2)1{
	append using data`i'.dta
	erase data`i'.dta
}

label var estbeta1 "Beta1" 
label var estbeta2 "Beta2"
scatter estbeta1 estbeta2, ///
 title("Scatterplot Estimates of Beta1 and Beta2") ///
 yline(0.5)
 
graph save Graph Figure1_Janko.gph, replace

clear

local i = 0
forvalues i = -1(0.2)1{
	simulate b1r=r(b1) b2r=r(b2), saving(data`i'.dta,replace) reps(60): endog, beta2(`i')zmult(.8) umult(.1)
	
}

clear
forvalues i = -1(0.2)1{
	append using data`i'.dta
	erase data`i'.dta
}

label var b1r "Beta1" 
label var b2r "Beta2"
scatter b1r b2r, ///
 title("Scatterplot Estimates of Beta1 and Beta2")  ///
 yline(0.5)
 
graph save Graph Figure2_Janko.gph, replace
