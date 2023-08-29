double fn(double phiLH, void *params){
	struct fn_params *p = (struct fn_params *) params;
	
	double thetaLH = p->thetaLH;
	double psiLH = p->psiLH;
	double Bcube_center_x = p->Bcube_center_x;
	double Bcube_center_y = p->Bcube_center_y;
	double Bcube_center_z = p->Bcube_center_z;
	double center_vec_x = p->center_vec_x;
	double center_vec_y = p->center_vec_y;
	double center_vec_z = p->center_vec_z;
	double random_vec_x = p->random_vec_x;
	double random_vec_y = p->random_vec_y;
	double random_vec_z = p->random_vec_z;
	
	// rhat is the unit vector along the line of sight in the local triad, we calculate it from the center_vec
	// hatZ is the unit vector along the local magfield
	// vecY will be the perpendicular vector to hatZ, calculated from the cross prod with a random vector
	// hatY is the unit vector of vecY
	int j;
	double Bcube_center[3]; Bcube_center[0]=Bcube_center_x ; Bcube_center[1]=Bcube_center_y ; Bcube_center[2]=Bcube_center_z ;
	double center_vec[3] ; center_vec[0] = center_vec_x; center_vec[1] = center_vec_y; center_vec[2] = center_vec_z;
	double random_vec[3] ; random_vec[0] = random_vec_x; random_vec[1] = random_vec_y; random_vec[2] = random_vec_z;
	
	double hatZ[3], rhat[3], local_B_proj[3], hatY[3], vecY[3], Lhat0[3], Lhat[3], filament_vec_proj[3], cross_product[3];
	double local_magfield_mod = 0.0, centers_mod=0.0, vecY_mod=0.0;
	long double dot_product=0.0, dot_product2=0.0 ;
	for(j=0;j<3;j++) local_magfield_mod += pow(Bcube_center[j],2);
	for(j=0;j<3;j++) hatZ[j] = Bcube_center[j] / sqrt(local_magfield_mod) ;
	for(j=0;j<3;j++) centers_mod += pow(center_vec[j],2) ;
	for(j=0;j<3;j++) rhat[j] = center_vec[j] / sqrt(centers_mod) ;
	for(j=0;j<3;j++) dot_product += Bcube_center[j] * rhat[j] ;
	for(j=0;j<3;j++) local_B_proj[j] = Bcube_center[j]  - dot_product * rhat[j] ;
	// this is cross product between the random vector and the vector along the magnetic field. By definition, vecY is perp to the vector along the local magfield
	vecY[0] = (hatZ[1]*random_vec[2] - hatZ[2]*random_vec[1]) ;
	vecY[1] = (hatZ[2]*random_vec[0] - hatZ[0]*random_vec[2]) ;
	vecY[2] = (hatZ[0]*random_vec[1] - hatZ[1]*random_vec[0]) ;
	for(j=0;j<3;j++) vecY_mod += pow(vecY[j],2) ;
	for(j=0;j<3;j++) hatY[j] = vecY[j] / sqrt(vecY_mod) ;
	// rotate hatZ around hatY by theta_LH using Rodrigues formula
	cross_product[0] = (hatY[1]*hatZ[2] - hatY[2]*hatZ[1]) ;
	cross_product[1] = (hatY[2]*hatZ[0] - hatY[0]*hatZ[2]) ;
	cross_product[2] = (hatY[0]*hatZ[1] - hatY[1]*hatZ[0]) ;
	dot_product = 0.0 ;
	for(j=0;j<3;j++) dot_product += hatY[j]*hatZ[j] ;
	for(j=0;j<3;j++) Lhat0[j] = hatZ[j]*cos(thetaLH) + cross_product[j]*sin(thetaLH) + hatY[j]*dot_product*(1.0 - cos(thetaLH));
	// We rotate Lhat0 around hatZ by phi using Rodrigues formula
	cross_product[0] = (hatZ[1]*Lhat0[2] - hatZ[2]*Lhat0[1]) ;
	cross_product[1] = (hatZ[2]*Lhat0[0] - hatZ[0]*Lhat0[2]) ;
	cross_product[2] = (hatZ[0]*Lhat0[1] - hatZ[1]*Lhat0[0]) ;
	dot_product = 0.0 ;
	for(j=0;j<3;j++) dot_product += hatZ[j]*Lhat0[j] ;
	for(j=0;j<3;j++) Lhat[j] = Lhat0[j]*cos(phiLH) + cross_product[j]*sin(phiLH) + hatZ[j]*dot_product*(1.0 - cos(phiLH)) ;
	// now Lhat is the vector of the filament
	// project the vector along the long axis of the filament (Lhat) towards rhat
	dot_product = 0.0 ;
	for(j=0;j<3;j++) dot_product += Lhat[j]*rhat[j] ;
	for(j=0;j<3;j++) filament_vec_proj[j] = Lhat[j] - dot_product * rhat[j] ;
	// this is to calculate the angle between filament_vec_proj and local_B_proj, i.e. psiLH
	cross_product[0] = (filament_vec_proj[1]*local_B_proj[2] - filament_vec_proj[2]*local_B_proj[1]) ;
	cross_product[1] = (filament_vec_proj[2]*local_B_proj[0] - filament_vec_proj[0]*local_B_proj[2]) ;
	cross_product[2] = (filament_vec_proj[0]*local_B_proj[1] - filament_vec_proj[1]*local_B_proj[0]) ;
	dot_product = 0.0 ;
	for(j=0;j<3;j++) dot_product += rhat[j]*cross_product[j] ;
	for(j=0;j<3;j++) dot_product2 += filament_vec_proj[j]*local_B_proj[j];
	return (double) atan2l(dot_product,dot_product2) - psiLH ;
}