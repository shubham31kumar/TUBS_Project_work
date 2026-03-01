#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

// Assignment 2: LBM Wind Tunnel
// Right triangle obstacle with momentum exchange force measurement
//
// Wind tunnel layout:
// - Periodic boundaries: left <-> right (inlet/outlet)
// - Velocity bounce-back: top and bottom walls (u=U_inlet, v=0)
// - Delay bounce-back: triangle obstacle (static, u_body=v_body=0)
//
// --------
// -++++++-   top wall: velocity BB
// -++++++-   fluid domain + triangle obstacle
// -++++++-   bottom wall: velocity BB
// --------
// +: lattice node, -: ghost node
//
// geo flags: 0=fluid, 1=delayBB (obstacle), 2=velocityBB (wall)
//
// D2Q9 push scheme with A-B pattern


class windTunnelLBM {
    private:
    int rows;
    int cols;
    int d;
    float uBC;

    //D2Q9
    //  
    // \ | /  fmp f0p fpp         fmpS f0pS fppS
    // - + -  fm0 f00 fp0     ->  fm0S f00S fp0S
    // / | \  fmm f0m fpm         fmmS f0mS fpmS
    float omega;
    // Structure of Arrays, A-B pattern
    std::vector<std::vector<float>> fmp;
    std::vector<std::vector<float>> f0p;
    std::vector<std::vector<float>> fpp;
    std::vector<std::vector<float>> fm0;
    std::vector<std::vector<float>> f00;
    std::vector<std::vector<float>> fp0;
    std::vector<std::vector<float>> fmm;
    std::vector<std::vector<float>> f0m;
    std::vector<std::vector<float>> fpm;

    std::vector<std::vector<float>> fmpS;
    std::vector<std::vector<float>> f0pS;
    std::vector<std::vector<float>> fppS;
    std::vector<std::vector<float>> fm0S;
    std::vector<std::vector<float>> f00S;
    std::vector<std::vector<float>> fp0S;
    std::vector<std::vector<float>> fmmS;
    std::vector<std::vector<float>> f0mS;
    std::vector<std::vector<float>> fpmS;

    std::vector<std::vector<int>> geo;

    //macroscopic variables
    std::vector<std::vector<float>> u;
    std::vector<std::vector<float>> v;
    std::vector<std::vector<float>> rho;

    // Force accumulators (momentum exchange)
    float Fx;
    float Fy;

    pybind11::array_t<float> vectorToNumpy(const std::vector<std::vector<float>>& vec)
     const {pybind11::array_t<float> array({rows,cols});
     auto buf = array.request();
     float* ptr=static_cast<float*>(buf.ptr);
     for (int i=0; i<rows; ++i){
        for(int j=0;j<cols; ++j){
            ptr[i*cols+j]=vec[j][i];
        }
     }
     return array;
     }

    pybind11::array_t<int> geoToNumpy()
     const {pybind11::array_t<int> array({rows,cols});
     auto buf = array.request();
     int* ptr=static_cast<int*>(buf.ptr);
     for (int i=0; i<rows; ++i){
        for(int j=0;j<cols; ++j){
            ptr[i*cols+j]=geo[j][i];
        }
     }
     return array;
     }

    void init_geometry(){
        // Triangle front face at x = 3*d, vertically centered
        int cx = 3 * d;
        int cy = rows / 2;            // vertical center of domain
        int y_top = cy - d / 2;       // top of front edge
        int y_bot = cy + d / 2;       // bottom of front edge
        float triangle_length = 1.5f * d;

        for(int x = 0; x < cols; x++){
            for(int y = 0; y < rows; y++){
                rho[x][y] = 1.0f;
                u[x][y] = uBC;
                v[x][y] = 0.0f;

                // 1. Tunnel walls (top and bottom)
                if(y == 0 || y == rows - 1){
                    geo[x][y] = 2; // velocity BB wall
                }
                // 2. Right-angle triangular obstacle, centered at cy
                //    - Vertical front face at x=cx from y_top to y_bot (height d)
                //    - Horizontal bottom edge from cx to cx+1.5d at y=y_bot
                //    - Hypotenuse from (cx, y_top) to (cx+1.5d, y_bot)
                else if(x >= cx && x <= cx + (int)triangle_length){
                    // hypotenuse line: y_hyp goes from y_top at x=cx to y_bot at x=cx+1.5d
                    float y_hyp = (float)y_top + ((float)d) * ((float)(x - cx) / triangle_length);

                    if(y >= (int)y_hyp && y <= y_bot){
                        geo[x][y] = 1; // delay BB obstacle
                        u[x][y] = 0.0f;
                        v[x][y] = 0.0f;
                    } else {
                        geo[x][y] = 0;
                    }
                }
                else {
                    geo[x][y] = 0;
                }

                // Initialize equilibrium distributions for fluid nodes
                if(geo[x][y] == 0 || geo[x][y] == 2){
                    float ux = u[x][y];
                    float uy = v[x][y];
                    float u2 = ux*ux + uy*uy;
                    f00[x][y] = (4.0f/9.0f)  * rho[x][y] * (1.0f - 1.5f*u2);
                    fp0[x][y] = (1.0f/9.0f)  * rho[x][y] * (1.0f + 3.0f*ux + 4.5f*ux*ux - 1.5f*u2);
                    fm0[x][y] = (1.0f/9.0f)  * rho[x][y] * (1.0f - 3.0f*ux + 4.5f*ux*ux - 1.5f*u2);
                    f0p[x][y] = (1.0f/9.0f)  * rho[x][y] * (1.0f + 3.0f*uy + 4.5f*uy*uy - 1.5f*u2);
                    f0m[x][y] = (1.0f/9.0f)  * rho[x][y] * (1.0f - 3.0f*uy + 4.5f*uy*uy - 1.5f*u2);
                    fpp[x][y] = (1.0f/36.0f) * rho[x][y] * (1.0f + 3.0f*(ux+uy) + 4.5f*(ux+uy)*(ux+uy) - 1.5f*u2);
                    fmm[x][y] = (1.0f/36.0f) * rho[x][y] * (1.0f - 3.0f*(ux+uy) + 4.5f*(ux+uy)*(ux+uy) - 1.5f*u2);
                    fmp[x][y] = (1.0f/36.0f) * rho[x][y] * (1.0f + 3.0f*(-ux+uy) + 4.5f*(-ux+uy)*(-ux+uy) - 1.5f*u2);
                    fpm[x][y] = (1.0f/36.0f) * rho[x][y] * (1.0f + 3.0f*(ux-uy) + 4.5f*(ux-uy)*(ux-uy) - 1.5f*u2);
                } else {
                    // Obstacle: zero-velocity equilibrium
                    f00[x][y] = 4.0f/9.0f;
                    fp0[x][y] = 1.0f/9.0f;
                    fm0[x][y] = 1.0f/9.0f;
                    f0p[x][y] = 1.0f/9.0f;
                    f0m[x][y] = 1.0f/9.0f;
                    fpp[x][y] = 1.0f/36.0f;
                    fmm[x][y] = 1.0f/36.0f;
                    fmp[x][y] = 1.0f/36.0f;
                    fpm[x][y] = 1.0f/36.0f;
                }

                // Copy to S arrays
                f00S[x][y] = f00[x][y];
                fp0S[x][y] = fp0[x][y];
                fm0S[x][y] = fm0[x][y];
                f0pS[x][y] = f0p[x][y];
                f0mS[x][y] = f0m[x][y];
                fppS[x][y] = fpp[x][y];
                fmmS[x][y] = fmm[x][y];
                fmpS[x][y] = fmp[x][y];
                fpmS[x][y] = fpm[x][y];
            }
        }
    }

    public:
    windTunnelLBM(int c, int r, float om, int d_in, float uBC_in): 
    cols(c),rows(r),omega(om),d(d_in),uBC(uBC_in),
    u(c,std::vector<float>(r,0.0f)),
    v(c,std::vector<float>(r,0.0f)),
    rho(c,std::vector<float>(r,1.0f)),

    fmp(c,std::vector<float>(r,1.0f/36.0f)),
    f0p(c,std::vector<float>(r,1.0f/9.0f)),
    fpp(c,std::vector<float>(r,1.0f/36.0f)),
    fm0(c,std::vector<float>(r,1.0f/9.0f)),
    f00(c,std::vector<float>(r,4.0f/9.0f)),
    fp0(c,std::vector<float>(r,1.0f/9.0f)),
    fmm(c,std::vector<float>(r,1.0f/36.0f)),
    f0m(c,std::vector<float>(r,1.0f/9.0f)),
    fpm(c,std::vector<float>(r,1.0f/36.0f)),

    fmpS(c,std::vector<float>(r,1.0f/36.0f)),
    f0pS(c,std::vector<float>(r,1.0f/9.0f)),
    fppS(c,std::vector<float>(r,1.0f/36.0f)),
    fm0S(c,std::vector<float>(r,1.0f/9.0f)),
    f00S(c,std::vector<float>(r,4.0f/9.0f)),
    fp0S(c,std::vector<float>(r,1.0f/9.0f)),
    fmmS(c,std::vector<float>(r,1.0f/36.0f)),
    f0mS(c,std::vector<float>(r,1.0f/9.0f)),
    fpmS(c,std::vector<float>(r,1.0f/36.0f)),
    geo(c,std::vector<int>(r,0)),
    Fx(0.0f),Fy(0.0f)
    {
        init_geometry();
    }

    ~windTunnelLBM(){}

    pybind11::array_t<float> getRho() const {return vectorToNumpy(rho);}
    pybind11::array_t<float> getU() const {return vectorToNumpy(u);}
    pybind11::array_t<float> getV() const {return vectorToNumpy(v);}
    pybind11::array_t<int> getGeo() const {return geoToNumpy();}
    float getFx() const {return Fx;}
    float getFy() const {return Fy;}
    int getD() const {return d;}
    float getUBC() const {return uBC;}
    float getOmega() const {return omega;}

    void setEq(int x, int y,float rho0, float ux, float uy){
                    rho[x][y]=rho0;
                    u[x][y]=ux;
                    v[x][y]=uy;
                    float u2 = ux*ux + uy*uy;
                    f00[x][y] = (4.0f/9.0f)  * rho0 * (1.0f - 1.5f*u2);
                    fp0[x][y] = (1.0f/9.0f)  * rho0 * (1.0f + 3.0f*ux + 4.5f*ux*ux - 1.5f*u2);
                    fm0[x][y] = (1.0f/9.0f)  * rho0 * (1.0f - 3.0f*ux + 4.5f*ux*ux - 1.5f*u2);
                    f0p[x][y] = (1.0f/9.0f)  * rho0 * (1.0f + 3.0f*uy + 4.5f*uy*uy - 1.5f*u2);
                    f0m[x][y] = (1.0f/9.0f)  * rho0 * (1.0f - 3.0f*uy + 4.5f*uy*uy - 1.5f*u2);
                    fpp[x][y] = (1.0f/36.0f) * rho0 * (1.0f + 3.0f*(ux+uy) + 4.5f*(ux+uy)*(ux+uy) - 1.5f*u2);
                    fmm[x][y] = (1.0f/36.0f) * rho0 * (1.0f - 3.0f*(ux+uy) + 4.5f*(ux+uy)*(ux+uy) - 1.5f*u2);
                    fmp[x][y] = (1.0f/36.0f) * rho0 * (1.0f + 3.0f*(-ux+uy) + 4.5f*(-ux+uy)*(-ux+uy) - 1.5f*u2);
                    fpm[x][y] = (1.0f/36.0f) * rho0 * (1.0f + 3.0f*(ux-uy) + 4.5f*(ux-uy)*(ux-uy) - 1.5f*u2);
    }

    void setFluid(int x, int y){geo[x][y]=0;}
    void setDelayBB(int x, int y){geo[x][y]=1;}
    void setVelocityBB(int x, int y){geo[x][y]=2;}

    void run(int steps){
        for (int t=0; t<steps; t++){//time loop

            Fx=0.0f;
            Fy=0.0f;

            for( int x=1; x<cols-1;x++){
                for (int y=1; y<rows-1;y++){
                    if(geo[x][y]==0){//fluid node
                    rho[x][y]=f00[x][y]+(((fmm[x][y]+fpp[x][y])+(fmp[x][y]+fpm[x][y]))+((fm0[x][y]+fp0[x][y])+(f0p[x][y]+f0m[x][y])));
                    u[x][y]=((((-fmm[x][y]+fpp[x][y])+(-fmp[x][y]+fpm[x][y]))+((-fm0[x][y]+fp0[x][y]))))/rho[x][y];
                    v[x][y]=((((-fmm[x][y]+fpp[x][y])+(fmp[x][y]-fpm[x][y]))+((f0p[x][y]-f0m[x][y]))))/rho[x][y];
                    // push scheme collide+stream
                    fmmS[x-1][y-1]=fmm[x][y]+omega*((rho[x][y]*(1 - 3*u[x][y] + 3*(u[x][y]*u[x][y]))*(1 - 3*v[x][y] + 3*(v[x][y]*v[x][y])))/36.-fmm[x][y]);
                    f0mS[x][y-1]=f0m[x][y]+omega*(-0.05555555555555555*((-2 + 3*(u[x][y]*u[x][y]))*rho[x][y]*(1 + 3*(v[x][y]*v[x][y]) - 3*v[x][y]))-f0m[x][y]);
                    fpmS[x+1][y-1]=fpm[x][y]+omega*((rho[x][y]*(1 + 3*(u[x][y]*u[x][y]) + 3*u[x][y])*(1 + 3*(v[x][y]*v[x][y]) - 3*v[x][y]))/36.-fpm[x][y]);
                    fm0S[x-1][y]=fm0[x][y]+omega*(-0.05555555555555555*((-2 + 3*(v[x][y]*v[x][y]))*rho[x][y]*(1 + 3*(u[x][y]*u[x][y]) - 3*u[x][y]))-fm0[x][y]);
                    f00S[x][y]=f00[x][y]+omega*(((-2 + 3*(u[x][y]*u[x][y]))*(-2 + 3*(v[x][y]*v[x][y]))*rho[x][y])/9.-f00[x][y]);
                    fp0S[x+1][y]=fp0[x][y]+omega*(-0.05555555555555555*((-2 + 3*(v[x][y]*v[x][y]))*rho[x][y]*(1 + 3*(u[x][y]*u[x][y]) + 3*u[x][y]))-fp0[x][y]);
                    fmpS[x-1][y+1]=fmp[x][y]+omega*((rho[x][y]*(1 + 3*(u[x][y]*u[x][y]) - 3*u[x][y])*(1 + 3*(v[x][y]*v[x][y]) + 3*v[x][y]))/36.-fmp[x][y]);
                    f0pS[x][y+1]=f0p[x][y]+omega*(-0.05555555555555555*((-2 + 3*(u[x][y]*u[x][y]))*rho[x][y]*(1 + 3*(v[x][y]*v[x][y]) + 3*v[x][y]))-f0p[x][y]);
                    fppS[x+1][y+1]=fpp[x][y]+omega*((rho[x][y]*(1 + 3*(u[x][y]*u[x][y]) + 3*u[x][y])*(1 + 3*(v[x][y]*v[x][y]) + 3*v[x][y]))/36.-fpp[x][y]);
                    }
                    else if(geo[x][y]==1){//delay bounce-back: static obstacle
                    fmmS[x-1][y-1]=fpp[x][y];
                    f0mS[x][y-1]=f0p[x][y];
                    fpmS[x+1][y-1]=fmp[x][y];
                    fm0S[x-1][y]=fp0[x][y];
                    fp0S[x+1][y]=fm0[x][y];
                    fmpS[x-1][y+1]=fpm[x][y];
                    f0pS[x][y+1]=f0m[x][y];
                    fppS[x+1][y+1]=fmm[x][y];

                    // Momentum exchange: F = sum c_i * (f_in + f_out)
                    // For each direction from solid toward fluid neighbor:
                    //   f_out = population heading outward (toward fluid)
                    //   f_in  = population heading inward (from fluid, opposite direction)
                    // (+1,0)
                    if(x+1<cols-1 && geo[x+1][y]==0){
                        Fx += (fp0[x][y]+fm0[x][y]);
                    }
                    // (-1,0)
                    if(x-1>0 && geo[x-1][y]==0){
                        Fx -= (fm0[x][y]+fp0[x][y]);
                    }
                    // (0,+1)
                    if(y+1<rows-1 && geo[x][y+1]==0){
                        Fy += (f0p[x][y]+f0m[x][y]);
                    }
                    // (0,-1)
                    if(y-1>0 && geo[x][y-1]==0){
                        Fy -= (f0m[x][y]+f0p[x][y]);
                    }
                    // (+1,+1)
                    if(x+1<cols-1 && y+1<rows-1 && geo[x+1][y+1]==0){
                        Fx += (fpp[x][y]+fmm[x][y]);
                        Fy += (fpp[x][y]+fmm[x][y]);
                    }
                    // (-1,-1)
                    if(x-1>0 && y-1>0 && geo[x-1][y-1]==0){
                        Fx -= (fmm[x][y]+fpp[x][y]);
                        Fy -= (fmm[x][y]+fpp[x][y]);
                    }
                    // (+1,-1)
                    if(x+1<cols-1 && y-1>0 && geo[x+1][y-1]==0){
                        Fx += (fpm[x][y]+fmp[x][y]);
                        Fy -= (fpm[x][y]+fmp[x][y]);
                    }
                    // (-1,+1)
                    if(x-1>0 && y+1<rows-1 && geo[x-1][y+1]==0){
                        Fx -= (fmp[x][y]+fpm[x][y]);
                        Fy += (fmp[x][y]+fpm[x][y]);
                    }

                    }
                    else if(geo[x][y]==2){//velocity bounce-back: wall
                    fmmS[x-1][y-1]=fpp[x][y]-6*1.0/36.0*(u[x][y]+v[x][y]);
                    f0mS[x][y-1]=f0p[x][y]-6*1.0/9.0*v[x][y];
                    fpmS[x+1][y-1]=fmp[x][y]-6*1.0/36*(-u[x][y]+v[x][y]);
                    fm0S[x-1][y]=fp0[x][y]-6*1.0/9.0*u[x][y];
                    fp0S[x+1][y]=fm0[x][y]+6*1.0/9.0*u[x][y];
                    fmpS[x-1][y+1]=fpm[x][y]-6*1.0/36*( u[x][y]-v[x][y]);
                    f0pS[x][y+1]=f0m[x][y]+6*1.0/9.0*v[x][y];
                    fppS[x+1][y+1]=fmm[x][y]+6*1.0/36.0*(u[x][y]+v[x][y]);
                    }
                }
            }

            //Periodic boundary conditions (left <-> right)
            for (int x=1; x<cols-1;x++){
                f0pS[x][1]=f0pS[x][rows-1];
                fmpS[x][1]=fmpS[x][rows-1];
                fppS[x][1]=fppS[x][rows-1];

                f0mS[x][rows-2]=f0mS[x][0];
                fmmS[x][rows-2]=fmmS[x][0];
                fpmS[x][rows-2]=fpmS[x][0];
            }

            for (int y=1; y<rows-1;y++){
                fp0S[1][y]=fp0S[cols-1][y];
                fpmS[1][y]=fpmS[cols-1][y];
                fppS[1][y]=fppS[cols-1][y];

                fm0S[cols-2][y]=fm0S[0][y];
                fmmS[cols-2][y]=fmmS[0][y];
                fmpS[cols-2][y]=fmpS[0][y];
            }

            //corners
            fppS[1][1]=fppS[cols-1][rows-1];
            fpmS[1][rows-2]=fpmS[cols-1][0];
            fmpS[cols-2][1]=fmpS[0][rows-1];
            fmmS[cols-2][rows-2]=fmmS[0][0];

            std::swap(f00S,f00);
            std::swap(fp0S,fp0);
            std::swap(fm0S,fm0);
            std::swap(f0mS,f0m);
            std::swap(fpmS,fpm);
            std::swap(fmmS,fmm);
            std::swap(f0pS,f0p);
            std::swap(fppS,fpp);
            std::swap(fmpS,fmp);
        }
    }

};


PYBIND11_MODULE(windTunnelLBM,m){
    pybind11::class_<windTunnelLBM>(m,"windTunnelLBM")
    .def(pybind11::init<int,int,float,int,float>())
    .def("getRho",&windTunnelLBM::getRho)
    .def("getU",&windTunnelLBM::getU)
    .def("getV",&windTunnelLBM::getV)
    .def("getGeo",&windTunnelLBM::getGeo)
    .def("getFx",&windTunnelLBM::getFx)
    .def("getFy",&windTunnelLBM::getFy)
    .def("getD",&windTunnelLBM::getD)
    .def("getUBC",&windTunnelLBM::getUBC)
    .def("getOmega",&windTunnelLBM::getOmega)
    .def("setEq",&windTunnelLBM::setEq)
    .def("run",&windTunnelLBM::run)
    .def("setFluid",&windTunnelLBM::setFluid)
    .def("setDelayBB",&windTunnelLBM::setDelayBB)
    .def("setVelocityBB",&windTunnelLBM::setVelocityBB);
}
