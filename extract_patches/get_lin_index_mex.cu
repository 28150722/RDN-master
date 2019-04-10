#include "mex.h"
#include "cuda.h"
#include "gpu/mxGPUArray.h"
#include "cuda_runtime.h"
#include "tmwtypes.h"

// Block Size
static const int BS = 16;

__global__ void cuda_get_pp_by_pn (//const int      K,
                                   const int      L,
                                   const int      Pn,
                                   const int      N,
                                   const float    *p,
                                   //const float    *rcc,
                                   //const uint32_T *rci,
                                   float          *pp)
{
  //// make sure we're working within the range of pp

  // the index along dim1 of pp: 0 or 1
  int dim_pnt = threadIdx.z;

  // the index along dim2 of pp: ml
  int Ps2 = (2*Pn+1)*(2*Pn+1);
  int sub_ml    = threadIdx.x;
  int blkcnt_ml = blockIdx.x;
  int l        = BS*blkcnt_ml + sub_ml;
  if (l >= L ) return;

  // the index along dim3 of pp: n
  int sub_n    = threadIdx.y;
  int blkcnt_n = blockIdx.y;
  int n        = BS*blkcnt_n + sub_n;
  if (n >= N) return;


  if(dim_pnt == 0){
	  for(int i = 0; i < 2*Pn + 1; i++){
		  for(int j = 0; j < 2*Pn + 1; j++){
			*(pp + i*(2*Pn + 1) + j + 2*Ps2*l + 2*Ps2*L*n) = *(p + 2*l + 2*L*n) + i - Pn;
		  }
	  }
  }else if(dim_pnt == 1){
	  for(int i = 0; i < 2*Pn + 1; i++){
		  for(int j = 0; j < 2*Pn + 1; j++){
			*(pp + i*(2*Pn + 1) + j + Ps2 + 2*Ps2*l + 2*Ps2*L*n) = *(p + 1 + 2*l + 2*L*n) + j - Pn;
		  }
	  }
  }

  __syncthreads();

}

__global__ void cuda_get_ind_val (const int   H,
                                  const int   W,
                                  const int   Pn,
								  const int   L,
                                  const int   N,
                                  const float *I,
                                  const float *pp,
                                  float       *f)
{
  //// make sure we're working within the range of pp

  // the index along dim2 of pp: ml
  int sub_ml    = threadIdx.x;
  int blkcnt_ml = blockIdx.x;
  int l        = BS*blkcnt_ml + sub_ml;
  //int Ps2L = 2*Pn+1;
  if (l >= L) return;

  // the index along dim3 of pp: n
  int sub_n    = threadIdx.y;
  int blkcnt_n = blockIdx.y;
  int n        = BS*blkcnt_n + sub_n;
  if (n >= N) return;


  for(int i = 0; i < 2*Pn + 1; i++){
	  for(int j = 0; j < 2*Pn + 1; j++){
		int ind_x = *(pp + i*(2*Pn + 1) + j + (2*Pn + 1)*(2*Pn + 1)*2*l + (2*Pn + 1)*(2*Pn + 1)*2*L*n) - 1;
		int ind_y = *(pp + i*(2*Pn + 1) + j + (2*Pn + 1)*(2*Pn + 1) + (2*Pn + 1)*(2*Pn + 1)*2*l + (2*Pn + 1)*(2*Pn + 1)*2*L*n) - 1;
		ind_x = (ind_x < W) ? (ind_x) : (W - 1);
		ind_y = (ind_y < H) ? (ind_y) : (H - 1);
		ind_x = (ind_x > 0) ? (ind_x) : 0;
		ind_y = (ind_y > 0) ? (ind_y) : 0;
		*(f + i*(2*Pn + 1) + j + (2*Pn + 1)*(2*Pn + 1)*l + (2*Pn + 1)*(2*Pn + 1)*L*n) = *(I + W*H*n + ind_x*H + ind_y);
	  }
  }
  __syncthreads();

}


// [I,p,rcc,rci] = check_and_get_input(nin,in); Helper
void check_and_get_input (int              nin, 
                          mxArray    const *in[],
                          mxGPUArray const *&I, 
                          mxGPUArray const *&p)
						 // mxGPUArray const *&pn)
                          //mxGPUArray const *&rcc,
                          //mxGPUArray const *&rci)
{
  //if (nin != 4)
  // / mexErrMsgTxt("Incorrect arguments. [f,ind] = get_pixval(I, p, rcc, rci)");

  //// check if gpuArray
  if ( mxIsGPUArray( in[0] ) == 0 ) mexErrMsgTxt("I must be a gpuArray.");
  if ( mxIsGPUArray( in[1] ) == 0 ) mexErrMsgTxt("p must be a gpuArray.");
  //if ( mxIsGPUArray( in[2] ) == 0 ) mexErrMsgTxt("pn must be a gpuArray.");
  //if ( mxIsGPUArray( in[3] ) == 0 ) mexErrMsgTxt("rci must be a gpuArray."); 
  
  //// fetch the results
  I   = mxGPUCreateFromMxArray( in[0] );
  p   = mxGPUCreateFromMxArray( in[1] );
  //pn = mxGPUCreateFromMxArray( in[2] );
  //rci = mxGPUCreateFromMxArray( in[3] );
  //// check the types
  if (mxGPUGetClassID(I)   != mxSINGLE_CLASS ) mexErrMsgTxt("I must be the type single.");
  if (mxGPUGetClassID(p)   != mxSINGLE_CLASS ) mexErrMsgTxt("p must be the type single.");
  //if (mxGPUGetClassID(pn) != mxUINT32_CLASS) mexErrMsgTxt("pn must be the type unint32.");
  //if (mxGPUGetClassID(rci) != mxUINT32_CLASS ) mexErrMsgTxt("rci must be the type uint32.");
}

// pp = get_pp_by_rc(p,rcc,rci); Get all the points pp by random combination
// Get all the poins from the patch size
void get_pp_by_pn (mxGPUArray const *p, 
                   //mxGPUArray const *rcc, 
                   //mxGPUArray const *rci,
				   const int Pn,
                   mxGPUArray       *pp)
{
  //// raw pointer
  const float    *ptr_p   = (const float*)    ( mxGPUGetDataReadOnly(p) );
  //const float    *ptr_rcc = (const float*)    ( mxGPUGetDataReadOnly(rcc) );
  //const uint32_T *ptr_rci = (const uint32_T*) ( mxGPUGetDataReadOnly(rci) );
  //const uint32_T *ptr_pn = (const uint32_T *) (mxGPUGetDataReadOnly(pn));
  float          *ptr_pp  = (float*)          ( mxGPUGetData(pp) );

  //// auxiliary 
  //const int K  = *( 0 + mxGPUGetDimensions(rcc) ); // rcc [K, ML]
  //const int ML = *( 1 + mxGPUGetDimensions(rcc) );
  
  //const int Pn = ptr_pn[0];
  const int L  = *( 1 + mxGPUGetDimensions(p)   ); // p [2,L,N]
  const int N  = *( 2 + mxGPUGetDimensions(p)   ); // p [2,L,N]
  

  //// block and thread partition
  dim3 num_thd( BS, BS ,2);
  dim3 num_blk( (L+BS-1)/BS, (N+BS-1)/BS );

#ifndef NDEBUG
  mexPrintf("In get_pp_by_rc\n");
  //mexPrintf("K = %d\n", K);
  mexPrintf("L = %d\n", L);
  mexPrintf("N = %d\n", N);
#endif // !NDEBUG

  cuda_get_pp_by_pn<<<num_blk, num_thd>>>(L,Pn,N, ptr_p, ptr_pp);
}

// [f,ind] = get_ind_val(I,pp); Get the values and the index 
void get_ind_val (const int pn,
				  mxGPUArray const *I, 
                  mxGPUArray const *pp,
                  mxGPUArray       *f)
{
  // thread 
  dim3 num_thd(BS,BS);
  // block
  const int Ps2 = *( mxGPUGetDimensions(pp) ); // p [2,L,N]
  const int L = *( 2 + mxGPUGetDimensions(pp) );
  const int N  = *( 3 + mxGPUGetDimensions(pp) ); 
  dim3 num_blk( (L+BS-1)/BS, (N+BS-1)/BS );
  // image size
  const int H = *( 0 + mxGPUGetDimensions(I) ); // I [H, W, 3, N]
  const int W = *( 1 + mxGPUGetDimensions(I) ); 
  // raw pointer
  const float    *ptr_I   = (const float*) ( mxGPUGetDataReadOnly(I) );
  const float    *ptr_pp  = (const float*) ( mxGPUGetDataReadOnly(pp) );
  float          *ptr_f   = (float*)       ( mxGPUGetData(f) );

#ifndef NDEBUG
  mexPrintf("In get_ind_val\n");
  mexPrintf("H = %d\n", H);
  mexPrintf("W = %d\n", W);
  mexPrintf("N = %d\n", N);
#endif // !NDEBUG

  cuda_get_ind_val<<<num_blk, num_thd>>>(H,W,pn,L, N, ptr_I,ptr_pp, ptr_f);
}



// [f,ind] = get_pixval(I, p, rcc, rci)
// f:   [MLN]     features
// ind: [MLN]     the linear index
// I:   [H,W,3,N] image array
// p:   [2,L,N]   points
// rcc: [K, ML]   combination coefficients
// rci: [K, ML]   non zero elements index
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs,  mxArray const *prhs[])
{
  //// Prepare the Input
  mxGPUArray const *I;
  mxGPUArray const *p;
  //mxGPUArray const *rcc;
  //mxGPUArray const *rci;
  //mxGPUArray const *pn_; //patch size
  //check_and_get_input(nin, in,  I,p,rcc,rci);

  check_and_get_input(nrhs,prhs,I,p);

  //// Create the Output
  const mxGPUArray* ddd1 = mxGPUCreateFromMxArray(prhs[1]);
  const mwSize *dim_p = mxGPUGetDimensions(ddd1);

  const mxGPUArray* ddd2 = mxGPUCreateFromMxArray(prhs[2]);
  const mwSize *dim_pn = mxGPUGetDimensions(ddd2);
  mwSize pn = *(dim_pn);
  mwSize N = *(dim_p + 2); 
  mwSize L = *(dim_p + 1);
  mwSize dimo[3];

  mwSize ps = 2*pn+1;  //patch size
  dimo[0] = ps*ps;
  dimo[1] = L;
  dimo[2] = N;
  mxGPUArray *f   = mxGPUCreateGPUArray(3, dimo, mxSINGLE_CLASS, mxREAL, // [PsPsLN]
                                        MX_GPU_DO_NOT_INITIALIZE); 
  //plhs[0]          = mxGPUCreateMxArrayOnGPU(f);



  //// do the job

  // get all the points pp: [2, Pn*Pn*L, N]
  mwSize pp_dim[4];
  pp_dim[0] = ps*ps;
  pp_dim[1] = 2;
  pp_dim[2] = L;
  pp_dim[3] = N;
  mxGPUArray *pp = mxGPUCreateGPUArray (4, pp_dim, mxSINGLE_CLASS, mxREAL, // [MLN]
                                        MX_GPU_DO_NOT_INITIALIZE);
  // get all the points coordinates from origin images
  get_pp_by_pn (p,pn,pp);

  // get the linear index and the values 
  get_ind_val (pn, I,pp, f);
  plhs[0] = mxGPUCreateMxArrayOnGPU(f);
  // cleanup !!!
  mxGPUDestroyGPUArray(I);
  mxGPUDestroyGPUArray(p);
  mxGPUDestroyGPUArray(ddd1);
  mxGPUDestroyGPUArray(ddd2);
  mxGPUDestroyGPUArray(pp);
  mxGPUDestroyGPUArray(f);

  return;
}
