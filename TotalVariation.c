/*
// Decomposition of f into u(smooth), v(piecewise smooth), and w(noise).
//
// f = u + v + w. 
//
// E(u,v) = \int (f-u-v)^2 + a * |\nabla u| + b * |\nabla v|^2
//
// [INPUT]
//
// prhs[0] : input image (double).
// prhs[1] : alpha. coefficient for the regularisation on u (double).
// prhs[2] : beta. coefficient for the regularisation on v (double).
// prhs[3] : number of iterations (int).
// prhs[4] : flag for graphical plotting (default = 0, plot = 1). (optional) 
// prhs[5] : time step (dt). default = 0.1 (optional)
// prhs[6] : epsilon (eps). default = 1e-20 (optional)
//
// [OUTPUT]
//
// plhs[0] : u.
// plhs[1] : v.
// plhs[2] : w.
// plhs[3] : energy (total).
// plhs[4] : energy (data fidelity).
// plhs[5] : energy (regularisation u).
// plhs[6] : energy (regularisation v).
//
// [USAGE]
// 
// [ u, v, e, e_data, e_reg ] = TotalVariation( im, alpha, beta, nIter, fPlot );
// [ u, v, e, e_data, e_reg ] = TotalVariation( im, 0.1, 0.1, 100, 1 );
// [ u, v, e, e_data, e_reg ] = TotalVariation( im, 0.1, 0.1, 100 );
*/

#include <string.h>

#include "mex.h"
#include "math.h"
#include "matrix.h"

#include "CImg.h"

#define T( U, r, c, nRow ) U[ (r) + (c) * nRow ]
// #define T( U, r, c, d, nRow, nCol ) U[ (r) + (c) * nRow + (d) * nRow * nCol ]

#ifndef MAX 
    #define MAX( a, b ) ( ( (a) > (b) ) ? (a) : (b) )
#endif

#ifndef MIN
    #define MIN( a, b ) ( ( (a) < (b) ) ? (a) : (b) )
#endif

using namespace std;
using namespace cimg_library;

void mexFunction(int nlhs, mxArray *plhs[ ],int nrhs, const mxArray *prhs[ ]) 
{
  /* input data. */
  double    *pImage;
  double    dA, dB;
  int       nIter, r, c;
  int       nRow, nCol, nLength;
  int       nPlot = 0;
  double    h = 1;
  double    dt = 0.001;
  double    eps = 1e-20;

  pImage    = (double*) mxGetData( prhs[0] );
  nRow      = (int) mxGetM( prhs[0] );
  nCol      = (int) mxGetN( prhs[0] );
  nLength   = nRow * nCol;
  
  dA        = (double) mxGetScalar( prhs[1] );
  dB        = (double) mxGetScalar( prhs[2] );
  nIter     = (int) mxGetScalar( prhs[3] );
  
  if( nrhs == 5 ) {

    nPlot   = (int) mxGetScalar( prhs[4] );
  }

  if( nrhs >= 6 ) {
    dt      = (double) mxGetScalar( prhs[5] );
    eps     = (double) mxGetScalar( prhs[6] );
  }

  fprintf(stdout, "row: %d, col: %d, length: %d\n", nRow, nCol, nLength );
  fprintf(stdout, "alpha: %f, beta: %f\n", dA, dB );
  fprintf(stdout, "iter: %d, plot: %d, dt: %f, eps: %f\n", nIter, nPlot, dt, eps );
  fflush (stdout);

  CImg <double>     I (pImage,nRow,nCol);
  CImg <double>     D (nRow,nCol);  // data term.
  CImg <double>     R (nRow,nCol);  // regularization term.
  CImg <double>     U (nRow,nCol);  // U: output. ( I = U + V + W )
  CImg <double>     V (nRow,nCol);  // V: output. ( I = U + V + W )
  CImg <double>     W (nRow,nCol);  // V: output. ( I = U + V + W )
  
  CImg_3x3 (N,double);
  
  double    *pErrTotal  = new double [nIter];
  double    *pErrData   = new double [nIter];
  double    *pErrReg1   = new double [nIter];
  double    *pErrReg2   = new double [nIter];
  double    eTotal, eData, eReg1, eReg2;

  U = I;
  V.fill (0);

  int   i;

  for (i = 0; i < nIter; i++ ) {

    eData   = 0;
    eReg1   = 0;
    eReg2   = 0;

    // minimise on U
    cimg_for3x3 (U,r,c,0,0,N,double) {

      D (r,c) = I (r,c) - V (r,c); 

      // ROF scheme.
      R (r,c) = (Nnc-Ncc) / sqrt (eps + pow (Nnc-Ncc,2) + pow ((Ncn-Ncp)/2,2)); 
      R (r,c) = R (r,c) - (Ncc-Npc) / sqrt (eps + pow (Ncc-Npc,2) + pow ((Npn-Npp)/2,2)); 
      R (r,c) = R (r,c) + (Ncn-Ncc) / sqrt (eps + pow (Ncn-Ncc,2) + pow ((Nnc-Npc)/2,2)); 
      R (r,c) = R (r,c) - (Ncc-Ncp) / sqrt (eps + pow (Ncc-Ncp,2) + pow ((Nnp-Npp)/2,2)); 
      
      U (r,c) = U (r,c) + dt * 2 * D (r,c) + dt * dA * R (r,c);
      
      eReg1 = eReg1 + sqrt (pow (Nnc-Ncc,2) + pow (Ncn-Ncc,2));
    }

  
    // minimise on V 
    cimg_for3x3 (V,r,c,0,0,N,double) {

      D (r,c) = I (r,c) - U (r,c); 
      
      R (r,c) = (Nnc - 2 * Ncc + Npc + Ncn - 2 * Ncc + Ncp); 
    
      V (r,c) = V (r,c) + dt * 2 * D (r,c) + dt * dB * 2 * R (r,c);
      W (r,c) = I (r,c) - U (r,c) - V (r,c);

      eData = eData + pow (D (r,c),2);
      eReg2 = eReg2 + pow (Nnc-Ncc,2) + pow (Ncn-Ncc,2);
    }
    
    eReg1 = dA * eReg1;
    eReg2 = dB * eReg2;
  
    pErrData[i]     = eData / nLength;
    pErrReg1[i]     = eReg1 / nLength;
    pErrReg2[i]     = eReg2 / nLength;
    pErrTotal[i]    = pErrData[i] + pErrReg1[i] + pErrReg2[i];
      
    fprintf (stdout, "i=%.4d, e=%.5f\n", i, pErrTotal[i]);
    fflush (stdout);
      
    if ((i>0) && (abs(pErrTotal[i-1] - pErrTotal[i]) < 1e-8)) {
 
      break;
    }
  }

  const mwSize  szImage[2]  = {nRow,nCol};
  const mwSize  szError[2]  = {1,i};

  plhs[0] = mxCreateNumericArray (2,szImage,(mxClassID)mxDOUBLE_CLASS,mxREAL);
  plhs[1] = mxCreateNumericArray (2,szImage,(mxClassID)mxDOUBLE_CLASS,mxREAL);
  plhs[2] = mxCreateNumericArray (2,szImage,(mxClassID)mxDOUBLE_CLASS,mxREAL);
  plhs[3] = mxCreateNumericArray (2,szError,(mxClassID)mxDOUBLE_CLASS,mxREAL );
  plhs[4] = mxCreateNumericArray (2,szError,(mxClassID)mxDOUBLE_CLASS,mxREAL );
  plhs[5] = mxCreateNumericArray (2,szError,(mxClassID)mxDOUBLE_CLASS,mxREAL );
  plhs[6] = mxCreateNumericArray (2,szError,(mxClassID)mxDOUBLE_CLASS,mxREAL );

  double    *pU             = (double*) mxGetData (plhs[0]);
  double    *pV             = (double*) mxGetData (plhs[1]);
  double    *pW             = (double*) mxGetData (plhs[2]);
  double    *pErrorTotal    = (double*) mxGetData (plhs[3]);
  double    *pErrorData     = (double*) mxGetData (plhs[4]);
  double    *pErrorReg1     = (double*) mxGetData (plhs[5]);
  double    *pErrorReg2     = (double*) mxGetData (plhs[6]);

  memcpy (pU, U.data(), U.size()*sizeof(double));
  memcpy (pV, V.data(), V.size()*sizeof(double));
  memcpy (pW, W.data(), W.size()*sizeof(double));
  memcpy (pErrorTotal, pErrTotal, i*sizeof(double));
  memcpy (pErrorData, pErrData, i*sizeof(double));
  memcpy (pErrorReg1, pErrReg1, i*sizeof(double));
  memcpy (pErrorReg2, pErrReg2, i*sizeof(double));

  delete [] pErrTotal;
  delete [] pErrData;
  delete [] pErrReg1;
  delete [] pErrReg2;
}
