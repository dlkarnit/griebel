//
//  A numerical simulation of the Navier-Stokes equation derived from the
//  work of Michael Griebel "Numerical Simulation in Fluid Dynamics".
//  Developed to satisfy the requirements of MEEM 5240 - Computational
//  Fluid Dynamics and Heat Transfer at Michigan Technological University.
//
//  Developed with Visual Studio Community 2017
//
// Copyright 2016 Duncan L. Karnitz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http ://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "stdafx.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#include <direct.h>
#include <omp.h>
#include <windows.h>

static const double      TAU = 1.0; // safety factor for time step stability, in (0,1]

static const double      UIN = 100.0; // inlet velocity m/s
static const double      PIN = 0.0; // inlet pressure
static const double     POUT = 0.0; // outlet pressure

static const double Reynolds = 1.0e-3; // reynolds number for the flow
static const double Duration = 1.0; // sec

static const double        X = 2.0; // m
static const double        Y = 0.4; // m
static const double       DX = 0.025; // m

static const long  NPadInlet = 7; // number of cells to use for defining the inlet position

static const long       NMax = 1000000; // maximum number of time steps
static const long  WriteFreq = 1; // number of steps per data write

class T2D {
private:
	T2D(const T2D& copy);
	const T2D& operator=(const T2D& copy);

public:
	T2D(long nx, long ny, double t0)
		: vNX(nx), vNY(ny), vData(new double[nx*ny])
	{
		std::fill(vData, vData + (nx*ny), t0);
	}

	~T2D()
	{
		delete vData;
	}

	long size() const
	{
		return vNX*vNY;
	}

	double operator[](long i) const
	{
		return vData[i];
	}

	double& operator[](long i)
	{
		return vData[i];
	}

	long index(long i, long j) const
	{
		return i + vNX*j;
	}

	double operator()(long i, long j) const
	{
		return vData[i + vNX*j];
	}

	double& operator()(long i, long j)
	{
		return vData[i + vNX*j];
	}

	void print()
	{
		printf_s("\n");
		for (long j = vNY - 1; j >= 0; --j) {
			for (long i = 0; i < vNX; ++i) {
				printf_s("%f ", (*this)(i, j));
			}
			printf_s("\n");
		}
	}

private:
	long vNX;
	long vNY;
	double* vData;
};

std::pair<double,double> StableTimeStep(const T2D& u, const T2D& v, const long nx, const long ny, const double dx, const double dy, const double Re)
{
	double uMax = 0.0;
	for (long i = 0; i < u.size(); ++i) uMax = (std::max)(uMax, fabs(u[i]));

	double vMax = 0.0;
	for (long i = 0; i < v.size(); ++i) vMax = (std::max)(vMax, fabs(v[i]));

	// Stability condition
	const double dt = TAU*(std::min)((0.5*Re)/(1 / (dx*dx) + 1 / (dy*dy)), (std::min)(dx / uMax, dy / vMax)); // sec

	const double gamma = (std::max)(uMax*dt / dx, vMax*dt / dy);

	return std::make_pair(dt,gamma);
}

void SetBound(T2D& u, T2D& v, const long nx, const long ny)
{
	// Apply boundary conditions for outlet
#pragma omp parallel for
	for (long j = 1; j < ny - 1; ++j) {
		u(nx, j) = u(nx - 1, j);
	}
//#pragma omp parallel for
	//for (long j = 1; j < ny; ++j) {
	//	v(nx - 1, j) = v(nx - 2, j);
	//}

	// Apply boundary conditions for inlet
#pragma omp parallel for
	for (long j = NPadInlet; j < ny - NPadInlet; ++j) {
		u(0, j) = UIN;
		u(1, j) = UIN;
	}
//#pragma omp parallel for
	//for (long j = NPadInlet + 1; j < ny - NPadInlet; ++j) {
	//	v(0, j) = v(1, j);
	//}

	// Apply boundary conditions for no slip
#pragma omp parallel for
	for (long j = 0; j <= NPadInlet; ++j) { // Below inlet
		v(0, j) = -v(1, j);
	}
#pragma omp parallel for
	for (long j = ny - NPadInlet; j < ny; ++j) { // Above inlet
		v(0, j) = -v(1, j);
	}
#pragma omp parallel for
	for (long i = 0; i < nx; ++i) { // Top and bottom
		u(i, 0) = -u(i, 1);
		u(i, ny - 1) = -u(i, ny - 2);
	}

	// Apply solid surface boundary conditions
#pragma omp parallel for
	for (long i = 0; i < nx; ++i) { // Top and bottom
		v(i, 0) = 0.0;
		v(i, 1) = 0.0;
		v(i, ny) = 0.0;
		v(i, ny - 1) = 0.0;
	}
#pragma omp parallel for
	for (long j = 0; j < NPadInlet; ++j) { // Below inlet
		u(0, j) = 0.0;
		u(1, j) = 0.0;
	}
#pragma omp parallel for
	for (long j = ny - NPadInlet; j < ny; ++j) { // Above inlet
		u(0, j) = 0.0;
		u(1, j) = 0.0;
	}
}

void Sweep(const T2D& A, const T2D& b, T2D& x, double& normL2, const long nx, const long ny, const double dx, const double w)
{
	double norm = 0.0;

	for (long step = 0; step < 2; ++step) {
#pragma omp parallel for reduction(+:norm)
		for (long j = 1; j < ny - 1; ++j) {
			for (long i = 1; i < nx - 1; ++i) {
				if ((i + j) % 2 == step) {

					const double xPrev = x(i, j);

					const double sum =
						x(i - 1, j) +
						x(i + 1, j) +
						x(i, j - 1) +
						x(i, j + 1);

					const double xNew = (1.0 - w) * xPrev + (w / A(i, j)) * ((sum / (dx*dx)) - b(i,j));

					norm += (xNew - xPrev)*(xNew - xPrev);

					x(i, j) = xNew;
				}
			}
		}
	}

	normL2 = norm;
}

void WriteVTKFromMAC(const char* fname, long nx, long ny, double dx, const T2D& P, const T2D& U, const T2D& V, const double t)
{
	const long nXCells = nx - 2;
	const long nYCells = ny - 2;
	const long nXPoints = nXCells + 1;
	const long nYPoints = nYCells + 1;

	FILE* fp = NULL;
	errno_t open_status = fopen_s(&fp, fname, "w");

	fprintf_s(fp, "# vtk DataFile Version 3.0\n");
	fprintf_s(fp, "vtk output\n");
	fprintf_s(fp, "ASCII\n");
	fprintf_s(fp, "DATASET RECTILINEAR_GRID\n");
	fprintf_s(fp, "FIELD FieldData 1\n");
	fprintf_s(fp, "TIME 1 1 float\n");
	fprintf_s(fp, "%f\n", t);
	fprintf_s(fp, "DIMENSIONS %ld %ld %ld\n", nXPoints, nYPoints, 1l);

	fprintf_s(fp, "X_COORDINATES %ld float\n", nXPoints);
	for (long i = 1; i <= nXPoints; ++i) {
		fprintf_s(fp, "%f ", i*dx);
	}
	fprintf_s(fp, "\n");

	fprintf_s(fp, "Y_COORDINATES %ld float\n", nYPoints);
	for (long j = 1; j <= nYPoints; ++j) {
		fprintf_s(fp, "%f ", j*dx);
	}
	fprintf_s(fp, "\n");

	fprintf_s(fp, "Z_COORDINATES %ld float\n", 1l);
	fprintf_s(fp, "0\n");

	// Write pressure as cell data
	fprintf_s(fp, "CELL_DATA %ld\n", nXCells*nYCells);
	fprintf_s(fp, "FIELD FieldData 1\n");

	// Reverse loop order, format requires x to be fastest changing
	fprintf_s(fp, "p 1 %ld float\n", nXCells*nYCells);
	for (long j = 1; j <= nYCells; ++j) {
		for (long i = 1; i <= nXCells; ++i) {
			fprintf_s(fp, "%f ", P(i,j));
		}
	}
	fprintf_s(fp, "\n");

	fprintf_s(fp, "POINT_DATA %ld\n", nXPoints*nYPoints);
	fprintf_s(fp, "FIELD FieldData 2\n");

	// Write U values at cell vertices. Since U is stored at vertical
	// cell walls we need to do some interpolation along the Y direction.
	fprintf_s(fp, "u 1 %ld float\n", nXPoints*nYPoints);
	for (long j = 1; j <= nYPoints; ++j) {
		for (long i = 1; i <= nXPoints; ++i) {
			fprintf_s(fp, "%f ", 0.5*(U(i, j - 1) + U(i, j)));
		}
	}
	fprintf_s(fp, "\n");

	// Write V values at cell vertices. Since V is stored at horizontal
	// cell walls we need to do some interpolation along the X direction.
	fprintf_s(fp, "v 1 %ld float\n", nXPoints*nYPoints);
	for (long j = 1; j <= nYPoints; ++j) {
		for (long i = 1; i <= nXPoints; ++i) {
			fprintf_s(fp, "%f ", 0.5*(V(i, j) + V(i - 1, j)));
		}
	}
	fprintf_s(fp, "\n");

	errno_t close_status = fclose(fp);
}

int main()
{
	SetCurrentDirectoryW(L"D:\\nozzle");

	omp_set_num_threads(2);

	char fname[128];

	// Stability requires dx < 2/k where k is velocity of transport
	const double dx = DX;
	const double dy = DX;

	const double  Re = Reynolds;
	const double Rei = 1 / Re;

	const long nx = static_cast<long>(X / dx);
	const long ny = static_cast<long>(Y / dy);

	if (dx != dy) {
		printf_s("Invalid assumption: dx == dy");
		return 1;
	}

	T2D p(nx, ny, 0.0);
	T2D u(nx + 1, ny, 0.0);
	T2D v(nx, ny + 1, 0.0);

	T2D f(nx + 1, ny, 0.0);
	T2D g(nx, ny + 1, 0.0);

	// Define LHS diagonal of 2D Laplacian
	T2D A(nx, ny, 4.0 / (dx*dx));
	T2D b(nx, ny, 0.0);

	// Define parameters for SOR
	double normL2 = 0.0;
	const long nSweeps = nx*ny*1000;
	const double w = 1.75;

	// Define parametrs for time stepping loop
	long n = 0;
	double t = 0.0;
	printf_s("Step    Time     Percent Complete           (dt,gamma)\n");

	while (t <= Duration && n <= NMax) {

		// Set boundaries before determining stable time step size
		SetBound(u, v, nx, ny);

		std::pair<double,double> params = StableTimeStep(u, v, nx, ny, dx, dy, Re);
		const double    dt = params.first;
		const double gamma = params.second;

		printf_s("%ld     %f       %f           (%f,%f)\n", n, t, t / Duration, dt, gamma);

		// Update F
#pragma omp parallel for
		for (long j = 1; j < ny - 1; ++j) {
			for (long i = 1; i < nx; ++i) {
				const double d2udx2 = (1 / (dx*dx))*(u(i + 1, j) - 2 * u(i, j) + u(i - 1, j));
				const double d2udy2 = (1 / (dy*dy))*(u(i, j + 1) - 2 * u(i, j) + u(i, j - 1));

				const double du2dx = (0.25 / dx)*((u(i, j) + u(i + 1, j))*(u(i, j) + u(i + 1, j)) - (u(i - 1, j) + u(i, j))*(u(i - 1, j) + u(i, j)))
					+ (0.25*gamma / dx)*(fabs(u(i, j) + u(i + 1, j))*(u(i, j) - u(i + 1, j)) - fabs(u(i - 1, j) + u(i, j))*(u(i - 1, j) - u(i, j)));

				const double duvdy = (0.25 / dy)*(((v(i - 1, j + 1) + v(i, j + 1))*(u(i, j) + u(i, j + 1))) - ((v(i - 1, j) + v(i, j))*(u(i, j - 1) + u(i, j))))
					+ (0.25*gamma / dy)*(fabs(v(i - 1, j + 1) + v(i, j + 1))*(u(i, j) - u(i, j + 1)) - fabs(v(i - 1, j) + v(i, j))*(u(i, j - 1) - u(i, j)));

				f(i, j) = u(i, j) + dt*(Rei*(d2udx2 + d2udy2) - du2dx - duvdy);
			}
		}
#pragma omp parallel for
		for (long i = 1; i < nx; ++i) {
			f(i, 0) = u(i, 0);
			f(i, ny - 1) = u(i, ny - 1);
		}
#pragma omp parallel for
		for (long j = 1; j < ny - 1; ++j) {
			f(0, j) = u(0, j);
			f(nx, j) = u(nx, j);
		}

		// Update G
#pragma omp parallel for
		for (long j = 1; j < ny; ++j) {
			for (long i = 1; i < nx - 1; ++i) {
				const double d2vdx2 = (1 / (dx*dx))*(v(i + 1, j) - 2 * v(i, j) + v(i - 1, j));
				const double d2vdy2 = (1 / (dy*dy))*(v(i, j + 1) - 2 * v(i, j) + v(i, j - 1));

				const double duvdx = (0.25 / dx)*(((u(i + 1, j) + u(i + 1, j - 1))*(v(i, j) + v(i + 1, j))) - ((u(i, j - 1) + u(i, j))*(v(i - 1, j) + v(i, j))))
					+ (0.25*gamma / dx)*(fabs(u(i + 1, j) + u(i + 1, j - 1))*(v(i, j) - v(i + 1, j)) - fabs(u(i, j - 1) + u(i, j))*(v(i - 1, j) - v(i, j)));

				const double dv2dy = (0.25 / dy)*((v(i, j) + v(i, j + 1))*(v(i, j) + v(i, j + 1)) - (v(i, j - 1) + v(i, j))*(v(i, j - 1) + v(i, j)))
					+ (0.25*gamma / dy)*(fabs(v(i, j) + v(i, j + 1))*(v(i, j) - v(i, j + 1)) - fabs(v(i, j - 1) + v(i, j))*(v(i, j - 1) - v(i, j)));

				g(i, j) = v(i, j) + dt*(Rei*(d2vdx2 + d2vdy2) - duvdx - dv2dy);
			}
		}
#pragma omp parallel for
		for (long i = 1; i < nx - 1; ++i) {
			g(i, 0) = v(i, 0);
			g(i, ny) = v(i, ny);
		}
#pragma omp parallel for
		for (long j = 1; j < ny; ++j) {
			g(0, j) = v(0, j);
			g(nx - 1, j) = v(nx - 1, j);
		}

		// Update RHS
#pragma omp parallel for
		for (long j = 1; j < ny - 1; ++j) {
			for (long i = 1; i < nx - 1; ++i) {
				b(i, j) = (1 / dt)*((f(i + 1, j) - f(i, j))/dx + (g(i, j + 1) - g(i, j))/dy);
			}
		}

		// Solve P
		for (long s = 1; s <= nSweeps; ++s) {

			// Below inlet
#pragma omp parallel for
			for (long j = 1; j < NPadInlet; ++j) {
				p(0, j) = p(1, j);
			}

			// Inlet
#pragma omp parallel for
			for (long j = NPadInlet; j < ny - NPadInlet; ++j) {
				p(0, j) = p(1, j);
			}

			// Above inlet
#pragma omp parallel for
			for (long j = ny - NPadInlet; j < ny - 1; ++j) {
				p(0, j) = p(1, j);
			}

			// Outlet
#pragma omp parallel for
			for (long j = 1; j < ny - 1; ++j) { // Outlet
				p(nx - 1, j) = POUT;
			}

			// Reflective
#pragma omp parallel for
			for (long i = 1; i < nx - 1; ++i) {
				p(i, 0) = p(i, 1);
				p(i, ny - 1) = p(i, ny - 2);
			}

			Sweep(A, b, p, normL2, nx, ny, dx, w);

			if (normL2 < 5e-15) break;
		}

		// Update U
#pragma omp parallel for
		for (long j = 1; j < ny - 1; ++j) {
			for (long i = 1; i < nx; ++i) {
				u(i, j) = f(i, j) - dt*(p(i, j) - p(i - 1, j))/dx;
			}
		}

		// Update V
#pragma omp parallel for
		for (long j = 1; j < ny; ++j) {
			for (long i = 1; i < nx - 1; ++i) {
				v(i, j) = g(i, j) - dt*(p(i, j) - p(i, j - 1))/dy;
			}
		}

		SetBound(u, v, nx, ny);

		if (n % WriteFreq == 0) {
			sprintf_s(fname, "nozzle_%ld.vtk", n);
			WriteVTKFromMAC(fname, nx, ny, dx, p, u, v, t);
		}

		t += dt;
		n += 1;
	}

	return 0;
}


